# Copyright (c) 2025 HOUMO AI
#
# File: build.py
# Description:
#  Qwen2.5-VL  Model Build and Test Tool - Python script for building and testing
# Qwen2.5-VL models.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# SPDX-License-Identifier: Apache-2.0
import os
import numpy as np
import time
import psutil
import threading
import multiprocessing
import argparse

import logging

logging.basicConfig(level="INFO")

HOUMO_TARGET = os.getenv("HOUMO_TARGET")
assert HOUMO_TARGET in ["xh2"], f"Unsupported HOUMO_TARGET: {HOUMO_TARGET}"

HOUMO_CORE_NUM = os.getenv("HOUMO_CORE_NUM", 2)
GOLDEN_THRESH = 0.98


def sanitize_name(name: str):
    return name.replace(":", "_").replace("/", "_")


def cosine_distance(data1, data2):
    if data1.shape != data2.shape:
        print(f"[error] shape not equal {data1.shape} vs {data2.shape}")
        return -1
    v1_d = data1.flatten().astype("float64")
    v2_d = data2.flatten().astype("float64")
    v1_d[v1_d == np.inf] = np.finfo(np.float16).max
    v2_d[v2_d == np.inf] = np.finfo(np.float16).max
    v1_d[v1_d == -np.inf] = np.finfo(np.float16).min
    v2_d[v2_d == -np.inf] = np.finfo(np.float16).min
    v1_norm = v1_d / np.linalg.norm(v1_d)
    v2_norm = v2_d / np.linalg.norm(v2_d)
    cosine_dist = np.dot(v1_norm, v2_norm)
    if np.isnan(cosine_dist):
        return -1
    return cosine_dist


class ProcessMemoryMonitor:
    """
    Monitors the memory usage of the current Python process in real-time using psutil.
    """

    def __init__(self, interval=2, log_file=None):
        """
        Initializes the monitor.
        Args:
            interval (int): Time between measurements in seconds.
            log_file (str, optional): Path to a file to log results. If None, prints to console.
        """
        self.process = psutil.Process(os.getpid())
        self.interval = interval
        self.log_file = log_file
        self.is_monitoring = False
        self.peak_memory_mb = 0

    def get_memory_info(self):
        """
        Gets current memory usage information.
        Returns:
            dict: A dictionary containing memory usage data.
        """
        memory_info = self.process.memory_info()
        rss_mb = memory_info.rss / (1024 * 1024)  # Resident Set Size in MB
        percent = self.process.memory_percent()  # Percentage of system memory
        return {"rss_mb": rss_mb, "percent": percent}

    def start(self):
        """Starts the monitoring loop in a separate daemon thread."""
        self.is_monitoring = True
        self.peak_memory_mb = 0
        self.monitor_thread = threading.Thread(target=self._monitor_loop)
        self.monitor_thread.daemon = True  # Thread will exit when main program does
        self.monitor_thread.start()
        print(f"Memory monitoring started (interval: {self.interval}s)")

    def _monitor_loop(self):
        """The internal loop that runs in the thread."""
        while self.is_monitoring:
            mem_info = self.get_memory_info()
            self.peak_memory_mb = max(self.peak_memory_mb, mem_info["rss_mb"])

            timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
            log_message = f"{timestamp} - RSS: {mem_info['rss_mb']:.2f} MB, System%: {mem_info['percent']:.2f}%"

            # Output to console or file
            if self.log_file:
                with open(self.log_file, "a") as f:
                    f.write(log_message + "\n")

            time.sleep(self.interval)

    def stop(self):
        """Stops the monitoring loop and prints peak usage."""
        self.is_monitoring = False
        if hasattr(self, "monitor_thread"):
            self.monitor_thread.join(
                timeout=1
            )  # Wait a moment for the thread to finish
        print(f"[Monitoring stopped. Peak RSS: {self.peak_memory_mb:.2f} MB]")


def _validate_adjust_flash_attention(flash_vals: tuple, context_length: int) -> tuple:
    """Validates and adjusts FlashAttention parameter values."""
    llm_val, vit_val = flash_vals

    # Validate LLM (Prefill & Decode) FlashAttention parameter
    # Values: 0=off, 1/2=on
    if llm_val not in [0, 1, 2]:
        raise ValueError(
            f"Prefill&Decode FlashAttention values only support 0/1/2, current value:{llm_val}"
        )

    # Validate ViT (Vision Transformer) FlashAttention parameter
    # Values: 0=off, 1=on
    if vit_val not in [0, 1]:
        raise ValueError(
            f"ViT FlashAttention values only support 0/1, current value:{vit_val}"
        )

    if context_length < 2048:
        llm_val = 0

    return (llm_val, vit_val)


def get_args() -> argparse.Namespace:
    """Parse commandline."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_dir",
        dest="model_dir",
        type=str,
        default=os.path.join("output", HOUMO_TARGET, "hmquant"),
        help="path to the model dir",
    )
    parser.add_argument(
        "--model_name",
        dest="model_name",
        type=str,
        default="qwen2.5-vl",
        help="output houmo model name",
    )
    parser.add_argument(
        "--batch",
        dest="batch",
        type=int,
        default=1,
        help="batch size",
    )
    parser.add_argument(
        "--j",
        dest="j",
        type=int,
        default=multiprocessing.cpu_count(),
        help="build parallel jobs",
    )
    parser.add_argument(
        "--ncore",
        dest="ncore",
        type=int,
        default=HOUMO_CORE_NUM,
        help="core number",
    )
    parser.add_argument(
        "--context_length",
        dest="context_length",
        type=int,
        default=2048,
        help="context_length",
    )
    parser.add_argument(
        "--ndevice",
        dest="ndevice",
        type=int,
        default=1,
        help="device number",
    )
    parser.add_argument(
        "--model_size",
        dest="model_size",
        type=str,
        default="7b",
        choices=["3b", "7b"],
        help="model size",
    )
    parser.add_argument(
        "--stage",
        dest="stage",
        type=str,
        default="build",
        choices=["build", "test", "all"],
        help="build stage",
    )
    parser.add_argument(
        "--output_dir",
        dest="output_dir",
        type=str,
        default=os.path.join("output", HOUMO_TARGET),
        help="build output dir",
    )
    parser.add_argument(
        "--prefill_length",
        dest="prefill_length",
        type=int,
        default=256,
        help="prefill_length",
    )
    parser.add_argument(
        "--flash_attention",
        dest="flash_attention",
        nargs=2,
        type=int,
        default=(0, 1),
        help="FlashAttention optimization switches: "
        "1st int = prefill/decode model switch (0=off, 1/2=on), "
        "2nd int = ViT model switch (0=off, 1=on); "
        "e.g., --flash_attention 0 1 (prefill&decode=0, ViT=1)",
    )

    args = parser.parse_args()
    args.flash_attention = _validate_adjust_flash_attention(
        args.flash_attention, args.context_length
    )
    return args


def build_llm(
    model_name,
    model_dir,
    model_path,
    output_dir,
    profile,
    ncore,
    ndevice,
    context_length,
    j,
    batch=None,
    tso=False,
    flash_attention=0,
    prefill_length=0,
):
    import tcim
    import json

    kwargs = {}
    custom_msg = {}

    kwargs["modify_llm"] = {}
    kwargs["enable_xh2_stable_output"] = tso
    if prefill_length:
        kwargs["modify_llm"]["fill-length"] = prefill_length
        custom_msg["prefill_length"] = prefill_length
    if flash_attention:
        kwargs["flash_attention"] = flash_attention
        custom_msg["flash_attention"] = flash_attention
    if ndevice:
        kwargs["ndevice"] = ndevice
    if batch:
        kwargs["modify_llm"]["batch"] = batch
        custom_msg["batch"] = batch
    if context_length:
        kwargs["modify_llm"]["context-length"] = context_length
        custom_msg["context_length"] = context_length
    kwargs["custom_msg"] = json.dumps(custom_msg, ensure_ascii=False)

    start = time.time()
    print(f"\n===> {model_name} build start...\n kwargs:{kwargs}")
    decode_model = os.path.join(model_dir, model_path)
    tcim.build_from_hmonnx(
        decode_model,
        weights=os.path.join(model_dir, "weight.npy"),
        output_name=model_name,
        ncore=ncore,
        target=HOUMO_TARGET,
        output_dir=output_dir,
        work_dir=os.path.join(output_dir, "tcim", model_name),
        llm_opt=True,
        j=j,
        **kwargs,
    )
    profile["build"] = time.time() - start
    print(f'{model_name} build completed in {profile["build"]:.3f} s.', flush=True)


def build_vit(
    model_name, model_dir, model_path, output_dir, profile, ncore, j, flash_attention
):
    import tcim

    kwargs = {}
    if HOUMO_TARGET == "xh2" and flash_attention:
        import json

        kwargs["flash_attention"] = flash_attention
        custom_msg = {}
        custom_msg["flash_attention"] = flash_attention
        kwargs["custom_msg"] = json.dumps(custom_msg, ensure_ascii=False)

    start = time.time()
    print(f"\n===> {model_name} build start... \n kwargs:{kwargs}")
    decode_model = os.path.join(model_dir, model_path)
    tcim.build_from_hmonnx(
        decode_model,
        weights=os.path.join(model_dir, "weight.npy"),
        output_name=model_name,
        ncore=ncore,
        target=HOUMO_TARGET,
        output_dir=output_dir,
        work_dir=os.path.join(output_dir, "tcim", model_name),
        j=j,
        **kwargs,
    )
    profile["build"] = time.time() - start
    print(f'{model_name} build completed in {profile["build"]:.3f} s.', flush=True)


def test(model_name, model_dir, output_dir, profile, batch=1, prefix=None):
    import tcim_lite

    print(f"\n===> {model_name} test start...")
    # load model
    model_path = os.path.join(output_dir, f"{model_name}.hmm")
    start = time.time()
    print(model_path)
    module = tcim_lite.runtime.load(model_path)
    profile["load"] = time.time() - start
    print(f'{model_name} load completed in {profile["load"]:.3f} s.', flush=True)

    # set input
    profile["set_input"] = 0
    if prefix is None:
        prefix = model_name
    input_num = module.get_num_inputs()
    for id in range(input_num):
        input_name = module.get_input_name(id)
        input_info = module.get_input_info(input_name)
        print(
            f"input_info[{input_name}] shape = {input_info.shape}, dtype = {input_info.dtype}, format = {input_info.format.name}"
        )
        input_data_path = os.path.join(
            model_dir, f"hmquant_{prefix}_{sanitize_name(input_name)}_input.npy"
        )
        input_data = np.load(input_data_path).astype(input_info.dtype)
        input_data = np.concatenate([input_data for i in range(batch)], axis=0)
        print(
            f"golden input[{input_name}] shape = {input_data.shape}, dtype = {input_data.dtype}"
        )
        start = time.time()
        module.set_input(input_name, input_data)
        profile["set_input"] += time.time() - start
    print(
        f'{model_name} set {input_num} inputs completed in {profile["set_input"]*1000:.3f} ms.'
    )

    # infer model
    start = time.time()
    module.run()
    module.sync()
    profile["infer"] = time.time() - start
    print(f'{model_name} infer completed in {profile["infer"]*1000:.3f} ms.')

    # get output and compare with golden
    profile["get_output"] = 0
    result_check = True
    output_num = module.get_num_outputs()
    for id in range(output_num):
        output_name = module.get_output_name(id)
        output_info = module.get_output_info(output_name)
        print(
            f"output_info[{output_name}] shape = {output_info.shape}, dtype = {output_info.dtype}, format = {output_info.format.name}"
        )
        start = time.time()
        output_data = module.get_output(output_name).numpy()
        profile["get_output"] += time.time() - start
        print(
            f"output[{output_name}] shape = {output_data.shape}, dtype = {output_data.dtype}"
        )
        output_data_path = os.path.join(
            model_dir, f"hmquant_{prefix}_{sanitize_name(output_name)}_output.npy"
        )
        if os.path.exists(output_data_path):
            golden_output = np.load(output_data_path)
            golden_output = np.concatenate(
                [golden_output for i in range(batch)], axis=0
            )
        else:
            result_check = False
            print(
                f"[warning] compare canceled while golden data not found -> {output_data_path}"
            )
            continue
        if golden_output.shape == output_data.shape:
            cosine_dist = cosine_distance(golden_output, output_data)
            is_match = (golden_output == output_data).all()
            print(
                f"[compare] golden output [{output_name}] match={is_match}, similarity={cosine_dist:.6f}"
            )
            if is_match:
                continue
            if cosine_dist < GOLDEN_THRESH:
                result_check = False
        else:
            result_check = False
            print(
                f"[compare] golden output [{output_name}] shape not match {golden_output.shape} vs {output_data.shape}"
            )
    print(
        f'{model_name} get {output_num} ouputs completed in {profile["get_output"]*1000:.3f} ms.'
    )
    if not result_check:
        print("[error] result check failed.")
        exit(-1)
    print(f"<=== {model_name} test success.")


if __name__ == "__main__":
    # Create and start the monitor
    memory_monitor = ProcessMemoryMonitor(interval=2)
    memory_monitor.start()

    # parse args
    args = get_args()
    print(args)
    curdir = os.getcwd()
    model_dir = args.model_dir
    model_name = args.model_name
    if args.model_size == "3b":
        nblocks = 36
    elif args.model_size == "7b":
        nblocks = 28
    output_dir = args.output_dir
    ncore = args.ncore
    batch = args.batch
    ndevice = args.ndevice
    context_length = args.context_length
    j = args.j
    llm_flash_attention, vit_flash_attention = args.flash_attention
    profile = {}

    # build model
    if args.stage == "build" or args.stage == "all":
        import platform

        arch = platform.machine()
        if arch != "x86_64":
            print(f"[error] tcim not support platform: {arch}")
            exit(0)
        model_path = f"hmquant_{model_name}_with_act.onnx"
        build_vit(
            "qwen2.5-vl_visual",
            os.path.join(model_dir, "visual"),
            model_path,
            output_dir,
            profile,
            ncore,
            j,
            flash_attention=vit_flash_attention,
        )
        build_llm(
            "qwen2.5-vl_prefill",
            os.path.join(model_dir, "prefill"),
            model_path,
            output_dir,
            profile,
            ncore,
            ndevice,
            context_length,
            j,
            flash_attention=llm_flash_attention,
            prefill_length=args.prefill_length,
        )
        build_llm(
            "qwen2.5-vl_decode",
            os.path.join(model_dir, "decoder"),
            model_path,
            output_dir,
            profile,
            ncore,
            ndevice,
            context_length,
            j,
            flash_attention=llm_flash_attention,
        )

    # test model
    if args.stage == "test" or args.stage == "all":
        part_dir = os.path.join(model_dir, "prefill")
        test("qwen2.5-vl_prefill", part_dir, output_dir, profile, prefix=model_name)
        part_dir = os.path.join(model_dir, "decoder")
        test("qwen2.5-vl_decode", part_dir, output_dir, profile, prefix=model_name)
        part_dir = os.path.join(model_dir, "visual")
        test("qwen2.5-vl_visual", part_dir, output_dir, profile, prefix=model_name)

    memory_monitor.stop()
