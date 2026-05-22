# Copyright 2025 HOUMO AI
#
# File: build_multidevices_pipeline.py
# Description:
#   Build multi-devices pipeline for Qwen3 models.
#   This script implements a pipeline for splitting and building Qwen3 models across multiple devices.
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
import onnx
import psutil
import threading
import multiprocessing
import argparse
import glob
import logging
from pathlib import Path

logging.basicConfig(level="INFO")

HOUMO_TARGET = os.getenv("HOUMO_TARGET")
assert HOUMO_TARGET in ["xh2"], f"Unsupported HOUMO_TARGET: {HOUMO_TARGET}"

HOUMO_CORE_NUM = os.getenv("HOUMO_CORE_NUM", 2)
GOLDEN_THRESH = 0.98


def sanitize_name(name: str):
    return name.replace(":", "_").replace("/", "_")


def cosine_distance(data1, data2) -> float:
    """
    Calculate cosine distance between two data arrays.

    Args:
        data1: First data array
        data2: Second data array

    Returns:
        float: Cosine distance between the two arrays (-1 if shapes don't match or result is NaN)
    """
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


def get_args() -> argparse.Namespace:
    """Parse command line arguments for the multi-device pipeline builder."""
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
        default="qwen3",
        help="output houmo model name",
    )
    parser.add_argument(
        "--ncore",
        dest="ncore",
        type=int,
        default=HOUMO_CORE_NUM,
        help="core number",
    )
    parser.add_argument(
        "--ndevice",
        dest="ndevice",
        type=int,
        default=2,
        help="block number",
    )
    parser.add_argument(
        "--output_dir",
        dest="output_dir",
        type=str,
        default=os.path.join("output", HOUMO_TARGET),
        help="build output dir",
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
        "--context_length",
        dest="context_length",
        type=int,
        default=32768,
        help="context_length",
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
        type=int,
        default=2,
        choices=[0, 1, 2],
        help="flash attention optimization",
    )

    args = parser.parse_args()
    if args.context_length < 2048:
        args.flash_attention = 0
    args = parser.parse_args()
    return args


def get_n_blocks(model_path):
    model = onnx.load(model_path, load_external_data=False)
    total_inputs = model.graph.input
    # get kvcache num
    n_kvcaches, cache_start_idx = 0, 0
    for i in range(len(total_inputs)):
        if "kcache" in total_inputs[i].name or "vcache" in total_inputs[i].name:
            n_kvcaches += 1
        else:
            cache_start_idx += 1

    assert (
        n_kvcaches % 2 == 0
    ), f"n_kvcaches({n_kvcaches}) must be even, please check your model."
    n_blocks = n_kvcaches // 2

    return n_blocks, cache_start_idx


def clip(raw_path, split_paths, ndevice, model_name):
    """
    Split a model into two parts at a specific layer boundary.

    Args:
        raw_path (str): Path to the original ONNX model
        part1_path (str): Path to save the first part of the model
        part2_path (str): Path to save the second part of the model
        nblocks (int): Total number of blocks in the model
    """
    n_blocks, cache_start_idx = get_n_blocks(raw_path)
    assert (
        n_blocks % 2 == 0
    ), f"n_blocks({n_blocks}) must be even, please check your model."
    model = onnx.load(raw_path, load_external_data=False)

    mid_layer_names = []
    n_kvcache_per_stage = n_blocks // ndevice
    for i in range(ndevice):
        if i == ndevice - 1:
            mid_layer_names.append(model.graph.output[0].name)
        else:
            mid_layer_names.append(f"add_{2 * n_kvcache_per_stage * (i + 1) - 1}")

    stages_inputs = []
    stages_outputs = []
    for i in range(ndevice):
        outputs = []
        if i == ndevice - 1:
            outputs.append(model.graph.output[0].name)
        else:
            outputs.append(f"add_{2 * n_kvcache_per_stage * (i + 1) - 1}")
        stages_outputs.append(outputs)
    for i in range(ndevice):
        inputs = []
        if i == 0:
            for j in range(cache_start_idx):
                inputs.append(model.graph.input[j].name)
            for j in range(n_kvcache_per_stage):
                inputs.append(
                    f"model_layers_{j + i * n_kvcache_per_stage}_self_attn_kcache_input"
                )
                inputs.append(
                    f"model_layers_{j + i * n_kvcache_per_stage}_self_attn_vcache_input"
                )
        else:
            for j in range(cache_start_idx):
                if j == 0:
                    inputs.append(stages_outputs[i - 1][0])
                else:
                    inputs.append(model.graph.input[j].name)
            for j in range(n_kvcache_per_stage):
                inputs.append(
                    f"model_layers_{j + i * n_kvcache_per_stage}_self_attn_kcache_input"
                )
                inputs.append(
                    f"model_layers_{j + i * n_kvcache_per_stage}_self_attn_vcache_input"
                )
        stages_inputs.append(inputs)

    for i in range(ndevice):
        extract_file_path = split_paths[i]
        extract_dir = os.path.dirname(extract_file_path)
        os.makedirs(extract_dir, exist_ok=True)
        onnx.utils.extract_model(
            Path(raw_path),
            Path(extract_file_path),
            input_names=stages_inputs[i],
            output_names=stages_outputs[i],
            check_model=False,
            infer_shapes=False,
        )

        extract_model = onnx.load(extract_file_path)
        onnx.save(
            extract_model,
            extract_file_path,
            save_as_external_data=True,
            all_tensors_to_one_file=True,
            location=f"{Path(extract_file_path).stem}_external_data",
        )
        if os.path.exists(extract_file_path.replace(".onnx", ".onnx.data")):
            os.remove(extract_file_path.replace(".onnx", ".onnx.data"))
        print(f"save clip model part{i} to {extract_file_path}.")


def build(
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
    prefill_length=256,
):
    import tcim
    import json

    kwargs = dict()
    kwargs["modify_llm"] = dict()
    custom_msg = dict()
    custom_msg["prefill_length"] = prefill_length
    if batch:
        kwargs["modify_llm"]["batch"] = batch
        custom_msg["batch"] = batch
    if HOUMO_TARGET == "xh2":
        kwargs["enable_xh2_stable_output"] = tso
        kwargs["flash_attention"] = flash_attention
        custom_msg["flash_attention"] = flash_attention
        if ndevice:
            kwargs["ndevice"] = ndevice
        if context_length:
            kwargs["modify_llm"]["context-length"] = context_length
            custom_msg["context_length"] = context_length
    kwargs["custom_msg"] = json.dumps(custom_msg, ensure_ascii=False)

    start = time.time()
    print(f"\n===> {model_name} build start...")
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


if __name__ == "__main__":
    args = get_args()
    curdir = os.getcwd()
    model_dir = args.model_dir
    model_name = args.model_name
    output_dir = args.output_dir
    ncore = args.ncore
    batch = args.batch
    ndevice = args.ndevice
    profile = {}

    decode_dirs = sorted(
        path
        for path in glob.glob(os.path.join(model_dir, "*decode*"))
        if os.path.isdir(path)
    )
    if not decode_dirs:
        raise FileNotFoundError(
            f'No subdirectory containing "decode" found under: {model_dir}'
        )
    decode_dir = os.path.abspath(decode_dirs[0])
    prefill_dir = os.path.join(model_dir, "prefill")

    # clip model to 2 parts
    onnx_files = glob.glob(f"{prefill_dir}/hmquant_*.onnx")
    prefill_raw_path = os.path.abspath(onnx_files[0]) if onnx_files else ""
    if not prefill_raw_path or not os.path.exists(prefill_raw_path):
        raise FileNotFoundError(
            f"No ONNX file found in {prefill_dir} with pattern hmquant_*.onnx"
        )
    prefill_split_paths = [
        os.path.join(prefill_dir, f"hmquant_{model_name}_part{k}_with_act.onnx")
        for k in range(ndevice)
    ]
    clip(prefill_raw_path, prefill_split_paths, ndevice, model_name)

    onnx_files = glob.glob(f"{decode_dir}/hmquant_*.onnx")
    decode_raw_path = os.path.abspath(onnx_files[0]) if onnx_files else ""
    if not decode_raw_path or not os.path.exists(decode_raw_path):
        raise FileNotFoundError(
            f"No ONNX file found in {decode_dir} with pattern hmquant_*.onnx"
        )
    decode_split_paths = [
        os.path.join(decode_dir, f"hmquant_{model_name}_part{k}_with_act.onnx")
        for k in range(ndevice)
    ]
    clip(decode_raw_path, decode_split_paths, ndevice, model_name)

    # build model
    for i in range(ndevice):
        model_path = f"hmquant_{model_name}_part{i}_with_act.onnx"
        build(
            f"{model_name}_prefill_part{i}",
            prefill_dir,
            model_path,
            output_dir,
            profile,
            ncore,
            1,
            args.context_length,
            args.j,
            batch,
            flash_attention=args.flash_attention,
            prefill_length=args.prefill_length,
        )
        model_path = f"hmquant_{model_name}_part{i}_with_act.onnx"
        build(
            f"{model_name}_decode_part{i}",
            decode_dir,
            model_path,
            output_dir,
            profile,
            ncore,
            1,
            args.context_length,
            args.j,
            batch,
            flash_attention=args.flash_attention,
            prefill_length=args.prefill_length,
        )

    model_path = os.path.basename(decode_raw_path)
    build(
        f"{model_name}_decode",
        decode_dir,
        model_path,
        output_dir,
        profile,
        ncore,
        ndevice,
        args.context_length,
        args.j,
        batch,
        flash_attention=args.flash_attention,
        prefill_length=args.prefill_length,
    )
