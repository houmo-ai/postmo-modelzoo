# Copyright (c) 2025 HOUMO AI
#
# File: ptq.py
# Description:
#   Post-Training Quantization Tool - Python script for quantizing
# Qwen3 models using post-training quantization techniques.
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
import argparse
import os
import shutil
import os.path as osp
from pathlib import Path
import time
import psutil
import threading

import torch
import torch.nn as nn
import gc
from loguru import logger
from safetensors.torch import load_file as load_safetensors_file
from safetensors.torch import save_file as save_safetensors_file
from tqdm import tqdm
from transformers import AutoConfig, AutoModelForCausalLM

HOUMO_DATASETS_PATH = os.getenv('HOUMO_DATASETS_PATH', '')
HOUMO_TARGET = os.getenv('HOUMO_TARGET', '')

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
        self.include_children = True

    def get_memory_info(self):
        """
        Gets current memory usage information.
        Returns:
            dict: A dictionary containing memory usage data.
        """
        rss = self.process.memory_info().rss
        if self.include_children:
            for child in self.process.children(recursive=True):
                try:
                    rss += child.memory_info().rss
                except psutil.NoSuchProcess:
                    continue
        rss_mb = rss / (1024 * 1024)  # Resident Set Size in MB
        percent = self.process.memory_percent()   # Percentage of system memory
        return {'rss_mb': rss_mb, 'percent': percent}

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
            self.peak_memory_mb = max(self.peak_memory_mb, mem_info['rss_mb'])

            timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
            log_message = f"{timestamp} - RSS: {mem_info['rss_mb']:.2f} MB, System%: {mem_info['percent']:.2f}%"
            # Output to console or file
            if self.log_file:
                with open(self.log_file, 'a') as f:
                    f.write(log_message + '\n')

            time.sleep(self.interval)
        timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
        logger.info(f"{timestamp} - Max RSS: {self.peak_memory_mb:.2f} MB, System%: {self.process.memory_percent():.2f}%")

    def stop(self):
        """Stops the monitoring loop and prints peak usage."""
        self.is_monitoring = False
        if hasattr(self, 'monitor_thread'):
            self.monitor_thread.join(timeout=1) # Wait a moment for the thread to finish
        print(f"[Monitoring stopped. Peak RSS: {self.peak_memory_mb:.2f} MB]")

def check_gpu():
    import subprocess

    try:
        result = subprocess.run(
            "nvidia-smi --query-gpu=count --format=csv,noheader,nounits | wc -l",
            shell=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
            text=True
        )
        if result.returncode == 0 and int(result.stdout.strip()) > 0:
            return True
        return False
    except Exception as e:
        print(f"Not install GPU driver, error msg: {e}")
        return False

def str2bool(v):
    if isinstance(v, bool):
       return v
    if v.lower() in ('yes', 'true', 't', 'y', '1',""):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def msg_output_format(title):
    padding_str = "*" * 10
    title = f"{padding_str} {title} {padding_str}"
    return title

def gptq_quant_llm(args):
    from datasets import load_dataset
    from gptqmodel import GPTQModel, QuantizeConfig
    import json

    model_name = os.path.basename(args.model)
    quant_path = os.path.join(args.work_dir, "{}_gptqmodel_4bit".format(model_name))

    calibration_dataset = []
    if args.calib_data:
        cnt = 0
        cnt_start = 0
        with open(args.calib_data, encoding='utf-8') as file:
            for line in file:
                if cnt >= cnt_start:
                    calibration_dataset.append(json.loads(line)['text'])
                cnt = cnt + 1
                if cnt == cnt_start + 512:
                    break
    else:
        calibration_dataset = load_dataset(
            'wikitext', 'wikitext-2-raw-v1', split='train'
        ).select(range(512))["text"]

    quant_config = QuantizeConfig(
        bits=4, group_size=64, hessian_mse=True, rotation="hadamard", offload_to_disk=False,
    )

    model = GPTQModel.load(args.model, quant_config)

    # increase `batch_size` to match gpu/vram specs to speed up quantization
    model.quantize(calibration_dataset, batch_size=1)

    model.save(quant_path)

def houmo_quant_llm(args):
    out_dir = Path(args.work_dir)
    hf_model_dir = args.model

    hf_model_path = osp.normpath(osp.abspath(args.model))
    cfg_name = Path(hf_model_path).name
    if not args.skip_quarot:
        cfg_name += "_quarot"

    work_dir = Path(out_dir) / cfg_name
    work_dir.mkdir(exist_ok=True, parents=True)
    config = AutoConfig.from_pretrained(hf_model_dir)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16

    native_model = AutoModelForCausalLM.from_pretrained(
        hf_model_dir,
        torch_dtype=dtype,
        device_map="cpu",
        config=config,
        trust_remote_code=True,
        attn_implementation="eager",
    )

    if native_model.config.tie_word_embeddings:
        old_torchscript = native_model.config.torchscript
        native_model.config.torchscript = True
        native_model.tie_weights()
        native_model.config.tie_word_embeddings = False
        native_model.config.torchscript = old_torchscript

    native_model.eval()
    native_model.to(dtype)

    quant_methods = []
    if not args.skip_quarot:
        torch.cuda.reset_peak_memory_stats()
        quant_methods.append("quarot")
        quant_name = "_".join(quant_methods)
        filename = work_dir / f"{quant_name}-state-dict.safetensors"
        if not os.path.exists(filename):
            from xh_model_zoo.xh_llm.quarot.quantizer_utils import quarot

            logger.info(msg_output_format("Start quarot quantization"))
            native_model = quarot(native_model, device=device)
            logger.info(msg_output_format("End quarot quantization"))

            state_dict = native_model.state_dict()
            logger.info(msg_output_format(f"Saving checkpoint to: {filename}"))
            # torch.save(state_dict, filename)
            save_safetensors_file(state_dict, filename)
            logger.info(f"Save checkpoint to: {filename}")
        else:
            state_dict = load_safetensors_file(filename)
            native_model.load_state_dict(state_dict)
            logger.info(msg_output_format(f"Load state_dict from {filename}"))

        consumption = torch.cuda.max_memory_allocated()
        unit = "B"
        if consumption > 1024:
            consumption = consumption / 1024
            unit = "k"
            if consumption > 1024:
                consumption = consumption / 1024
                unit = "M"
            consumption = round(consumption, 2)
        logger.info(f"GPU memory cost for export {consumption}{unit}")

    if len(quant_methods) != 0:
        quant_name = "_".join(quant_methods)
        filename = work_dir / f"{quant_name}-state-dict.safetensors"
        state_dict = native_model.state_dict()
        # del native_model
        for k in tqdm(state_dict):
            paths = k.split(".")
            v = state_dict[k]
            if paths[-1] == "quant_weight":
                if v.min().item() >= -pow(2, 7) and v.max().item() <= pow(2, 7) - 1:
                    v = v.to(torch.int8)
                elif v.min().item() >= -pow(2, 15) and v.max().item() <= pow(2, 15) - 1:
                    v = v.to(torch.int16)
                else:
                    v = v.to(torch.float32)
            else:
                v = v.to(torch.float16)

            state_dict[k] = v
        logger.info(msg_output_format(f"Saving checkpoint to: {filename}"))
        save_safetensors_file(state_dict, filename)
        logger.info(f"Save checkpoint to: {filename}")

    try:
        del native_model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
        logger.info("native_model released and GPU memory cleaned.")
    except Exception as e:
        logger.warning(f"Failed to release native_model: {e}")


def houmo_export_llm(args):
    hf_model_path = osp.normpath(osp.abspath(args.model))
    from xh_model_zoo.xh_llm import LLMConverter
    from xh_model_zoo.xh_llm.models.qwen3moe import Qwen3MoeConvertConfig

    from xhquant.api import DeviceType, xhquant_init, QuantScheme, get_root_logger  # isort:skip
    from xh_model_zoo.utils.memory_tracker import MemoryTracker  # isort:skip
    from xh_model_zoo.utils.time_profiler import TimeProfiler

    import copy

    hf_model_path = osp.normpath(osp.abspath(args.model))
    cfg_name = Path(hf_model_path).name
    src_cfg_name = copy.deepcopy(cfg_name)
    quant_weight = None
    if args.gptqmodel:
        quant_path = os.path.join(args.work_dir, "{}_gptqmodel_4bit".format(src_cfg_name))
        hf_model_path = osp.normpath(osp.abspath(quant_path))
    else:
        if not args.skip_quarot:
            cfg_name += "_quarot"
            quant_weight = os.path.join(args.work_dir, f"{cfg_name}", f"{cfg_name[len(src_cfg_name)+1:]}-state-dict.safetensors")


    quant_scheme = QuantScheme(target_device=DeviceType.XH2a, quant_type=args.quant_type)
    # quant_scheme.nodes["lm_head"] = "w8a8h1_sefp"
    config = Qwen3MoeConvertConfig(
        batch_size=1,
        context_length=args.context_length,
        input_sequence_length=args.input_sequence_length,
        quant_scheme=quant_scheme,
        quant_weight=quant_weight,
    )

    prefix = "{}-XH2a-{}-{}".format(
        src_cfg_name, format_number(args.context_length), args.quant_type
    )
    work_dir = Path(args.work_dir) / prefix
    work_dir.mkdir(exist_ok=True, parents=True)
    log_file = work_dir / "convert.log"
    xhquant_init(log_file, debug=args.debug)
    logger = get_root_logger()
    with TimeProfiler("convert", logger), MemoryTracker("cuda", "convert", logger):
        LLMConverter.from_pretrained(hf_model_path, "Qwen3MoeForCausalLM", config, str(work_dir))

def move_models(
    work_dir: Path,
    source: str = "prefill",
    model: str = "prefill",
    target_name: str = "hmquant_qwen3moe_with_act.onnx",
):
    source_dir = work_dir / "hmquant/{}".format(source)
    matched_files = list(source_dir.glob("*{}.onnx".format(model)))

    if not matched_files:
        raise FileNotFoundError(f"No matching ONNX files found in {source_dir}")

    target_path = source_dir / target_name
    if target_path.exists():
        target_path.unlink()

    shutil.move(matched_files[0], target_path)
    return target_path


def format_number(n):
    if n >= 1024 * 1024:
        return f"{n // (1024 * 1024)}m"
    elif n >= 1024:
        return f"{n // 1024}k"
    else:
        return f"0k"


def move_llm(args):
    work_dir = Path(args.work_dir)
    dest_dir = Path(args.out_dir)
    dest_dir.mkdir(parents=True, exist_ok=True)
    model_name = os.path.basename(args.model)
    hm_model_name = "hmquant_{}_with_act.onnx".format(args.model_name)
    hmm_model_dir = "{}-XH2a-{}-{}".format(
        model_name, format_number(args.context_length), args.quant_type
    )
    logger.info(
        msg_output_format("Start move from {} to {}").format(
            work_dir / hmm_model_dir, dest_dir
        )
    )

    shutil.move(
        work_dir / hmm_model_dir / "hmonnx/prefill", dest_dir / "hmquant/prefill"
    )
    move_models(dest_dir, "prefill", target_name=hm_model_name)
    shutil.move(
        work_dir / hmm_model_dir / "hmonnx/decode", dest_dir / "hmquant/decoder"
    )
    move_models(dest_dir, "decoder", "decoder", target_name=hm_model_name)
    shutil.move(
        work_dir / hmm_model_dir / "token_embedding.pt",
        dest_dir / "hmquant/quant_embedding.pt",
    )
    logger.info(msg_output_format("Start remove work_dir: {}".format(work_dir)))
    shutil.rmtree(work_dir, ignore_errors=True)

if HOUMO_TARGET == 'xh2':

    def parse_args():
        parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
        parser.add_argument("--model", type=str, default="qwen3-30b-a3b")
        parser.add_argument("--model-name", type=str, default="qwen3", help="output hmonnx model name")
        parser.add_argument("--calib_data", type=str,
                            default="../../../hmodel/xh2/examples/xh_gen_data/gen_qwen3_30b_EBSS.jsonl",
                            help="calibration dataset choose")
        parser.add_argument("--work-dir", type=str, default="work_dirs/")
        parser.add_argument("--out-dir", type=str, default="output/{}".format(HOUMO_TARGET))
        parser.add_argument("--skip-quarot", action="store_true", help="skip_quarot")
        parser.add_argument("--w-bits", type=int, default=4)
        parser.add_argument("--debug", action="store_true", help="debug mode")
        parser.add_argument("--context-length", type=int, default=8192, help="max sequence length")
        parser.add_argument("--input-sequence-length", type=int, default=256, help="input sequence length")
        parser.add_argument("--quant-type", default="w4a8h0_ssfp", help="quant type, default is w4a8_ssfp")
        parser.add_argument("--gptqmodel", action="store_true", help="use gptqmodel to quant")
        args = parser.parse_args()
        return args

    def main():
        args = parse_args()
        if args.gptqmodel:
            gptq_quant_llm(args)
        else:
            houmo_quant_llm(args)
        houmo_export_llm(args)
        move_llm(args)

if __name__ == "__main__":
    if not check_gpu():
        print("Error: Not found GPU device.")
        exit(-1)
    memory_monitor = ProcessMemoryMonitor(interval=2, log_file="./cpu_memory.log")
    memory_monitor.start()
    main()
    memory_monitor.stop()
