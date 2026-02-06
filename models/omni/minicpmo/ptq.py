#!/usr/bin/python3
# -*- coding: utf-8 -*-
# Copyright 2025 HOUMO AI
#
# File: quant_compile.py
# Description:
#   Post-training quantization of the minicpmo model.
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
import gc
from loguru import logger

from xh_model_zoo.xh_llm import LLMConverter

from xhquant.api import DeviceType, QuantScheme, get_root_logger, xhquant_init  # isort:skip
from xh_model_zoo.utils.memory_tracker import MemoryTracker  # isort:skip
from xh_model_zoo.utils.time_profiler import TimeProfiler  # isort:skip

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

def cleanup_cuda():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()

def houmo_export_llm(args, component="vision"):
    hf_model_path = osp.normpath(osp.abspath(args.model))
    model_name = args.model_name
    audio_path = os.path.join(args.model, "assets/demo.wav")
    video_path = os.path.join(args.model, "assets/Skiing.mp4")
    quant_type = args.quant_type
    quant_scheme = QuantScheme(target_device=DeviceType.XH2a, quant_type=quant_type)
    if component == "vision":
        from xh_model_zoo.xh_llm.models.minicpmo.minicpmo_vision_convert_config import MinicpmoVisionConvertConfig
        config = MinicpmoVisionConvertConfig(
            quant_scheme=quant_scheme,
            video=video_path,
            audio=audio_path,
            debug=args.debug,
            valid=args.valid,
            image_slice_max_size=[40, 40],
        )
        architecture = "MiniCPMOVisionEncoder"
    elif component == "audio":
        from xh_model_zoo.xh_llm.models.minicpmo.minicpmo_audio_convert_config import MinicpmoAudioConvertConfig
        config = MinicpmoAudioConvertConfig(
            quant_scheme=quant_scheme,
            video=video_path,
            audio=audio_path,
            debug=args.debug,
            valid=args.valid,
            image_slice_max_size=[40,40],
        )
        architecture = "MiniCPMWhisperEncoder"
    elif component == "llm":
        from xh_model_zoo.xh_llm.models.minicpmo.minicpmo_llm_convert_config import MinicpmoLLMConvertConfig
        config = MinicpmoLLMConvertConfig(
            quant_scheme=quant_scheme,
            video=video_path,
            audio=audio_path,
            debug=args.debug,
            valid=args.valid,
            context_length=args.context_length,
            input_sequence_length=args.input_sequence_length,
            image_slice_max_size=[40, 40],
        )
        architecture = "MiniCPMOLLMEncoder"
    elif component == "tts":
        from xh_model_zoo.xh_llm.models.minicpmo.minicpmo_tts_convert_config import MinicpmoTTSConvertConfig
        config = MinicpmoTTSConvertConfig(
            quant_scheme=quant_scheme,
            video=video_path,
            audio=audio_path,
            debug=args.debug,
            valid=args.valid,
            context_length=args.context_length,
            input_sequence_length=args.input_sequence_length,
            image_slice_max_size=[40, 40],
        )
        architecture = "MiniCPMOTTS"
    elif component == "tts_dvae":
        from xh_model_zoo.xh_llm.models.minicpmo.minicpmo_tts_dvae_convert_config import MinicpmoTTSDVAEConvertConfig
        config = MinicpmoTTSDVAEConvertConfig(
            quant_scheme=quant_scheme,
            video=video_path,
            audio=audio_path,
            debug=args.debug,
            valid=args.valid,
            image_slice_max_size=[40, 40],
        )
        architecture = "MiniCPMOTTSDVAEEncoder"
    elif component == "tts_vocos":
        from xh_model_zoo.xh_llm.models.minicpmo.minicpmo_tts_vocos_convert_config import MinicpmoTTSVocosConvertConfig
        config = MinicpmoTTSVocosConvertConfig(
            quant_scheme=quant_scheme,
            video=video_path,
            audio=audio_path,
            debug=args.debug,
            valid=args.valid,
            image_slice_max_size=[40, 40],
        )
        architecture = "MiniCPMOTTSVOCOS"
    else:
        raise ValueError(f"Unsupported component: {component}")

    prefix = f"{model_name}-{component}-{quant_type}"
    work_dir = Path("work_dirs") / prefix
    work_dir.mkdir(exist_ok=True, parents=True)
    log_file = work_dir / f"{component}-convert.log"
    xhquant_init(log_file, debug=args.debug)
    logger = get_root_logger()
    with TimeProfiler("convert", logger), MemoryTracker("cuda:0", "convert", logger):
        LLMConverter.from_pretrained(hf_model_path, architecture, config, str(work_dir))

    cleanup_cuda()

def move_models(
    work_dir: Path,
    source: str = "prefill",
    model: str = "prefill",
    target_name: str = "hmquant_minicpmo_with_act.onnx",
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
        return "0k"


def move_hmonnx(args):
    def remove_meta_info(info_path: Path):
        if info_path.exists():
            info_path.unlink()
    work_dir = Path(args.work_dir)
    dest_dir = Path(args.out_dir)
    model_name = args.model_name
    hm_model_name = "hmquant_{}_with_act.onnx".format(args.model_name)
    START_MOVE_MSG = "Start move from {} to {}"
    ### visual ###
    hmm_model_dir = "{}-vision-{}".format(
        model_name, args.quant_type
    )
    logger.info(
        msg_output_format(START_MOVE_MSG).format(
            work_dir / hmm_model_dir, dest_dir
        )
    )
    shutil.move(
        work_dir / hmm_model_dir / "hmonnx/vision", dest_dir / "hmquant/visual"
    )
    move_models(dest_dir, "visual", "vision", target_name=hm_model_name)
    remove_meta_info(Path(dest_dir / "hmquant/visual/meta_info.json"))
    ### audio ###
    hmm_model_dir = "{}-audio-{}".format(
        model_name, args.quant_type
    )
    logger.info(
        msg_output_format(START_MOVE_MSG).format(
            work_dir / hmm_model_dir, dest_dir
        )
    )
    shutil.move(
        work_dir / hmm_model_dir / "hmonnx/audio", dest_dir / "hmquant/audio"
    )
    move_models(dest_dir, "audio", "audio", target_name=hm_model_name)
    remove_meta_info(Path(dest_dir / "hmquant/audio/meta_info.json"))
    ### llm ###
    hmm_model_dir = "{}-llm-{}".format(
        model_name, args.quant_type
    )
    logger.info(
        msg_output_format(START_MOVE_MSG).format(
            work_dir / hmm_model_dir, dest_dir
        )
    )
    shutil.move(
        work_dir / hmm_model_dir / "hmonnx/llm/prefill_onnx", dest_dir / "hmquant/prefill"
    )
    move_models(dest_dir, "prefill", "llm_prefill", target_name=hm_model_name)
    shutil.move(
        work_dir / hmm_model_dir / "hmonnx/llm/decode_onnx", dest_dir / "hmquant/decoder"
    )
    move_models(dest_dir, "decoder", "llm_decode", target_name=hm_model_name)
    shutil.move(
        work_dir / hmm_model_dir / "token_embedding.pt",
        dest_dir / "hmquant/quant_embedding.pt",
    )
    ### tts ###
    hmm_model_dir = "{}-tts-{}".format(
        model_name, args.quant_type
    )
    logger.info(
        msg_output_format(START_MOVE_MSG).format(
            work_dir / hmm_model_dir, dest_dir
        )
    )
    shutil.move(
        work_dir / hmm_model_dir / "hmonnx/tts/prefill_onnx", dest_dir / "hmquant/tts_prefill"
    )
    move_models(dest_dir, "tts_prefill", "tts_prefill", target_name=hm_model_name)
    shutil.move(
        work_dir / hmm_model_dir / "hmonnx/tts/decode_onnx", dest_dir / "hmquant/tts_decoder"
    )
    move_models(dest_dir, "tts_decoder", "tts_decode", target_name=hm_model_name)
    shutil.move(
        work_dir / hmm_model_dir / "quant_embedding_tts.pt",
        dest_dir / "hmquant/quant_embedding_tts.pt",
    )
    for i in range(4):
        shutil.move(
            work_dir / hmm_model_dir / f"quant_embedding_tts_code_{i}.pt",
            dest_dir / f"hmquant/quant_embedding_tts_code_{i}.pt",
        )
    ### dvae ###
    hmm_model_dir = "{}-tts_dvae-{}".format(
        model_name, args.quant_type
    )
    logger.info(
        msg_output_format(START_MOVE_MSG).format(
            work_dir / hmm_model_dir, dest_dir
        )
    )
    shutil.move(
        work_dir / hmm_model_dir / "hmonnx/tts_dvae", dest_dir / "hmquant/dvae"
    )
    move_models(dest_dir, "dvae", "dvae_part1", target_name="hmquant_dvae_part1_with_act.onnx")
    move_models(dest_dir, "dvae", "dvae_part2", target_name="hmquant_dvae_part2_with_act.onnx")
    remove_meta_info(Path(dest_dir / "hmquant/dvae/meta_info.json"))
    ### vocos ###
    hmm_model_dir = "{}-tts_vocos-{}".format(
        model_name, args.quant_type
    )
    logger.info(
        msg_output_format(START_MOVE_MSG).format(
            work_dir / hmm_model_dir, dest_dir
        )
    )
    shutil.move(
        work_dir / hmm_model_dir / "hmonnx/tts_vocos", dest_dir / "hmquant/vocos"
    )
    move_models(dest_dir, "vocos", "vocos", target_name="hmquant_vocos_with_act.onnx")
    remove_meta_info(Path(dest_dir / "hmquant/vocos/meta_info.json"))

    logger.info(msg_output_format("Start remove work_dir: {}".format(work_dir)))
    shutil.rmtree(work_dir, ignore_errors=True)

if HOUMO_TARGET == 'xh2':

    def parse_args():
        parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
        parser.add_argument("--model", type=str, default="./MiniCPM-o-2_6", help="input hf model path")
        parser.add_argument("--model-name", type=str, default="minicpmo", help="output hmonnx model name")
        parser.add_argument("--work-dir", type=str, default="work_dirs/")
        parser.add_argument("--out-dir", type=str, default="output/{}".format(HOUMO_TARGET))
        parser.add_argument("--valid", type=bool, default=False, help="check hmonnx mode")
        parser.add_argument("--debug", action="store_true", help="debug mode")
        parser.add_argument("--context-length", type=int, default=4096, help="max sequence length")
        parser.add_argument("--input-sequence-length", type=int, default=256, help="input sequence length")
        parser.add_argument("--quant-type", default="w8a8h0_sefp", help="quant type, default is w8a8h0_sefp")
        args = parser.parse_args()
        return args

    def main():
        args = parse_args()
        for component in ["vision", "audio", "llm", "tts", "tts_dvae", "tts_vocos"]:
            houmo_export_llm(args, component=component)
        move_hmonnx(args)

if __name__ == "__main__":
    if not check_gpu():
        print("Error: Not found GPU device.")
        exit(-1)
    memory_monitor = ProcessMemoryMonitor(interval=2, log_file="./cpu_memory.log")
    memory_monitor.start()
    main()
    memory_monitor.stop()