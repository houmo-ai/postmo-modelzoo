# Copyright (c) 2025 HOUMO AI
#
# File: ptq.py
# Description:
#   Post-Training Quantization Tool - Python script for quantizing
# Qwen3.5 models using post-training quantization techniques.
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
"""End-to-end Qwen3.5 VL PTQ: rotate + GPTQ, then vision + LLM HMONNX export.

Run from this directory: ``python ptq.py --model /path/to/Qwen3.5-9B ...``
See src/README.md for script-level equivalents.
"""
import argparse
import json
import os
import shutil
import sys
import threading
import time
from pathlib import Path

import psutil

# Same package directory as this file (works when cwd is models/llm/qwen3.5)
_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

from quant_pipeline import (
    export_llm,
    export_llm_moe,
    move_llm,
    move_llm_moe,
    quant_llm,
    quant_llm_moe,
)

HOUMO_DATASETS_PATH = os.getenv("HOUMO_DATASETS_PATH", "")
HOUMO_TARGET = os.getenv("HOUMO_TARGET", "")


class ProcessMemoryMonitor:
    """Monitors the memory usage of the current Python process via psutil."""

    def __init__(self, interval=2, log_file=None):
        self.process = psutil.Process(os.getpid())
        self.interval = interval
        self.log_file = log_file
        self.is_monitoring = False
        self.peak_memory_mb = 0

    def get_memory_info(self):
        memory_info = self.process.memory_info()
        rss_mb = memory_info.rss / (1024 * 1024)
        percent = self.process.memory_percent()
        return {"rss_mb": rss_mb, "percent": percent}

    def start(self):
        self.is_monitoring = True
        self.peak_memory_mb = 0
        self.monitor_thread = threading.Thread(target=self._monitor_loop)
        self.monitor_thread.daemon = True
        self.monitor_thread.start()
        print(f"Memory monitoring started (interval: {self.interval}s)")

    def _monitor_loop(self):
        while self.is_monitoring:
            mem_info = self.get_memory_info()
            self.peak_memory_mb = max(self.peak_memory_mb, mem_info["rss_mb"])
            timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
            log_message = (
                f"{timestamp} - RSS: {mem_info['rss_mb']:.2f} MB, "
                f"System%: {mem_info['percent']:.2f}%"
            )
            if self.log_file:
                with open(self.log_file, "a") as f:
                    f.write(log_message + "\n")
            time.sleep(self.interval)

    def stop(self):
        self.is_monitoring = False
        if hasattr(self, "monitor_thread"):
            self.monitor_thread.join(timeout=1)
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
            text=True,
        )
        if result.returncode == 0 and int(result.stdout.strip()) > 0:
            return True
        return False
    except Exception as e:
        print(f"Not install GPU driver, error msg: {e}")
        return False


def parse_args():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="HF checkpoint directory (Qwen3.5 VL instruct)",
    )
    parser.add_argument(
        "--model-name",
        type=str,
        default="",
        help=(
            "HMONNX / layout name (vision --model_name, output folders). "
            "Default: last path component of --model (e.g. .../Qwen3.5-9B → Qwen3.5-9B)"
        ),
    )
    parser.add_argument(
        "--work-dir",
        type=str,
        default="./work_dirs",
        help="step1–2 outputs: <model-name>_rotated_fp and <model-name>_gptq_4bit under this root",
    )
    parser.add_argument(
        "--cleanup-work-dir",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "after move_llm finishes, remove the entire --work-dir directory tree "
            "(default: on; --no-cleanup-work-dir to keep checkpoints)"
        ),
    )
    parser.add_argument(
        "--out-dir",
        type=str,
        default="./output",
        help="step3–4 outputs: <model-name>/vision_export/... + <model-name>_llm_export/",
    )
    parser.add_argument(
        "--move-only",
        action="store_true",
        help=(
            "only run move_llm: layout $HOUMO_TARGET/hmquant (default segment xh2 if unset) "
            "under --out-dir; needs --out-dir; uses effective --model-name (or --model stem); "
            "skips quant, export, GPU check"
        ),
    )
    parser.add_argument("--seed", type=int, default=1024)
    parser.add_argument(
        "--resume",
        action="store_true",
        help=(
            "skip rotate/GPTQ subprocesses when the expected HF dirs already exist "
            "(under --work-dir, or paths from --rotated-model-dir / --gptq-model-dir)"
        ),
    )
    parser.add_argument(
        "--export-only",
        action="store_true",
        help="do not run quantization; only run export (requires existing rotated + GPTQ checkpoints)",
    )
    parser.add_argument(
        "--rotated-model-dir",
        type=str,
        default=None,
        help="HF dir for rotated fp weights (vision export); default: <work-dir>/<model-name>_rotated_fp",
    )
    parser.add_argument(
        "--gptq-model-dir",
        type=str,
        default=None,
        help="HF dir for GPTQ weights (LLM export); default: <work-dir>/<model-name>_gptq_4bit",
    )
    parser.add_argument(
        "--debug", action="store_true", help="forward --debug to export scripts"
    )
    parser.add_argument(
        "--context-length",
        type=int,
        default=2048,
        help="LLM export context length",
    )
    parser.add_argument(
        "--max-pe-length",
        type=int,
        default=262144,
        help="LLM export rotary cache max_pe_length",
    )
    parser.add_argument(
        "--input-sequence-length",
        type=int,
        default=256,
        help="MoE LLM export --input-sequence-length",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda:0",
        help="CUDA device string for rotate / GPTQ subprocesses",
    )
    parser.add_argument(
        "--trust-remote-code",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="HF trust_remote_code for rotate + dense scripts",
    )
    parser.add_argument(
        "--calib_data",
        type=str,
        default=None,
        help="optional JSONL calibration for GPTQ (maps to --calibration-jsonl)",
    )
    parser.add_argument(
        "--self-attn-bits",
        type=int,
        default=4,
        choices=[2, 3, 4, 5, 8],
        help="MoE GPTQ self-attn q/k/v/o projection bit width",
    )
    parser.add_argument("--skip-export-vision", action="store_true")
    parser.add_argument("--skip-export-llm", action="store_true")
    parser.add_argument("--datasets-dir", type=str, default="../../../data/datasets")
    parser.add_argument(
        "--vision-image-path",
        type=str,
        default="../../../data/pic/beach.jpeg",
        help="sample image for vision export (--image_path); required if src/images/qwen2_vl_demo.jpeg is absent",
    )
    parser.add_argument(
        "--max_size_w",
        type=int,
        default=448,
        help="vision export max input width in pixels",
    )
    parser.add_argument(
        "--max_size_h",
        type=int,
        default=448,
        help="vision export max input height in pixels",
    )
    parser.add_argument(
        "--max_size_t",
        type=int,
        default=2,
        help="vision export max temporal size in frames",
    )
    args = parser.parse_args()
    mn = (args.model_name or "").strip()
    if not mn:
        args.model_name = Path(args.model).resolve().name
    else:
        args.model_name = mn
    return args


def _remove_work_dir(args) -> None:
    if not getattr(args, "cleanup_work_dir", True):
        return
    root = Path(args.work_dir).resolve()
    if not root.is_dir():
        return
    shutil.rmtree(root)
    print(f"[ptq] removed --work-dir: {root}")


def _detect_model_family(model_dir: str) -> str:
    config_path = Path(model_dir).resolve() / "config.json"
    if not config_path.is_file():
        return "dense"
    try:
        with config_path.open("r", encoding="utf-8") as f:
            cfg = json.load(f)
    except Exception:
        return "dense"

    model_type = str(cfg.get("model_type", "")).lower()
    text_type = str((cfg.get("text_config") or {}).get("model_type", "")).lower()
    archs = [str(x).lower() for x in (cfg.get("architectures") or [])]
    markers = [model_type, text_type, *archs]
    if any("moe" in x for x in markers):
        return "moe"
    return "dense"


def main(args=None):
    if args is None:
        args = parse_args()
    family = _detect_model_family(args.model)
    print(f"[ptq] detected model family: {family}")
    if args.move_only:
        if family == "moe":
            move_llm_moe(args)
        else:
            move_llm(args)
        _remove_work_dir(args)
        return
    if family == "moe":
        quant_llm_moe(args)
        export_llm_moe(args)
        move_llm_moe(args)
    else:
        quant_llm(args)
        export_llm(args)
        move_llm(args)
    _remove_work_dir(args)


if __name__ == "__main__":
    _args = parse_args()
    _monitor = None
    if not _args.move_only:
        if not check_gpu():
            print("Error: Not found GPU device.")
            sys.exit(-1)
        _monitor = ProcessMemoryMonitor(interval=2)
        _monitor.start()
    try:
        main(_args)
    finally:
        if _monitor is not None:
            _monitor.stop()
