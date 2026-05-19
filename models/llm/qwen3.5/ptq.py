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
from pathlib import Path

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
from hmatc.utils.monitor import ProcessMemoryMonitor
from hmatc.utils.utils import (
    check_gpu,
    first_not_none,
    get_model_configs,
    parse_context_length,
)

HOUMO_DATASETS_PATH = os.getenv("HOUMO_DATASETS_PATH", "")
HOUMO_TARGET = os.getenv("HOUMO_TARGET")
assert HOUMO_TARGET in ["xh2"], f"Unsupported HOUMO_TARGET: {HOUMO_TARGET}"
DEFAULT_CONFIG_PATH = os.path.join(os.path.dirname(__file__), "config.yaml")


def get_default_model_dir(model_config: dict) -> str:
    repo_ids = model_config.get("modelscope_repo", [])
    if repo_ids:
        return repo_ids[0].rsplit("/", maxsplit=1)[-1]
    model_name = model_config.get("model_name", "qwen3.6").upper()
    model_size = model_config.get("model_size", "35b-a3b").upper()
    return f"{model_name}-{model_size}"


def parse_args():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "--config",
        dest="config_path",
        type=str,
        default=DEFAULT_CONFIG_PATH,
        help="path to config.yaml",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="HF checkpoint directory (Qwen3.5 VL instruct)",
    )
    parser.add_argument(
        "--model-name",
        type=str,
        default=None,
        help=(
            "HMONNX / layout name (vision --model_name, output folders). "
            "Default: last path component of --model (e.g. .../Qwen3.5-9B → Qwen3.5-9B)"
        ),
    )
    parser.add_argument(
        "--model-size",
        type=str,
        default=None,
        help="model size",
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
        default=None,
        help="LLM export context length",
    )
    parser.add_argument(
        "--max-pe-length",
        type=int,
        default=None,
        help="LLM export rotary cache max_pe_length, default follows context_length",
    )
    parser.add_argument(
        "--llm-export-full-output-valid",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="run full-output comparison during LLM export validation; disabled by default to reduce VRAM",
    )
    parser.add_argument(
        "--input-sequence-length",
        type=int,
        default=None,
        help="MoE LLM export --input-sequence-length",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
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
        nargs="+",
        default=None,
        help="vision export max input width in pixels (multiple values allowed)",
    )
    parser.add_argument(
        "--max_size_h",
        type=int,
        nargs="+",
        default=None,
        help="vision export max input height in pixels (multiple values allowed)",
    )
    parser.add_argument(
        "--max_size_t",
        type=int,
        nargs="+",
        default=None,
        help="vision export max temporal size in frames (multiple values allowed)",
    )
    args = parser.parse_args()
    default_model_size, default_model_name, model_configs = get_model_configs(
        args.config_path
    )
    args.model_name = first_not_none(args.model_name, default_model_name)
    args.model_size = first_not_none(args.model_size, default_model_size)
    model_config = model_configs.get(args.model_name, {}).get(args.model_size, {})
    args.model = first_not_none(args.model, get_default_model_dir(model_config))
    args.context_length = first_not_none(
        args.context_length,
        parse_context_length(model_config.get("context_length", "256k")),
    )
    args.max_pe_length = first_not_none(args.max_pe_length, args.context_length)
    args.input_sequence_length = first_not_none(
        args.input_sequence_length, model_config.get("prefill_length", 256)
    )
    args.max_size_w = first_not_none(
        args.max_size_w, [model_config.get("max_size_w", 896)]
    )
    args.max_size_h = first_not_none(
        args.max_size_h, [model_config.get("max_size_h", 896)]
    )
    args.max_size_t = first_not_none(
        args.max_size_t, [model_config.get("max_size_t", 2)]
    )
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


if __name__ == "__main__":
    args = parse_args()
    print(args)

    _family = _detect_model_family(args.model)
    print(f"[ptq] detected model family: {_family}")

    assert check_gpu() is True, "Error: Not found GPU device."
    with ProcessMemoryMonitor(interval=2, quiet=True) as monitor:
        if _family == "moe":
            quant_llm_moe(args)
            export_llm_moe(args)
            move_llm_moe(args)
        else:
            quant_llm(args)
            export_llm(args)
            move_llm(args)
        _remove_work_dir(args)
    print(
        f"\n=== All quantization steps completed. Peak memory: {monitor.peak_memory_mb:.2f} MB ==="
    )
