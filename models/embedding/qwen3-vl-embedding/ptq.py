# Copyright (c) 2026 HOUMO AI
#
# File: ptq.py
# Description:
#   Post-Training Quantization Tool - Python script for quantizing
# Qwen3-VL-Embedding models using post-training quantization techniques.
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
import psutil
import random
import numpy as np
from typing import Dict

import torch
import gc
from loguru import logger

from xhquant.api import (
    DeviceType,
    QuantScheme,
    xhquant_init,
)

from hmatc.utils.monitor import ProcessMemoryMonitor
from hmatc.utils.utils import (
    check_gpu,
    first_not_none,
    get_model_configs,
    parse_context_length,
)

from xh_model_zoo.xh_llm.models.builder import MODELS
from xh_model_zoo.xh_llm.models.eval_model_type import EvalModelType
from xh_model_zoo.xh_llm.models.qwen3_vl import (
    Qwen3_VLConvertConfig,
    Qwen3_VLEmbeddingConverterXH2a,
    VisualConfig,
)

from xh_model_zoo.utils.memory_tracker import MemoryTracker  # isort:skip
from xh_model_zoo.utils.time_profiler import TimeProfiler  # isort:skip

HOUMO_DATASETS_PATH = os.getenv(
    "HOUMO_DATASETS_PATH",
    str(Path(__file__).resolve().parents[3] / "data" / "datasets"),
)
HOUMO_PIC_PATH = os.getenv(
    "HOUMO_PIC_PATH", str(Path(__file__).resolve().parents[3] / "data" / "pic")
)
HOUMO_TARGET = os.getenv("HOUMO_TARGET", "")
DEFAULT_CONFIG_PATH = os.path.join(os.path.dirname(__file__), "config.yaml")

def get_default_model_dir(model_config: dict) -> str:
    repo_ids = model_config.get("modelscope_repo", [])
    if repo_ids:
        return repo_ids[0].rsplit("/", maxsplit=1)[-1]
    model_name = model_config.get("model_name", "qwen3-vl-embedding").upper()
    model_size = model_config.get("model_size", "8b").upper()
    return f"{model_name}-{model_size}"

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def msg_output_format(title):
    padding_str = "*" * 10
    title = f"{padding_str} {title} {padding_str}"
    return title


def cleanup_cuda():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()


def cleanup_cpu():
    gc.collect()
    try:
        import ctypes

        ctypes.CDLL("libc.so.6").malloc_trim(0)
    except Exception:
        pass

def houmo_export_llm(args):
    hf_model_path = osp.normpath(osp.abspath(args.model))

    ops = dict(
        MatMul=dict(
            act_scheme=dict(bits=8, fp_mode="sefp"),
            act_schema_2=dict(bits=16, fp_mode="sefp"),
        )
    )

    quant_scheme = QuantScheme(
        target_device=DeviceType.XH2a, quant_type=args.quant_type, ops=ops
    )
    
    config = Qwen3_VLConvertConfig(
        batch_size=args.batch_size,
        context_length=args.context_length,
        input_sequence_length=args.input_sequence_length,
        quant_scheme=quant_scheme,
        quant_weight="",
        gptqmodel_cfg="",
        max_pe_length=32768,
        visual_config=VisualConfig(
            image_max_size_h=args.max_size_h,
            image_max_size_w=args.max_size_w,
            image_max_size_t=args.max_size_t,
            temporal_patch_size=2,
            patch_size=16,
            sample_image_path=args.sample_image_path,
            sample_video_path="",
        ),
    )

    prefix = "{}-XH2a-{}".format(args.model_name, format_number(args.context_length))
    work_dir = Path(args.work_dir) / prefix
    work_dir.mkdir(exist_ok=True, parents=True)
    log_file = work_dir / "convert.log"
    xhquant_init(log_file, debug=args.debug)

    logger.info(f"Exporting Qwen3-VL-Embedding backbone from {hf_model_path}")
    logger.info(f"Output directory: {work_dir}")

    with TimeProfiler("convert", logger), MemoryTracker("cuda:0", "convert", logger):
        Qwen3_VLEmbeddingConverterXH2a.convert(hf_model_path, config, work_dir)

def move_models(
    work_dir: Path,
    source: str = "prefill",
    model: str = "prefill",
    target_name: str = "hmquant_qwen3-vl-embedding_with_act.onnx",
):
    source_dir = work_dir / "hmquant/{}".format(source)
    matched_files = list(source_dir.glob("*{}*.onnx".format(model)))

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
    model_name = os.path.basename(args.model_name)
    hm_model_name = "hmquant_{}_with_act.onnx".format(model_name)
    hmm_model_dir = "{}-XH2a-{}".format(model_name, format_number(args.context_length))
    logger.info(
        msg_output_format("Start move from {} to {}").format(
            work_dir / hmm_model_dir, dest_dir
        )
    )
    #shutil.move(work_dir / hmm_model_dir / "hmonnx", dest_dir / "hmquant/prefill")
    prefill_dest_dir = dest_dir / "hmquant/prefill"
    if not prefill_dest_dir.exists():
        prefill_dest_dir.mkdir(parents=True, exist_ok=True)
    os.system("mv {}/hmonnx/*prefill* {}".format(str(work_dir / hmm_model_dir), str(prefill_dest_dir)))
    move_models(dest_dir, "prefill", target_name=hm_model_name)
    decode_dest_dir = dest_dir / "hmquant/decode"
    if not decode_dest_dir.exists():
        decode_dest_dir.mkdir(parents=True, exist_ok=True)
    os.system("mv {}/hmonnx/*decode* {}".format(str(work_dir / hmm_model_dir), str(decode_dest_dir)))
    move_models(dest_dir, "decode", "decode", target_name=hm_model_name)
    shutil.move(
        work_dir / hmm_model_dir / "token_embedding.pt",
        dest_dir / "hmquant/quant_embedding.pt",
    )
    vision_dest_dir = dest_dir / "hmquant/visual"
    if not vision_dest_dir.exists():
        vision_dest_dir.mkdir(parents=True, exist_ok=True)
    os.system("mv {}/hmonnx/*vision* {}".format(str(work_dir / hmm_model_dir), str(vision_dest_dir)))
    move_models(dest_dir, "visual", "vision", target_name=hm_model_name)
    logger.info(msg_output_format("Start remove work_dir: {}".format(work_dir)))
    shutil.rmtree(work_dir, ignore_errors=True)

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
    parser.add_argument("--model", type=str, default=None)
    parser.add_argument(
        "--model-name",
        type=str,
        default=None,
        help="output hmonnx model name",
    )
    parser.add_argument(
        "--model-size",
        type=str,
        default=None,
        help="model size",
    )
    parser.add_argument("--work-dir", type=str, default="work_dirs/")
    parser.add_argument(
        "--out-dir", type=str, default="output/{}".format(HOUMO_TARGET)
    )
    parser.add_argument("--debug", action="store_true", help="debug mode")
    parser.add_argument(
        "--context-length", type=int, default=8192, help="max sequence length"
    )
    parser.add_argument(
        "--sample-image-path", 
        type=str, 
        default=f"{HOUMO_PIC_PATH}/beach.jpeg", 
        help="sample image for vision export (--image_path)",
    )
    parser.add_argument(
        "--input-sequence-length",
        type=int,
        default=None,
        help="input sequence length",
    )
    parser.add_argument(
        "--quant-type",
        default=None,
        help="quant type, default is w8a8h0_ssfp",
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
        parse_context_length(model_config.get("context_length", "8k")),
    )
    args.input_sequence_length = first_not_none(
        args.input_sequence_length, model_config.get("prefill_length", 256)
    )
    args.quant_type = first_not_none(
        args.quant_type, model_config.get("quant_type", "w8a8h0_ssfp")
    )
    args.max_size_w = model_config.get("max_size_w", 896)
    args.max_size_h = model_config.get("max_size_h", 896)
    args.max_size_t = model_config.get("max_size_t", 2)
    args.batch_size = model_config.get("batch_size", 1)
    return args

if __name__ == "__main__":
    assert check_gpu() is True, "Error: Not found GPU device."

    args = parse_args()
    set_seed(42)
    with ProcessMemoryMonitor(interval=2, quiet=True) as monitor:
        houmo_export_llm(args)
        move_llm(args)
    print(
        f"\n=== All quantization steps completed. Peak memory: {monitor.peak_memory_mb:.2f} MB ==="
    )
