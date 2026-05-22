# Copyright (c) 2025 HOUMO AI
#
# File: qwen3_5_moe_xh2a_export_hmonnx.py
# Description:
#   Export script: Qwen3.5-MoE LLM -> prefill/decode HMONNX via LLMConverter.
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
import os.path as osp
from pathlib import Path

from xh_model_zoo.xh_llm import LLMConverter
from xh_model_zoo.xh_llm.models.qwen3_5_moe import Qwen3_5MoeConvertConfig

from xhquant.api import (
    DeviceType,
    QuantScheme,
    get_root_logger,
    xhquant_init,
)  # isort:skip
from xh_model_zoo.utils.memory_tracker import MemoryTracker  # isort:skip
from xh_model_zoo.utils.time_profiler import TimeProfiler  # isort:skip


def main(args):
    hf_model_path = osp.normpath(osp.abspath(args.model))
    model_name = Path(hf_model_path).name
    target_device = DeviceType.XH2a
    quant_type = args.quant_type
    quant_scheme = QuantScheme(target_device=target_device, quant_type=quant_type)

    config = Qwen3_5MoeConvertConfig(
        batch_size=1,
        context_length=args.context_length,
        input_sequence_length=args.input_sequence_length,
        quant_scheme=quant_scheme,
        quant_weight=args.quant_weight,
        num_logits_to_keep=args.num_logits_to_keep,
        linear_attention_mode=args.linear_attention_mode,
        linear_chunk_size=args.linear_chunk_size,
    )

    output_root = Path(args.output_dir)
    work_dir = output_root / f"{model_name}_llm_export"
    work_dir.mkdir(exist_ok=True, parents=True)
    log_file = work_dir / "convert.log"
    xhquant_init(log_file, debug=args.debug)
    logger = get_root_logger()
    logger.info(f"model: {hf_model_path}")
    logger.info(f"quant_weight: {args.quant_weight}")
    logger.info(f"output: {work_dir}")

    # Detect architecture from config.json automatically
    architecture = args.architecture  # may be None → auto-detect

    with TimeProfiler("convert", logger), MemoryTracker("cuda", "convert", logger):
        LLMConverter.from_pretrained(hf_model_path, architecture, config, str(work_dir))

    logger.info(f"Done. Artifacts in: {work_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Export Qwen3.5-MoE to prefill/decode HMONNX",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--model",
        type=str,
        default="/data01/nfs_shared/Qwen3.5-35B-A3B",
        help="HuggingFace model directory",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="work_dirs",
        help="Output root directory. Artifacts will be written to <output-dir>/<model_name>_llm_export/",
    )
    parser.add_argument(
        "--architecture",
        type=str,
        default=None,
        help="Architecture string (auto-detected if None). "
        "Use 'Qwen3_5MoeForConditionalGeneration' or 'Qwen3_5MoeForCausalLM'",
    )
    parser.add_argument(
        "--context-length",
        type=int,
        default=2048,
        help="Maximum context length (kv cache size)",
    )
    parser.add_argument(
        "--input-sequence-length", type=int, default=256, help="Prefill chunk size"
    )
    parser.add_argument(
        "--quant-type", type=str, default="w8a8h0_sefp", help="Quantisation type string"
    )
    parser.add_argument(
        "--quant-weight",
        type=str,
        default=None,
        help="Path to GPTQModel quantised weights (optional)",
    )
    parser.add_argument(
        "--num-logits-to-keep",
        type=int,
        default=1,
        help="How many final logit positions to keep (1 = last only)",
    )
    parser.add_argument(
        "--linear-attention-mode",
        type=str,
        default="auto",
        choices=["auto", "chunk", "recurrent"],
        help="Linear attention computation mode",
    )
    parser.add_argument(
        "--linear-chunk-size",
        type=int,
        default=64,
        help="Chunk size for linear attention",
    )
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()
    main(args)
