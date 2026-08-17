# Copyright (c) 2025 HOUMO AI
#
# File: build.py
# Description:
#  Qwen3-TTS Model Build and Test Tool.
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
import argparse
import multiprocessing
from loguru import logger

from hmatc.exec.xh2_exec import Xh2Exec
from hmatc.utils.utils import (
    find_hmonnx_file,
    first_not_none,
    get_model_configs,
    get_platform,
    parse_context_length,
)

HOUMO_TARGET = os.getenv("HOUMO_TARGET")
assert HOUMO_TARGET in ["xh2"], f"Unsupported HOUMO_TARGET: {HOUMO_TARGET}"

HOUMO_CORE_NUM = int(os.getenv("HOUMO_CORE_NUM", 2))
DEFAULT_CONFIG_PATH = os.path.join(os.path.dirname(__file__), "config.yaml")


def get_default_parallel_jobs():
    return max(1, int(multiprocessing.cpu_count() * 0.75))


def get_args() -> argparse.Namespace:
    """Parse commandline."""
    parser = argparse.ArgumentParser()
    # fmt: off
    parser.add_argument("--config", dest="config_path", type=str, default=DEFAULT_CONFIG_PATH, help="path to config.yaml")
    parser.add_argument("--model_dir", dest="model_dir", type=str, default=os.path.join("output", HOUMO_TARGET, "hmquant"), help="path to the model dir")
    parser.add_argument("--model_name", dest="model_name", type=str, default=None, help="output houmo model name")
    parser.add_argument("--model_size", dest="model_size", type=str, default=None, help="model size")
    parser.add_argument("--batch", dest="batch", type=int, default=None, help="batch size")
    parser.add_argument("--j", dest="j", type=int, default=get_default_parallel_jobs(), help="build parallel jobs. Default is about 75%% of CPU count.")
    parser.add_argument("--ncore", dest="ncore", type=int, default=None, help="core number")
    parser.add_argument("--context_length", dest="context_length", type=int, default=None, help="context_length")
    parser.add_argument("--prefill_length", dest="prefill_length", type=int, default=None, help="prefill_length")
    parser.add_argument("--ndevice", dest="ndevice", type=int, default=None, help="device number")
    parser.add_argument("--stage", dest="stage", type=str, default="build", choices=["build", "test", "all"], help="build stage")
    parser.add_argument("--output_dir", dest="output_dir", type=str, default=os.path.join("output", HOUMO_TARGET), help="build output dir")
    parser.add_argument("--enable_stable_opt", dest="enable_stable_opt", action="store_true", default=False, help="enable stable output")
    parser.add_argument("--enable_common_subgraph", dest="enable_common_subgraph", action="store_true", default=False, help="enable common subgraph optimization")
    parser.add_argument("--flash_attention", dest="flash_attention", nargs=2, type=int, default=(2, 1), help="FlashAttention modes for LLM and non-LLM; both support 0/1/2")
    parser.add_argument("--models", dest="models", nargs="+", type=str, default=None, help="specify which sub-models to build (default: all)")
    # fmt: on

    args = parser.parse_args()
    default_model_size, default_model_name, model_configs = get_model_configs(args.config_path)
    args.model_name = first_not_none(args.model_name, default_model_name)
    args.model_size = first_not_none(args.model_size, default_model_size)
    model_config = model_configs.get(args.model_name, {}).get(args.model_size, {})
    args.ncore = first_not_none(args.ncore, model_config.get("ncore", HOUMO_CORE_NUM))
    args.batch = first_not_none(args.batch, model_config.get("batch", 1))
    args.ndevice = first_not_none(args.ndevice, model_config.get("ndevice", 1))
    args.prefill_length = first_not_none(args.prefill_length, model_config.get("prefill_length", 256))
    args.context_length = first_not_none(
        args.context_length,
        parse_context_length(model_config.get("context_length", "2k")),
    )
    if args.context_length < 2048:
        _, others_flash_attention = args.flash_attention
        args.flash_attention = (0, others_flash_attention)
    return args


if __name__ == "__main__":
    args = get_args()
    logger.info(args)

    model_dir = args.model_dir
    model_name = args.model_name
    model_size = args.model_size
    output_dir = args.output_dir
    ncore = args.ncore
    batch = args.batch
    ndevice = args.ndevice
    context_length = args.context_length
    tso = args.enable_stable_opt
    j = args.j
    llm_flash_attention, others_flash_attention = args.flash_attention

    if args.stage == "build" or args.stage == "all":
        assert get_platform() == "x86_64", f"Only supported for compilation on the x86_64 platform."

        build_configs = [
            # Non-LLM sub-models
            {
                "name": "text_projection",
                "flash_attn": others_flash_attention,
            },
            {
                "name": "speech_tokenizer",
                "flash_attn": others_flash_attention,
            },
            # LLM sub-model: Stateful Decoder
            {
                "name": "stateful_decoder",
                "llm_opt": True,
                "flash_attn": llm_flash_attention,
                "llm_batch": batch,
                "ndevice": ndevice,
            },
            # LLM sub-model: Code Predictor
            {
                "name": "code_predictor_prefill",
                "is_prefill": True,
                "llm_opt": True,
                "flash_attn": llm_flash_attention,
                "ndevice": ndevice,
            },
            {
                "name": "code_predictor_decode",
                "llm_opt": True,
                "flash_attn": llm_flash_attention,
                "llm_batch": batch,
                "ndevice": ndevice,
            },
            # LLM sub-model: Talker
            {
                "name": "talker_prefill",
                "is_prefill": True,
                "llm_opt": True,
                "flash_attn": llm_flash_attention,
                "context_length": context_length,
                "prefill_length": args.prefill_length,
                "ndevice": ndevice,
                "enable_common_subgraph": args.enable_common_subgraph,
            },
            {
                "name": "talker_decode",
                "llm_opt": True,
                "llm_batch": batch,
                "flash_attn": llm_flash_attention,
                "context_length": context_length,
                "ndevice": ndevice,
            },
        ]
        if "base" in model_size.lower():
            build_configs += [
                {
                    "name": "speaker_encoder",
                    "flash_attn": others_flash_attention,
                },
                {
                    "name": "speech_tokenizer_encoder",
                    "flash_attn": others_flash_attention,
                },
            ]

        for cfg in build_configs:
            name = cfg["name"]
            if args.models and name not in args.models:
                logger.info(f"Skipping {name} (not in --models list)")
                continue
            kwargs = {k: v for k, v in cfg.items() if k not in ["name"]}
            Xh2Exec.build_from_hmonnx(
                hmonnx=find_hmonnx_file(os.path.join(model_dir, name)),
                hmm_name=f"{model_name}-{model_size}_{name}",
                output=output_dir,
                ncore=ncore,
                enable_xh2_stable_output=tso,
                parallel_jobs=j,
                **kwargs,
            )

    logger.info("\n=== All builds completed. ===")
