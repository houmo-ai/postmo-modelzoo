# Copyright 2025 HOUMO AI
#
# File: ptq.py
# Description:
#   Implementation for post-training quantization of Gemma 4 model.
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

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
import random
import argparse
import torch
from xhquant.api import xhquant_init
from hmatc.utils.utils import first_not_none, get_model_configs, logger

HOUMO_TARGET = os.getenv("HOUMO_TARGET")
assert HOUMO_TARGET in ["xh2"], f"Unsupported HOUMO_TARGET: {HOUMO_TARGET}"

DEFAULT_CONFIG_PATH = os.path.join(os.path.dirname(__file__), "config.yaml")


def get_default_model_dir(model_config: dict) -> str:
    repo_ids = model_config.get("modelscope_repo", [])
    if repo_ids:
        return repo_ids[0].rsplit("/", maxsplit=1)[-1]
    model_name = model_config.get("model_name", "gemma4")
    model_size = model_config.get("model_size", "26b-a4b")
    return f"{model_name}-{model_size}"


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # fmt: off
    parser.add_argument("--config", dest="config_path", type=str, default=DEFAULT_CONFIG_PATH, help="path to config.yaml")
    parser.add_argument("--model", type=str, default=None, help="path to the model directory")
    parser.add_argument("--model-name", type=str, default=None, help="model name for output files")
    parser.add_argument("--model-size", type=str, default=None, help="model size identifier for output files")
    parser.add_argument("--out-dir", type=str, default=f"./output/{HOUMO_TARGET}", help="output directory")
    parser.add_argument("--max_size_w", type=int, default=None, help="max image width for visual model")
    parser.add_argument("--max_size_h", type=int, default=None, help="max image height for visual model")
    parser.add_argument("--context-length", type=int, default=2048, help="max sequence length")
    parser.add_argument("--prefill-chunk-length", type=int, default=256, help="prefill chunk length")
    parser.add_argument("--nsamples", type=int, default=512, help="number of calibration samples")
    parser.add_argument("--seqlen", type=int, default=1024, help="sequence length for calibration")
    parser.add_argument("--mse", type=float, default=2.4, help="MSE threshold for quantization")
    parser.add_argument("--bits", type=int, default=4, choices=[4], help="quantization bits")
    parser.add_argument("--group-size", type=int, default=64, help="group size for quantization")
    parser.add_argument("--hessian-mse", action=argparse.BooleanOptionalAction, default=True, help="enable Hessian MSE assisted optimization")
    parser.add_argument("--calibration-jsonl", type=str, default="./calib_EBSS.jsonl", help="path to calibration dataset in jsonl format")
    parser.add_argument("--calibration-text-key", type=str, default="text", help="key for text field in calibration jsonl")
    parser.add_argument("--seed", type=int, default=42, help="random seed for reproducibility")
    parser.add_argument("--audio-sampling-rate", type=int, default=16000, help="audio sampling rate")
    parser.add_argument("--assistant-model", type=str, default=None, help="path to assistant model directory")

    args = parser.parse_args()

    default_model_size, default_model_name, model_configs = get_model_configs(args.config_path)
    args.model_name = first_not_none(args.model_name, default_model_name)
    args.model_size = first_not_none(args.model_size, default_model_size)
    model_config = model_configs.get(args.model_name, {}).get(args.model_size, {})
    args.model = first_not_none(args.model, get_default_model_dir(model_config))
    args.max_size_w = first_not_none(args.max_size_w, model_config.get("max_size_w", 448))
    args.max_size_h = first_not_none(args.max_size_h, model_config.get("max_size_h", 448))
    # fmt: on

    random.seed(args.seed)
    torch.manual_seed(args.seed)

    xhquant_init(logger=logger)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float16
    logger.info(f"device={device} dtype={dtype}")

    if args.model_size in ["26b-a4b"]:
        from ptq_moe import quant_moe, run_gptqmodel

        # gptq 4bit
        gptqmodel_path = f"{args.model}-gptq-4bit"
        if not os.path.exists(gptqmodel_path):
            run_gptqmodel(args, device, dtype)
        else:
            logger.warning(f"Using existing GPTQ model => {gptqmodel_path}.")

        if args.assistant_model is not None:
            from ptq_mtp import quant_mtp

            quant_mtp(args, device)
        else:
            quant_moe(args, device, dtype)
    elif args.model_size in ["e2b", "e4b"]:
        from ptq_e import quant_e

        quant_e(args, device, dtype)
    else:
        raise ValueError(f"Unsupported model size: {args.model_size}")
