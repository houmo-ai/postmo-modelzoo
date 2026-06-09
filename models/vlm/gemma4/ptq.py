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
import re

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
import shutil
import argparse
import torch
from xhquant.api import get_root_logger
from hmatc.utils.utils import first_not_none, get_model_configs

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


def is_quantized_model(model_dir: str) -> bool:
    quantize_config = os.path.join(model_dir, "quantize_config.json")
    return os.path.exists(quantize_config)


def remove_path(path: str) -> None:
    if os.path.isdir(path) and not os.path.islink(path):
        shutil.rmtree(path)
    elif os.path.exists(path):
        os.remove(path)


def flatten_exported_hmquant_dir(export_output_dir: str, hmquant_dir: str) -> None:
    if not os.path.isdir(export_output_dir):
        raise FileNotFoundError(
            f"Export output directory not found: {export_output_dir}"
        )

    pattern = re.compile(r"^hmquant_xh2.*_\d{8}$")
    candidates = [
        entry
        for entry in os.listdir(export_output_dir)
        if os.path.isdir(os.path.join(export_output_dir, entry))
        and pattern.match(entry)
    ]
    if not candidates:
        raise FileNotFoundError(
            f"No hmquant_xh2*_YYYYMMDD directory found under {export_output_dir}"
        )

    exported_hmquant_dir = os.path.join(export_output_dir, sorted(candidates)[-1])
    if os.path.exists(hmquant_dir) and not os.path.isdir(hmquant_dir):
        remove_path(hmquant_dir)
    os.makedirs(hmquant_dir, exist_ok=True)

    for name in os.listdir(exported_hmquant_dir):
        src = os.path.join(exported_hmquant_dir, name)
        dst = os.path.join(hmquant_dir, name)
        remove_path(dst)
        shutil.move(src, dst)

    shutil.rmtree(export_output_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # fmt: off
    parser.add_argument("--config", dest="config_path", type=str, default=DEFAULT_CONFIG_PATH, help="path to config.yaml")
    parser.add_argument("--model", type=str, default=None, help="path to the model directory")
    parser.add_argument("--model-name", type=str, default=None, help="model name for output files")
    parser.add_argument("--model-size", type=str, default=None, help="model size identifier for output files")
    parser.add_argument("--out-dir", type=str, default=f"./output/{HOUMO_TARGET}", help="output directory")
    parser.add_argument("--context-length", type=int, default=2048, help="max sequence length")
    parser.add_argument("--prefill-chunk-length", type=int, default=256, help="prefill chunk length")
    parser.add_argument("--seed", type=int, default=42, help="random seed for reproducibility")
    parser.add_argument("--debug", action="store_true", help="enable debug mode")
    args = parser.parse_args()

    default_model_size, default_model_name, model_configs = get_model_configs(args.config_path)
    args.model_name = first_not_none(args.model_name, default_model_name)
    args.model_size = first_not_none(args.model_size, default_model_size)
    model_config = model_configs.get(args.model_name, {}).get(args.model_size, {})
    args.model = first_not_none(args.model, get_default_model_dir(model_config))
    # fmt: on

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if args.model_size in ["26b-a4b"]:
        quant_config_path = "configs/gemma4_26b_a4b_full.yaml"
    elif args.model_size in ["e2b"]:
        quant_config_path = "configs/gemma4_e2b_full.yaml"
    elif args.model_size in ["e4b"]:
        quant_config_path = "configs/gemma4_e4b_full.yaml"
    elif args.model_size in ["31b"]:
        quant_config_path = "configs/gemma4_31b_full.yaml"
    else:
        raise ValueError(f"Unsupported model size: {args.model_size}")

    from xhmodel_merak.xh_llm.workflows import AutoLLMWorkflow

    workflow = AutoLLMWorkflow.from_config(
        hf_model_dir=args.model,
        config_path=quant_config_path,
        seed=args.seed,
        debug=args.debug,
    )

    logger = get_root_logger()

    quant_output_dir = os.path.join(args.out_dir, "hmquant", "quantized_model")
    if not is_quantized_model(args.model) and os.path.exists(quant_output_dir):
        logger.warning(
            f"Output directory already exists: {quant_output_dir}, removing it."
        )
        shutil.rmtree(quant_output_dir)

    config_overrides = None
    if is_quantized_model(args.model):
        config_overrides = dict(
            quant=dict(
                algorithm="existing_hf",
                artifact_format="gptqmodel_hf",
                output_format="gptqmodel_hf",
                existing_hf_model_dir=args.model,
            )
        )

    quant_result = workflow.quant(
        output_dir=quant_output_dir,
        device=device,
        config_overrides=config_overrides,
    )

    export_output_dir = os.path.join(args.out_dir, "hmquant", "exported_model")
    if os.path.exists(export_output_dir):
        logger.warning(
            f"Output directory already exists: {export_output_dir}, removing it."
        )
        shutil.rmtree(export_output_dir)

    config_overrides = {
        "export.model.context_max_length": args.context_length,
        "export.model.prefill_chunk_length": args.prefill_chunk_length,
    }

    export_result = workflow.export(
        quant_result=quant_result,
        output_dir=str(export_output_dir),
        device=device,
        config_overrides=config_overrides,
    )
    flatten_exported_hmquant_dir(
        export_output_dir=export_output_dir,
        hmquant_dir=os.path.join(args.out_dir, "hmquant"),
    )
