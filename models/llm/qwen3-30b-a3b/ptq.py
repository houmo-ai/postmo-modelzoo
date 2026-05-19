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

import torch
import torch.nn as nn
import gc
from loguru import logger
from safetensors.torch import load_file as load_safetensors_file
from safetensors.torch import save_file as save_safetensors_file
from tqdm import tqdm
from transformers import AutoConfig, AutoModelForCausalLM
from hmatc.utils.monitor import ProcessMemoryMonitor
from hmatc.utils.utils import (
    check_gpu,
    first_not_none,
    get_model_configs,
    parse_context_length,
)

HOUMO_TARGET = os.getenv("HOUMO_TARGET")
assert HOUMO_TARGET in ["xh2"], f"Unsupported HOUMO_TARGET: {HOUMO_TARGET}"

HOUMO_DATASETS_PATH = os.getenv("HOUMO_DATASETS_PATH", "")
DEFAULT_CONFIG_PATH = os.path.join(os.path.dirname(__file__), "config.yaml")


def get_default_model_dir(model_config: dict) -> str:
    repo_ids = model_config.get("modelscope_repo", [])
    if repo_ids:
        return repo_ids[0].rsplit("/", maxsplit=1)[-1]
    model_name = model_config.get("model_name", "qwen3")
    model_size = model_config.get("model_size", "30b-a3b")
    return f"{model_name}-{model_size}"


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ("yes", "true", "t", "y", "1", ""):
        return True
    elif v.lower() in ("no", "false", "f", "n", "0"):
        return False
    else:
        raise argparse.ArgumentTypeError("Boolean value expected.")


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
        with open(args.calib_data, encoding="utf-8") as file:
            for line in file:
                if cnt >= cnt_start:
                    calibration_dataset.append(json.loads(line)["text"])
                cnt = cnt + 1
                if cnt == cnt_start + 512:
                    break
    else:
        calibration_dataset = load_dataset(
            "wikitext", "wikitext-2-raw-v1", split="train"
        ).select(range(512))["text"]

    quant_config = QuantizeConfig(
        bits=4,
        group_size=64,
        hessian_mse=True,
        rotation="hadamard",
        offload_to_disk=False,
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

    from xhquant.api import (
        DeviceType,
        xhquant_init,
        QuantScheme,
        get_root_logger,
    )  # isort:skip
    from xh_model_zoo.utils.memory_tracker import MemoryTracker  # isort:skip
    from xh_model_zoo.utils.time_profiler import TimeProfiler

    import copy

    hf_model_path = osp.normpath(osp.abspath(args.model))
    cfg_name = Path(hf_model_path).name
    src_cfg_name = copy.deepcopy(cfg_name)
    quant_weight = None
    if args.gptqmodel:
        quant_path = os.path.join(
            args.work_dir, "{}_gptqmodel_4bit".format(src_cfg_name)
        )
        hf_model_path = osp.normpath(osp.abspath(quant_path))
    else:
        if not args.skip_quarot:
            cfg_name += "_quarot"
            quant_weight = os.path.join(
                args.work_dir,
                f"{cfg_name}",
                f"{cfg_name[len(src_cfg_name)+1:]}-state-dict.safetensors",
            )

    quant_scheme = QuantScheme(
        target_device=DeviceType.XH2a, quant_type=args.quant_type
    )
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
        LLMConverter.from_pretrained(
            hf_model_path, "Qwen3MoeForCausalLM", config, str(work_dir)
        )


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


def parse_args():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "--config",
        type=str,
        default=DEFAULT_CONFIG_PATH,
        help="path to config.yaml",
    )
    parser.add_argument("--model", type=str, default=None)
    parser.add_argument(
        "--model-name", type=str, default=None, help="output hmonnx model name"
    )
    parser.add_argument("--model-size", type=str, default=None, help="model size")
    parser.add_argument(
        "--calib_data",
        type=str,
        default=None,
        help="calibration dataset choose",
    )
    parser.add_argument("--work-dir", type=str, default="work_dirs/")
    parser.add_argument("--out-dir", type=str, default="output/{}".format(HOUMO_TARGET))
    parser.add_argument("--skip-quarot", action="store_true", help="skip_quarot")
    parser.add_argument("--w-bits", type=int, default=4)
    parser.add_argument("--debug", action="store_true", help="debug mode")
    parser.add_argument(
        "--context-length", type=int, default=None, help="max sequence length"
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
        help="quant type, default is w4a8_ssfp",
    )
    parser.add_argument(
        "--gptqmodel", action="store_true", help="use gptqmodel to quant"
    )
    args = parser.parse_args()
    default_model_size, default_model_name, model_configs = get_model_configs(
        args.config
    )
    args.model_name = first_not_none(args.model_name, default_model_name)
    args.model_size = first_not_none(args.model_size, default_model_size)
    model_config = model_configs.get(args.model_name, {}).get(args.model_size, {})
    args.model = first_not_none(args.model, get_default_model_dir(model_config))
    args.context_length = first_not_none(
        args.context_length,
        parse_context_length(model_config.get("context_length", "32k")),
    )
    args.input_sequence_length = first_not_none(
        args.input_sequence_length, model_config.get("prefill_length", 256)
    )
    args.quant_type = first_not_none(
        args.quant_type, model_config.get("quant_type", "w4a8h0_ssfp")
    )
    if args.calib_data is None:
        args.calib_data = (
            "../../../hmodel/xh2/examples/xh_gen_data/gen_qwen3_30b_EBSS.jsonl"
        )
    return args


if __name__ == "__main__":
    assert check_gpu() is True, "Error: Not found GPU device."
    args = parse_args()
    with ProcessMemoryMonitor(interval=2, quiet=True) as monitor:
        if args.gptqmodel:
            gptq_quant_llm(args)
        else:
            houmo_quant_llm(args)
        houmo_export_llm(args)
        move_llm(args)
    print(
        f"\n=== All quantization steps completed. Peak memory: {monitor.peak_memory_mb:.2f} MB ==="
    )
