# Copyright (c) 2025 HOUMO AI
#
# File: quant_pipline.py
# Description:
#   Quantization Pipeline Module - Python script implementing the
# quantization pipeline for Qwen3 models.
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
import torch
import shutil
from pathlib import Path
from loguru import logger
from tqdm import tqdm
import os
import os.path as osp
from pathlib import Path
from transformers import AutoConfig, AutoModelForCausalLM
from safetensors.torch import load_file as load_safetensors_file
from safetensors.torch import save_file as save_safetensors_file

from xhquant.api import (
    DeviceType,
    xhquant_init,
    QuantScheme,
    get_root_logger,
)  # isort:skip

from xh_model_zoo.xh_llm import LLMConverter
from xh_model_zoo.xh_llm.models.qwen3_legacy import Qwen3LegacyConvertConfig
from xh_model_zoo.utils.memory_tracker import MemoryTracker  # isort:skip
from xh_model_zoo.utils.time_profiler import TimeProfiler  # isort:skip


def msg_output_format(title):
    padding_str = "*" * 10
    title = f"{padding_str} {title} {padding_str}"
    return title


def _patch_wikitext2_if_local(calib_data: str | None, datasets_dir: str | None, model_name: str):
    """Monkey-patch datasets.load_dataset to use a local wikitext-2-raw-v1 copy.

    Search priority: calib_data > datasets_dir/wikitext-2-raw-v1 > ./wikitext-2-raw-v1.
    Falls back silently to the original HuggingFace Hub download if not found.
    """
    import datasets as _ds
    from datasets import load_from_disk

    candidates = []
    if calib_data:
        candidates.append(Path(calib_data))
    if datasets_dir:
        candidates.append(Path(datasets_dir) / "wikitext-2-raw-v1")
    candidates.append(Path("wikitext-2-raw-v1"))

    local_wikitext_path = None
    for cand in candidates:
        cand = cand.resolve()
        # May point directly to wikitext-2-raw-v1/ or to its parent
        if (cand / "dataset_dict.json").is_file():
            local_wikitext_path = cand
            break
        if (cand / "wikitext-2-raw-v1" / "dataset_dict.json").is_file():
            local_wikitext_path = cand / "wikitext-2-raw-v1"
            break

    if local_wikitext_path is None:
        return  # No local data found, fall through to original HuggingFace Hub download

    logger.info(f"Found local wikitext-2-raw-v1 at {local_wikitext_path}, monkey-patching datasets.load_dataset")

    _original_load = _ds.load_dataset

    def _hijacked_load_dataset(path, name=None, **kwargs):
        if path == "wikitext" and name == "wikitext-2-raw-v1":
            full = load_from_disk(str(local_wikitext_path))
            split = kwargs.get("split", None)
            if split == "train":
                return full["train"]
            elif split == "test":
                return full["test"]
            return full
        return _original_load(path, name, **kwargs)

    _ds.load_dataset = _hijacked_load_dataset


def quant_llm(args):
    out_dir = Path(args.work_dir)
    hf_model_dir = args.model

    hf_model_path = osp.normpath(osp.abspath(args.model))
    model_name = Path(hf_model_path).name
    cfg_name = model_name
    if not args.skip_quarot:
        cfg_name += "_quarot"
    if not args.skip_gptq:
        cfg_name += "_gptq"

    work_dir = Path(out_dir) / cfg_name
    work_dir.mkdir(exist_ok=True, parents=True)
    config = AutoConfig.from_pretrained(hf_model_dir)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16
    # # for qwen2-0.5b 1.5b & 3b
    # if args.tie_embed_head:
    #     config.tie_word_embeddings = False

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

    if not args.skip_gptq:
        # 尝试使用本地 wikitext-2-raw-v1 数据集（若存在），避免从 HuggingFace Hub 下载
        _patch_wikitext2_if_local(args.calib_data, args.datasets_dir, hf_model_dir)

        from xh_model_zoo.xh_llm.quarot.quantizer_utils import gptq

        gptq_config = dict(
            calib_dataset="wikitext2",
            calib_samples=128,
            seqlen=2048,
            w_clip=True,
            w_bits=args.w_bits,
            w_asym=False,
            w_groupsize=64,
            percdamp=0.01,
            act_order=False,
            int8_down_proj=False,
            heading_gptq=True,
        )

        torch.cuda.reset_peak_memory_stats()
        logger.info(msg_output_format("Start gptq quantization"))
        quant_methods.append("gptq")
        layers_cache_dir = work_dir / "layers_cache"
        layers_cache_dir.mkdir(exist_ok=True, parents=True)

        native_model = gptq(
            native_model,
            args=args,
            model_name=hf_model_dir,
            **gptq_config,
            device=device,
            layers_cache_dir=str(layers_cache_dir),
            cache_dir=args.datasets_dir,
        )
        logger.info(msg_output_format("End gptq quantization"))

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


def export_llm(args):
    hf_model_path = osp.normpath(osp.abspath(args.model))
    model_name = Path(hf_model_path).name
    target_device = DeviceType.XH2a
    quant_type = args.quant_type
    quant_scheme = QuantScheme(target_device=DeviceType.XH2a, quant_type=quant_type)
    model_name = os.path.basename(args.model)
    model_dir = os.path.join(args.work_dir, "{}_quarot_gptq".format(model_name))
    quant_weight = os.path.join(model_dir, "quarot_gptq-state-dict.safetensors")
    config = Qwen3LegacyConvertConfig(
        batch_size=1,
        context_length=args.context_length,
        input_sequence_length=args.input_sequence_length,
        quant_scheme=quant_scheme,
        quant_weight=quant_weight,
        mix_search=args.mix_search,
        num_logits_to_keep=args.num_logits_to_keep,
    )

    prefix = f"{model_name}-{target_device}-{args.context_length//1024}k-{quant_type}"
    work_dir = Path("work_dirs") / prefix
    work_dir.mkdir(exist_ok=True, parents=True)
    log_file = work_dir / "convert.log"
    xhquant_init(log_file, debug=args.debug)
    logger = get_root_logger()
    with TimeProfiler("convert", logger), MemoryTracker("cuda:0", "convert", logger):
        LLMConverter.from_pretrained(
            hf_model_path, "Qwen3Embedding", config, str(work_dir)
        )


def move_models(
    work_dir: Path,
    source: str = "prefill",
    model: str = "prefill",
    target_name: str = "hmquant_qwen3_with_act.onnx",
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
        work_dir / hmm_model_dir / "token_embedding.pt",
        dest_dir / "hmquant/quant_embedding.pt",
    )
    logger.info(msg_output_format("Start remove work_dir: {}".format(work_dir)))
    shutil.rmtree(work_dir, ignore_errors=True)
