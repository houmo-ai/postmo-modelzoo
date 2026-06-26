# Copyright (c) 2025 HOUMO AI
#
# File: quant_pipline.py
# Description:
#   Quantization Pipeline Module - Python script implementing the
# quantization pipeline for Gte models.
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
import torch.nn as nn
import shutil
from typing import Dict, List, Optional
from pathlib import Path
from loguru import logger
from tqdm import tqdm
import os
import os.path as osp
from pathlib import Path
from transformers import AutoConfig, Qwen2ForCausalLM, AutoModelForCausalLM
from safetensors.torch import load_file as load_safetensors_file
from safetensors.torch import save_file as save_safetensors_file

from xhquant.api import (
    DeviceType,
    xhquant_init,
    QuantScheme,
    get_root_logger,
)  # isort:skip

from xh_model_zoo.xh_llm import LLMConverter
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
        if (cand / "dataset_dict.json").is_file():
            local_wikitext_path = cand
            break
        if (cand / "wikitext-2-raw-v1" / "dataset_dict.json").is_file():
            local_wikitext_path = cand / "wikitext-2-raw-v1"
            break

    if local_wikitext_path is None:
        return

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


def _patch_tokenizer_to_use_fast():
    """Monkey-patch AutoTokenizer.from_pretrained to force use_fast=True.

    The gte model directory lacks merges.txt, which the slow Python tokenizer
    (Qwen2Tokenizer, triggered by use_fast=False) requires.  The fast tokenizer
    reads the self-contained tokenizer.json instead, so it works without
    merges.txt.  We override any use_fast=False to True to avoid the crash.
    """
    import transformers

    _original = transformers.AutoTokenizer.from_pretrained

    def _patched(pretrained_model_name_or_path, *args, **kwargs):
        if kwargs.get("use_fast") is False:
            kwargs["use_fast"] = True
        return _original(pretrained_model_name_or_path, *args, **kwargs)

    transformers.AutoTokenizer.from_pretrained = _patched


def _get_work_dir(args) -> Path:
    out_dir = Path(args.work_dir)
    hf_model_path = osp.normpath(osp.abspath(args.model))
    model_name = Path(hf_model_path).name
    cfg_name = model_name

    if not args.skip_quarot:
        cfg_name += "_quarot"
    if not args.skip_gptq:
        cfg_name += "_gptq"

    work_dir = out_dir / cfg_name
    work_dir.mkdir(exist_ok=True, parents=True)
    return work_dir


def _load_and_prepare_model(hf_model_dir: str, device: str) -> nn.Module:
    dtype = torch.bfloat16
    config = AutoConfig.from_pretrained(hf_model_dir)

    native_model: nn.Module = Qwen2ForCausalLM.from_pretrained(
        hf_model_dir,
        torch_dtype=dtype,
        device_map="cpu",
        config=config,
        trust_remote_code=True,
        attn_implementation="eager",
    )

    if native_model.config.use_cache:
        native_model.config.use_cache = False

    if native_model.config.tie_word_embeddings:
        native_model.config.torchscript = True
        native_model.tie_weights()
        native_model.config.tie_word_embeddings = False

    native_model.eval()
    native_model.to(device)

    return native_model


def _calculate_gpu_memory() -> str:
    consumption = torch.cuda.max_memory_allocated()
    unit = "B"

    if consumption > 1024:
        consumption /= 1024
        unit = "k"

        if consumption > 1024:
            consumption /= 1024
            unit = "M"

    consumption = round(consumption, 2)
    return f"{consumption}{unit}"


def _apply_quarot_quantization(
    model: nn.Module, work_dir: Path, device: str, quant_methods: List[str]
) -> nn.Module:
    torch.cuda.reset_peak_memory_stats()
    quant_methods.append("quarot")
    quant_name = "_".join(quant_methods)
    filename = work_dir / f"{quant_name}-state-dict.safetensors"

    if not os.path.exists(filename):
        from xh_model_zoo.xh_llm.quarot.quantizer_utils import quarot

        logger.info(msg_output_format("Start quarot quantization"))
        model = quarot(model, device=device)
        logger.info(msg_output_format("End quarot quantization"))

        state_dict = model.state_dict()
        logger.info(msg_output_format(f"Saving checkpoint to: {filename}"))
        save_safetensors_file(state_dict, filename)
        logger.info(f"Save checkpoint to: {filename}")
    else:
        state_dict = load_safetensors_file(filename)
        model.load_state_dict(state_dict)
        logger.info(msg_output_format(f"Load state_dict from {filename}"))

    memory_usage = _calculate_gpu_memory()
    logger.info(f"GPU memory cost for export {memory_usage}")

    return model


def _apply_gptq_quantization(
    model: nn.Module,
    args,
    work_dir: Path,
    hf_model_dir: str,
    device: str,
    quant_methods: List[str],
) -> nn.Module:
    # 使 AutoTokenizer 跳过 use_fast=False（gte 模型缺少 merges.txt）
    _patch_tokenizer_to_use_fast()
    # 尝试使用本地 wikitext-2-raw-v1 数据集（若存在），避免从 HuggingFace Hub 下载
    _patch_wikitext2_if_local(getattr(args, 'calibration_dataset', None), args.datasets_dir, hf_model_dir)

    from xh_model_zoo.xh_llm.quarot.quantizer_utils import gptq

    # GPTQ config
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
        heading_gptq=False,
    )

    torch.cuda.reset_peak_memory_stats()
    logger.info(msg_output_format("Start gptq quantization"))
    quant_methods.append("gptq")

    layers_cache_dir = work_dir / "layers_cache"
    layers_cache_dir.mkdir(exist_ok=True, parents=True)

    # gptq quant
    model = gptq(
        model,
        args=args,
        model_name=hf_model_dir,
        **gptq_config,
        device=device,
        layers_cache_dir=str(layers_cache_dir),
        cache_dir=args.datasets_dir,
    )

    logger.info(msg_output_format("End gptq quantization"))

    memory_usage = _calculate_gpu_memory()
    logger.info(f"GPU memory cost for export {memory_usage}")

    return model


def _optimize_state_dict(
    state_dict: Dict[str, torch.Tensor],
) -> Dict[str, torch.Tensor]:
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

    return state_dict


def _save_final_quantized_model(
    model: nn.Module, work_dir: Path, quant_methods: List[str]
) -> None:
    quant_name = "_".join(quant_methods)
    filename = work_dir / f"{quant_name}-state-dict.safetensors"

    state_dict = model.state_dict()
    optimized_state_dict = _optimize_state_dict(state_dict)

    logger.info(msg_output_format(f"Saving checkpoint to: {filename}"))
    save_safetensors_file(optimized_state_dict, filename)
    logger.info(f"Save checkpoint to: {filename}")


def quant_llm(args):
    work_dir = _get_work_dir(args)
    hf_model_dir = args.model

    device = "cuda" if torch.cuda.is_available() else "cpu"

    native_model = _load_and_prepare_model(hf_model_dir, device)

    quant_methods = []

    # Quarot quant
    if not args.skip_quarot:
        native_model = _apply_quarot_quantization(
            native_model, work_dir, device, quant_methods
        )

    # GPTQ quant
    if not args.skip_gptq:
        native_model = _apply_gptq_quantization(
            native_model, args, work_dir, hf_model_dir, device, quant_methods
        )

    # save quant model
    if len(quant_methods) != 0:
        _save_final_quantized_model(native_model, work_dir, quant_methods)


def export_llm(args):
    from xh_model_zoo.xh_llm.models.qwen2_ste import SteQwen2ConvertConfig

    hf_model_path = osp.normpath(osp.abspath(args.model))
    model_name = Path(hf_model_path).name
    target_device = DeviceType.XH2a
    quant_type = args.quant_type
    quant_scheme = QuantScheme(target_device=DeviceType.XH2a, quant_type=quant_type)

    prefix = f"{model_name}-{target_device}-{args.context_length//1024}k-{quant_type}"
    work_dir = Path(args.work_dir) / prefix
    work_dir.mkdir(exist_ok=True, parents=True)

    hf_model_dir = args.model
    dtype = torch.bfloat16
    config = AutoConfig.from_pretrained(hf_model_dir)
    native_model: nn.Module = Qwen2ForCausalLM.from_pretrained(
        hf_model_dir,
        torch_dtype=dtype,
        device_map="cpu",
        config=config,
        trust_remote_code=False,
        attn_implementation="eager",
    )

    if native_model.config.use_cache:
        native_model.config.use_cache = False

    token_embedding = native_model.model.get_input_embeddings()
    token_embedding_file = Path(work_dir) / "token_embedding.pt"
    torch.save(
        token_embedding.state_dict()["weight"].float().half(), str(token_embedding_file)
    )

    config = SteQwen2ConvertConfig(
        batch_size=args.batch,
        context_length=args.context_length,
        quant_scheme=quant_scheme,
    )

    log_file = work_dir / "convert.log"
    xhquant_init(log_file, debug=args.debug)
    logger = get_root_logger()
    with TimeProfiler("convert", logger), MemoryTracker("cuda:0", "convert", logger):
        dst_path = os.path.join(".", "gte_qwen2_1.5b_inst")
        if not os.path.exists(dst_path):
            os.symlink(src=str(hf_model_path), dst=str(dst_path))
        LLMConverter.from_pretrained(dst_path, "gte_qwen2", config, work_dir)
        if os.path.exists(dst_path) and Path(dst_path).is_symlink():
            Path(dst_path).unlink()


def move_llm(args):
    target_device = DeviceType.XH2a
    prefix_model_name = "gte_qwen2_1.5b_inst"
    model_name = os.path.basename(args.model)
    quant_type = args.quant_type

    prefix = f"{model_name}-{target_device}-{args.context_length//1024}k-{quant_type}"
    work_dir = Path(args.work_dir) / prefix
    dest_dir = Path(args.out_dir)

    prefix = f"{prefix_model_name}-{target_device}-batch_{args.batch}-{args.context_length//1024}k-{quant_type}_prefill"
    hmm_model_dir = work_dir / "hmonnx" / "golden" / "prefill"
    for file in os.listdir(hmm_model_dir):
        if os.path.isdir(os.path.join(hmm_model_dir, file)):
            continue
        else:
            src_file = os.path.join(hmm_model_dir, file)
            dst_file = os.path.join(hmm_model_dir, "step_0", file)
            shutil.move(src_file, dst_file)

    hmm_model_dir = os.path.join(hmm_model_dir, "step_0")
    logger.info(
        msg_output_format("Start move from {} to {}").format(
            hmm_model_dir, dest_dir / "hmquant" / "prefill"
        )
    )

    shutil.move(hmm_model_dir, dest_dir / "hmquant" / "prefill")

    shutil.move(
        work_dir / "token_embedding.pt",
        dest_dir / "hmquant/quant_embedding.pt",
    )

    out_path = dest_dir / "hmquant" / "prefill"
    for file in os.listdir(out_path):
        if os.path.isdir(os.path.join(out_path, file)):
            shutil.rmtree(os.path.join(out_path, file))
        else:
            if prefix in file and file.endswith(("onnx", "npy")):
                dst_file = os.path.join(out_path, file.replace(prefix, args.model_name))
                src_file = os.path.join(out_path, file)
                shutil.move(src_file, dst_file)

    logger.info(msg_output_format("Start remove work_dir: {}".format(work_dir)))
    shutil.rmtree(work_dir, ignore_errors=True)
