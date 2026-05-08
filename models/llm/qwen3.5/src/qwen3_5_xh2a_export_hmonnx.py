# Copyright (c) 2025 HOUMO AI
#
# File: qwen3_5_xh2a_export_hmonnx.py
# Description:
#   Export Qwen3.5 LLM prefill/decode graphs to ONNX/HMONNX for XH2a.
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

import json
import re
import shutil
import subprocess
import sys
import time
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Tuple

project_root = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import torch
import torch.nn as nn
import xhquant.utils.suppress_printing
from transformers import AutoConfig
from xhquant.api import (
    CacheTensor,
    Config,
    ConfigDict,
    HMONNXGoldenInference,
    PrecisionMode,
    ptq_quantize,
    set_random_seed,
)
from xhquant.xhonnxruntime import AutoOffloadGraphModel, HMONNXGraphGoldenInference

try:
    from ._export_validation import (
        collect_hf_references,
        cleanup_memory,
        run_conversion_validation,
    )
    from .common import decode_next_token, get_root_logger, xhquant_llm_init
except ImportError:
    from _export_validation import (
        collect_hf_references,
        cleanup_memory,
        run_conversion_validation,
    )
    from common import decode_next_token, get_root_logger, xhquant_llm_init

from xh_model_zoo.xh_llm.models.builder import MODELS
from xh_model_zoo.xh_llm.models.eval_model_type import EvalModelType
from xh_model_zoo.xh_llm.models.qwen3_5 import XHQwen3_5Model

torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = False

DEFAULT_NO_SPLIT_MODULES = [
    "XHTrace_Qwen3_5DecoderLayer",
    "XHTrace_Qwen3_5SparseMoeBlock",
    "MoeBlock",
]

DTYPE_NAME_MAP = {
    "auto": "auto",
    "fp16": "float16",
    "float16": "float16",
    "half": "float16",
    "fp32": "float32",
    "float32": "float32",
    "bf16": "bfloat16",
    "bfloat16": "bfloat16",
    "bfp16": "bfloat16",
}


def _flatten_inputs(inputs):
    flat = []
    for arg in inputs:
        if isinstance(arg, (list, tuple)):
            flat.extend(arg)
        else:
            flat.append(arg)
    return flat


def _extract_quant_method(config: AutoConfig) -> Optional[str]:
    quantization_config = getattr(config, "quantization_config", None)
    quant_method = getattr(quantization_config, "quant_method", None)
    if isinstance(quantization_config, dict):
        quant_method = quantization_config.get("quant_method", quant_method)
    return None if quant_method is None else str(quant_method).lower()


def _detect_source_quant_method(hf_model_dir: str) -> Optional[str]:
    config = AutoConfig.from_pretrained(hf_model_dir, trust_remote_code=True)
    return _extract_quant_method(config)


def _resolve_compute_dtype_name(raw_dtype: str, hf_model_dir: str) -> str:
    normalized = _normalize_dtype_name(raw_dtype)
    if normalized != "auto":
        return normalized
    return "float16"


def _detect_export_quant_type(hf_model_dir: str) -> str:
    quant_method = _detect_source_quant_method(hf_model_dir)
    return "w4a8h0_ssfp" if quant_method == "gptq" else "w8a8h1_sefp"


def _detect_release_wmix_amix(hf_model_dir: str) -> str:
    quant_method = _detect_source_quant_method(hf_model_dir)
    return "w4_a8" if quant_method == "gptq" else "w8_a8"


def _normalize_dtype_name(dtype_name: str) -> str:
    key = dtype_name.strip().lower()
    normalized = DTYPE_NAME_MAP.get(key, None)
    if normalized is None:
        valid_values = ", ".join(sorted(DTYPE_NAME_MAP.keys()))
        raise ValueError(
            f"Unsupported dtype: {dtype_name}. Valid values: {valid_values}"
        )
    return normalized


def _normalize_max_memory_spec(max_memory: Dict[Any, Any]) -> Dict[Any, Any]:
    normalized: Dict[Any, Any] = {}
    for raw_key, value in max_memory.items():
        key = raw_key
        if isinstance(raw_key, str):
            stripped_key = raw_key.strip()
            lowered_key = stripped_key.lower()
            if stripped_key.isdigit():
                key = int(stripped_key)
            elif (
                lowered_key.startswith("cuda:")
                and lowered_key.split(":", 1)[1].isdigit()
            ):
                key = int(lowered_key.split(":", 1)[1])
            elif lowered_key in {"cpu", "mps", "disk"}:
                key = lowered_key
        normalized[key] = value
    return normalized


def _get_default_device() -> torch.device:
    return torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")


def _get_no_split_modules(args) -> List[str]:
    raw = getattr(args, "no_split_modules", "")
    modules = [module.strip() for module in raw.split(",") if module.strip()]
    return modules if modules else list(DEFAULT_NO_SPLIT_MODULES)


def _de_dispatch_to_cpu(model: torch.nn.Module):
    import accelerate.hooks

    accelerate.hooks.remove_hook_from_module(model, recurse=True)
    if hasattr(model, "hf_device_map"):
        delattr(model, "hf_device_map")
    model.to("cpu")
    cleanup_memory()


def _format_dtype(dtype: Optional[torch.dtype]) -> str:
    if dtype is None:
        return "None"
    return str(dtype).replace("torch.", "")


def _infer_first_floating_dtype(module: torch.nn.Module) -> Optional[torch.dtype]:
    for tensor in module.parameters(recurse=True):
        if tensor.is_floating_point():
            return tensor.dtype
    for tensor in module.buffers(recurse=True):
        if tensor.is_floating_point():
            return tensor.dtype
    return None


def _log_module_dtype(
    tag: str, module: Optional[torch.nn.Module], expected_dtype: torch.dtype, logger
) -> None:
    if module is None:
        logger.warning(f"{tag} is None, cannot inspect dtype.")
        return
    observed_dtype = _infer_first_floating_dtype(module)
    logger.info(
        f"{tag} first floating tensor dtype: {_format_dtype(observed_dtype)}, "
        f"target dtype: {_format_dtype(expected_dtype)}"
    )
    if observed_dtype is not None and observed_dtype != expected_dtype:
        logger.warning(
            f"{tag} dtype mismatch: observed={_format_dtype(observed_dtype)}, "
            f"target={_format_dtype(expected_dtype)}"
        )


def _get_cache_dtype(caches: Any) -> Optional[torch.dtype]:
    if caches is None or len(caches) == 0:
        return None
    cache0 = caches[0]
    if isinstance(cache0, torch.Tensor):
        return cache0.dtype
    if hasattr(cache0, "dtype"):
        return cache0.dtype
    return None


def _build_prompt_inputs(
    tokenizer, device, max_len: int, prompt: str, system_prompt: str
):
    if hasattr(tokenizer, "apply_chat_template"):
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt},
        ]
        text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        input_ids = tokenizer([text], return_tensors="pt").input_ids
    else:
        input_ids = tokenizer(prompt, return_tensors="pt").input_ids

    if input_ids.shape[1] > max_len:
        input_ids = input_ids[:, :max_len]
    return input_ids.to(device)


def _build_linear_attn_mask(
    valid_len: int, mask_info, device: torch.device
) -> torch.Tensor:
    mask = torch.zeros(mask_info.shape, dtype=mask_info.dtype, device=device)
    if valid_len > 0:
        slices = [slice(None)] * mask.dim()
        slices[-1] = slice(0, valid_len)
        mask[tuple(slices)] = 1
    return mask


def _build_inputs_embeds(
    token_embedding: nn.Module,
    input_ids: torch.Tensor,
    target_seq_len: int,
    pad_token_id: int,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    input_ids = input_ids.to(device)
    if input_ids.shape[1] < target_seq_len:
        pad_len = target_seq_len - input_ids.shape[1]
        pad_ids = torch.full(
            (input_ids.shape[0], pad_len),
            pad_token_id,
            dtype=torch.long,
            device=device,
        )
        input_ids = torch.cat([input_ids, pad_ids], dim=1)
    elif input_ids.shape[1] > target_seq_len:
        input_ids = input_ids[:, :target_seq_len]
    inputs_embeds = token_embedding(input_ids).to(dtype)
    return inputs_embeds


def _is_cache_input_name(name: str) -> bool:
    return (
        name.startswith(
            (
                "past_key_cache_",
                "past_value_cache_",
                "past_conv_cache_",
                "past_recurrent_state_",
            )
        )
        or "kcache_input" in name
        or "vcache_input" in name
    )


def _ensure_cache_tensor(tensor: torch.Tensor) -> CacheTensor:
    if isinstance(tensor, CacheTensor):
        return tensor
    return CacheTensor(tensor)


def _alloc_cache_inputs(
    hm_model: HMONNXGoldenInference, device: torch.device
) -> Dict[str, torch.Tensor]:
    cache_inputs: Dict[str, torch.Tensor] = {}
    for name in hm_model.get_input_names():
        if _is_cache_input_name(name):
            info = hm_model.get_input(name)
            cache_inputs[name] = _ensure_cache_tensor(
                torch.zeros(info.shape, dtype=info.dtype, device=device)
            )
    return cache_inputs


def _resolve_input_name(
    hm_model: HMONNXGoldenInference, candidates: Tuple[str, ...], fallback=None
) -> str:
    input_names = hm_model.get_input_names()
    for name in candidates:
        if name in input_names:
            return name
    if fallback is not None:
        return fallback(hm_model)
    raise ValueError(f"None of {candidates} found in inputs: {input_names}")


def _infer_inputs_embeds_name(hm_model: HMONNXGoldenInference) -> str:
    for name in hm_model.get_input_names():
        info = hm_model.get_input(name)
        if (
            info.dtype in (torch.float16, torch.float32, torch.bfloat16)
            and len(info.shape) == 3
        ):
            return name
    return hm_model.get_input_names()[0]


def _run_hmonnx_with_golden(
    hm_model: HMONNXGoldenInference,
    input_feed: Dict[str, torch.Tensor],
) -> Tuple[Tuple[torch.Tensor, ...], Dict[str, torch.Tensor]]:
    outputs = hm_model.run(input_feed)
    if not isinstance(outputs, (tuple, list)):
        outputs = (outputs,)
    output_names = hm_model.get_output_names()
    output_map = {name: out for name, out in zip(output_names, outputs)}
    return tuple(outputs), output_map


def _to_context_length_str(context_length) -> str:
    if isinstance(context_length, str):
        ctx = context_length.strip().lower()
        if ctx.endswith("k"):
            return ctx
        if ctx.isdigit():
            value = int(ctx)
            return f"{value // 1024}k" if value % 1024 == 0 else ctx
        return ctx
    if context_length % 1024 == 0:
        return f"{context_length // 1024}k"
    return str(context_length)


def _extract_model_size_from_candidates(*candidates) -> str:
    for text in candidates:
        match = re.search(r"(\d+[Bb])", str(text))
        if match:
            return match.group(1).upper()
    return "unknown"


def _with_model_size_in_name(modelscope_name: str, model_size: str) -> str:
    normalized_name = str(modelscope_name).strip()
    normalized_size = str(model_size).strip().lower()
    if not normalized_name or normalized_size == "unknown":
        return normalized_name
    if re.search(
        rf"(?<![0-9a-z]){re.escape(normalized_size)}(?![0-9a-z])",
        normalized_name.lower(),
    ):
        return normalized_name
    return f"{normalized_name}_{model_size.lower()}"


def _build_release_prefix(cfg) -> str:
    from datetime import datetime as _dt

    release_cfg = cfg.get("release", {})
    xh_version = str(release_cfg.get("xh_version", "xh2"))
    modelscope_name = str(release_cfg.get("modelscope_name", "qwen3_5"))
    model_size = _extract_model_size_from_candidates(
        release_cfg.get("model_size", ""),
        getattr(cfg, "hf_model_dir", ""),
        getattr(cfg, "cfg_name", ""),
    )
    release_model_name = _with_model_size_in_name(modelscope_name, model_size)
    wmix_amix = str(release_cfg.get("wmix_amix", "w8_a8"))
    prefill_length = int(
        release_cfg.get(
            "prefill_length", cfg.model.wrap_cfg.get("input_sequence_length", 256)
        )
    )
    context_length = _to_context_length_str(
        release_cfg.get(
            "context_length", cfg.model.wrap_cfg.get("max_sequence_length", 2048)
        )
    )
    date_val = release_cfg.get("date", None)
    date_str = _dt.now().strftime("%Y%m%d") if date_val is None else str(date_val)
    prefix = f"hmquant_{xh_version}_{release_model_name}_{wmix_amix}_{prefill_length}_{context_length}_{date_str}"
    return prefix.lower()


def _copy_path(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if src.is_dir():
        if dst.exists():
            shutil.rmtree(dst)
        shutil.copytree(src, dst, symlinks=True)
    else:
        shutil.copy2(src, dst)


def _find_external_data(onnx_path: Path):
    candidates = [
        onnx_path.with_name(f"{onnx_path.stem}_external_data"),
        onnx_path.with_name(onnx_path.stem.replace("_with_act", "_external_data")),
    ]
    for path in candidates:
        if path.exists():
            return path
    return None


def _ensure_step0_layout(golden_dir: Path, logger) -> None:
    step0_dir = golden_dir / "step_0"
    step0_dir.mkdir(exist_ok=True, parents=True)
    for item in list(golden_dir.iterdir()):
        if item.name == "step_0":
            continue
        if item.name.endswith(".onnx") or item.name.endswith("_external_data"):
            continue
        if item.is_dir() and item.name.startswith("step_"):
            continue
        if item.name == "logits.npy" or item.name.startswith("hmquant_"):
            target = step0_dir / item.name
            if target.exists():
                if target.is_dir():
                    shutil.rmtree(target)
                else:
                    target.unlink()
            logger.info(f"Move golden artifact into step_0: {item} -> {target}")
            shutil.move(str(item), str(target))


def _cleanup_hmonnx_in_golden(golden_dir: Path, release_prefix: str, logger) -> None:
    _ensure_step0_layout(golden_dir, logger)

    for step_dir in sorted(golden_dir.glob("step_*")):
        if not step_dir.is_dir():
            continue
        if step_dir.name != "step_0":
            logger.info(f"Cleanup: removing extra golden step dir {step_dir}")
            shutil.rmtree(step_dir)
            continue
        for fpath in step_dir.iterdir():
            if fpath.is_dir() and not fpath.is_symlink():
                continue
            if fpath.name.endswith(".onnx") or fpath.name.endswith("_external_data"):
                logger.info(f"Cleanup: removing {fpath}")
                fpath.unlink()

    for fpath in golden_dir.iterdir():
        if fpath.name.startswith(release_prefix):
            continue
        if fpath.name == "step_0":
            continue
        if fpath.is_dir() and not fpath.is_symlink():
            continue
        if fpath.name.endswith(".onnx"):
            logger.info(f"Cleanup: removing {fpath}")
            fpath.unlink()


def _load_token_embedding(embed_path: Path) -> nn.Module:
    try:
        obj = torch.load(str(embed_path), map_location="cpu", weights_only=False)
    except TypeError:
        obj = torch.load(str(embed_path), map_location="cpu")
    if isinstance(obj, nn.Module):
        obj.eval()
        return obj
    if isinstance(obj, dict):
        if "weight" not in obj:
            raise ValueError(
                f"Unsupported token embedding state dict format: {embed_path}"
            )
        emb = nn.Embedding(obj["weight"].shape[0], obj["weight"].shape[1])
        emb.load_state_dict(obj)
        emb.eval()
        return emb
    raise TypeError(f"Unsupported token embedding object type: {type(obj)}")


def _create_golden_session(
    onnx_file: str,
    golden_dir: Path,
    device: torch.device,
    multi_gpu: bool,
    max_memory: Optional[Dict],
    logger,
) -> HMONNXGoldenInference:
    golden_dir.mkdir(exist_ok=True, parents=True)
    if multi_gpu:
        session = HMONNXGraphGoldenInference(onnx_file)
        session.save_golden = True
        session.golden_dir = golden_dir
        session.initialize()
        graph_module = session._session.graph_module
        logger.info(
            f"Applying AutoOffloadGraphModel to HMONNX graph '{Path(onnx_file).name}', "
            f"max_memory={max_memory}"
        )
        AutoOffloadGraphModel.from_graph_model(graph_module, max_memory=max_memory)
        session._device = torch.device("cpu")
        session._exec_device = torch.device("cpu")
    else:
        session = HMONNXGoldenInference(onnx_file)
        session.exec_device = device
        session.to(device)
        session.save_golden = True
        session.golden_dir = golden_dir
        session.initialize()
    return session


def _generate_golden(
    cfg,
    args,
    input_ids_full: torch.Tensor,
    tokenizer,
    prefill_onnx_file: str,
    decode_onnx_file: str,
    logger,
) -> Path:
    multi_gpu = getattr(args, "golden_multi_gpu", False)
    golden_max_memory = getattr(args, "golden_max_memory", None)

    device = torch.device("cpu") if multi_gpu else _get_default_device()
    dtype = getattr(torch, cfg.dtype)

    token_embedding_file = Path(cfg.work_dir) / "token_embedding.pt"
    token_embedding = _load_token_embedding(token_embedding_file).to(device).to(dtype)
    token_embedding.eval()

    pad_token_id = tokenizer.pad_token_id
    if pad_token_id is None:
        pad_token_id = tokenizer.eos_token_id
    if pad_token_id is None:
        raise ValueError("pad_token_id and eos_token_id are both None.")

    valid_len = input_ids_full.shape[1]
    release_prefix = _build_release_prefix(cfg)
    release_dir = Path(cfg.work_dir) / release_prefix
    prefill_dir = release_dir / "prefill"
    decode_dir = release_dir / "decode"
    prefill_dir.mkdir(exist_ok=True, parents=True)
    decode_dir.mkdir(exist_ok=True, parents=True)
    logger.info(f"Release prefix: {release_prefix}")
    logger.info(f"Release directory: {release_dir}")

    prefill_onnx_path = Path(prefill_onnx_file)
    decode_onnx_path = Path(decode_onnx_file)
    named_prefill_onnx = prefill_dir / f"{release_prefix}_prefill_with_act.onnx"
    named_decode_onnx = decode_dir / f"{release_prefix}_decode_with_act.onnx"

    quant_embedding_file = release_dir / "quant_embedding.pt"
    if token_embedding_file.exists() and not quant_embedding_file.exists():
        shutil.copy2(token_embedding_file, quant_embedding_file)
        logger.info(f"Copied quant_embedding.pt to {quant_embedding_file}")

    logger.info("Generating prefill golden...")
    prefill_session = _create_golden_session(
        str(prefill_onnx_path),
        prefill_dir,
        device,
        multi_gpu,
        golden_max_memory,
        logger,
    )
    if hasattr(prefill_session, "legacy_mode"):
        prefill_session.legacy_mode = False

    prefill_inputs_name = _resolve_input_name(
        prefill_session,
        ("inputs_embeds", "input_1"),
        fallback=_infer_inputs_embeds_name,
    )
    prefill_past_seq_name = _resolve_input_name(
        prefill_session, ("past_seq_length", "valid_length")
    )
    prefill_current_seq_name = _resolve_input_name(
        prefill_session, ("current_input_length", "current_length")
    )
    prefill_mask_name = _resolve_input_name(
        prefill_session, ("linear_attn_mask", "attention_mask", "attn_mask")
    )

    prefill_inputs_info = prefill_session.get_input(prefill_inputs_name)
    prefill_mask_info = prefill_session.get_input(prefill_mask_name)
    prefill_past_seq_info = prefill_session.get_input(prefill_past_seq_name)
    prefill_current_seq_info = prefill_session.get_input(prefill_current_seq_name)

    prefill_inputs_embeds = _build_inputs_embeds(
        token_embedding,
        input_ids_full,
        prefill_inputs_info.shape[1],
        pad_token_id,
        device,
        prefill_inputs_info.dtype,
    )
    prefill_linear_attn_mask = _build_linear_attn_mask(
        valid_len, prefill_mask_info, device
    )
    prefill_batch = prefill_inputs_info.shape[0]
    prefill_past_seq_length = torch.tensor(
        [0] * prefill_batch, dtype=prefill_past_seq_info.dtype, device=device
    )
    prefill_current_input_length = torch.tensor(
        [valid_len] * prefill_batch, dtype=prefill_current_seq_info.dtype, device=device
    )

    prefill_seq_len = prefill_inputs_info.shape[1]
    prefill_position_ids = torch.arange(
        0, prefill_seq_len, device=device, dtype=torch.int64
    )

    prefill_cache_inputs = _alloc_cache_inputs(prefill_session, device)
    prefill_input_feed: Dict[str, torch.Tensor] = {}
    for name in prefill_session.get_input_names():
        if name == prefill_inputs_name:
            prefill_input_feed[name] = prefill_inputs_embeds
        elif name == prefill_past_seq_name:
            prefill_input_feed[name] = prefill_past_seq_length
        elif name == prefill_current_seq_name:
            prefill_input_feed[name] = prefill_current_input_length
        elif name == prefill_mask_name:
            prefill_input_feed[name] = prefill_linear_attn_mask
        elif name in ("time_position_ids", "hight_position_ids", "width_position_ids"):
            info = prefill_session.get_input(name)
            prefill_input_feed[name] = prefill_position_ids.to(
                dtype=info.dtype
            ).reshape(info.shape)
        else:
            prefill_input_feed[name] = prefill_cache_inputs[name]

    _, prefill_output_map = _run_hmonnx_with_golden(prefill_session, prefill_input_feed)
    prefill_logits = prefill_output_map["logits"]
    if args.num_logits_to_keep == 0:
        prefill_logits = prefill_logits[:, valid_len - 1 : valid_len, :]
    next_token_id, next_token_text = decode_next_token(tokenizer, prefill_logits)
    logger.info(f"Prefill golden next token: {next_token_id} {next_token_text}")

    if multi_gpu:
        prefill_output_map = {
            key: value.cpu() if isinstance(value, torch.Tensor) else value
            for key, value in prefill_output_map.items()
        }
        prefill_cache_inputs = {
            key: value.cpu() if isinstance(value, torch.Tensor) else value
            for key, value in prefill_cache_inputs.items()
        }

    del prefill_session
    cleanup_memory()

    logger.info("Generating decode golden...")
    decode_session = _create_golden_session(
        str(decode_onnx_path),
        decode_dir,
        device,
        multi_gpu,
        golden_max_memory,
        logger,
    )
    if hasattr(decode_session, "legacy_mode"):
        decode_session.legacy_mode = False

    decode_inputs_name = _resolve_input_name(
        decode_session,
        ("inputs_embeds", "input_1"),
        fallback=_infer_inputs_embeds_name,
    )
    decode_past_seq_name = _resolve_input_name(
        decode_session, ("past_seq_length", "valid_length")
    )
    decode_current_seq_name = _resolve_input_name(
        decode_session, ("current_input_length", "current_length")
    )
    decode_mask_name = _resolve_input_name(
        decode_session, ("linear_attn_mask", "attention_mask", "attn_mask")
    )

    decode_inputs_info = decode_session.get_input(decode_inputs_name)
    decode_mask_info = decode_session.get_input(decode_mask_name)
    decode_past_seq_info = decode_session.get_input(decode_past_seq_name)
    decode_current_seq_info = decode_session.get_input(decode_current_seq_name)

    decode_input_ids = next_token_id.to(device)
    decode_inputs_embeds = _build_inputs_embeds(
        token_embedding,
        decode_input_ids,
        decode_inputs_info.shape[1],
        pad_token_id,
        device,
        decode_inputs_info.dtype,
    )
    decode_linear_attn_mask = _build_linear_attn_mask(1, decode_mask_info, device)
    decode_batch = decode_inputs_info.shape[0]
    decode_past_seq_length = torch.tensor(
        [valid_len] * decode_batch, dtype=decode_past_seq_info.dtype, device=device
    )
    decode_current_input_length = torch.tensor(
        [1] * decode_batch, dtype=decode_current_seq_info.dtype, device=device
    )

    decode_input_feed: Dict[str, torch.Tensor] = {}
    for name in decode_session.get_input_names():
        if name == decode_inputs_name:
            decode_input_feed[name] = decode_inputs_embeds
        elif name == decode_past_seq_name:
            decode_input_feed[name] = decode_past_seq_length
        elif name == decode_current_seq_name:
            decode_input_feed[name] = decode_current_input_length
        elif name == decode_mask_name:
            decode_input_feed[name] = decode_linear_attn_mask
        elif name in ("time_position_ids", "hight_position_ids", "width_position_ids"):
            info = decode_session.get_input(name)
            decode_pos = torch.tensor(
                [valid_len], device=device, dtype=info.dtype
            ).reshape(info.shape)
            decode_input_feed[name] = decode_pos
        elif name.startswith("past_conv_cache_"):
            idx = name.split("_")[-1]
            decode_input_feed[name] = prefill_output_map[f"conv_cache_out_{idx}"]
        elif name.startswith("past_recurrent_state_"):
            idx = name.split("_")[-1]
            decode_input_feed[name] = prefill_output_map[f"recurrent_state_out_{idx}"]
        else:
            if name in prefill_cache_inputs:
                decode_input_feed[name] = prefill_cache_inputs[name]
            else:
                info = decode_session.get_input(name)
                tensor = torch.zeros(info.shape, dtype=info.dtype, device=device)
                if _is_cache_input_name(name):
                    tensor = _ensure_cache_tensor(tensor)
                decode_input_feed[name] = tensor

    _, decode_output_map = _run_hmonnx_with_golden(decode_session, decode_input_feed)
    if isinstance(decode_output_map, dict) and "logits" in decode_output_map:
        import numpy as np

        np.save(
            str(decode_dir / "logits.npy"),
            decode_output_map["logits"].detach().cpu().numpy(),
        )

    del decode_session
    cleanup_memory()

    if prefill_onnx_path.exists() and not named_prefill_onnx.exists():
        _copy_path(prefill_onnx_path, named_prefill_onnx)
    if decode_onnx_path.exists() and not named_decode_onnx.exists():
        _copy_path(decode_onnx_path, named_decode_onnx)

    prefill_ext = _find_external_data(prefill_onnx_path)
    decode_ext = _find_external_data(decode_onnx_path)
    if prefill_ext is not None:
        _copy_path(prefill_ext, prefill_dir / prefill_ext.name)
    if decode_ext is not None:
        _copy_path(decode_ext, decode_dir / decode_ext.name)

    _cleanup_hmonnx_in_golden(prefill_dir, release_prefix, logger)
    _cleanup_hmonnx_in_golden(decode_dir, release_prefix, logger)

    golden_meta = {
        "release_prefix": release_prefix,
        "zip_name": f"{release_prefix}.zip",
        "zip_cmd": f"zip -r -y {release_prefix}.zip {release_prefix}/",
        "config": (
            str(Path(cfg.config_file).name) if hasattr(cfg, "config_file") else ""
        ),
        "dtype": cfg.dtype,
        "hf_model": getattr(cfg, "hf_model_dir", ""),
        "quant_embedding": "quant_embedding.pt",
        "prefill_onnx": (
            str(named_prefill_onnx.relative_to(release_dir))
            if named_prefill_onnx.exists()
            else None
        ),
        "decode_onnx": (
            str(named_decode_onnx.relative_to(release_dir))
            if named_decode_onnx.exists()
            else None
        ),
    }
    with (release_dir / "golden_meta_info.json").open("w", encoding="utf-8") as fout:
        json.dump(golden_meta, fout, ensure_ascii=False, indent=2)

    logger.info(f"Golden generation done. Release dir: {release_dir}")
    logger.info(f"To package: {golden_meta['zip_cmd']}")
    return release_dir


def _package_release_dir(release_dir: Path, logger) -> Optional[Path]:
    if not release_dir.exists():
        return None
    zip_file = release_dir.parent / f"{release_dir.name}.zip"
    if zip_file.exists():
        zip_file.unlink()
    logger.info(f"Packing release: {release_dir} -> {zip_file}")
    subprocess.run(
        ["zip", "-r", "-y", zip_file.name, release_dir.name],
        cwd=str(release_dir.parent),
        check=True,
    )
    return zip_file


def _prepare_quanted_graph(xh_model, data_batch, cfg, dtype, logger):
    xh_model.interactive_mode = True
    xh_model.set_exec_device(torch.device("cpu"))
    xh_model.to("cpu")
    xh_model.to(dtype)
    logger.info("Convert to frontend graph")
    xh_model.convert_to_fronted_graph(data_batch, release_wraped_model=True)
    cleanup_memory()
    logger.info("Convert to quanted graph")
    xh_model.convert_to_quant_graph(cfg.target_device)
    cleanup_memory()

    calib_data = _flatten_inputs(xh_model.prepare_inputs_for_graph(data_batch))
    logger.info("PTQ quantization with calibration data prepared on CPU")
    ptq_quantize(
        xh_model.quanted_model,
        [calib_data],
        PrecisionMode.ALIGNED,
        [torch.device("cpu")],
        auto_release_unused_parameters=True,
    )
    cleanup_memory()
    xh_model.change_eval_type(EvalModelType.QUANTED_ALIGNED)
    xh_model.to("cpu")
    xh_model.to(dtype)


def _export_onnx(
    xh_model,
    data_batch,
    onnx_output_dir,
    prefix,
    dtype,
    logger,
    valid=False,
    tokenizer=None,
):
    logger.info("Start exporting...")
    xh_model.to("cpu")
    cleanup_memory()
    xh_model.convert_to_export_graph(data_batch)
    xh_model.change_eval_type(EvalModelType.EXPORTED)

    if valid:
        xh_model.to("cpu")
        xh_model.to(dtype)
        with torch.no_grad():
            outs = xh_model.test_step(data_batch)
            exported_logits = outs.logits.detach()
        if tokenizer is not None:
            next_token_id, next_token_text = decode_next_token(
                tokenizer, exported_logits
            )
            logger.info(f"Exported model next token: {next_token_id} {next_token_text}")

    onnx_file = xh_model.to_export_onnx(data_batch, onnx_output_dir, prefix)[0]
    return onnx_file


def _export_single_graph(
    cfg,
    args,
    mode: Literal["prefill", "decode"],
    input_ids: torch.Tensor,
    tokenizer,
    onnx_output_dir: Path,
    logger,
) -> str:
    cfg_name = cfg.cfg_name
    dtype = getattr(torch, cfg.dtype)
    input_sequence_length = cfg.model.wrap_cfg.input_sequence_length

    logger.info(f"{'=' * 20} Initializing model for {mode} export {'=' * 20}")

    model_cfg = deepcopy(cfg.model)
    qwen3_5_model: XHQwen3_5Model = MODELS.build(model_cfg)
    logger.info(f"[{mode}] Loading HF model with device_map='auto'")
    native_model = qwen3_5_model.get_hf_model(
        device_map="auto",
        use_safetensors=True,
        torch_dtype=dtype,
    )
    _log_module_dtype(f"[{mode}] HF model", native_model, dtype, logger)
    _de_dispatch_to_cpu(native_model)
    qwen3_5_model.init_wrap_model(native_model)
    _log_module_dtype(f"[{mode}] wrap model", qwen3_5_model.wrap_model, dtype, logger)
    del native_model

    qwen3_5_model.change_eval_type(EvalModelType.WRAPED)
    qwen3_5_model.set_exec_device(torch.device("cpu"))
    qwen3_5_model.to("cpu")
    qwen3_5_model.to(dtype)

    if mode == "prefill":
        qwen3_5_model.set_linear_attention_mode("chunk")
        qwen3_5_model.set_input_sequence_length(input_sequence_length)
        input_ids_pad = torch.cat(
            [
                input_ids,
                torch.full(
                    (input_ids.shape[0], input_sequence_length - input_ids.shape[1]),
                    tokenizer.pad_token_id,
                    dtype=torch.long,
                ),
            ],
            dim=-1,
        )
        data_batch = {
            "input_ids": input_ids_pad,
            "past_seq_length": [0],
            "current_input_length": [input_ids.shape[1]],
        }
        onnx_prefix = f"{cfg_name}_prefill"
    else:
        qwen3_5_model.set_linear_attention_mode("recurrent")
        qwen3_5_model.set_input_sequence_length(1)
        data_batch = {
            "input_ids": input_ids[:, :1],
            "past_seq_length": [input_ids.shape[-1]],
            "current_input_length": [1],
        }
        onnx_prefix = f"{cfg_name}_decode"

    _prepare_quanted_graph(qwen3_5_model, data_batch, cfg, dtype, logger)

    if getattr(args, "valid_exported", False):
        with torch.no_grad():
            outs = qwen3_5_model.test_step(data_batch)
            logits = outs.logits.detach()
        next_token_id, next_token_text = decode_next_token(tokenizer, logits)
        logger.info(
            f"{mode.capitalize()} quanted next token: {next_token_id} {next_token_text}"
        )

    # XH2a runtime only supports fp16; cast bf16 model to fp16 before ONNX export
    export_dtype = dtype
    if dtype == torch.bfloat16:
        logger.info(
            "Casting quantized model from bf16 to fp16 for ONNX export (XH2a requires fp16)"
        )
        export_dtype = torch.float16
        qwen3_5_model.to(torch.float16)
        data_batch = {
            k: (
                v.to(torch.float16)
                if isinstance(v, torch.Tensor) and v.is_floating_point()
                else v
            )
            for k, v in data_batch.items()
        }

    onnx_file = _export_onnx(
        qwen3_5_model,
        data_batch,
        str(onnx_output_dir),
        onnx_prefix,
        export_dtype,
        logger,
        valid=getattr(args, "valid_exported", False),
        tokenizer=tokenizer,
    )

    logger.info(f"Cleaning up {mode} model")
    qwen3_5_model.release_exported_model()
    qwen3_5_model.release_quanted_model()
    qwen3_5_model.release_frontend_model()
    qwen3_5_model.release_wraped_model()
    del qwen3_5_model
    cleanup_memory()

    return onnx_file


def _copy_hf_configs(hf_model_dir: str, work_dir: str, logger) -> Path:
    hf_config_dir = Path(work_dir) / "hf_config"
    hf_config_dir.mkdir(exist_ok=True, parents=True)
    for candidates in [
        ["chat_template.json", "chat_template.jinja"],
        ["config.json"],
        ["configuration.json"],
        ["generation_config.json"],
        ["tokenizer_config.json"],
        ["vocab.json"],
        ["tokenizer.json"],
        ["special_tokens_map.json"],
        ["merges.txt"],
        ["tokenizer.model"],
    ]:
        for cfg_file in candidates:
            src = Path(hf_model_dir) / cfg_file
            if src.exists():
                shutil.copyfile(src, hf_config_dir / cfg_file)
                break
        else:
            logger.warning(
                f"Skip copying hf config: missing {candidates} in {hf_model_dir}"
            )
    return hf_config_dir


def _build_normalized_meta(meta_info: ConfigDict, cfg, args) -> Dict[str, Any]:
    quant_type = _detect_export_quant_type(args.hf_model_dir)
    return dict(
        create_time=meta_info.create_time,
        device=str(cfg.target_device),
        model_name=Path(args.hf_model_dir).name,
        hf_model_path=args.hf_model_dir,
        architecture="Qwen3_5ForConditionalGeneration",
        source_quant_method=meta_info.get("source_quant_method", None),
        quant_type=quant_type,
        quant_scheme=dict(
            target_device=str(cfg.target_device),
            quant_type=quant_type,
        ),
        dtype=cfg.dtype,
        prefill_onnx=meta_info.prefill_onnx_file,
        decode_onnx=meta_info.decode_onnx_file,
        hf_config=meta_info.hf_config,
        token_embedding_file=meta_info.token_embedding_file,
        max_context_tokens=cfg.model.wrap_cfg.max_sequence_length,
        pad_token_id=meta_info.get("pad_token_id", None),
        kv_cache=dict(
            shape=meta_info.get("kv_cache_shape", None),
            num_decoder_layers=meta_info.get("num_full_attention_layers", None),
        ),
        linear_cache=dict(
            conv_shape=meta_info.get("conv_cache_shape", None),
            recurrent_shape=meta_info.get("recurrent_state_shape", None),
            num_decoder_layers=meta_info.get("num_linear_attention_layers", None),
        ),
    )


def _prepare_export_context(cfg, args, logger):
    dtype = getattr(torch, cfg.dtype)
    no_split_modules = _get_no_split_modules(args)
    gpu_device = _get_default_device()

    cfg.hf_model_dir = args.hf_model_dir
    cfg.model.hf_model = args.hf_model_dir
    cfg.model.wrap_cfg.max_pe_length = args.max_pe_length
    cfg.model.wrap_cfg.max_sequence_length = args.max_sequence_length
    cfg.model.wrap_cfg.num_logits_to_keep = args.num_logits_to_keep
    cfg.model.wrap_cfg.support_long_context_over_fp16_limit = getattr(
        args, "support_long_context_over_fp16_limit", False
    )

    source_quant = _detect_source_quant_method(args.hf_model_dir)
    if source_quant == "gptq":
        logger.info("GPTQ source detected: overriding w_schema to 4-bit")
        cfg.model.quant_config.w_schema.bits = 4
        if "fallback_w_schema" not in cfg.model.quant_config:
            cfg.model.quant_config["fallback_w_schema"] = dict(
                fp_mode="ssfp",
                hidden_bit=False,
                bits=8,
            )

    qwen3_5_model: XHQwen3_5Model = MODELS.build(cfg.model)
    tokenizer = qwen3_5_model.get_tokenizer()
    logger.info("Loading HF model with device_map='auto'")
    native_model = qwen3_5_model.get_hf_model(
        device_map="auto",
        use_safetensors=True,
        torch_dtype=dtype,
    )
    _log_module_dtype("HF model", native_model, dtype, logger)

    source_quant_method = _detect_source_quant_method(args.hf_model_dir)
    meta_info = ConfigDict(
        dict(
            create_time=time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()),
            config=str(Path(cfg.config_file).relative_to(cfg.work_dir)),
            hf_model=cfg.hf_model_dir,
            source_quant_method=source_quant_method,
            dtype=cfg.dtype,
            pad_token_id=getattr(native_model.config, "eos_token_id", None),
        )
    )

    hf_config_dir = _copy_hf_configs(cfg.hf_model_dir, cfg.work_dir, logger)
    meta_info.hf_config = str(hf_config_dir.relative_to(cfg.work_dir))

    input_ids = _build_prompt_inputs(
        tokenizer,
        "cpu",
        qwen3_5_model.input_sequence_length,
        prompt=args.prompt,
        system_prompt=args.system_prompt,
    )
    input_sequence_length = cfg.model.wrap_cfg.input_sequence_length
    valid_len = input_ids.shape[1]

    hf_refs = dict(hf_logits=None, hf_generated_ids=None)
    if args.valid:
        compare_full_output = bool(getattr(args, "valid_compare_full_output", True))
        compare_max_new_tokens = int(getattr(args, "valid_compare_max_new_tokens", 64))
        hf_refs = collect_hf_references(
            native_model=native_model,
            tokenizer=tokenizer,
            input_ids=input_ids,
            gpu_device=gpu_device,
            dtype=dtype,
            num_logits_to_keep=args.num_logits_to_keep,
            compare_full_output=compare_full_output,
            compare_max_new_tokens=compare_max_new_tokens,
            logger=logger,
        )

    _de_dispatch_to_cpu(native_model)

    # Qwen3.5: ForConditionalGeneration.model.language_model or ForCausalLM.model
    if hasattr(native_model.model, "language_model"):
        token_embedding = native_model.model.language_model.get_input_embeddings()
    else:
        token_embedding = native_model.model.get_input_embeddings()
    token_embedding_file = Path(cfg.work_dir) / "token_embedding.pt"
    torch.save(token_embedding.state_dict(), str(token_embedding_file))
    meta_info.token_embedding_file = str(token_embedding_file.relative_to(cfg.work_dir))

    qwen3_5_model.init_wrap_model(native_model)
    del native_model
    qwen3_5_model.to(dtype)
    _log_module_dtype("Wrap model", qwen3_5_model.wrap_model, dtype, logger)
    logger.info(
        f"Wrap cache dtype: conv={_format_dtype(_get_cache_dtype(getattr(qwen3_5_model, 'past_conv_caches', None)))}, "
        f"recurrent={_format_dtype(_get_cache_dtype(getattr(qwen3_5_model, 'past_recurrent_states', None)))}"
    )

    if args.valid:
        run_conversion_validation(
            qwen3_5_model=qwen3_5_model,
            cfg=cfg,
            args=args,
            tokenizer=tokenizer,
            input_ids=input_ids,
            valid_len=valid_len,
            input_sequence_length=input_sequence_length,
            dtype=dtype,
            no_split_modules=no_split_modules,
            gpu_device=gpu_device,
            hf_logits=hf_refs["hf_logits"],
            hf_generated_ids=hf_refs["hf_generated_ids"],
            logger=logger,
        )
    else:
        logger.info("Skip precision checks (enable with --valid)")

    if (
        qwen3_5_model.past_key_caches is not None
        and len(qwen3_5_model.past_key_caches) > 0
    ):
        meta_info.kv_cache_shape = list(qwen3_5_model.past_key_caches[0].shape)
        meta_info.num_full_attention_layers = len(qwen3_5_model.past_key_caches)
    if (
        hasattr(qwen3_5_model, "past_conv_caches")
        and qwen3_5_model.past_conv_caches is not None
        and len(qwen3_5_model.past_conv_caches) > 0
    ):
        meta_info.conv_cache_shape = list(qwen3_5_model.past_conv_caches[0].shape)
        meta_info.num_linear_attention_layers = len(qwen3_5_model.past_conv_caches)
    if (
        hasattr(qwen3_5_model, "past_recurrent_states")
        and qwen3_5_model.past_recurrent_states is not None
        and len(qwen3_5_model.past_recurrent_states) > 0
    ):
        meta_info.recurrent_state_shape = list(
            qwen3_5_model.past_recurrent_states[0].shape
        )

    qwen3_5_model.release_exported_model()
    qwen3_5_model.release_quanted_model()
    qwen3_5_model.release_frontend_model()
    qwen3_5_model.release_wraped_model()
    del qwen3_5_model
    cleanup_memory()

    return meta_info, tokenizer, input_ids.cpu()


def _load_export_meta_info(work_dir: Path, logger) -> Dict[str, Any]:
    meta_file = work_dir / "export_meta_info.json"
    if not meta_file.exists():
        logger.warning(
            f"export_meta_info.json not found under {work_dir}, fallback to directory scan for ONNX files."
        )
        return {}
    with open(meta_file, "r", encoding="utf-8") as file:
        return json.load(file)


def _resolve_existing_onnx_file(
    work_dir: Path,
    explicit_onnx_file: Optional[str],
    export_meta_info: Dict[str, Any],
    meta_key: str,
    onnx_subdir: str,
    logger,
) -> Path:
    if explicit_onnx_file is not None:
        onnx_file = Path(explicit_onnx_file)
        if not onnx_file.exists():
            raise FileNotFoundError(f"Specified ONNX file not found: {onnx_file}")
        return onnx_file.resolve()

    meta_rel_path = export_meta_info.get(meta_key, None)
    if isinstance(meta_rel_path, str):
        onnx_file = work_dir / meta_rel_path
        if onnx_file.exists():
            return onnx_file.resolve()
        logger.warning(
            f"{meta_key} from export_meta_info.json not found on disk: {onnx_file}"
        )

    onnx_dir = work_dir / onnx_subdir
    candidates = sorted(
        path.resolve() for path in onnx_dir.glob("*.onnx") if path.is_file()
    )
    if len(candidates) == 1:
        return candidates[0]
    if len(candidates) == 0:
        raise FileNotFoundError(f"No ONNX file found in {onnx_dir}")
    raise RuntimeError(
        f"Found multiple ONNX files in {onnx_dir}: {candidates}. "
        f"Please specify --{meta_key} explicitly."
    )


def _prepare_golden_only_context(cfg, args, logger):
    work_dir = Path(cfg.work_dir)
    tokenizer_source_dir = work_dir / "hf_config"
    tokenizer_source = (
        str(tokenizer_source_dir)
        if tokenizer_source_dir.exists()
        else args.hf_model_dir
    )

    cfg.hf_model_dir = tokenizer_source
    cfg.model.hf_model = tokenizer_source
    cfg.model.wrap_cfg.max_pe_length = args.max_pe_length
    cfg.model.wrap_cfg.max_sequence_length = args.max_sequence_length
    cfg.model.wrap_cfg.num_logits_to_keep = args.num_logits_to_keep
    cfg.model.wrap_cfg.support_long_context_over_fp16_limit = getattr(
        args, "support_long_context_over_fp16_limit", False
    )

    qwen3_5_model: XHQwen3_5Model = MODELS.build(cfg.model)
    tokenizer = qwen3_5_model.get_tokenizer()
    input_ids = _build_prompt_inputs(
        tokenizer,
        "cpu",
        qwen3_5_model.input_sequence_length,
        prompt=args.prompt,
        system_prompt=args.system_prompt,
    )
    del qwen3_5_model
    cleanup_memory()
    return tokenizer, input_ids


def _run_golden_generation(
    cfg,
    args,
    input_ids: torch.Tensor,
    tokenizer,
    prefill_onnx_file: str,
    decode_onnx_file: str,
    logger,
):
    if not getattr(args, "golden", False):
        logger.info(
            "Skip HMONNX golden generation (enable with --golden or --golden_only)"
        )
        return None

    logger.info("=" * 60)
    logger.info("Generating HMONNX golden (prefill + decode)")
    logger.info("=" * 60)
    release_dir = _generate_golden(
        cfg, args, input_ids, tokenizer, prefill_onnx_file, decode_onnx_file, logger
    )
    if getattr(args, "package_release", False):
        _package_release_dir(release_dir, logger)
    return release_dir


def _golden_only_impl(cfg, args):
    logger = get_root_logger()
    work_dir = Path(cfg.work_dir)
    token_embedding_file = work_dir / "token_embedding.pt"
    if not token_embedding_file.exists():
        raise FileNotFoundError(
            f"token_embedding.pt not found in existing work_dir: {token_embedding_file}. "
            "Please reuse a completed export directory."
        )

    export_meta_info = _load_export_meta_info(work_dir, logger)
    prefill_onnx_file = _resolve_existing_onnx_file(
        work_dir,
        args.prefill_onnx_file,
        export_meta_info,
        "prefill_onnx_file",
        "prefill_onnx",
        logger,
    )
    decode_onnx_file = _resolve_existing_onnx_file(
        work_dir,
        args.decode_onnx_file,
        export_meta_info,
        "decode_onnx_file",
        "decode_onnx",
        logger,
    )

    logger.info(f"Reusing existing prefill ONNX: {prefill_onnx_file}")
    logger.info(f"Reusing existing decode ONNX: {decode_onnx_file}")

    tokenizer, input_ids = _prepare_golden_only_context(cfg, args, logger)
    _run_golden_generation(
        cfg,
        args,
        input_ids,
        tokenizer,
        str(prefill_onnx_file),
        str(decode_onnx_file),
        logger,
    )


def _export_impl(cfg, args):
    logger = get_root_logger()

    meta_info, tokenizer, input_ids = _prepare_export_context(cfg, args, logger)
    cleanup_memory()

    prefill_onnx_dir = Path(cfg.work_dir) / "prefill_onnx"
    decode_onnx_dir = Path(cfg.work_dir) / "decode_onnx"
    prefill_onnx_dir.mkdir(exist_ok=True, parents=True)
    decode_onnx_dir.mkdir(exist_ok=True, parents=True)

    logger.info("=" * 60)
    logger.info("Exporting PREFILL graph (chunk mode)")
    logger.info("=" * 60)
    prefill_onnx_file = _export_single_graph(
        cfg, args, "prefill", input_ids, tokenizer, prefill_onnx_dir, logger
    )
    meta_info.prefill_onnx_file = str(Path(prefill_onnx_file).relative_to(cfg.work_dir))

    logger.info("=" * 60)
    logger.info("Exporting DECODE graph (recurrent mode)")
    logger.info("=" * 60)
    decode_onnx_file = _export_single_graph(
        cfg, args, "decode", input_ids, tokenizer, decode_onnx_dir, logger
    )
    meta_info.decode_onnx_file = str(Path(decode_onnx_file).relative_to(cfg.work_dir))

    release_dir = _run_golden_generation(
        cfg, args, input_ids, tokenizer, prefill_onnx_file, decode_onnx_file, logger
    )
    if release_dir is not None:
        meta_info.release_dir = str(Path(release_dir).relative_to(cfg.work_dir))

    export_meta_path = Path(cfg.work_dir) / "export_meta_info.json"
    with export_meta_path.open("w", encoding="utf-8") as file:
        json.dump(meta_info, file, ensure_ascii=False, indent=4)

    normalized_meta = _build_normalized_meta(meta_info, cfg, args)
    with (Path(cfg.work_dir) / "meta.json").open("w", encoding="utf-8") as file:
        json.dump(normalized_meta, file, ensure_ascii=False, indent=4)


def main(args):
    cfg = Config.fromfile(args.config)
    cfg_name = Path(args.config).stem
    if getattr(args, "golden_only", False) and getattr(args, "existing_work_dir", None):
        cfg.work_dir = str(Path(args.existing_work_dir))
    elif args.work_dir is not None:
        cfg.work_dir = str(Path(args.work_dir))
    else:
        cfg.work_dir = str(
            Path("./work_dirs")
            / "qwen3_5_27b"
            / f"{cfg_name}_{Path(args.hf_model_dir).stem}"
        )
    Path(cfg.work_dir).mkdir(exist_ok=True, parents=True)

    log_file = Path(cfg.work_dir) / f"{cfg_name}_debug.log"
    cfg.dtype = _resolve_compute_dtype_name(args.dtype, args.hf_model_dir)
    args.dtype = cfg.dtype
    cfg.debug = args.debug
    if getattr(args, "golden_only", False):
        args.golden = True

    if getattr(args, "golden_max_memory", None) is not None:
        args.golden_max_memory = _normalize_max_memory_spec(
            json.loads(args.golden_max_memory)
        )

    if not hasattr(cfg, "release") or cfg.release is None:
        cfg.release = {}
    if cfg.release.get("wmix_amix", None) is None:
        cfg.release["wmix_amix"] = _detect_release_wmix_amix(args.hf_model_dir)
    for key in ("xh_version", "modelscope_name", "wmix_amix", "date"):
        cli_val = getattr(args, f"release_{key}", None)
        if cli_val is not None:
            cfg.release[key] = cli_val

    set_random_seed(args.seed)

    xhquant_llm_init(log_file, cfg.debug)
    logger = get_root_logger()

    logger.info(f"Config:\n{cfg.pretty_text}")
    config_file = Path(cfg.work_dir) / Path(args.config).name
    cfg.config_file = str(config_file)
    cfg.cfg_name = cfg_name
    cfg.target_device = cfg.get("target_device", "XH2a")
    if getattr(args, "golden_only", False) and config_file.exists():
        logger.info(f"Reusing existing config snapshot: {config_file}")
    else:
        cfg.dump(config_file)

    xhquant.utils.suppress_printing.disable_printing = True
    start_time = time.time()
    if getattr(args, "golden_only", False):
        _golden_only_impl(cfg, args)
    else:
        _export_impl(cfg, args)
    logger.info(f"{cfg_name} completed in {time.time() - start_time:.4f} s")


def parse_arguments():
    import argparse

    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "--config", type=str, default="configs/qwen3_5/qwen3_5_27b_xh2a.py"
    )
    parser.add_argument("--hf_model_dir", type=str, default="weights/Qwen3.5-27B")
    parser.add_argument(
        "--dtype",
        type=str,
        default="fp16",
        help="compute dtype for HF/wrap/export: auto/fp32/fp16/bf16",
    )
    parser.add_argument("--work_dir", type=str, default=None)
    parser.add_argument("--debug", action="store_true", help="debug mode")
    parser.add_argument("--seed", type=int, default=1024)
    parser.add_argument("--prompt", type=str, default="你多大了？用中文回答。")
    parser.add_argument(
        "--system-prompt", type=str, default="You are a helpful assistant."
    )
    parser.add_argument(
        "--no_split_modules",
        type=str,
        default="",
        help="comma-separated no_split_module_classes for accelerate dispatch (default: auto)",
    )
    parser.add_argument("--max_sequence_length", type=int, default=2048)
    parser.add_argument("--max_pe_length", type=int, default=262144)
    parser.add_argument(
        "--support_long_context_over_fp16_limit",
        action="store_true",
        help=(
            "Use precomputed rotary cache (offline RoPE) so exported graphs can "
            "support position_id > 65504. Default uses online rotary computation."
        ),
    )
    parser.add_argument(
        "--valid",
        action="store_true",
        help="run precision checks (HF/wrap/frontend/quant)",
    )
    parser.add_argument(
        "--valid_exported",
        action="store_true",
        help="validate exported graph before ONNX save",
    )
    parser.add_argument("--num_logits_to_keep", type=int, default=1)
    parser.add_argument(
        "--valid_compare_full_output",
        dest="valid_compare_full_output",
        action="store_true",
        default=True,
    )
    parser.add_argument(
        "--no-valid_compare_full_output",
        dest="valid_compare_full_output",
        action="store_false",
    )
    parser.add_argument("--valid_compare_max_new_tokens", type=int, default=64)
    parser.add_argument(
        "--golden", action="store_true", help="generate HMONNX golden after export"
    )
    parser.add_argument(
        "--golden_only",
        action="store_true",
        help="skip HF load/quant/export and only generate golden from existing work_dir ONNX files",
    )
    parser.add_argument(
        "--existing_work_dir",
        type=str,
        default=None,
        help="existing export work_dir containing token_embedding.pt and prefill_onnx/decode_onnx for --golden_only",
    )
    parser.add_argument("--prefill_onnx_file", type=str, default=None)
    parser.add_argument("--decode_onnx_file", type=str, default=None)
    parser.add_argument(
        "--golden_multi_gpu",
        action="store_true",
        help="Enable multi-GPU golden inference via HMONNXGraphGoldenInference + AutoOffloadGraphModel.",
    )
    parser.add_argument(
        "--golden_max_memory",
        type=str,
        default=None,
        help='Per-device memory limits for multi-GPU golden (JSON string). Example: \'{"0":"70GiB","1":"70GiB","cpu":"160GiB"}\'.',
    )
    parser.add_argument(
        "--package_release",
        action="store_true",
        help="zip the release directory after golden generation",
    )
    parser.add_argument("--release_xh_version", type=str, default=None)
    parser.add_argument("--release_modelscope_name", type=str, default=None)
    parser.add_argument("--release_wmix_amix", type=str, default=None)
    parser.add_argument("--release_date", type=str, default=None)
    return parser


if __name__ == "__main__":
    parser = parse_arguments()
    args = parser.parse_args()
    main(args)
