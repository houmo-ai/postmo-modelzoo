#!/usr/bin/python3
# -*- coding: utf-8 -*-
# Copyright 2026 HOUMO AI
#
# File: _thinker_gptq_view.py
# Description:
#   GPTQ checkpoint view helpers for the Qwen3-Omni thinker sub-model.
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
import shutil
from pathlib import Path
from typing import Any, Dict

from safetensors.torch import load_file as load_safetensors_file
from safetensors.torch import save_file as save_safetensors_file
from transformers import AutoConfig, AutoTokenizer


_MIRRORED_FILES = (
    "generation_config.json",
    "tokenizer_config.json",
    "tokenizer.json",
    "vocab.json",
    "merges.txt",
    "special_tokens_map.json",
    "added_tokens.json",
)

_TEXT_WEIGHT_PREFIXES = (
    "thinker.model.",
    "thinker.lm_head",
)


def is_qwen3_omni_gptq_checkpoint(model_dir: str) -> bool:
    config_path = Path(model_dir) / "config.json"
    if not config_path.exists():
        return False

    try:
        config_payload = json.loads(config_path.read_text())
    except Exception:
        return False

    architectures = config_payload.get("architectures", [])
    quantization_config = config_payload.get("quantization_config", {})
    if not isinstance(architectures, list) or not isinstance(quantization_config, dict):
        return False

    quant_method = str(quantization_config.get("quant_method", "")).lower()
    checkpoint_format = str(quantization_config.get("checkpoint_format", "")).lower()
    return (
        "Qwen3OmniMoeForConditionalGeneration" in architectures
        and (quant_method == "gptq" or checkpoint_format.startswith("gptq"))
    )


def is_qwen3_omni_checkpoint(model_dir: str) -> bool:
    config_path = Path(model_dir) / "config.json"
    if not config_path.exists():
        return False

    try:
        config_payload = json.loads(config_path.read_text())
    except Exception:
        return False

    architectures = config_payload.get("architectures", [])
    return isinstance(architectures, list) and "Qwen3OmniMoeForConditionalGeneration" in architectures


def _write_json(file_path: Path, payload: Dict[str, Any]) -> None:
    file_path.parent.mkdir(parents=True, exist_ok=True)
    with open(file_path, "w", encoding="utf-8") as file_obj:
        json.dump(payload, file_obj, indent=2, ensure_ascii=False)


def _ensure_link_or_copy(src: Path, dst: Path) -> None:
    if dst.is_symlink() or dst.exists():
        if dst.is_symlink() and dst.resolve() == src.resolve():
            return
        if dst.is_dir() and not dst.is_symlink():
            shutil.rmtree(dst)
        else:
            dst.unlink()

    dst.parent.mkdir(parents=True, exist_ok=True)
    try:
        dst.symlink_to(src)
    except OSError:
        shutil.copyfile(src, dst)


def _resolve_special_token_ids(root_dir: Path, root_config) -> Dict[str, int]:
    tokenizer = AutoTokenizer.from_pretrained(str(root_dir), trust_remote_code=True)

    bos_token_id = getattr(tokenizer, "bos_token_id", None)
    if bos_token_id is None:
        bos_token_id = getattr(root_config, "im_start_token_id", None)

    eos_token_id = getattr(tokenizer, "eos_token_id", None)
    if eos_token_id is None:
        eos_token_id = getattr(root_config, "im_end_token_id", None)

    pad_token_id = getattr(tokenizer, "pad_token_id", None)
    if pad_token_id is None:
        pad_token_id = eos_token_id

    token_ids = {}
    if bos_token_id is not None:
        token_ids["bos_token_id"] = int(bos_token_id)
    if eos_token_id is not None:
        token_ids["eos_token_id"] = int(eos_token_id)
    if pad_token_id is not None:
        token_ids["pad_token_id"] = int(pad_token_id)
    return token_ids


def _build_qwen3omni_thinker_config_payload(root_dir: Path) -> Dict[str, Any]:
    root_config = AutoConfig.from_pretrained(str(root_dir), trust_remote_code=True)
    text_config = root_config.thinker_config.text_config.to_dict()
    quantization_config = dict(getattr(root_config, "quantization_config", {}) or {})
    talker_config = getattr(root_config, "talker_config", None)

    text_config["architectures"] = ["Qwen3MoeForCausalLM"]
    text_config["model_type"] = "qwen3_moe"
    text_config["tie_word_embeddings"] = False
    if quantization_config:
        quantization_config["block_name_to_quantize"] = "model.layers"
        text_config["quantization_config"] = quantization_config
    accept_hidden_layer = getattr(talker_config, "accept_hidden_layer", None)
    if accept_hidden_layer is not None:
        text_config["accept_hidden_layer"] = int(accept_hidden_layer)
    text_config.update(_resolve_special_token_ids(root_dir, root_config))
    text_config["xh_qwen3omni_thinker_qwen3moe_compat"] = True
    return text_config


def _is_text_thinker_weight(weight_name: str) -> bool:
    return any(weight_name.startswith(prefix) for prefix in _TEXT_WEIGHT_PREFIXES)


def _is_existing_view_complete(view_dir: Path) -> bool:
    index_path = view_dir / "model.safetensors.index.json"
    config_path = view_dir / "config.json"
    if not index_path.exists() or not config_path.exists():
        return False
    try:
        index_payload = json.loads(index_path.read_text())
    except Exception:
        return False
    weight_map = index_payload.get("weight_map", {})
    if not isinstance(weight_map, dict) or not weight_map:
        return False
    return all((view_dir / shard_name).exists() for shard_name in set(weight_map.values()))


def prepare_qwen3_omni_thinker_text_view(model_dir: str, view_dir: Path) -> Path:
    root_dir = Path(model_dir)
    if not is_qwen3_omni_checkpoint(str(root_dir)):
        raise ValueError(f"{root_dir} is not a Qwen3-Omni checkpoint")

    if _is_existing_view_complete(view_dir):
        return view_dir

    view_dir.mkdir(parents=True, exist_ok=True)
    _write_json(view_dir / "config.json", _build_qwen3omni_thinker_config_payload(root_dir))

    root_index_path = root_dir / "model.safetensors.index.json"
    if not root_index_path.exists():
        raise FileNotFoundError(f"missing sharded safetensors index: {root_index_path}")

    root_index = json.loads(root_index_path.read_text())
    root_weight_map = root_index.get("weight_map", {})
    if not isinstance(root_weight_map, dict):
        raise ValueError(f"invalid weight_map in {root_index_path}")

    view_weight_map: Dict[str, str] = {}
    shard_to_keys: Dict[str, Dict[str, str]] = {}
    for key, shard_name in root_weight_map.items():
        if not _is_text_thinker_weight(key):
            continue
        stripped_key = key[len("thinker.") :]
        view_weight_map[stripped_key] = shard_name
        shard_to_keys.setdefault(shard_name, {})[key] = stripped_key

    if not view_weight_map:
        raise RuntimeError(f"no thinker text weights found in {root_index_path}")

    for shard_name, key_mapping in sorted(shard_to_keys.items()):
        shard_src = root_dir / shard_name
        shard_dst = view_dir / shard_name
        if shard_dst.is_symlink() or shard_dst.exists():
            shard_dst.unlink()

        shard_tensor_map = load_safetensors_file(str(shard_src), device="cpu")
        filtered_tensor_map = {
            stripped_key: shard_tensor_map[original_key]
            for original_key, stripped_key in key_mapping.items()
            if original_key in shard_tensor_map
        }
        if not filtered_tensor_map:
            raise RuntimeError(f"no thinker text tensors found in shard {shard_src}")
        save_safetensors_file(filtered_tensor_map, str(shard_dst))

    total_size = sum((view_dir / shard_name).stat().st_size for shard_name in shard_to_keys)
    _write_json(
        view_dir / "model.safetensors.index.json",
        {
            "metadata": {"total_size": total_size},
            "weight_map": view_weight_map,
        },
    )

    for file_name in _MIRRORED_FILES:
        src = root_dir / file_name
        if src.exists():
            _ensure_link_or_copy(src, view_dir / file_name)

    return view_dir


def prepare_qwen3_omni_thinker_gptq_view(model_dir: str, view_dir: Path) -> Path:
    root_dir = Path(model_dir)
    if not is_qwen3_omni_gptq_checkpoint(str(root_dir)):
        raise ValueError(f"{root_dir} is not a Qwen3-Omni GPTQ checkpoint")
    return prepare_qwen3_omni_thinker_text_view(model_dir, view_dir)