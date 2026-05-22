#
# Copyright (c) 2025 HOUMO AI
#
# File: example_qwen35_moe_vl_rotate_fp.py
# Description:
#   Rotate Qwen3.5 VL fp weights and align vision projection (MoE-aware).
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
import json
import os
import random
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from transformers import AutoModelForImageTextToText, AutoProcessor

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from gptqmodel.models.definitions.qwen3_5 import Qwen3_5QModel
from gptqmodel.models.definitions.qwen3_5_moe import Qwen3_5MoeQModel
from gptqmodel.models.definitions.qwen3_vl import Qwen3_VLQModel
from gptqmodel.quantization.rotation.rotation import (
    fuse_layer_norms_qwen3_5,
    fuse_vision_layer_norms_qwen3_5_vl,
    rotate_model_qwen3_5_vl,
    rotate_vision_model_qwen3_5_vl,
)
from gptqmodel.utils.model import get_module_by_name_prefix

DEFAULT_PROMPT = "请简要描述这张图片中的主要内容。"


def parse_args():
    parser = argparse.ArgumentParser(
        description="Rotate Qwen3.5 VL LLM weights and align vision output projection"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="./Qwen3.5-9B",
        help="source HF model directory",
    )
    parser.add_argument(
        "--out",
        type=str,
        default="./Qwen3.5-9B-rotated-fp",
        help="output HF model directory",
    )
    parser.add_argument(
        "--llm-rotation",
        type=str,
        default="hadamard",
        choices=["none", "hadamard", "block_hadamard", "random"],
        help="rotation mode for LLM weights; use `none` to skip LLM rotation",
    )
    parser.add_argument(
        "--vision-rotation",
        type=str,
        default="last",
        choices=["last", "full"],
        help="vision rotation mode: `last` keeps the original logic and only aligns the final vision output projection; `full` also rotates equivalent earlier vision layers",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="device used to materialize rotation matrices",
    )
    parser.add_argument(
        "--torch-dtype",
        type=str,
        default="auto",
        choices=["auto", "float16", "bfloat16", "float32", "float64"],
        help="dtype used when loading the source model",
    )
    parser.add_argument(
        "--validate",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="validate logits before and after rotation on one synthetic multimodal sample",
    )
    parser.add_argument(
        "--validate-device",
        type=str,
        default=None,
        help="device for validation forward; defaults to --device",
    )
    parser.add_argument(
        "--validate-forward-dtype",
        type=str,
        default="model",
        choices=["model", "float32", "float64"],
        help="dtype used only for validation forward/generation",
    )
    parser.add_argument(
        "--validate-image-size",
        type=int,
        default=224,
        help="synthetic validation image size",
    )
    parser.add_argument(
        "--validate-prompt", type=str, default=DEFAULT_PROMPT, help="validation prompt"
    )
    parser.add_argument(
        "--validate-generation-tokens",
        type=int,
        default=8,
        help="greedy generation length used for functional equivalence validation",
    )
    parser.add_argument(
        "--atol", type=float, default=5e-2, help="absolute tolerance for validation"
    )
    parser.add_argument(
        "--rtol", type=float, default=5e-2, help="relative tolerance for validation"
    )
    parser.add_argument(
        "--max-shard-size",
        type=str,
        default="5GB",
        help="max shard size used by save_pretrained",
    )
    parser.add_argument("--trust-remote-code", action="store_true", default=False)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def resolve_torch_dtype(dtype_name: str):
    if dtype_name == "auto":
        return "auto"
    mapping = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
        "float64": torch.float64,
    }
    return mapping[dtype_name]


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_model_config(model_path: str):
    config_path = Path(model_path) / "config.json"
    if not config_path.exists():
        return {}
    with config_path.open("r", encoding="utf-8") as f:
        return json.load(f)


def detect_qwen35_text_variant(model_path: str):
    config = load_model_config(model_path)
    text_config = config.get("text_config", {})
    model_type = str(
        config.get("model_type") or text_config.get("model_type") or ""
    ).lower()
    if "qwen3_5_moe" in model_type:
        return "moe"
    if "qwen3_5" in model_type:
        return "dense"
    return "unknown"


def get_loader_bridge_cls(text_variant: str):
    if text_variant == "moe":
        return Qwen3_5MoeQModel
    return Qwen3_5QModel


def get_multimodal_text_layout_cls(text_variant: str):
    if text_variant == "moe":
        return Qwen3_5MoeQModel
    return Qwen3_VLQModel


def maybe_clone_lm_head(model, lm_head_name: str):
    if not getattr(model.config, "tie_word_embeddings", False):
        return

    model.config.tie_word_embeddings = False
    text_config = getattr(model.config, "text_config", None)
    if text_config is not None:
        text_config.tie_word_embeddings = False

    lm_head, _ = get_module_by_name_prefix(model, lm_head_name)
    lm_head.weight = torch.nn.Parameter(lm_head.weight.data.clone())


def build_validation_inputs(processor, prompt: str, image_size: int, seed: int):
    generator = np.random.default_rng(seed)
    image = Image.fromarray(
        generator.integers(0, 256, size=(image_size, image_size, 3), dtype=np.uint8)
    )
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "text", "text": prompt},
            ],
        }
    ]
    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    return processor(
        text=[text], images=[image], videos=None, padding=True, return_tensors="pt"
    )


def move_batch_to_device(batch, device: str, float_dtype: torch.dtype):
    moved = {}
    for key, value in batch.items():
        if not torch.is_tensor(value):
            moved[key] = value
            continue
        if value.is_floating_point():
            moved[key] = value.to(device=device, dtype=float_dtype)
        else:
            moved[key] = value.to(device=device)
    return moved


def resolve_validation_dtype(model, dtype_name: str):
    if dtype_name == "model":
        return next(model.parameters()).dtype
    mapping = {
        "float32": torch.float32,
        "float64": torch.float64,
    }
    return mapping[dtype_name]


def compute_logits(model, batch, device: str, forward_dtype: torch.dtype | None = None):
    original_dtype = next(model.parameters()).dtype
    effective_dtype = forward_dtype or original_dtype
    model = model.to(device=device, dtype=effective_dtype)
    batch = move_batch_to_device(batch, device=device, float_dtype=effective_dtype)
    with torch.inference_mode():
        logits = model(**batch, use_cache=False).logits.detach().float().cpu()
    model = model.to(device="cpu", dtype=original_dtype)
    if device.startswith("cuda") and torch.cuda.is_available():
        torch.cuda.empty_cache()
    return logits


def generate_tokens(
    model,
    batch,
    device: str,
    max_new_tokens: int,
    forward_dtype: torch.dtype | None = None,
):
    original_dtype = next(model.parameters()).dtype
    effective_dtype = forward_dtype or original_dtype
    model = model.to(device=device, dtype=effective_dtype)
    batch = move_batch_to_device(batch, device=device, float_dtype=effective_dtype)
    with torch.inference_mode():
        generated = (
            model.generate(
                **batch,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                use_cache=True,
            )
            .detach()
            .cpu()
        )
    model = model.to(device="cpu", dtype=original_dtype)
    if device.startswith("cuda") and torch.cuda.is_available():
        torch.cuda.empty_cache()
    return generated


def validate_equivalence(model, processor, args):
    validate_device = args.validate_device or args.device
    validate_dtype = resolve_validation_dtype(model, args.validate_forward_dtype)
    batch = build_validation_inputs(
        processor=processor,
        prompt=args.validate_prompt,
        image_size=args.validate_image_size,
        seed=args.seed,
    )
    print("[2/5] Running pre-rotation validation forward ...")
    ref_logits = compute_logits(
        model, batch, validate_device, forward_dtype=validate_dtype
    )
    ref_generated = generate_tokens(
        model,
        batch,
        validate_device,
        args.validate_generation_tokens,
        forward_dtype=validate_dtype,
    )
    return batch, ref_logits, ref_generated


def compare_logits(model, batch, ref_logits, ref_generated, args):
    validate_device = args.validate_device or args.device
    validate_dtype = resolve_validation_dtype(model, args.validate_forward_dtype)
    print("[4/5] Running post-rotation validation forward ...")
    rotated_logits = compute_logits(
        model, batch, validate_device, forward_dtype=validate_dtype
    )
    max_abs_diff = (rotated_logits - ref_logits).abs().max().item()
    mean_abs_diff = (rotated_logits - ref_logits).abs().mean().item()
    last_token_equal = bool(
        torch.equal(
            rotated_logits[:, -1].argmax(dim=-1), ref_logits[:, -1].argmax(dim=-1)
        )
    )

    rotated_generated = generate_tokens(
        model,
        batch,
        validate_device,
        args.validate_generation_tokens,
        forward_dtype=validate_dtype,
    )

    prompt_length = batch["input_ids"].shape[1]
    ref_new_tokens = ref_generated[:, prompt_length:]
    rotated_new_tokens = rotated_generated[:, prompt_length:]
    generation_equal = bool(torch.equal(ref_new_tokens, rotated_new_tokens))

    print(
        f"Validation full_logits max_abs_diff={max_abs_diff:.6e}, mean_abs_diff={mean_abs_diff:.6e}"
    )
    print(f"Validation last_token_equal={last_token_equal}")
    print(
        f"Validation generation_equal={generation_equal}, generated_tokens={args.validate_generation_tokens}"
    )
    if not generation_equal:
        raise AssertionError(
            "Rotated model greedy generation differs from reference. "
            f"last_token_equal={last_token_equal}, generation_equal={generation_equal}, "
            f"full_logits_max_abs_diff={max_abs_diff:.6e}, full_logits_mean_abs_diff={mean_abs_diff:.6e}"
        )


def main():
    args = parse_args()
    set_seed(args.seed)
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

    torch_dtype = resolve_torch_dtype(args.torch_dtype)
    rotation_device = torch.device(args.device)

    print("[1/5] Loading processor and source fp model ...")
    text_variant = detect_qwen35_text_variant(args.model)
    loader_bridge_cls = get_loader_bridge_cls(text_variant)
    text_layout_cls = get_multimodal_text_layout_cls(text_variant)
    loader_bridge_cls.before_model_load(loader_bridge_cls, load_quantized_model=False)
    processor = AutoProcessor.from_pretrained(
        args.model, trust_remote_code=args.trust_remote_code
    )
    model = AutoModelForImageTextToText.from_pretrained(
        args.model,
        trust_remote_code=args.trust_remote_code,
        device_map="cpu",
        torch_dtype=torch_dtype,
    )
    model.eval()

    lm_head_name = Qwen3_VLQModel.lm_head
    layers_node = text_layout_cls.extract_layers_node()
    pre_lm_head_norm_module_name = text_layout_cls.pre_lm_head_norm_module

    should_rotate_llm = args.llm_rotation != "none"
    if text_variant == "moe" and should_rotate_llm:
        print(
            "[config] Detected Qwen3.5-MoE text tower; skipping LLM rotation and keeping vision-only rotation "
            "to avoid dense-only text rotation path mismatches."
        )
        should_rotate_llm = False

    effective_vision_rotation = args.vision_rotation
    if not should_rotate_llm and effective_vision_rotation == "last":
        print(
            "[config] `--vision-rotation last` only aligns the final vision projection for LLM rotation. "
            "Promoting it to `full` because LLM rotation is disabled."
        )
        effective_vision_rotation = "full"

    if should_rotate_llm:
        maybe_clone_lm_head(model, lm_head_name)

    validation_batch = None
    ref_logits = None
    ref_generated = None
    if args.validate:
        validation_batch, ref_logits, ref_generated = validate_equivalence(
            model, processor, args
        )

    if effective_vision_rotation == "full" and should_rotate_llm:
        print(
            "[3/5] Fusing vision/text norms, rotating full vision tower, and rotating LLM ..."
        )
        model = fuse_vision_layer_norms_qwen3_5_vl(
            model=model,
            pre_lm_head_norm_module_name=pre_lm_head_norm_module_name,
            layers_node=layers_node,
            lm_head_name=lm_head_name,
        )
    elif effective_vision_rotation == "full":
        print(
            "[3/5] Fusing vision norms and rotating full vision tower only (LLM rotation skipped) ..."
        )
        model = fuse_vision_layer_norms_qwen3_5_vl(
            model=model,
            pre_lm_head_norm_module_name=pre_lm_head_norm_module_name,
            layers_node=layers_node,
            lm_head_name=lm_head_name,
        )
    else:
        print(
            "[3/5] Fusing text norms, rotating LLM, and aligning vision output projection ..."
        )

    if should_rotate_llm:
        model = fuse_layer_norms_qwen3_5(
            model=model,
            pre_lm_head_norm_module_name=pre_lm_head_norm_module_name,
            layers_node=layers_node,
            lm_head_name=lm_head_name,
        )

    if effective_vision_rotation == "full":
        model, _ = rotate_vision_model_qwen3_5_vl(
            model=model,
            rotate_mode=(
                "hadamard" if args.llm_rotation == "none" else args.llm_rotation
            ),
            device=rotation_device,
            lm_head_name=lm_head_name,
            layers_node=layers_node,
        )

    if should_rotate_llm:
        model, _ = rotate_model_qwen3_5_vl(
            model=model,
            rotate_mode=args.llm_rotation,
            device=rotation_device,
            lm_head_name=lm_head_name,
            layers_node=layers_node,
        )
    model.eval()

    if args.validate:
        compare_logits(model, validation_batch, ref_logits, ref_generated, args)

    print(f"[5/5] Saving rotated fp model to: {args.out}")
    os.makedirs(args.out, exist_ok=True)
    model.save_pretrained(
        args.out, safe_serialization=True, max_shard_size=args.max_shard_size
    )
    processor.save_pretrained(args.out)
    print("Finished.")


if __name__ == "__main__":
    main()
