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
import json
import time
import random
import argparse
import tempfile
import torch
import torch.nn as nn
from datetime import datetime
from pathlib import Path
from typing import Any
from functools import lru_cache
from PIL import Image
from datasets import load_dataset
from transformers import (
    Gemma4ForConditionalGeneration,
    GemmaTokenizer,
    Gemma4Processor,
    AutoTokenizer,
    AutoProcessor,
)
from xhquant.utils import MemoryTracker, TimeProfiler
from xhquant.api import (
    Config,
    xhquant_init,
    get_root_logger,
    DeviceType,
    convert_onnx_to_hmonnx,
)
from xhmodel_merak.xh_llm import (
    AutoLLMConfig,
    AutoLLMModel,
    format_model_name,
)
from xhmodel_merak.xh_llm.models.gemma4_moe import (
    XHGemma4MoeWithMaskConfig,
    XHGemma4MoeWithMaskModel,
)
from xhmodel_merak.xh_llm.models.gemma4_moe.gemma4_moe_visual_model import (
    Gemma4MoeVisionWrapper,
    XHGemma4MoeVisualModel,
)

HOUMO_TARGET = os.getenv("HOUMO_TARGET", "xh2")
HOUMO_CORE_NUM = int(os.getenv("HOUMO_CORE_NUM", 2))

DEFAULT_IMAGE_SIZE = (448, 448)  # WH
OFFICIAL_POOLING_KERNEL_SIZE = 3
DIRECT_TOKEN_POOLING_KERNEL_SIZE = 1
MAX_SOFT_TOKENS = 280


try:
    BICUBIC = Image.Resampling.BICUBIC
except AttributeError:
    BICUBIC = Image.BICUBIC


def _build_cfg_from_model(args):
    cfg = dict(
        chip_arch="XH2a",
        model=dict(
            model_type="Gemma4ForConditionalGeneration_with_mask",
            hf_model=f"{args.model}-gptq-4bit",
            fallback_hf_model=args.model,
            model_name=args.model_name,
            context_max_length=args.context_length,
            prefill_chunk_length=256,
            use_cache=True,
            num_logits_to_keep=1,
            quant_scheme=dict(
                quant_type="w4a8h1_sefp",
                ops=dict(),
            ),
            visual_config=dict(
                model_type="Gemma4ForConditionalGeneration_visual",
                hf_model=f"{args.model}-gptq-4bit",
                model_name=args.model_name,
                max_size_w=DEFAULT_IMAGE_SIZE[0],
                max_size_h=DEFAULT_IMAGE_SIZE[1],
                upsample_token=False,
                fuse_norm=True,
                quant_scheme=dict(
                    quant_type="w8a8h1_sefp",
                    ops=dict(),
                ),
            ),
        ),
    )
    cfg = format_model_name(cfg)
    return Config(cfg)


def _build_visual_cfg_from_model(args):
    cfg = dict(
        chip_arch="XH2a",
        model=dict(
            model_type="Gemma4ForConditionalGeneration_visual",
            hf_model=args.model,
            model_name=args.model_name,
            max_size_w=DEFAULT_IMAGE_SIZE[0],
            max_size_h=DEFAULT_IMAGE_SIZE[1],
            upsample_token=False,
            fuse_norm=True,
            quant_scheme=dict(
                quant_type="w8a8h1_sefp",
                ops=dict(),
            ),
        ),
    )
    return Config(cfg)


def resolve_pooling_kernel_size(upsample_token: bool) -> int:
    return (
        OFFICIAL_POOLING_KERNEL_SIZE
        if upsample_token
        else DIRECT_TOKEN_POOLING_KERNEL_SIZE
    )


def build_visual_variant_name(
    upsample_token: bool,
    target_image_size: tuple[int, int] = DEFAULT_IMAGE_SIZE,
) -> str:
    token_tag = "upsample_token" if upsample_token else "no_upsample_token"
    return f"{token_tag}_{target_image_size[0]}x{target_image_size[1]}"


def prepare_visual_input_image(
    image_path: str | Path,
    upsample_token: bool,
    target_image_size: tuple[int, int] = DEFAULT_IMAGE_SIZE,
) -> tuple[Image.Image, dict[str, Any]]:
    resolved_image_path = Path(image_path).resolve()
    image = Image.open(resolved_image_path).convert("RGB")
    original_size = image.size

    if image.size != target_image_size:
        image = image.resize(target_image_size, BICUBIC)

    preprocess_meta = {
        "source_image": str(resolved_image_path),
        "original_image_size": list(original_size),
        "target_image_size": list(target_image_size),
        "upsample_token": upsample_token,
        "processor_pooling_kernel_size": resolve_pooling_kernel_size(upsample_token),
        "max_soft_tokens": MAX_SOFT_TOKENS,
        "vision_variant": build_visual_variant_name(upsample_token, target_image_size),
    }
    return image, preprocess_meta


def prepare_vision_input_image(
    image_path: str | Path,
    upsample_token: bool,
    target_image_size: tuple[int, int] = DEFAULT_IMAGE_SIZE,
) -> tuple[Image.Image, dict[str, Any]]:
    return prepare_visual_input_image(image_path, upsample_token, target_image_size)


def extract_valid_patch_tokens(
    pixel_values: torch.Tensor,
    pixel_position_ids: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    padding_positions = (pixel_position_ids == -1).all(dim=-1)
    valid_mask = ~padding_positions
    return (
        pixel_values[:, valid_mask[0]],
        pixel_position_ids[:, valid_mask[0]],
        valid_mask[0],
    )


def patch_gemma4_rmsnorm_for_export(fuse_norm: bool) -> None:
    from transformers.models.gemma4.modeling_gemma4 import Gemma4RMSNorm
    from xhquant.backend.xh2a import torch_ops_xh2a_rmsnorm

    def _safe_norm(self, hidden_states: torch.Tensor):
        max_val = hidden_states.abs().amax(dim=-1, keepdim=True).clamp(min=1.0)
        scaled = hidden_states / max_val
        variance = (scaled * scaled).mean(-1, keepdim=True) + self.eps
        return hidden_states * torch.reciprocal(max_val * torch.sqrt(variance))

    def _get_export_weight(self, hidden_states: torch.Tensor):
        if self.with_scale:
            return self.weight.float()
        export_weight = getattr(self, "_xh2a_export_unit_weight", None)
        hidden_size = hidden_states.shape[-1]
        if (
            export_weight is None
            or export_weight.shape != (hidden_size,)
            or export_weight.dtype != hidden_states.dtype
            or export_weight.device != hidden_states.device
        ):
            export_weight = torch.ones(
                hidden_size, device=hidden_states.device, dtype=hidden_states.dtype
            )
            if "_xh2a_export_unit_weight" in self._buffers:
                self._buffers["_xh2a_export_unit_weight"] = export_weight
            else:
                self.register_buffer(
                    "_xh2a_export_unit_weight", export_weight, persistent=False
                )
        return export_weight

    def _exportable_forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states_fp32 = hidden_states.float()
        if torch.onnx.is_in_onnx_export():
            weight = _get_export_weight(self, hidden_states_fp32)
            normed_output = torch_ops_xh2a_rmsnorm(
                hidden_states_fp32, weight, self.eps, -1, "normal", 0, True
            )
        else:
            if not self.with_scale:
                _get_export_weight(self, hidden_states_fp32)
            normed_output = _safe_norm(self, hidden_states_fp32)
            if self.with_scale:
                normed_output = normed_output * self.weight.float()
        return normed_output.type_as(hidden_states)

    if fuse_norm:
        Gemma4RMSNorm.forward = _exportable_forward
    else:
        Gemma4RMSNorm._norm = _safe_norm


@lru_cache(maxsize=1)
def build_xh2a_custom_translation_table() -> dict[Any, Any]:
    from torch.onnx._internal.exporter._registration import _get_overload

    from xhquant.export.onnx.xh2a_onnx_registry import xh2a_default_registry

    custom_translation_table = {}
    for qualified_name, aten_overload_func in xh2a_default_registry.items():
        target = _get_overload(qualified_name)
        if target is None:
            continue
        for overload_func in aten_overload_func.overloads:
            custom_translation_table[target] = overload_func
    return custom_translation_table


def export_onnx_with_xh2a_custom_ops(
    model: nn.Module,
    export_args: tuple[torch.Tensor, ...],
    onnx_file: str | Path,
    input_names: list[str],
    output_names: list[str],
) -> None:
    onnx_file = Path(onnx_file)
    onnx_program = torch.onnx.export(
        model,
        export_args,
        None,
        input_names=input_names,
        output_names=output_names,
        opset_version=18,
        do_constant_folding=True,
        custom_translation_table=build_xh2a_custom_translation_table(),
        dynamo=True,
        optimize=True,
        verbose=True,
    )
    onnx_program.save(str(onnx_file))


def _create_default_image() -> Path:
    image_path = Path(tempfile.gettempdir()) / "gemma4_moe_visual_default.png"
    if not image_path.exists():
        Image.new(
            "RGB", (DEFAULT_IMAGE_SIZE[0], DEFAULT_IMAGE_SIZE[1]), color=(255, 255, 255)
        ).save(image_path)
    return image_path


def rename_hmquant_dir(
    base_dir: str | Path, dry_run: bool = False
) -> list[tuple[Path, Path]]:
    """Rename directories matching hmquant_xh2_{model}_{quant}_{seq}_{context}_{date} to hmquant.

    Args:
        base_dir: The directory to search for matching directories.
        dry_run: If True, only print what would be renamed without actually renaming.

    Returns:
        List of (old_path, new_path) tuples for renamed directories.

    Example:
        hmquant_xh2_gemma-4-26b-a4b_w8a8_256_2k_20260506 -> hmquant
    """
    import re

    base_path = Path(base_dir)
    # Format: hmquant_xh2_{model_name}_{quant_type}_{seq_len}_{context_len}_{date}
    # date format: YYYYMMDD (8 digits)
    pattern = re.compile(r"^hmquant_xh2_.+_\d{8}$")

    renamed = []
    for item in base_path.iterdir():
        if item.is_dir() and pattern.match(item.name):
            new_path = item.parent / "hmquant"

            if new_path.exists():
                print(f"Warning: Target directory already exists: {new_path}")
                continue

            renamed.append((item, new_path))

            if dry_run:
                print(f"[Dry-run] Would rename: {item} -> {new_path}")
            else:
                item.rename(new_path)
                print(f"Renamed: {item} -> {new_path}")

    return renamed


def configure_gemma4_visual_processor(processor: Any, upsample_token: bool) -> int:
    pooling_kernel_size = resolve_pooling_kernel_size(upsample_token)
    processor.image_processor.max_soft_tokens = MAX_SOFT_TOKENS
    processor.image_processor.pooling_kernel_size = pooling_kernel_size
    processor.image_seq_length = (
        MAX_SOFT_TOKENS
        if upsample_token
        else DEFAULT_IMAGE_SIZE[0] // 28 * DEFAULT_IMAGE_SIZE[1] // 28
    )
    return pooling_kernel_size


def get_jsonl_texts(path, nsamples, text_key="text"):
    samples = []
    skipped_blank = 0
    with open(path, "r", encoding="utf-8") as file:
        for line_no, line in enumerate(file, start=1):
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            text = record.get(text_key)
            if not isinstance(text, str):
                raise ValueError(f"Invalid `{text_key}` at {path}:{line_no}")
            if not text.strip():
                skipped_blank += 1
                logger.warning("Skipping blank `%s` at %s:%s", text_key, path, line_no)
                continue
            samples.append(text)
            if len(samples) >= nsamples:
                break

    if len(samples) == 0:
        raise ValueError(f"No calibration samples found in {path}")
    if skipped_blank:
        logger.warning(
            "Skipped %s blank calibration sample(s) from %s", skipped_blank, path
        )
    if len(samples) < nsamples:
        logger.warning(
            "Requested %s calibration sample(s) from %s, but only found %s valid sample(s)",
            nsamples,
            path,
            len(samples),
        )
    return samples


def get_wikitext2(nsamples, seqlen):
    traindata = load_dataset("wikitext", "wikitext-2-raw-v1", split="train").filter(
        lambda x: len(x["text"]) >= seqlen
    )
    return [example["text"] for example in traindata.select(range(nsamples))]


def build_calibration_dataset(
    calibration_jsonl, seqlen, nsamples, calibration_text_key
):
    if calibration_jsonl:
        calibration_dataset = get_jsonl_texts(
            path=calibration_jsonl,
            nsamples=nsamples,
            text_key=calibration_text_key,
        )
        source_name = os.path.basename(calibration_jsonl)
        return calibration_dataset, f"jsonl:{source_name}"

    return get_wikitext2(nsamples=nsamples, seqlen=seqlen), "wikitext2"


def run_gptqmodel(args, device):
    from gptqmodel import GPTQModel, QuantizeConfig

    logger = get_root_logger()
    calibration_dataset, calibration_source = build_calibration_dataset(
        calibration_jsonl=args.calibration_jsonl,
        seqlen=args.seqlen,
        nsamples=args.nsamples,
        calibration_text_key=args.calibration_text_key,
    )
    logger.info(
        f"Calibration samples ({calibration_source}): {len(calibration_dataset)}"
    )
    quantize_config = QuantizeConfig(
        bits=args.bits,
        group_size=args.group_size,
        hessian_mse=args.hessian_mse,
        mse=args.mse,
        device=device,
        offload_to_disk=False,
        offload_to_disk_path=None,
    )

    logger.info("Loading fp model ...")
    load_kwargs = dict(trust_remote_code=False)
    model = GPTQModel.load(args.model, quantize_config, **load_kwargs)

    logger.info("Running GPTQ quantization ...")
    model.quantize(calibration_dataset, batch_size=1)
    output_dir = f"{args.model}-gptq-4bit"
    model.save(output_dir)
    logger.info(f"Quantized model saved to: {output_dir}")


def export_llm(cfg, output_dir: str | Path, device, dtype):
    output_dir = Path(output_dir)
    logger = get_root_logger()
    model_cfg: XHGemma4MoeWithMaskConfig = AutoLLMConfig.from_pretrained(cfg.model)
    xh_model: XHGemma4MoeWithMaskModel = AutoLLMModel.from_pretrained(model_cfg)

    if hasattr(xh_model, "visual"):
        try:
            delattr(xh_model, "visual")
        except AttributeError:
            pass
    if hasattr(xh_model, "_models") and isinstance(xh_model._models, dict):
        xh_model._models.pop("visual", None)

    with TimeProfiler("convert", logger), MemoryTracker(
        device, "convert2hmonnx", logger
    ):
        xh_model.export_hmonnx(str(output_dir))


def export_vision(cfg, output_dir: str | Path, device, dtype):
    output_dir = Path(output_dir)
    logger = get_root_logger()
    model_cfg = AutoLLMConfig.from_pretrained(cfg.model)
    xh_visual_model: XHGemma4MoeVisualModel = AutoLLMModel.from_pretrained(model_cfg)
    logger.info(f"Vision export dir: {output_dir}")

    logger.info(f"Loading Gemma4 model from {xh_visual_model.hf_model_dir} ...")
    model = Gemma4ForConditionalGeneration.from_pretrained(
        xh_visual_model.hf_model_dir,
        torch_dtype=torch.float16,
        device_map="cpu",
        trust_remote_code=True,
        attn_implementation="eager",
    ).eval()

    processor: Gemma4Processor = AutoProcessor.from_pretrained(
        xh_visual_model.hf_model_dir, trust_remote_code=True
    )
    processor_pooling_kernel_size = configure_gemma4_visual_processor(
        processor,
        xh_visual_model.config.upsample_token,
    )
    vision_tower = model.model.vision_tower
    embed_vision = model.model.embed_vision
    vision_config = model.config.vision_config
    vision_config._attn_implementation = "eager"
    vision_config.pooling_kernel_size = processor_pooling_kernel_size
    patch_gemma4_rmsnorm_for_export(xh_visual_model.config.fuse_norm)

    vision_wrapper = (
        Gemma4MoeVisionWrapper(vision_tower, embed_vision, vision_config)
        .eval()
        .to(device)
        .to(dtype)
    )
    image_path = _create_default_image()
    processed_image, _ = prepare_visual_input_image(
        image_path,
        upsample_token=xh_visual_model.config.upsample_token,
        target_image_size=(
            xh_visual_model.config.max_size_w,
            xh_visual_model.config.max_size_h,
        ),
    )
    inputs = processor.apply_chat_template(
        [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": processed_image},
                    {"type": "text", "text": "Describe."},
                ],
            }
        ],
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
    )
    pixel_values = inputs["pixel_values"].to(device=device, dtype=dtype)
    image_position_ids = inputs["image_position_ids"].to(device=device)
    with torch.no_grad():
        vision_wrapper.precompute_constants(pixel_values, image_position_ids)
    pixel_values_valid, _, _ = extract_valid_patch_tokens(
        pixel_values, image_position_ids
    )

    vision_wrapper_fp32 = vision_wrapper.float().cpu()
    pixel_values_cpu = pixel_values_valid.float().cpu()
    input_names = ["pixel_values"]
    output_names = ["image_embeds"]
    export_args = (pixel_values_cpu,)

    onnx_file = output_dir / "visual.onnx"
    export_onnx_with_xh2a_custom_ops(
        vision_wrapper_fp32,
        export_args,
        onnx_file,
        input_names,
        output_names,
    )

    str_datetime = datetime.now().strftime("%Y%m%d")
    hmonnx_name = f"hmquant_xh2_{args.model_name}_{cfg.model.max_size_w}_{str_datetime}_visual_with_act.onnx"
    hmonnx_file = output_dir / hmonnx_name
    convert_onnx_to_hmonnx(
        onnx_file,
        [pixel_values_cpu],
        DeviceType.XH2a,
        str(hmonnx_file),
    )
    logger.info("Vision export done.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # fmt: off
    parser.add_argument("--model", type=str, default="./models/gemma-4-26B-A4B-it", help="path to the model directory")
    parser.add_argument("--model-name", type=str, default="gemma-4-26b-a4b", help="model name for output files")
    parser.add_argument("--out-dir", type=str, default=f"./output/{HOUMO_TARGET}", help="output directory")      
    parser.add_argument("--context-length", type=int, default=2048, help="max sequence length")
    parser.add_argument("--nsamples", type=int, default=512, help="number of calibration samples")
    parser.add_argument("--seqlen", type=int, default=1024, help="sequence length for calibration")
    parser.add_argument("--mse", type=float, default=2.4, help="MSE threshold for quantization")
    parser.add_argument("--bits", type=int, default=4, choices=[4], help="quantization bits")
    parser.add_argument("--group-size", type=int, default=64, help="group size for quantization")
    parser.add_argument("--hessian-mse", action=argparse.BooleanOptionalAction, default=True, help="enable Hessian MSE assisted optimization")
    parser.add_argument("--calibration-jsonl", type=str, default="./calib_EBSS.jsonl", help="path to calibration dataset in jsonl format")
    parser.add_argument("--calibration-text-key", type=str, default="text", help="key for text field in calibration jsonl")
    parser.add_argument("--seed", type=int, default=42, help="random seed for reproducibility")
    # fmt: on
    args = parser.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)

    output_dir = Path(args.out_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    xhquant_init()
    logger = get_root_logger()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float16
    logger.info(f"device={device} dtype={dtype}")

    # gpqmodel 4bit
    run_gptqmodel(args, device)

    # export llm
    cfg = _build_cfg_from_model(args)
    export_llm(cfg, output_dir, device, dtype)
    # rename hmquant_xh2_{model_name}_{quant_type}_{seq_len}_{context_len}_{datetime} -> hmquant
    rename_hmquant_dir(output_dir)

    # export visual
    cfg = _build_visual_cfg_from_model(args)
    output_dir = output_dir / "hmquant" / "visual"
    output_dir.mkdir(parents=True, exist_ok=True)
    export_vision(cfg, output_dir, device, dtype)
