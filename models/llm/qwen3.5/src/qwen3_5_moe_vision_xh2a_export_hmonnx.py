#
# Copyright (c) 2025 HOUMO AI
#
# File: qwen3_5_moe_vision_xh2a_export_hmonnx.py
# Description:
#   Export script: Qwen3.5-MoE vision encoder -> ONNX/HMONNX.
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

"""
Qwen3.5-MoE Vision Model xh2a export script.

Exports the Qwen3.5-MoE ViT (vision encoder) to ONNX and HMONNX format.
Reuses qwen3_5 vision export utilities; only the model class and config differ.

Usage:
    # Basic export (default 448x448, batch=1)
    python examples/llm/qwen3_5_moe/qwen3_5_moe_vision_xh2a_export_hmonnx.py \\
        --hf_model_dir /data01/nfs_shared/Qwen3.5-35B-A3B

    # Export with validation
    python examples/llm/qwen3_5_moe/qwen3_5_moe_vision_xh2a_export_hmonnx.py \\
        --hf_model_dir /data01/nfs_shared/Qwen3.5-35B-A3B \\
        --valid

    # Custom resolution
    python examples/llm/qwen3_5_moe/qwen3_5_moe_vision_xh2a_export_hmonnx.py \\
        --hf_model_dir /data01/nfs_shared/Qwen3.5-35B-A3B \\
        --max_size_w 896 --max_size_h 896
"""

import tempfile
from pathlib import Path

import accelerate.hooks
import onnx
import torch
import torch.nn.functional as F
import xhquant.utils.suppress_printing
from qwen_vl_utils import process_vision_info
from xhquant.api import (
    ConfigDict,
    HMONNXInference,
    set_random_seed,
)
from xhquant.utils.onnxsim_large_model.simplify_large_onnx import simplify_large_onnx
from accelerate import dispatch_model, infer_auto_device_map
from accelerate.utils import get_balanced_memory

from xh_model_zoo.api import Config, get_root_logger, xhquant_llm_init
from xh_model_zoo.utils import MemoryTracker, TimeProfiler
from xh_model_zoo.xh_llm.models.builder import MODELS
from xh_model_zoo.xh_llm.models.qwen3_5 import Qwen3_5Processor
from xh_model_zoo.xh_llm.models.qwen3_5_moe import XHQwen3_5MoeVisionModel


DEFAULT_NO_SPLIT_MODULES = ["Qwen3_5MoeTextDecoderLayer", "Qwen3_5MoeVisionBlock"]
DEFAULT_CONFIG = "configs/qwen3_5_moe/qwen3_5_moe_vision_config.py"


# ─────────────────────── Device utilities ────────────────────────────


def _normalize_device_map_to_modules(model, device_map):
    module_names = set(dict(model.named_modules()).keys())
    normalized = {}
    for key, device in device_map.items():
        if key == "" or key in module_names:
            normalized[key] = device
            continue
        parts = key.split(".")
        while parts:
            candidate = ".".join(parts)
            if candidate in module_names:
                normalized.setdefault(candidate, device)
                break
            parts.pop()
    return normalized


def _auto_offload(model, no_split_module_classes=None, target_dtype=torch.float16):
    if no_split_module_classes is None:
        no_split_module_classes = []
    max_memory = get_balanced_memory(
        model, dtype=target_dtype, low_zero=False, max_memory=None,
        no_split_module_classes=list(no_split_module_classes),
    )
    device_map = infer_auto_device_map(
        model, dtype=target_dtype, max_memory=max_memory,
        no_split_module_classes=list(no_split_module_classes),
    )
    device_map = _normalize_device_map_to_modules(model, device_map)
    dispatch_model(model, device_map=device_map, offload_buffers=True)


def _remove_offload(model):
    accelerate.hooks.remove_hook_from_module(model, recurse=True)
    if hasattr(model, "hf_device_map"):
        delattr(model, "hf_device_map")


def _de_dispatch_to_cpu(model):
    _remove_offload(model)
    model.to("cpu")
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


# ──────────────────── Processor / Input utilities ────────────────────


def _build_processor(hf_model_dir):
    return Qwen3_5Processor.from_pretrained(hf_model_dir)


def _prepare_inputs(hf_model_dir, args):
    processor = _build_processor(hf_model_dir)
    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "image": args.image_path,
                    "resized_height": args.max_size_h,
                    "resized_width": args.max_size_w,
                },
                {"type": "text", "text": args.prompt},
            ],
        }
    ]
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs = process_vision_info(messages, image_patch_size=args.patch_size)
    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    )
    return processor, inputs


def _filter_generate_kwargs(inputs):
    exclude = {"hm_pixel_values", "pixel_values_videos", "image_grid_thw", "video_grid_thw"}
    return {k: v for k, v in inputs.items() if k not in exclude}


def _run_generate(native_model, inputs, processor, max_new_tokens, label, logger, **gen_kwargs):
    gen_inputs = _filter_generate_kwargs(inputs)
    with torch.no_grad():
        generated_ids = native_model.generate(**gen_inputs, max_new_tokens=max_new_tokens, **gen_kwargs)
    generated_ids_trimmed = [
        out_ids[len(in_ids):] for in_ids, out_ids in zip(gen_inputs["input_ids"], generated_ids)
    ]
    output_text = processor.batch_decode(
        generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False,
    )
    logger.info(f"***************** {label} output *****************")
    logger.info(output_text)


# ──────────────────────── Export functions ────────────────────────────


def _export_onnx(wraped_model, hm_pixel_values, out_onnx_file, logger):
    if Path(out_onnx_file).exists():
        logger.info(f"ONNX model already exists: {out_onnx_file}")
        return onnx.load(out_onnx_file, load_external_data=True)

    Path(out_onnx_file).parent.mkdir(exist_ok=True, parents=True)
    wraped_model.float().eval().cpu()

    with tempfile.TemporaryDirectory() as tmp_dir:
        onnx_file = str(Path(tmp_dir) / "visual.onnx")
        logger.info(f"Exporting ONNX model to {onnx_file}")
        torch.onnx.export(
            wraped_model,
            (hm_pixel_values.float().cpu(),),
            onnx_file,
            export_params=True,
            opset_version=18,
            do_constant_folding=True,
            input_names=["pixel_values"],
            output_names=["image_embeds"],
            verbose=True,
        )
        onnx_model = onnx.load(onnx_file, load_external_data=True)

    onnx_model, _ = simplify_large_onnx(onnx_model)
    onnx.save(
        onnx_model,
        out_onnx_file,
        save_as_external_data=True,
        all_tensors_to_one_file=True,
        location="visual_external_data",
        convert_attribute=True,
    )
    logger.info(f"ONNX model saved to: {out_onnx_file}")
    return onnx_model


def _convert_to_hmonnx(out_onnx_file, out_hmonnx_file, hm_pixel_values, quant_config, logger):
    if out_hmonnx_file.exists():
        logger.info(f"HMONNX model already exists: {out_hmonnx_file}")
        return
    from xhquant.api import DeviceType, convert_onnx_to_hmonnx
    out_hmonnx_file.parent.mkdir(exist_ok=True, parents=True)
    convert_onnx_to_hmonnx(
        str(out_onnx_file), [hm_pixel_values.float().cpu()],
        DeviceType.XH2a, str(out_hmonnx_file), quant_config,
    )
    logger.info(f"Convert ONNX to HMONNX success: {out_hmonnx_file}")


# ──────────────────────── Main Export Logic ───────────────────────────


def _export_impl(cfg, args):
    logger = get_root_logger()
    execution_device = torch.device(cfg.execution_device)

    # Build model
    cfg.model.wrap_cfg.max_size_w = args.max_size_w
    cfg.model.wrap_cfg.max_size_h = args.max_size_h
    cfg.model.wrap_cfg.max_size_t = args.max_size_t
    cfg.model.wrap_cfg.temporal_patch_size = args.temporal_patch_size
    cfg.model.wrap_cfg.patch_size = args.patch_size
    cfg.model.hf_model = args.hf_model_dir

    vision_model: XHQwen3_5MoeVisionModel = MODELS.build(cfg.model)
    native_model = vision_model.get_hf_model(device_map="auto")

    # Prepare processor & inputs
    processor, inputs = _prepare_inputs(args.hf_model_dir, args)
    inputs.to(execution_device)

    # Optional native generate before wrapping
    if args.valid:
        _run_generate(
            native_model, inputs, processor, args.max_new_tokens,
            "native model", logger, repetition_penalty=1.05,
        )

    # De-dispatch and init wrap model
    logger.info("De-dispatching native model to CPU for wrapping...")
    _de_dispatch_to_cpu(native_model)
    vision_model.init_wrap_model(native_model)
    wraped_model = vision_model.wrap_model

    # Prepare pixel values for export
    hm_pixel_values = inputs["hm_pixel_values"][0].type(wraped_model.dtype).to(wraped_model.device)
    if args.batch_size > 1:
        hm_pixel_values = hm_pixel_values.repeat(args.batch_size, 1, 1, 1, 1)

    # Export to ONNX
    out_onnx_file = str(Path(cfg.work_dir) / "onnx" / f"visual_{args.batch_size}.onnx")
    _export_onnx(wraped_model, hm_pixel_values, out_onnx_file, logger)

    # Convert to HMONNX
    out_hmonnx_file = Path(cfg.work_dir) / "vision" / "vision.onnx"
    _convert_to_hmonnx(out_onnx_file, out_hmonnx_file, hm_pixel_values, cfg.model.quant_config, logger)

    logger.info(f"Vision export done. Artifacts in: {cfg.work_dir}")

    # Validate HMONNX (optional)
    if args.valid:
        logger.info("HMONNX validation requested — re-dispatching for generate()...")
        _auto_offload(native_model, no_split_module_classes=DEFAULT_NO_SPLIT_MODULES)
        hm_session = HMONNXInference(str(out_hmonnx_file))
        hm_session.exec_device = execution_device

        # Patch native model visual with HMONNX wrapper
        spatial_merge_size = getattr(native_model.model.visual, "spatial_merge_size", 2)
        patch_size_val = getattr(native_model.model.visual, "patch_size", args.patch_size)

        class _HMONNXVisual(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.spatial_merge_size = spatial_merge_size
                self.patch_size = patch_size_val

            def forward(self, pixel_values):
                out = hm_session(pixel_values.half())
                return out[0] if isinstance(out, (list, tuple)) else out

        native_model.model.visual = _HMONNXVisual()
        inputs["pixel_values"] = inputs["hm_pixel_values"][0].half()
        _run_generate(
            native_model, inputs, processor, args.max_new_tokens,
            "hmonnx model", logger,
        )


# ──────────────────────────── Entry Point ────────────────────────────


def parse_arguments():
    import argparse
    parser = argparse.ArgumentParser(
        description="Export Qwen3.5-MoE ViT to ONNX/HMONNX",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--config", type=str, default=DEFAULT_CONFIG)
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./work_dirs",
        help="Root directory to store export artifacts",
    )
    parser.add_argument("--hf_model_dir", type=str, default="/data01/nfs_shared/Qwen3.5-35B-A3B")
    parser.add_argument(
        "--model_name",
        type=str,
        default=None,
        help="Override model name used for output directory (default: basename of --hf_model_dir)",
    )
    parser.add_argument("--max_size_w", type=int, default=448)
    parser.add_argument("--max_size_h", type=int, default=448)
    parser.add_argument("--max_size_t", type=int, default=2)
    parser.add_argument("--temporal_patch_size", type=int, default=2)
    parser.add_argument("--patch_size", type=int, default=16)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--image_path", type=str, default="images/qwen2_vl_demo.jpeg")
    parser.add_argument("--prompt", type=str, default="清晰描述图片中的内容。")
    parser.add_argument("--max_new_tokens", type=int, default=256)
    parser.add_argument("--valid", action="store_true", help="Validate HMONNX by running generate()")
    parser.add_argument("--seed", type=int, default=1024)
    parser.add_argument("--debug", action="store_true")
    return parser


def main(args):
    cfg = Config.fromfile(args.config)
    model_name = args.model_name or Path(args.hf_model_dir).name
    size_suffix = f"{args.max_size_w}x{args.max_size_h}x{args.max_size_t}"
    cfg.work_dir = str(Path(args.output_dir) / f"{model_name}_{size_suffix}")
    Path(cfg.work_dir).mkdir(exist_ok=True, parents=True)
    log_file = Path(cfg.work_dir) / "vision_export.log"
    cfg.device = "cuda:0"
    cfg.execution_device = "cuda:0" if torch.cuda.is_available() else "cpu"

    set_random_seed(args.seed)
    xhquant_llm_init(log_file, args.debug)
    logger = get_root_logger()
    logger.info(f"hf_model_dir: {args.hf_model_dir}")
    logger.info(f"output: {cfg.work_dir}")

    xhquant.utils.suppress_printing.disable_printing = True
    with TimeProfiler("vision export", logger), MemoryTracker(0, "vision_export", logger):
        _export_impl(cfg, args)


if __name__ == "__main__":
    parser = parse_arguments()
    args = parser.parse_args()
    main(args)
