# Copyright (c) 2025 HOUMO AI
#
# File: qwen3_5_vision_xh2a_export_hmonnx.py
# Description:
#   Export Qwen3.5 vision encoder to ONNX/HMONNX for XH2a.
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
Qwen3.5 Vision Model xh2a export script.

Exports the Qwen3.5 ViT (vision encoder) to ONNX and HMONNX format.
Supports:
  - GPTQ quantized model loading (--use_gptq_model)
  - QuaRot rotation (--quarot_weight_path, --gptq_model_rotate)
  - Variable image/video resolution (--max_size_w, --max_size_h, --max_size_t)
  - Batch export (--batch_size)
  - Optional end-to-end validation with the full VL model (--valid)
  - HF vs Wrap precision comparison (--compare_hf_wrap, --compare_per_block)
  - Configurable comparison dtype (--compare_dtype float16/float32/bfloat16)

Reference: examples/qwen3_vl/qwen3_vl_vision_xh2a_export_hmonnx.py

Usage:
    # Basic export (default 448x448, batch=1)
    python examples/qwen3.5/qwen3_5_vision_xh2a_export_hmonnx.py \\
        --config configs/qwen3_5/qwen3_5_instruct_vision_config.py \\
        --hf_model_dir weights/Qwen3.5-27B

    # Export with validation
    python examples/qwen3.5/qwen3_5_vision_xh2a_export_hmonnx.py \\
        --config configs/qwen3_5/qwen3_5_instruct_vision_config.py \\
        --hf_model_dir weights/Qwen3.5-27B \\
        --valid

    # Export with GPTQ model + rotation
    python examples/qwen3.5/qwen3_5_vision_xh2a_export_hmonnx.py \\
        --config configs/qwen3_5/qwen3_5_instruct_vision_config.py \\
        --hf_model_dir weights/Qwen3.5-27B-GPTQ \\
        --use_gptq_model --gptq_model_rotate

    # HF vs Wrap precision comparison (fp32)
    python examples/qwen3.5/qwen3_5_vision_xh2a_export_hmonnx.py \\
        --config configs/qwen3_5/qwen3_5_instruct_vision_config.py \\
        --hf_model_dir weights/Qwen3.5-27B \\
        --compare_hf_wrap

    # HF vs Wrap with per-block comparison in fp16
    python examples/qwen3.5/qwen3_5_vision_xh2a_export_hmonnx.py \\
        --config configs/qwen3_5/qwen3_5_instruct_vision_config.py \\
        --hf_model_dir weights/Qwen3.5-27B \\
        --compare_hf_wrap --compare_per_block --compare_dtype float16
"""

import shutil
import tempfile
import time
from pathlib import Path

import accelerate.hooks
import onnx
import onnxruntime as ort
import torch
import torch.nn as nn
import torch.nn.functional as F
from accelerate import dispatch_model, infer_auto_device_map
from accelerate.utils import get_balanced_memory
import xhquant.utils.suppress_printing
from qwen_vl_utils import process_vision_info
from torch import Tensor
from xhquant.api import (
    ConfigDict,
    HMONNXInference,
    QTensor,
    set_random_seed,
)
from xhquant.utils.onnxsim_large_model.simplify_large_onnx import simplify_large_onnx
from safetensors.torch import load_file as load_safetensors_file

from xh_model_zoo.api import Config, get_root_logger, xhquant_llm_init
from xh_model_zoo.utils import MemoryTracker, TimeProfiler
from xh_model_zoo.xh_llm.models.builder import MODELS
from xh_model_zoo.xh_llm.models.qwen3_5 import Qwen3_5Processor, XHQwen3_5VisionModel


# ─────────────────────────── Argument Parsing ──────────────────────────


DEFAULT_NO_SPLIT_MODULES = ["Qwen3_5TextDecoderLayer", "Qwen3_5VisionBlock"]


def _normalize_device_map_to_modules(model: nn.Module, device_map):
    module_names = set(dict(model.named_modules()).keys())
    normalized_device_map = {}

    for key, device in device_map.items():
        if key == "" or key in module_names:
            normalized_key = key
        else:
            parts = key.split(".")
            normalized_key = ""
            while parts:
                candidate = ".".join(parts)
                if candidate in module_names:
                    normalized_key = candidate
                    break
                parts.pop()

        existing_device = normalized_device_map.get(normalized_key)
        if existing_device is not None and existing_device != device:
            continue
        normalized_device_map[normalized_key] = device

    return normalized_device_map


def _auto_offload(model: nn.Module, no_split_module_classes=None, target_dtype: torch.dtype = torch.float16):
    if no_split_module_classes is None:
        no_split_module_classes = []
    if not isinstance(no_split_module_classes, (list, tuple)):
        no_split_module_classes = [no_split_module_classes]
    max_memory = get_balanced_memory(
        model,
        dtype=target_dtype,
        low_zero=False,
        max_memory=None,
        no_split_module_classes=list(no_split_module_classes),
    )
    device_map = infer_auto_device_map(
        model,
        dtype=target_dtype,
        max_memory=max_memory,
        no_split_module_classes=list(no_split_module_classes),
    )
    device_map = _normalize_device_map_to_modules(model, device_map)
    dispatch_model(model, device_map=device_map, offload_buffers=True)


def _remove_offload(model: nn.Module):
    accelerate.hooks.remove_hook_from_module(model, recurse=True)
    if hasattr(model, "hf_device_map"):
        delattr(model, "hf_device_map")


def parse_arguments():
    import argparse

    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument(
        "--config",
        type=str,
        default="configs/qwen3_5/qwen3_5_instruct_vision_config.py",
    )
    parser.add_argument(
        "--output_root",
        type=str,
        default="./work_dirs",
        help="Root directory; work_dir is <output_root>/<model_name>/vision_export/.",
    )
    parser.add_argument(
        "--clean_output",
        action="store_true",
        default=False,
        help="Remove onnx/, vision/, and debug/ under work_dir before export (avoids reusing stale artifacts).",
    )
    parser.add_argument("--hf_model_dir", type=str, default="weights/Qwen3.5-27B")
    parser.add_argument("--use_gptq_model", action="store_true", default=False)
    parser.add_argument("--gptq_model_rotate", action="store_true", default=False)
    parser.add_argument("--model_name", type=str, default="27B")
    parser.add_argument("--max_size_w", type=int, default=448)
    parser.add_argument("--max_size_h", type=int, default=448)
    parser.add_argument("--max_size_t", type=int, default=2)
    parser.add_argument("--temporal_patch_size", type=int, default=2)
    parser.add_argument("--patch_size", type=int, default=16)
    parser.add_argument("--debug", action="store_true", help="debug mode")
    parser.add_argument("--image_path", type=str, default="images/qwen2_vl_demo.jpeg")
    parser.add_argument("--prompt", type=str, default="清晰描述图片中的内容。")
    parser.add_argument("--seed", type=int, default=1024)
    parser.add_argument("--valid", action="store_true", help="evaluate the model")
    parser.add_argument("--max_new_tokens", type=int, default=2048)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--quarot_weight_path", type=str, default=None)
    parser.add_argument(
        "--compare_dtype",
        type=str,
        default="float32",
        choices=["float16", "float32", "bfloat16"],
        help="Data type for HF vs Wrap precision comparison (default: float32)",
    )
    parser.add_argument(
        "--compare_hf_wrap",
        action="store_true",
        default=False,
        help="Compare HF model output vs Wrap model output for precision validation",
    )
    parser.add_argument(
        "--compare_per_block",
        action="store_true",
        default=False,
        help="Enable per-block (per-layer) precision comparison between HF and Wrap",
    )
    return parser


# ─────────────────── Precision Comparison Utilities ───────────────────


def _compute_precision_metrics(hf_output, wrap_output, label=""):
    """Compute precision comparison metrics between two tensors."""
    diff = (hf_output - wrap_output).abs()
    max_abs = diff.max().item()
    mean_abs = diff.mean().item()
    cos_sim = F.cosine_similarity(
        hf_output.flatten().unsqueeze(0),
        wrap_output.flatten().unsqueeze(0),
    ).item()
    return {
        "label": label,
        "max_abs_error": max_abs,
        "mean_abs_error": mean_abs,
        "cosine_similarity": cos_sim,
    }


def _log_precision_metrics(logger, metrics):
    """Log precision comparison results."""
    label = metrics["label"]
    logger.info(f"  [{label}] max  abs error : {metrics['max_abs_error']:.6e}")
    logger.info(f"  [{label}] mean abs error : {metrics['mean_abs_error']:.6e}")
    logger.info(f"  [{label}] cosine similarity: {metrics['cosine_similarity']:.10f}")


def _collect_block_outputs(visual_model, forward_fn, block_attr="blocks"):
    """Collect per-block hidden_states via forward hooks.

    Returns:
        (forward_result, OrderedDict[block_name -> hidden_states_cpu_float32])
    """
    from collections import OrderedDict

    outputs = OrderedDict()

    def make_hook(name):
        def hook(module, args, output):
            if isinstance(output, tuple):
                outputs[name] = output[0].detach().cpu().float()
            elif isinstance(output, Tensor):
                outputs[name] = output.detach().cpu().float()
        return hook

    handles = []
    blocks = getattr(visual_model, block_attr)
    for idx, blk in enumerate(blocks):
        h = blk.register_forward_hook(make_hook(f"block_{idx}"))
        handles.append(h)

    result = forward_fn()

    for h in handles:
        h.remove()
    return result, outputs


def _run_hf_visual_forward(native_model, pixel_values, image_grid_thw, compare_dtype, compare_per_block, logger):
    """Run HF visual forward to collect reference output for comparison.

    Returns:
        (hf_embeds_ref, hf_block_outs_ref) — tensors on CPU in compare_dtype.
    """
    logger.info("Running HF visual forward (before wrapping) to collect reference output...")
    hf_visual = native_model.model.visual
    hf_device = next(hf_visual.parameters()).device
    hf_vis_dtype = next(hf_visual.parameters()).dtype
    hf_visual.eval()

    # Force eager attention for fair comparison with wrap model (which uses explicit matmul attention)
    if hasattr(hf_visual, "config") and hasattr(hf_visual.config, "_attn_implementation"):
        old_attn_impl = hf_visual.config._attn_implementation
        hf_visual.config._attn_implementation = "eager"
        logger.info(f"HF visual attention: {old_attn_impl} -> eager (forced for comparison)")

    pv = pixel_values.to(device=hf_device, dtype=compare_dtype)
    gt = image_grid_thw.to(device=hf_device)
    hf_visual.to(compare_dtype)

    hf_block_outs_ref = None
    if compare_per_block:
        with torch.no_grad():
            hf_out, hf_block_outs_ref = _collect_block_outputs(
                hf_visual, lambda: hf_visual(pv, grid_thw=gt), block_attr="blocks",
            )
    else:
        with torch.no_grad():
            hf_out = hf_visual(pv, grid_thw=gt)

    hf_embeds_ref = hf_out.pooler_output
    if isinstance(hf_embeds_ref, (list, tuple)):
        hf_embeds_ref = torch.cat(hf_embeds_ref, dim=0)
    hf_embeds_ref = hf_embeds_ref.detach().cpu().to(compare_dtype)

    del hf_out, pv, gt
    # Restore HF visual to original dtype for subsequent operations
    hf_visual.to(hf_vis_dtype)
    logger.info("HF visual reference output collected.")
    return hf_embeds_ref, hf_block_outs_ref


def _run_wrap_visual_forward(wraped_model, hm_pixel_values, compare_dtype, compare_per_block, logger):
    """Run Wrap visual forward for comparison.

    Returns:
        (wrap_embeds, wrap_block_outs) — tensors on CPU in compare_dtype.
    """
    logger.info("Running Wrap visual forward for comparison...")
    wrap_device = next(wraped_model.parameters()).device
    wraped_model.eval()
    wraped_model.to(compare_dtype)
    hm_pv = hm_pixel_values.to(device=wrap_device, dtype=compare_dtype)

    wrap_block_outs = None
    if compare_per_block:
        with torch.no_grad():
            wrap_out, wrap_block_outs = _collect_block_outputs(
                wraped_model, lambda: wraped_model(hm_pv), block_attr="blocks",
            )
    else:
        with torch.no_grad():
            wrap_out = wraped_model(hm_pv)

    if isinstance(wrap_out, tuple):
        wrap_embeds = wrap_out[0]
    else:
        wrap_embeds = wrap_out
    wrap_embeds = wrap_embeds.detach().cpu().to(compare_dtype)

    # Restore wrap model to fp16 for subsequent export operations
    wraped_model.to(torch.float16)
    return wrap_embeds, wrap_block_outs


def _compare_hf_wrap(hf_embeds_ref, hf_block_outs_ref, wrap_embeds, wrap_block_outs, compare_dtype, logger):
    """Compare HF vs Wrap visual model outputs and log metrics."""
    # Align shapes for comparison
    if hf_embeds_ref.dim() == 2 and wrap_embeds.dim() == 3:
        wrap_embeds_flat = wrap_embeds.reshape(-1, wrap_embeds.shape[-1])
    elif hf_embeds_ref.dim() == 3 and wrap_embeds.dim() == 3:
        wrap_embeds_flat = wrap_embeds
    else:
        wrap_embeds_flat = wrap_embeds.reshape(hf_embeds_ref.shape)

    min_len = min(hf_embeds_ref.shape[0], wrap_embeds_flat.shape[0])
    hf_cmp = hf_embeds_ref[:min_len]
    wrap_cmp = wrap_embeds_flat[:min_len]

    logger.info("=" * 70)
    logger.info(f"HF vs Wrap Vision Model Precision Comparison (dtype={compare_dtype})")
    logger.info("=" * 70)

    metrics = _compute_precision_metrics(hf_cmp, wrap_cmp, label="image_embeds (merged)")
    _log_precision_metrics(logger, metrics)

    # Per-block comparison
    if hf_block_outs_ref and wrap_block_outs:
        logger.info("-" * 70)
        logger.info("Per-block HF vs Wrap hidden_states comparison:")
        logger.info(f"{'Block':<12} {'MaxAbsDiff':>14} {'MeanAbsDiff':>14} {'CosSim':>14}")
        logger.info("-" * 56)

        first_bad_block = None
        for block_name in sorted(hf_block_outs_ref.keys()):
            if block_name not in wrap_block_outs:
                logger.warning(f"  {block_name}: missing in wrap outputs")
                continue

            hf_h = hf_block_outs_ref[block_name].to(compare_dtype)
            wrap_h = wrap_block_outs[block_name].to(compare_dtype)

            if hf_h.dim() != wrap_h.dim():
                if hf_h.dim() == 2 and wrap_h.dim() == 3:
                    wrap_h = wrap_h.reshape(-1, wrap_h.shape[-1])
                elif hf_h.dim() == 3 and wrap_h.dim() == 2:
                    hf_h = hf_h.reshape(-1, hf_h.shape[-1])

            min_seq = min(hf_h.shape[0], wrap_h.shape[0])
            hf_h = hf_h[:min_seq]
            wrap_h = wrap_h[:min_seq]

            blk_diff = (hf_h - wrap_h).abs().max().item()
            blk_mean = (hf_h - wrap_h).abs().mean().item()
            blk_cos = F.cosine_similarity(
                hf_h.flatten().unsqueeze(0), wrap_h.flatten().unsqueeze(0),
            ).item()

            status = "" if blk_diff < 1e-3 else " *** DIVERGED"
            logger.info(f"  {block_name:<12} {blk_diff:>14.6e} {blk_mean:>14.6e} {blk_cos:>14.10f}{status}")

            if blk_diff >= 1e-3 and first_bad_block is None:
                first_bad_block = block_name

        if first_bad_block is not None:
            logger.warning(f"First block with diff >= 1e-3: {first_bad_block}")
        else:
            logger.info("All blocks match within 1e-3 tolerance.")

    logger.info("=" * 70)


# ────────────────── Validation Wrap Models (ONNX / HMONNX) ───────────


class ONNXWrapModel(nn.Module):
    """Wraps the ONNX model for plugging back into the HF model for validation."""

    def __init__(self, model_path, spatial_merge_size, patch_size, device):
        super().__init__()
        self.session = ort.InferenceSession(model_path)
        self.spatial_merge_size = spatial_merge_size
        self.patch_size = patch_size
        self.spatial_merge_unit = self.spatial_merge_size ** 2
        self.device = device

    @property
    def dtype(self):
        return torch.float16

    def forward(self, pixel_values, grid_thw=None, **kwargs):
        from transformers.modeling_outputs import BaseModelOutputWithPooling

        dtype = pixel_values.dtype
        device = pixel_values.device
        pixel_values = pixel_values.float().cpu().numpy()
        out = self.session.run(None, {"pixel_values": pixel_values})
        image_embeds = torch.from_numpy(out[0]).to(dtype=dtype, device=device)
        return BaseModelOutputWithPooling(last_hidden_state=None, pooler_output=image_embeds)


class HMONNXWrapModel(nn.Module):
    """Wraps the HMONNX model for plugging back into the HF model for validation."""

    def __init__(self, model, spatial_merge_size, patch_size, device):
        super().__init__()
        self._model = model
        self.spatial_merge_size = spatial_merge_size
        self.patch_size = patch_size
        self.spatial_merge_unit = self.spatial_merge_size ** 2
        self.device = device

    @property
    def dtype(self):
        return torch.float16

    @torch.no_grad()
    def forward(self, pixel_values, grid_thw=None, **kwargs):
        from transformers.modeling_outputs import BaseModelOutputWithPooling

        (out,) = self._model(pixel_values.half())
        return BaseModelOutputWithPooling(last_hidden_state=None, pooler_output=out)


# ─────────────────────── Generate Helper ─────────────────────────────


def _filter_generate_kwargs(inputs):
    """Filter out keys not accepted by model.generate() (e.g. hm_pixel_values)."""
    GENERATE_BLACKLIST = {"hm_pixel_values"}
    return {k: v for k, v in inputs.items() if k not in GENERATE_BLACKLIST}


def _run_generate(native_model, inputs, processor, max_new_tokens, label, logger, **gen_kwargs):
    """Run model.generate() and log the decoded output text.

    Args:
        native_model: The HF model.
        inputs: Processor outputs (will be filtered for generate-incompatible keys).
        processor: The VL processor for decoding.
        max_new_tokens: Maximum generated tokens.
        label: Label for logging (e.g. "native model", "onnx model").
        logger: Logger instance.
        **gen_kwargs: Extra kwargs passed to generate().
    """
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


# ─────────────────────── Export Functions ─────────────────────────────


def _export_onnx(wraped_model, hm_pixel_values, out_onnx_file, logger):
    """Export the wrapped visual model to ONNX.

    Returns:
        The loaded onnx.ModelProto.
    """
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
    """Convert ONNX model to HMONNX."""
    if out_hmonnx_file.exists():
        logger.info(f"HMONNX model already exists: {out_hmonnx_file}")
        return

    from xhquant.api import DeviceType, convert_onnx_to_hmonnx

    out_hmonnx_file.parent.mkdir(exist_ok=True, parents=True)
    input_args = [hm_pixel_values.float().cpu()]
    convert_onnx_to_hmonnx(
        out_onnx_file,
        input_args,
        DeviceType.XH2a,
        str(out_hmonnx_file),
        quant_config,
    )
    logger.info(f"Convert ONNX to HMONNX success: {out_hmonnx_file}")


def _run_hmonnx_golden(out_hmonnx_file, hm_pixel_values, execution_device, golden_dir, logger):
    """Run HMONNX golden inference and save golden data."""
    from xhquant.api import HMONNXGoldenInference

    golden_dir.mkdir(exist_ok=True, parents=True)
    hm_model = HMONNXGoldenInference(out_hmonnx_file)
    hm_model.save_golden = True
    hm_model.exec_device = execution_device
    hm_model.golden_dir = golden_dir
    hm_model.forward(hm_pixel_values.half())
    logger.info(f"HMONNX golden saved to: {golden_dir}")


def _validate_onnx(native_model, inputs, processor, out_onnx_file, execution_device, args, logger):
    """Validate the exported ONNX model by plugging it into the full VL model."""
    visual_ref = native_model.model.visual
    onnx_infer_model = ONNXWrapModel(
        out_onnx_file,
        spatial_merge_size=getattr(visual_ref, "spatial_merge_size", 2),
        patch_size=getattr(visual_ref, "patch_size", args.patch_size),
        device=execution_device,
    )
    native_model.model.visual = onnx_infer_model
    inputs["pixel_values"] = inputs["hm_pixel_values"][0].half()
    _run_generate(native_model, inputs, processor, args.max_new_tokens, "onnx model", logger, repetition_penalty=1.05)


def _validate_hmonnx(native_model, inputs, processor, out_hmonnx_file, execution_device, args, logger):
    """Validate the HMONNX model by plugging it into the full VL model."""
    visual_ref = native_model.model.visual
    spatial_merge_size = getattr(visual_ref, "spatial_merge_size", 2)
    patch_size_val = getattr(visual_ref, "patch_size", args.patch_size)

    hm_session = HMONNXInference(out_hmonnx_file)
    hm_session.to("cpu")
    hm_session.exec_device = execution_device
    xh_model = HMONNXWrapModel(
        hm_session,
        spatial_merge_size=spatial_merge_size,
        patch_size=patch_size_val,
        device=execution_device,
    )

    native_model.model.visual = xh_model
    _run_generate(native_model, inputs, processor, args.max_new_tokens, "hmonnx model", logger)


# ──────────────────────── Prepare Inputs ─────────────────────────────


def _prepare_inputs(hf_model_dir, args):
    """Prepare processor and inputs from image/prompt.

    Returns:
        (processor, inputs)
    """
    processor = Qwen3_5Processor.from_pretrained(hf_model_dir)

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


# ──────────────────── Device Dispatch Utilities ──────────────────────


def _de_dispatch_to_cpu(model: nn.Module):
    """Remove accelerate dispatch hooks and move model to CPU."""
    _remove_offload(model)
    model.to("cpu")
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _resolve_no_split_module_classes(model: nn.Module, logger) -> list[str]:
    """Resolve no_split_module_classes from model metadata.

    Keep this logic in export script to avoid changing shared auto_offload utils.
    """
    resolved: list[str] = []

    def _collect(item):
        if item is None:
            return
        if isinstance(item, str):
            resolved.append(item)
            return
        if isinstance(item, type):
            resolved.append(item.__name__)
            return
        if isinstance(item, (list, tuple, set)):
            for sub_item in item:
                _collect(sub_item)
            return
        resolved.append(str(item))

    _collect(getattr(model, "_no_split_modules", None))
    resolved = [name for name in dict.fromkeys(resolved) if name]
    if not resolved:
        resolved = list(DEFAULT_NO_SPLIT_MODULES)
        logger.warning(f"Model._no_split_modules is empty, fallback to: {resolved}")
    else:
        logger.info(f"Use no_split_module_classes for auto_offload: {resolved}")
    return resolved


# ──────────────────────── Main Export Logic ───────────────────────────


def _export_impl(cfg, args):
    logger = get_root_logger()
    config_file = cfg.config_file
    cfg_name = cfg.cfg_name

    execution_device = torch.device(cfg.execution_device)
    device = torch.device(cfg.device)
    compare_dtype = getattr(torch, args.compare_dtype)

    # ── Build model ──
    cfg.model.wrap_cfg.max_size_w = args.max_size_w
    cfg.model.wrap_cfg.max_size_h = args.max_size_h
    cfg.model.wrap_cfg.max_size_t = args.max_size_t
    cfg.model.wrap_cfg.temporal_patch_size = args.temporal_patch_size
    cfg.model.wrap_cfg.patch_size = args.patch_size
    cfg.model.hf_model = args.hf_model_dir
    cfg.model.is_gptqmodel = args.use_gptq_model
    cfg.hf_model_dir = args.hf_model_dir

    qwen3_5_vision_model: XHQwen3_5VisionModel = MODELS.build(cfg.model)
    native_model = qwen3_5_vision_model.get_hf_model(device_map="auto")
    no_split_modules = _resolve_no_split_module_classes(native_model, logger)

    # ── Optional weight transforms (QuaRot / GPTQ) ──
    # Weight transforms require direct CPU parameter access; de-dispatch first.
    needs_weight_transforms = args.quarot_weight_path is not None or args.use_gptq_model
    if needs_weight_transforms:
        logger.info("De-dispatching model to CPU for weight transforms...")
        _de_dispatch_to_cpu(native_model)

    if args.quarot_weight_path is not None:
        if native_model.config.tie_word_embeddings:
            old_torchscript = native_model.config.torchscript
            native_model.config.torchscript = True
            native_model.tie_weights()
            native_model.config.tie_word_embeddings = False
            native_model.config.torchscript = old_torchscript
            native_model.config.text_config.tie_word_embeddings = False

        from xh_model_zoo.xh_llm.quarot import rotation_utils
        rotation_utils.fuse_layer_norms(native_model)
        state_dict = load_safetensors_file(args.quarot_weight_path)
        native_model.load_state_dict(state_dict)
        logger.info(f"Load state_dict from {args.quarot_weight_path}")

    if args.use_gptq_model:
        if args.gptq_model_rotate:
            from xh_model_zoo.xh_llm.quarot import rotation_utils
            rotation_utils.fuse_layer_norms(native_model, llm_rotate=False)
            rotation_utils.rotate_model(native_model, "hadamard", device=device, llm_rotate=False)
        else:
            raise NotImplementedError("Only support gptq model with rotation")

    # ── Prepare processor & inputs ──
    processor, inputs = _prepare_inputs(qwen3_5_vision_model.hf_model_dir, args)

    # Re-dispatch model to GPU(s) if it was de-dispatched for weight transforms;
    # otherwise the model is already auto-dispatched from get_hf_model(device_map="auto").
    if needs_weight_transforms:
        logger.info("Re-dispatching model with auto_offload after weight transforms...")
        _auto_offload(
            native_model,
            no_split_module_classes=no_split_modules,
            target_dtype=torch.float16,
        )

    inputs.to(execution_device)

    assert inputs["image_grid_thw"][0][-1] == args.max_size_w // args.patch_size, (
        f"inputs['image_grid_thw'][0][-1] = {inputs['image_grid_thw'][0][-1]}, "
        f"args.max_size_w = {args.max_size_w}, args.patch_size = {args.patch_size}"
    )
    assert inputs["image_grid_thw"][0][-2] == args.max_size_h // args.patch_size, (
        f"inputs['image_grid_thw'][0][-2] = {inputs['image_grid_thw'][0][-2]}, "
        f"args.max_size_h = {args.max_size_h}, args.patch_size = {args.patch_size}"
    )

    pixel_values = inputs["pixel_values"]
    image_grid_thw = inputs["image_grid_thw"].to(torch.long)

    # ── Step 1: HF visual reference (before wrapping, for comparison) ──
    hf_embeds_ref = None
    hf_block_outs_ref = None
    if args.compare_hf_wrap:
        hf_embeds_ref, hf_block_outs_ref = _run_hf_visual_forward(
            native_model, pixel_values, image_grid_thw,
            compare_dtype, args.compare_per_block, logger,
        )

    # ── Step 2: Native model generate (before wrapping) ──
    if args.valid:
        _run_generate(
            native_model, inputs, processor, args.max_new_tokens,
            "native model", logger, repetition_penalty=1.05,
        )

    # ── Step 3: De-dispatch to CPU, then init wrap model (transforms visual in-place) ──
    logger.info("De-dispatching native model to CPU for wrapping...")
    _de_dispatch_to_cpu(native_model)

    qwen3_5_vision_model.init_wrap_model(native_model)
    wraped_model = qwen3_5_vision_model.wrap_model

    # ── Step 4: HF vs Wrap precision comparison ──
    if args.compare_hf_wrap and hf_embeds_ref is not None:
        hm_pv_for_cmp = inputs["hm_pixel_values"][0].clone()
        wrap_embeds, wrap_block_outs = _run_wrap_visual_forward(
            wraped_model, hm_pv_for_cmp, compare_dtype, args.compare_per_block, logger,
        )
        _compare_hf_wrap(
            hf_embeds_ref, hf_block_outs_ref, wrap_embeds, wrap_block_outs,
            compare_dtype, logger,
        )
        del hf_embeds_ref, hf_block_outs_ref, hm_pv_for_cmp, wrap_embeds, wrap_block_outs

    # ── Step 5: Prepare hm_pixel_values for export ──
    hm_pixel_values = inputs["hm_pixel_values"][0].type(wraped_model.dtype).to(wraped_model.device)
    if args.batch_size > 1:
        hm_pixel_values = hm_pixel_values.repeat(args.batch_size, 1, 1, 1, 1)
    if args.max_size_t != 2:
        logger.warning(
            f"max_size_t is not 2, you are exporting a video onnx model. "
            f"Here only image onnx model is supported; video input may cause inference error."
        )
        hm_pixel_values = hm_pixel_values.repeat(1, 1, args.max_size_t // 2, 1, 1)

    # ── Step 6: Export to ONNX ──
    out_onnx_file = str(Path(cfg.work_dir) / "onnx" / f"visual_{args.batch_size}.onnx")
    _export_onnx(wraped_model, hm_pixel_values, out_onnx_file, logger)

    # Re-dispatch native model for validation (visual will be replaced by ONNX/HMONNX wrappers)
    if args.valid:
        logger.info("Auto offloading native model for validation...")
        _auto_offload(
            native_model,
            no_split_module_classes=no_split_modules,
            target_dtype=torch.float16,
        )

    # ── Step 7: Validate ONNX (optional) ──
    if args.valid:
        _validate_onnx(native_model, inputs, processor, out_onnx_file, execution_device, args, logger)

    # ── Step 8: Convert to HMONNX ──
    out_hmonnx_file = Path(cfg.work_dir) / "vision" / f"{cfg_name}.onnx"
    _convert_to_hmonnx(out_onnx_file, out_hmonnx_file, hm_pixel_values, cfg.model.quant_config, logger)

    # ── Step 9: HMONNX golden inference ──
    golden_dir = Path(cfg.work_dir) / "vision" / "golden"
    _run_hmonnx_golden(out_hmonnx_file, hm_pixel_values, execution_device, golden_dir, logger)

    # ── Step 10: Validate HMONNX (optional) ──
    if args.valid:
        _validate_hmonnx(native_model, inputs, processor, out_hmonnx_file, execution_device, args, logger)


# ──────────────────────────── Entry Point ────────────────────────────


def main(args):
    cfg = Config.fromfile(args.config)
    cfg_name = Path(args.config).stem

    # Output layout: only --model_name (plus --output_root) affects directory names; subdirs are fixed strings.
    cfg.work_dir = str(Path(args.output_root) / f"{args.model_name}" / "vision_export")

    work_path = Path(cfg.work_dir)
    if args.clean_output and work_path.exists():
        for sub in ("onnx", "vision", "debug"):
            p = work_path / sub
            if p.exists():
                shutil.rmtree(p)

    log_file = Path(cfg.work_dir) / f"{cfg_name}_debug.log"
    Path(cfg.work_dir).mkdir(exist_ok=True, parents=True)
    cfg.device = "cuda:0"
    cfg.dtype = "float16"
    cfg.debug = args.debug
    cfg.execution_device = "cuda:0" if torch.cuda.is_available() else "cpu"

    debug_output_dir = Path(cfg.work_dir) / "debug"
    debug_output_dir.mkdir(exist_ok=True, parents=True)

    seed = cfg.get("seed", 1024)
    set_random_seed(seed)

    xhquant_llm_init(log_file, cfg.debug)
    logger = get_root_logger()

    logger.info(f"Config:\n{cfg.pretty_text}")
    config_file = Path(cfg.work_dir) / Path(args.config).name
    cfg.config_file = str(config_file)
    cfg.cfg_name = cfg_name
    cfg.dump(config_file)

    xhquant.utils.suppress_printing.disable_printing = True
    with TimeProfiler(f"{cfg_name} export", logger), MemoryTracker(0, "export", logger):
        _export_impl(cfg, args)


if __name__ == "__main__":
    parser = parse_arguments()
    args = parser.parse_args()
    main(args)
