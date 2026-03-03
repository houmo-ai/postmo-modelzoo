# Copyright (c) 2025 HOUMO AI
#
# File: quant_pipeline.py
# Description:
#   Post-Training Quantization Tool - Python script for quantizing
# GLM-OCR models using post-training quantization techniques.
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

import gc
import json
import shutil
import time
import os

import tempfile
import time
from pathlib import Path
from loguru import logger

import onnx
import onnxruntime as ort
import torch
import torch.nn as nn
from pathlib import Path
from typing import List, Tuple

import torch
import xhquant.utils.suppress_printing
from xhquant.api import (
    ConfigDict,
    PrecisionMode,
    QTensor,
    ptq_quantize,
    set_random_seed,
    HMONNXInference,
)

from PIL import Image, ImageOps
from common import decode_next_token, xhquant_llm_init, get_root_logger
from common import xhquant_llm_init, get_root_logger
from xh_model_zoo.xh_llm.models.builder import MODELS
from xh_model_zoo.xh_llm.models.glm_ocr import XHGlmOcrVisionModel, GlmOcrProcessor
from xh_model_zoo.xh_llm.models.glm_ocr.utils import build_inputs, build_messages
from xh_model_zoo.utils.memory_tracker import MemoryTracker
from xh_model_zoo.utils.time_profiler import TimeProfiler
from xh_model_zoo.xh_llm.models.builder import MODELS
from xh_model_zoo.xh_llm.models.base_llm_model import LLMBaseModel
from xh_model_zoo.xh_llm.models.eval_model_type import EvalModelType
from xh_model_zoo.xh_llm.models.glm_ocr import (
    GlmOcrHFCompatible,
    XHGlmOcrLLMModel,
)
from torch import Tensor
from transformers import AutoModelForImageTextToText
from transformers.modeling_outputs import BaseModelOutputWithPooling


def _copy_hf_configs(src_dir: Path, dst_dir: Path, logger):
    dst_dir.mkdir(parents=True, exist_ok=True)
    hf_config_file_candidates = [
        ["chat_template.json", "chat_template.jinja"],
        ["config.json"],
        ["generation_config.json"],
        ["preprocessor_config.json"],
        ["tokenizer_config.json"],
        ["vocab.json"],
        ["tokenizer.json"],
        ["merges.txt"],
        ["special_tokens_map.json"],
    ]
    for candidates in hf_config_file_candidates:
        copied = False
        for cfg_file in candidates:
            src_file = src_dir / cfg_file
            if src_file.exists():
                shutil.copyfile(src_file, dst_dir / cfg_file)
                copied = True
                break
        if not copied:
            logger.warning(
                f"Skip copying hf config files, missing {candidates} in {src_dir}"
            )


def _flatten_args(args):
    flat_args = []
    for arg in args:
        if isinstance(arg, (List, Tuple)):
            flat_args.extend(arg)
        else:
            flat_args.append(arg)
    return flat_args


def _decode_new_tokens(processor, input_ids, generated_ids):
    if isinstance(generated_ids, torch.Tensor):
        return processor.decode(
            generated_ids[0][input_ids.shape[1] :], skip_special_tokens=False
        )
    if isinstance(generated_ids, (list, tuple)) and len(generated_ids) > 0:
        return processor.decode(
            generated_ids[0][input_ids.shape[1] :], skip_special_tokens=False
        )
    return ""


def _load_and_process_image(image_path: str, target_w: int, target_h: int):
    image = Image.open(image_path).convert("RGB")
    orig_w, orig_h = image.size
    if (orig_w, orig_h) != (target_w, target_h):
        scale = min(target_w / orig_w, target_h / orig_h)
        new_w = int(orig_w * scale)
        new_h = int(orig_h * scale)
        image = image.resize((new_w, new_h), Image.Resampling.BICUBIC)
        pad_w = target_w - new_w
        pad_h = target_h - new_h
        image = ImageOps.expand(
            image, border=(0, 0, pad_w, pad_h), fill=(114, 114, 114)
        )
    return image


def _build_stop_token_ids(processor, eos_token_id):
    stop_token_ids = set()
    if eos_token_id is not None:
        if isinstance(eos_token_id, (list, tuple)):
            stop_token_ids.update(int(v) for v in eos_token_id)
        else:
            stop_token_ids.add(int(eos_token_id))

    tokenizer_eos = getattr(processor.tokenizer, "eos_token_id", None)
    tokenizer_pad = getattr(processor.tokenizer, "pad_token_id", None)
    if tokenizer_eos is not None:
        stop_token_ids.add(int(tokenizer_eos))
    if tokenizer_pad is not None:
        stop_token_ids.add(int(tokenizer_pad))

    for token in ["<|user|>", "<|assistant|>", "<|observation|>", "<eop>"]:
        token_id = processor.tokenizer.convert_tokens_to_ids(token)
        if token_id is not None and int(token_id) >= 0:
            stop_token_ids.add(int(token_id))

    return sorted(stop_token_ids)


def _align_inputs_device(inputs, device):
    aligned = {}
    for key, value in inputs.items():
        if isinstance(value, Tensor):
            aligned[key] = value.to(device)
        else:
            aligned[key] = value
    return aligned


def _fix_image_token_id_if_needed(
    model_config, input_ids: Tensor, image_grid_thw: Tensor, logger
):
    current_id = int(model_config.image_token_id)
    current_count = int((input_ids == current_id).sum().item())
    spatial_merge_size = int(model_config.vision_config.spatial_merge_size)
    expected_count = int(
        (image_grid_thw.prod(-1) // (spatial_merge_size**2)).sum().item()
    )

    if current_count == expected_count:
        return

    unique_ids, counts = torch.unique(input_ids, return_counts=True)
    matched = unique_ids[counts == expected_count]
    if matched.numel() == 1:
        detected_id = int(matched[0].item())
        logger.warning(
            f"image_token_id mismatch: configured={current_id}, observed_count={current_count}, "
            f"expected={expected_count}. Auto-fix to detected image_token_id={detected_id}."
        )
        model_config.image_token_id = detected_id


def _safe_generate_text(
    model,
    inputs,
    processor,
    input_ids,
    max_new_tokens,
    logger,
    stage: str,
    stop_token_ids=None,
):

    llm_model = getattr(model, "_llm_model", None)
    if isinstance(llm_model, XHGlmOcrLLMModel):
        _reset_kv_cache_buffers(llm_model)
        llm_model.rope_deltas = None
    elif isinstance(model, XHGlmOcrLLMModel):
        _reset_kv_cache_buffers(model)
        model.rope_deltas = None

    model_device = input_ids.device
    try:
        model_device = next(model.parameters()).device
    except Exception:
        pass
    aligned_inputs = _align_inputs_device(inputs, model_device)

    with torch.no_grad():
        generate_kwargs = dict(max_new_tokens=max_new_tokens)
        if stop_token_ids is not None and len(stop_token_ids) > 0:
            generate_kwargs["eos_token_id"] = stop_token_ids
            if getattr(processor.tokenizer, "pad_token_id", None) is not None:
                generate_kwargs["pad_token_id"] = int(processor.tokenizer.pad_token_id)
        generated_ids = model.generate(**aligned_inputs, **generate_kwargs)
    return _decode_new_tokens(processor, input_ids, generated_ids)


def _safe_test_next_token(model, tokenizer, data_batch, logger, stage: str):
    try:
        outs = model.test_step(data_batch)
        logits = outs.logits.detach()
        return decode_next_token(tokenizer, logits)
    except Exception as exc:
        logger.warning(f"{stage} test_step failed, skip token validation: {exc}")
        return None, None


def _reset_kv_cache_buffers(model: XHGlmOcrLLMModel):
    for cache_name in ("past_key_caches", "past_value_caches"):
        caches = getattr(model, cache_name, None)
        if caches is None:
            continue
        for cache in caches:
            if isinstance(cache, Tensor):
                cache.zero_()


def _move_graph_tensor_constants_(graph, device, dtype=None) -> int:
    target_device = torch.device(device)
    moved = 0
    for module in graph.modules():
        for name, value in list(getattr(module, "_buffers", {}).items()):
            if isinstance(value, Tensor):
                need_move = value.device != target_device
                need_cast = (
                    dtype is not None
                    and value.is_floating_point()
                    and value.dtype != dtype
                )
                if need_move or need_cast:
                    new_val = value.to(
                        device=target_device, dtype=dtype if need_cast else value.dtype
                    )
                    module._buffers[name] = new_val
                    moved += 1

        buffer_names = set(getattr(module, "_buffers", {}).keys())
        for name, value in list(module.__dict__.items()):
            if name in buffer_names:
                continue
            if isinstance(value, Tensor):
                need_move = value.device != target_device
                need_cast = (
                    dtype is not None
                    and value.is_floating_point()
                    and value.dtype != dtype
                )
                if need_move or need_cast:
                    new_val = value.to(
                        device=target_device, dtype=dtype if need_cast else value.dtype
                    )
                    setattr(module, name, new_val)
                    moved += 1
    return moved


def _prepare_ptq_calibration_batches(
    glm_ocr_llm_model: XHGlmOcrLLMModel, data_prefill: dict, logger
):
    _reset_kv_cache_buffers(glm_ocr_llm_model)
    glm_ocr_llm_model.rope_deltas = None

    prefill_calib = glm_ocr_llm_model.prepare_inputs_for_graph(data_prefill)
    prefill_calib = _flatten_args(prefill_calib)
    calib_batches = [prefill_calib]

    try:
        prompt_len = int(data_prefill["input_ids"].shape[-1])
        decode_data = {
            "input_ids": data_prefill["input_ids"][:, :1],
            "past_seq_length": prompt_len,
        }
        decode_calib = glm_ocr_llm_model.prepare_inputs_for_graph(decode_data)
        decode_calib = _flatten_args(decode_calib)
        calib_batches.append(decode_calib)
        logger.info(
            f"PTQ calibration batches prepared: {len(calib_batches)} (prefill + decode)"
        )
    except Exception as exc:
        logger.warning(
            f"Build decode calibration batch failed, fallback to prefill only: {exc}"
        )
        logger.info(
            f"PTQ calibration batches prepared: {len(calib_batches)} (prefill only)"
        )

    return calib_batches


def _prepare_image_embeds(native_model, inputs, execution_device, dtype):
    visual = native_model.model.visual
    visual.to(execution_device)
    visual.to(dtype)

    with torch.no_grad():
        image_embeds = visual(
            inputs["pixel_values"].to(execution_device).to(dtype),
            grid_thw=inputs["image_grid_thw"].to(execution_device),
            return_dict=True,
        ).pooler_output
        if isinstance(image_embeds, (list, tuple)):
            image_embeds = torch.cat(image_embeds, dim=0)
        image_embeds = image_embeds.to(execution_device).to(dtype)
    return image_embeds


def xhmodel_export_onnx(
    xh_model: LLMBaseModel,
    tokenizer,
    data_batch,
    onnx_output_dir: str,
    cfg_name,
    execution_device,
    dtype,
    logger,
    valid: bool = True,
):
    logger = get_root_logger()

    xh_model.to("cpu")
    torch.cuda.empty_cache()

    logger.info("Start exporting graph.............")
    with TimeProfiler("export graph"):
        xh_model.convert_to_export_graph(data_batch)
    logger.info("Finish exported graph.")

    torch.cuda.empty_cache()
    xh_model.change_eval_type(EvalModelType.EXPORTED)

    if valid:
        xh_model.to(execution_device)
        xh_model.to(dtype)
        xh_model.set_exec_device(execution_device)
        try:
            with torch.no_grad():
                outs = xh_model.test_step(data_batch)
                logits = outs.logits.detach()
                next_tokens, next_token_str = decode_next_token(tokenizer, logits)
            logger.info(f"Exported model next token: {next_tokens} {next_token_str}")
        except Exception as exc:
            logger.warning(f"exported test_step failed, skip token validation: {exc}")

        xh_model.to("cpu")
        torch.cuda.empty_cache()

    logger.info("*************** Start exporting onnx ***************")
    with TimeProfiler("export onnx"):
        onnx_file = xh_model.to_export_onnx(data_batch, onnx_output_dir, cfg_name)[0]
    return onnx_file


def _export_impl(cfg, args):
    logger = get_root_logger()
    is_valid_model = args.valid
    config_file = cfg.config_file
    cfg_name = cfg.cfg_name

    device = torch.device(cfg.device)
    execution_device = torch.device(cfg.execution_device)
    dtype = getattr(torch, cfg.dtype)

    meta_info = ConfigDict(
        dict(
            create_time=time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()),
            config=str(Path(config_file).relative_to(cfg.work_dir)),
        )
    )

    model_cfg = ConfigDict(cfg.model.copy())
    target_w = int(model_cfg.pop("image_size_w", 672))
    target_h = int(model_cfg.pop("image_size_h", 672))

    glm_ocr_llm_model: XHGlmOcrLLMModel = MODELS.build(model_cfg)
    native_model = glm_ocr_llm_model.get_hf_model(
        attn_implementation=args.attn_implementation
    )

    hf_model_dir = cfg.hf_model_dir
    meta_info.hf_model = hf_model_dir

    hf_config_dir = Path(cfg.work_dir) / "hf_config"
    _copy_hf_configs(Path(hf_model_dir), hf_config_dir, logger)
    meta_info.hf_config = str(hf_config_dir.relative_to(cfg.work_dir))

    token_embedding = native_model.model.get_input_embeddings()
    token_embedding_file = Path(cfg.work_dir) / "token_embedding.pt"
    torch.save(token_embedding, str(token_embedding_file))
    meta_info.token_embedding_file = str(token_embedding_file.relative_to(cfg.work_dir))

    processor = GlmOcrProcessor.from_pretrained(hf_model_dir)
    image = _load_and_process_image(
        args.image_path, target_w=target_w, target_h=target_h
    )
    messages = build_messages(image, args.prompt)
    inputs = build_inputs(processor, messages, device=execution_device)
    tokenizer = processor.tokenizer

    _fix_image_token_id_if_needed(
        native_model.config, inputs["input_ids"], inputs["image_grid_thw"], logger
    )
    cfg_eos_token_id = getattr(native_model.config, "eos_token_id", None)
    if cfg_eos_token_id is None:
        cfg_eos_token_id = getattr(native_model.generation_config, "eos_token_id", None)
    stop_token_ids = _build_stop_token_ids(processor, cfg_eos_token_id)

    native_model.to(execution_device)  # pyright: ignore[reportArgumentType]
    native_model.to(dtype)

    image_embeds = _prepare_image_embeds(native_model, inputs, execution_device, dtype)
    native_model.cpu()

    glm_ocr_llm_model.init_wrap_model(native_model)
    del native_model

    glm_ocr_llm_model.change_eval_type(EvalModelType.WRAPED)
    glm_ocr_llm_model.token_embedding = torch.load(
        token_embedding_file, weights_only=False
    )

    xh2a_hf_compatible_model = None
    native_model = glm_ocr_llm_model.get_hf_model(
        attn_implementation=args.attn_implementation
    )
    xh2a_hf_compatible_model = GlmOcrHFCompatible.to_hf_compatible(
        native_model, glm_ocr_llm_model
    )
    xh2a_hf_compatible_model.to(execution_device)
    xh2a_hf_compatible_model.to(dtype)

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        gc.collect()

    if (
        glm_ocr_llm_model.past_key_caches is not None
        and len(glm_ocr_llm_model.past_key_caches) > 0
    ):
        meta_info.use_cache = True
        meta_info.kv_cache_shape = glm_ocr_llm_model.past_key_caches[0].shape
        meta_info.num_hidden_layers = len(glm_ocr_llm_model.past_key_caches)

    data_prefill = {
        "input_ids": inputs["input_ids"],
        "attention_mask": inputs["attention_mask"],
        "image_embeds": image_embeds,
        "past_seq_length": 0,
        "image_grid_thw": inputs["image_grid_thw"],
    }
    profile_root = (
        Path(args.profile_dir)
        if args.profile_dir is not None
        else Path(cfg.work_dir) / "node_profile"
    )

    glm_ocr_llm_model.change_eval_type(EvalModelType.WRAPED)
    glm_ocr_llm_model.to(dtype)
    glm_ocr_llm_model.to(device)

    glm_ocr_llm_model.cpu()
    logger.info("************* convert to frontend graph *************")
    glm_ocr_llm_model.convert_to_fronted_graph(data_prefill, release_wraped_model=False)

    wraped_output_text = None
    if is_valid_model and xh2a_hf_compatible_model is not None:
        glm_ocr_llm_model.to(execution_device)
        glm_ocr_llm_model.to(dtype)
        wraped_output_text = _safe_generate_text(
            xh2a_hf_compatible_model,
            inputs,
            processor,
            inputs["input_ids"],
            args.max_new_tokens,
            logger,
            "wraped",
            stop_token_ids,
        )
        if wraped_output_text is not None:
            logger.info("***************** wraped model output *****************")
            logger.info(wraped_output_text)

    if is_valid_model and xh2a_hf_compatible_model is not None:
        glm_ocr_llm_model.change_eval_type(EvalModelType.FRONTEND)
        glm_ocr_llm_model.to(execution_device)
        glm_ocr_llm_model.to(dtype)
        moved = _move_graph_tensor_constants_(
            glm_ocr_llm_model.frontend_model, execution_device, dtype=dtype
        )
        if moved > 0:
            logger.info(
                f"Moved {moved} frontend graph tensor constants to {execution_device}."
            )
        frontend_output_text = _safe_generate_text(
            xh2a_hf_compatible_model,
            inputs,
            processor,
            inputs["input_ids"],
            args.max_new_tokens,
            logger,
            "frontend-traced",
            stop_token_ids,
        )
        if frontend_output_text is not None:
            logger.info(
                "***************** frontend traced model output *****************"
            )
            logger.info(frontend_output_text)

        glm_ocr_llm_model.change_eval_type(EvalModelType.WRAPED)

    if args.profile_nodes:
        from glm_ocr_profile_nodes import profile_traced_quanted_graph_nodes

        profile_traced_quanted_graph_nodes(
            glm_ocr_llm_model=glm_ocr_llm_model,
            data_prefill=data_prefill,
            profile_root=profile_root,
            execution_device=execution_device,
            dtype=dtype,
            target_device=cfg.target_device,
            logger=logger,
            decode_steps=args.profile_decode_steps,
            tokenizer=tokenizer,
        )

    logger.info("************* convert to quanted graph *************")
    glm_ocr_llm_model.convert_to_quant_graph(cfg.target_device)

    glm_ocr_llm_model.change_eval_type(EvalModelType.QUANTED_ALIGNED)
    glm_ocr_llm_model.to(execution_device)
    glm_ocr_llm_model.to(dtype)
    moved = _move_graph_tensor_constants_(
        glm_ocr_llm_model.quanted_model, execution_device, dtype=dtype
    )
    if moved > 0:
        logger.info(
            f"Moved {moved} quanted graph tensor constants to {execution_device} before PTQ."
        )

    logger.info("*************** Start PTQ Quantize ***************")
    calib_batches = _prepare_ptq_calibration_batches(
        glm_ocr_llm_model, data_prefill, logger
    )
    ptq_quantize(
        glm_ocr_llm_model.quanted_model,
        calib_batches,
        PrecisionMode.ALIGNED,
        [execution_device],
    )
    logger.info("*************** Finished PTQ Quantize ***************")

    moved = _move_graph_tensor_constants_(
        glm_ocr_llm_model.quanted_model, execution_device, dtype=dtype
    )
    if moved > 0:
        logger.info(
            f"Moved {moved} quanted graph tensor constants to {execution_device} after PTQ."
        )

    glm_ocr_llm_model.change_eval_type(EvalModelType.QUANTED_ALIGNED)
    glm_ocr_llm_model.to(execution_device)
    glm_ocr_llm_model.to(dtype)

    if is_valid_model and xh2a_hf_compatible_model is not None:
        quanted_aligned_output_text = _safe_generate_text(
            xh2a_hf_compatible_model,
            inputs,
            processor,
            inputs["input_ids"],
            args.max_new_tokens,
            logger,
            "quanted-aligned",
            stop_token_ids,
        )
        if quanted_aligned_output_text is not None:
            logger.info(
                "***************** quanted (aligned) model output *****************"
            )
            logger.info(quanted_aligned_output_text)

    glm_ocr_llm_model.change_eval_type(EvalModelType.QUANTED_ALIGNED)

    prefill_onnx_dir = Path(cfg.work_dir) / "prefill_onnx"
    decode_onnx_dir = Path(cfg.work_dir) / "decode_onnx"
    prefill_onnx_dir.mkdir(exist_ok=True, parents=True)
    decode_onnx_dir.mkdir(exist_ok=True, parents=True)

    logger.info("*************** Start exporting prefill model ***************")
    prefill_onnx_file = xhmodel_export_onnx(
        glm_ocr_llm_model,
        tokenizer,
        data_prefill,
        str(prefill_onnx_dir),
        f"{cfg_name}_prefill",
        execution_device,
        dtype,
        logger,
        is_valid_model,
    )
    glm_ocr_llm_model.release_exported_model()
    glm_ocr_llm_model.change_eval_type(EvalModelType.QUANTED_ALIGNED)
    meta_info.prefill_onnx_file = str(Path(prefill_onnx_file).relative_to(cfg.work_dir))
    logger.info(f"save prefill onnx model to {prefill_onnx_file}")
    logger.info("*************** Finished export prefill model ***************")

    glm_ocr_llm_model.change_eval_type(EvalModelType.QUANTED_ALIGNED)
    glm_ocr_llm_model.set_input_sequence_length(1)

    data_decode = {
        "input_ids": torch.randint(0, 1000, (1, 1)),
        "past_seq_length": int(data_prefill["input_ids"].shape[-1]),
    }

    if is_valid_model:
        glm_ocr_llm_model.to(execution_device)
        next_token_id, next_token_text = _safe_test_next_token(
            glm_ocr_llm_model, tokenizer, data_decode, logger, "decode-quanted"
        )
        if next_token_id is not None:
            logger.info(
                f"Decode Quanted Model next token: {next_token_id} {next_token_text}"
            )

    torch.cuda.empty_cache()
    logger.info("*************** Start exporting decode model ***************")
    with TimeProfiler("export decode onnx", logger):
        decode_onnx_file = xhmodel_export_onnx(
            glm_ocr_llm_model,
            tokenizer,
            data_decode,
            str(decode_onnx_dir),
            f"{cfg_name}_decode",
            execution_device,
            dtype,
            logger,
            is_valid_model,
        )
    glm_ocr_llm_model.release_exported_model()
    meta_info.decode_onnx_file = str(Path(decode_onnx_file).relative_to(cfg.work_dir))
    logger.info(f"save decode onnx model to {decode_onnx_file}")
    logger.info("*************** Finished export decode model ***************")

    torch.save(
        {
            "input_ids": inputs["input_ids"].cpu(),
            "attention_mask": inputs["attention_mask"].cpu(),
            "image_grid_thw": inputs["image_grid_thw"].cpu(),
            "image_embeds": image_embeds.cpu(),
            "decode_input_ids": data_decode["input_ids"].cpu(),
            "decode_past_seq_length": int(data_decode["past_seq_length"]),
        },
        Path(cfg.work_dir) / "export_samples.pt",
    )

    patch_size = int(processor.image_processor.patch_size)
    meta_info.image_size_w = int(inputs["image_grid_thw"][0, 2].item()) * patch_size
    meta_info.image_size_h = int(inputs["image_grid_thw"][0, 1].item()) * patch_size
    meta_info.model_type = "glm_ocr"

    meta_file = str(Path(cfg.work_dir) / "export_meta_info.json")
    json.dump(meta_info, open(meta_file, "w"), indent=4)
    logger.info(f"Save meta info to {meta_file}")


def export_llm(args):
    from types import SimpleNamespace

    work_dir = args.work_dir
    cfg_name = "glm_ocr_llm_xh2a_2k_export"

    # Build quant_config and model dict inline (no external config file)
    quant_config = dict(
        inputs=dict(
            inputs_embeds=dict(quantizer=dict(qspec=dict(fake_dtype="float16"))),
            past_seq_length=dict(quantizer=dict(qspec=dict(fake_dtype="int32"))),
            current_input_length=dict(quantizer=dict(qspec=dict(fake_dtype="int32"))),
            position_ids=dict(quantizer=dict(qspec=dict(fake_dtype="float16"))),
        ),
        w_schema=dict(bits=8, fp_mode="sefp"),
        act_schema=dict(bits=16, fp_mode="sefp"),
        nodes_cfg=dict(
            lm_head=dict(
                w_schema=dict(bits=8, fp_mode="sefp"),
                act_schema=dict(bits=16, fp_mode="sefp"),
            )
        ),
    )

    model_dict = dict(
        type="XHGlmOcrLLMModel",
        hf_model=args.hf_model_dir,
        wrap_cfg=dict(
            max_sequence_length=args.max_sequence_length,
            input_sequence_length=args.input_sequence_length,
            use_cache=True,
            num_logits_to_keep=1,
            kv_cache=dict(cache_axis=2),
        ),
        quant_config=quant_config,
        frontend_type="TorchFX",
        export_cfg=dict(
            input_names=[
                "inputs_embeds",
                "position_ids",
                "past_seq_length",
                "current_input_length",
            ],
            output_names=["logits"],
        ),
        image_size_w=args.image_size_w,
        image_size_h=args.image_size_h,
    )

    # Construct a SimpleNamespace to keep _export_impl interface stable
    cfg = SimpleNamespace(
        device="cuda:0" if torch.cuda.is_available() else "cpu",
        execution_device="cuda:0" if torch.cuda.is_available() else "cpu",
        dtype="float16",
        debug=args.debug,
        seed=args.seed,
        work_dir=work_dir,
        hf_model_dir=args.hf_model_dir,
        target_device=args.target_device,
        model=model_dict,
        config_file=str(Path(work_dir) / f"{cfg_name}_runtime.py"),
        cfg_name=cfg_name,
    )

    log_file = Path(work_dir) / f"{cfg_name}_debug.log"
    Path(work_dir).mkdir(exist_ok=True, parents=True)

    set_random_seed(args.seed)

    xhquant_llm_init(log_file, args.debug)
    logger = get_root_logger()

    xhquant.utils.suppress_printing.disable_printing = True

    logger.info(f"Args: {args}")
    with TimeProfiler(f"{cfg_name} export", logger), MemoryTracker(0, "export", logger):
        _export_impl(cfg, args)


def load_and_process_image(image_path: str, target_w: int, target_h: int):
    image = Image.open(image_path).convert("RGB")
    orig_w, orig_h = image.size
    if (orig_w, orig_h) != (target_w, target_h):
        scale = min(target_w / orig_w, target_h / orig_h)
        new_w = int(orig_w * scale)
        new_h = int(orig_h * scale)
        image = image.resize((new_w, new_h), Image.Resampling.BICUBIC)
        pad_w = target_w - new_w
        pad_h = target_h - new_h
        image = ImageOps.expand(
            image, border=(0, 0, pad_w, pad_h), fill=(114, 114, 114)
        )
    return image


def _export_vision_impl(
    cfg_name, work_dir, device, execution_device, dtype, model_cfg, quant_config, args
):
    logger = get_root_logger()
    is_valid_model = args.valid
    max_size_w = args.image_size_w
    max_size_h = args.image_size_h

    # -------------------------------------------------------------------------
    # 1. Build vision model through MODELS registry (BaseModel workflow)
    # -------------------------------------------------------------------------
    glm_ocr_vision_model: XHGlmOcrVisionModel = MODELS.build(model_cfg)
    native_model = glm_ocr_vision_model.get_hf_model()
    hf_model_dir = glm_ocr_vision_model.hf_model_dir

    # -------------------------------------------------------------------------
    # 2. Prepare sample input using processor
    # -------------------------------------------------------------------------
    processor = GlmOcrProcessor.from_pretrained(hf_model_dir)
    image = load_and_process_image(
        args.image_path, target_w=max_size_w, target_h=max_size_h
    )
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": args.prompt},
            ],
        }
    ]
    inputs = build_inputs(processor, messages, device=execution_device)

    pixel_values = inputs["pixel_values"]

    if args.batch_size > 1:
        pixel_values = pixel_values.repeat(args.batch_size, 1)

    # -------------------------------------------------------------------------
    # 3. Validate native model BEFORE wrapping
    # -------------------------------------------------------------------------
    if is_valid_model:
        import accelerate

        accelerate.hooks.remove_hook_from_module(native_model, recurse=True)
        native_model.to(execution_device)
        with torch.no_grad():
            generated_ids = native_model.generate(
                **inputs, max_new_tokens=args.max_new_tokens
            )
        native_output = processor.decode(
            generated_ids[0][inputs["input_ids"].shape[1] :], skip_special_tokens=False
        )
        logger.info("***************** native model output *****************")
        logger.info(native_output)
        native_model.cpu()

    # -------------------------------------------------------------------------
    # 3.5. Wrap the vision model
    # -------------------------------------------------------------------------
    glm_ocr_vision_model.init_wrap_model(native_model)
    wraped_model = glm_ocr_vision_model.wrap_model

    # -------------------------------------------------------------------------
    # 4. Export ONNX via torch.onnx.export on wrapped model
    # -------------------------------------------------------------------------
    out_onnx_file = str(Path(work_dir) / "onnx" / f"visual_{args.batch_size}.onnx")
    Path(out_onnx_file).parent.mkdir(exist_ok=True, parents=True)

    if Path(out_onnx_file).exists():
        logger.info(f"onnx model already exists: {out_onnx_file}")
        onnx_model = onnx.load(out_onnx_file, load_external_data=True)
    else:
        wraped_model.float().eval().cpu()

        with tempfile.TemporaryDirectory() as tmp_dir:
            onnx_file = str(Path(tmp_dir) / "visual.onnx")
            logger.info(f"exporting onnx model to {onnx_file}")
            torch.onnx.export(
                wraped_model,
                (pixel_values.float().cpu(),),
                onnx_file,
                export_params=True,
                opset_version=18,
                do_constant_folding=True,
                input_names=["pixel_values"],
                output_names=["image_embeds"],
                verbose=True,
            )
            onnx_model = onnx.load(onnx_file, load_external_data=True)

        import onnx_graphsurgeon as gs

        ir_version = onnx_model.ir_version
        graph = gs.import_onnx(onnx_model)
        graph.toposort()
        graph.fold_constants()
        graph.cleanup()
        onnx_model = gs.export_onnx(graph)
        onnx_model.ir_version = ir_version

        onnx.save(
            onnx_model,
            out_onnx_file,
            save_as_external_data=True,
            all_tensors_to_one_file=True,
            location="visual_external_data",
            convert_attribute=True,
        )

    # -------------------------------------------------------------------------
    # 5. Convert ONNX to HMONNX
    # -------------------------------------------------------------------------
    out_hmonnx_file = Path(work_dir) / "vision" / f"{cfg_name}.onnx"
    out_hmonnx_file.parent.mkdir(exist_ok=True, parents=True)

    if not out_hmonnx_file.exists():
        from xhquant.api import DeviceType, convert_onnx_to_hmonnx

        input_args = [pixel_values.float().cpu()]
        convert_onnx_to_hmonnx(
            out_onnx_file,
            input_args,
            DeviceType.XH2a,
            str(out_hmonnx_file),
            quant_config,
        )
        logger.info(
            f"Convert onnx to hmonnx success, out hmonnx file to: {out_hmonnx_file}"
        )

    # -------------------------------------------------------------------------
    # 6. Validate HMONNX model (optional)
    # -------------------------------------------------------------------------
    if not is_valid_model:
        return

    class HMONNXWrapModel(nn.Module):
        def __init__(self, model, device, spatial_merge_size=2):
            super().__init__()
            self._model = model
            self.device = device
            self.spatial_merge_size = spatial_merge_size

        @property
        def dtype(self):
            return torch.float16

        @torch.no_grad()
        def forward(self, pixel_values, grid_thw=None, return_dict=True, **kwargs):
            out = self._model(pixel_values.half())
            if return_dict:
                return BaseModelOutputWithPooling(
                    last_hidden_state=out, pooler_output=out
                )
            return (out,)

    hm_session = HMONNXInference(out_hmonnx_file)
    hm_session.to("cpu")
    hm_session.exec_device = execution_device
    xh_model = HMONNXWrapModel(hm_session, device=execution_device)
    native_model.to(execution_device)
    native_model.model.visual = xh_model

    with torch.no_grad():
        generated_ids = native_model.generate(
            **inputs, max_new_tokens=args.max_new_tokens
        )
    hmonnx_output = processor.decode(
        generated_ids[0][inputs["input_ids"].shape[1] :], skip_special_tokens=False
    )
    logger.info("***************** hmonnx model output *****************")
    logger.info(hmonnx_output)


def export_vision(args):
    work_dir = args.work_dir
    cfg_name = "glm_ocr_vision_xh2a_export_hmonnx"

    log_file = Path(work_dir) / f"{cfg_name}_debug.log"
    Path(work_dir).mkdir(exist_ok=True, parents=True)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    execution_device = device
    dtype = getattr(torch, "float16")

    set_random_seed(args.seed)

    xhquant_llm_init(log_file, args.debug)
    logger = get_root_logger()

    # Build quant_config and model dict inline (no external config file)
    quant_config = dict(
        inputs=dict(
            pixel_values=dict(
                quantizer=dict(
                    qspec=dict(fake_dtype="float16"),
                )
            ),
        ),
        w_schema=dict(bits=8, fp_mode="sefp"),
        act_schema=dict(bits=16, fp_mode="sefp"),
    )

    model_cfg = ConfigDict(
        dict(
            type="XHGlmOcrVisionModel",
            hf_model=args.hf_model_dir,
            wrap_cfg=dict(
                max_size_w=args.image_size_w,
                max_size_h=args.image_size_h,
                max_size_t=args.max_size_t,
                patch_size=args.patch_size,
                temporal_patch_size=args.temporal_patch_size,
            ),
            quant_config=quant_config,
            export_cfg=dict(
                input_names=["pixel_values"],
                output_names=["image_embeds"],
            ),
        )
    )

    logger.info(f"Args: {args}")

    xhquant.utils.suppress_printing.disable_printing = True
    with TimeProfiler(f"{cfg_name} export", logger), MemoryTracker(0, "export", logger):
        _export_vision_impl(
            cfg_name,
            work_dir,
            device,
            execution_device,
            dtype,
            model_cfg,
            quant_config,
            args,
        )


def move_and_rename_folder(src_folder: str, dest_parent_folder: str, new_name: str):
    """
    Moves a folder to a new parent directory and renames it.

    Args:
        src_folder (str): The path to the folder to be moved.
        dest_parent_folder (str): The path to the destination parent directory.
        new_name (str): The new name for the folder.

    Example:
        move_and_rename_folder('/path/to/old_folder', '/path/to/new_parent', 'new_folder_name')
        # Result: /path/to/new_parent/new_folder_name
    """
    if not os.path.exists(src_folder):
        print(f"Error: Source folder '{src_folder}' does not exist.")
        return

    # Ensure the destination parent folder exists
    if not os.path.exists(dest_parent_folder):
        try:
            os.makedirs(dest_parent_folder)
            print(f"Created destination parent directory: {dest_parent_folder}")
        except OSError as e:
            print(f"Error creating destination directory: {e}")
            return

    # Construct the full destination path
    dest_path = os.path.join(dest_parent_folder, new_name)

    if os.path.exists(dest_path):
        print(f"Error: Destination '{dest_path}' already exists.")
        return

    try:
        shutil.move(src_folder, dest_path)
        print(
            f"Successfully moved and renamed:\n  Source: {src_folder}\n  Destination: {dest_path}"
        )
    except Exception as e:
        print(f"Error moving folder: {e}")


def rename_single_onnx_file(directory: str, new_filename: str):
    """
    Renames the single .onnx file in a directory to a new name.

    Args:
        directory (str): The directory to search in.
        new_filename (str): The new filename (should include .onnx extension).
    """
    if not os.path.isdir(directory):
        print(f"Error: Directory '{directory}' does not exist.")
        return

    # List all files ending with .onnx
    files = [
        f
        for f in os.listdir(directory)
        if f.endswith(".onnx") and os.path.isfile(os.path.join(directory, f))
    ]

    if len(files) == 0:
        print(f"Error: No .onnx files found in '{directory}'.")
    elif len(files) > 1:
        print(
            f"Error: Multiple .onnx files found in '{directory}': {files}. Please ensure only one exists."
        )
    else:
        old_path = os.path.join(directory, files[0])
        new_path = os.path.join(directory, new_filename)

        try:
            os.rename(old_path, new_path)
            # print(f"Successfully renamed '{files[0]}' to '{new_filename}' in '{directory}'")
        except Exception as e:
            print(f"Error renaming file: {e}")


def remove_folder(folder_path: str):
    """
    Removes a folder and all its contents.

    Args:
        folder_path (str): The path to the folder to be removed.
    """
    if not os.path.exists(folder_path):
        print(f"Error: Folder '{folder_path}' does not exist.")
        return

    if not os.path.isdir(folder_path):
        print(f"Error: '{folder_path}' is not a directory.")
        return

    try:
        shutil.rmtree(folder_path)
        print(f"Successfully removed folder: {folder_path}")
    except Exception as e:
        print(f"Error removing folder: {e}")


def move_llm(args):
    work_dir = args.work_dir
    dest_parent_folder = args.output_dir
    src_prefill_folder = os.path.join(work_dir, "prefill_onnx")
    src_decode_folder = os.path.join(work_dir, "decode_onnx")
    src_visual_folder = os.path.join(work_dir, "vision")
    src_embed_folder = os.path.join(work_dir, "token_embedding.pt")

    move_and_rename_folder(src_embed_folder, dest_parent_folder, "quant_embedding.pt")
    logger.info(
        f"Moved and renamed token embedding file to {dest_parent_folder}/quant_embedding.pt"
    )
    move_and_rename_folder(src_prefill_folder, dest_parent_folder, "prefill")
    rename_single_onnx_file(
        os.path.join(dest_parent_folder, "prefill"), "hmquant_glm-ocr_with_act.onnx"
    )
    logger.info(
        f"Moved and renamed prefill onnx model to {dest_parent_folder}/prefill/hmquant_glm-ocr_with_act.onnx"
    )
    move_and_rename_folder(src_decode_folder, dest_parent_folder, "decoder")
    rename_single_onnx_file(
        os.path.join(dest_parent_folder, "decoder"), "hmquant_glm-ocr_with_act.onnx"
    )
    logger.info(
        f"Moved and renamed decode onnx model to {dest_parent_folder}/decoder/hmquant_glm-ocr_with_act.onnx"
    )
    move_and_rename_folder(src_visual_folder, dest_parent_folder, "visual")
    rename_single_onnx_file(
        os.path.join(dest_parent_folder, "visual"), "hmquant_glm-ocr_with_act.onnx"
    )
    logger.info(
        f"Moved and renamed visual onnx model to {dest_parent_folder}/visual/hmquant_glm-ocr_with_act.onnx"
    )
    remove_folder(work_dir)
    logger.info(f"Removed temporary work directory: {work_dir}")
