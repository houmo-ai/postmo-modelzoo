#!/usr/bin/python3
# -*- coding: utf-8 -*-
# Copyright 2026 HOUMO AI
#
# File: _hmonnx_pipeline.py
# Description:
#   HMONNX pipeline helpers for the Qwen3-Omni model, including
#   monkey-patched inference and golden-data dumping.
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
import time
import types
from pathlib import Path
from enum import Enum
from typing import Any, Dict, Optional

import numpy as np
import soundfile as sf
import torch
import torch.nn as nn
from transformers import AutoConfig, AutoTokenizer, Qwen3OmniMoeForConditionalGeneration

from xhquant.api import CacheTensor
from xhquant.xhonnxruntime.hmonnx_inference import HMONNXInference

from xh_model_zoo.xh_llm.models.qwen3_omni.modeling_qwen3_omni_moe import _get_feat_extract_output_lengths
from xh_model_zoo.xh_llm.models.qwen3_omni.modeling_qwen3_omni_moe import (
    Qwen3OmniMoeTalkerCodePredictorOutputWithPast,
    Qwen3OmniMoeTalkerOutputWithPast,
)
from xh_model_zoo.xh_llm.models.qwen3_omni.monkey_patch import Qwen3OmniMoeThinkerForConditionalGeneration_forward
from xh_model_zoo.xh_llm.models.qwen3_omni.processing_qwen3_omni_moe import Qwen3OmniMoeProcessor

# Monkey-patch HMONNXInference to auto-set exec_device=cuda (GPU execution
# without moving all weights to GPU, avoiding OOM).
_orig_hmonnx_init = HMONNXInference.__init__


def _hmonnx_init_cuda_exec(self, onnx_file: str) -> None:
    _orig_hmonnx_init(self, onnx_file)
    if torch.cuda.is_available():
        self.exec_device = torch.device("cuda")


HMONNXInference.__init__ = _hmonnx_init_cuda_exec

SCRIPT_DIR = Path(__file__).resolve().parent
_HMONNX_RUNTIME_FIX_CACHE: Dict[str, Path] = {}
_GIB = 1024**3

def _ensure_mistral_common_reasoning_effort():
    try:
        import mistral_common.protocol.instruct.request as request_module
    except ImportError:
        return

    if hasattr(request_module, "ReasoningEffort"):
        return

    class ReasoningEffort(str, Enum):
        none = "none"
        high = "high"

    request_module.ReasoningEffort = ReasoningEffort

def _force_eager_moe_implementation(module, logger=None):
    visited_configs = set()
    updated = 0

    def _visit_config(config):
        nonlocal updated
        if config is None:
            return
        config_id = id(config)
        if config_id in visited_configs:
            return
        visited_configs.add(config_id)

        if hasattr(config, "_experts_implementation") and config._experts_implementation != "eager":
            config._experts_implementation = "eager"
            updated += 1

        config_dict = getattr(config, "__dict__", None)
        if not isinstance(config_dict, dict):
            return
        for value in config_dict.values():
            if hasattr(value, "__dict__"):
                _visit_config(value)

    _visit_config(getattr(module, "config", None))
    for submodule in module.modules():
        _visit_config(getattr(submodule, "config", None))

    if logger is not None and updated:
        logger.info(f"forced {updated} config nodes to use eager MoE experts")

def release_export_cuda_memory(logger=None, label: Optional[str] = None):
    import gc

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
    if logger is not None:
        suffix = f" after {label}" if label else ""
        logger.info(f"released export resources and cleared CUDA cache{suffix}")


def _pick_best_validation_single_gpu(logger=None) -> Optional[str]:
    if not torch.cuda.is_available():
        return None

    best_gpu_idx = None
    best_free_bytes = -1
    debug_entries = []

    for gpu_idx in range(torch.cuda.device_count()):
        free_bytes, total_bytes = torch.cuda.mem_get_info(gpu_idx)
        min_required_bytes = max(40 * _GIB, int(total_bytes * 0.9))
        debug_entries.append(
            f"cuda:{gpu_idx}: free={_format_bytes_as_gib_str(free_bytes)}, "
            f"required={_format_bytes_as_gib_str(min_required_bytes)}"
        )
        if free_bytes < min_required_bytes:
            continue
        if free_bytes > best_free_bytes:
            best_gpu_idx = gpu_idx
            best_free_bytes = free_bytes

    if logger is not None:
        logger.info("validation single-gpu candidates: " + "; ".join(debug_entries))

    if best_gpu_idx is None:
        return None
    return f"cuda:{best_gpu_idx}"


def _resolve_validation_device_map(device_map: str, logger=None) -> str:
    if device_map != "auto":
        return device_map
    if not torch.cuda.is_available():
        resolved = "cpu"
    else:
        resolved = _pick_best_validation_single_gpu(logger) or "auto"
    if logger is not None:
        logger.info(f"validation device_map resolved from auto to {resolved}")
    return resolved


def _format_bytes_as_gib_str(num_bytes: int) -> str:
    gib = max(num_bytes, 0) / _GIB
    return f"{gib:.1f}GiB"


def _build_safe_validation_max_memory(logger=None) -> Optional[Dict[Any, str]]:
    if not torch.cuda.is_available():
        return None

    reserve_bytes = 8 * _GIB
    min_free_bytes = 12 * _GIB
    max_memory: Dict[Any, str] = {}
    debug_entries = []

    for gpu_idx in range(torch.cuda.device_count()):
        free_bytes, _ = torch.cuda.mem_get_info(gpu_idx)
        if free_bytes < min_free_bytes:
            debug_entries.append(f"cuda:{gpu_idx}: skipped free={_format_bytes_as_gib_str(free_bytes)}")
            continue

        planner_bytes = free_bytes - reserve_bytes
        if planner_bytes < min_free_bytes:
            planner_bytes = int(free_bytes * 0.9)
        if planner_bytes <= 0:
            debug_entries.append(f"cuda:{gpu_idx}: skipped after reserve free={_format_bytes_as_gib_str(free_bytes)}")
            continue

        max_memory[gpu_idx] = _format_bytes_as_gib_str(planner_bytes)
        debug_entries.append(
            f"cuda:{gpu_idx}: planner={max_memory[gpu_idx]}, free={_format_bytes_as_gib_str(free_bytes)}"
        )

    try:
        import psutil

        cpu_available = psutil.virtual_memory().available
        cpu_budget = max(cpu_available - 16 * _GIB, 32 * _GIB)
        max_memory["cpu"] = _format_bytes_as_gib_str(cpu_budget)
    except Exception:
        max_memory["cpu"] = "64.0GiB"

    if logger is not None:
        logger.info("validation max_memory: " + "; ".join(debug_entries + [f"cpu: planner={max_memory['cpu']}"]))

    return max_memory


def _build_inputs_embeds_input_ids(model_kwargs: Optional[Dict[str, Any]]) -> Optional[torch.Tensor]:
    if not model_kwargs:
        return None

    inputs_embeds = model_kwargs.get("inputs_embeds")
    if not isinstance(inputs_embeds, torch.Tensor):
        return None

    batch_size = 1
    for value in model_kwargs.values():
        if isinstance(value, torch.Tensor):
            batch_size = int(value.shape[0])
            break

    return torch.ones((batch_size, 0), dtype=torch.long, device=inputs_embeds.device)


def _patch_talker_predictor_sampling(native_model, logger=None):
    talker = getattr(native_model, "talker", None)
    if talker is None or getattr(talker, "_imz_predictor_sampling_patched", False):
        return

    original_generate = talker.generate
    talker._imz_talker_do_sample = False

    def generate(self, *args, **kwargs):
        previous = getattr(self, "_imz_talker_do_sample", False)
        self._imz_talker_do_sample = bool(kwargs.get("do_sample", False))
        predictor = self.code_predictor
        original_predictor_generate = predictor.generate

        def predictor_generate(predictor_self, *p_args, **p_kwargs):
            p_kwargs["do_sample"] = False
            p_kwargs.pop("top_k", None)
            p_kwargs.pop("top_p", None)
            return original_predictor_generate(*p_args, **p_kwargs)

        if not self._imz_talker_do_sample:
            predictor.generate = types.MethodType(predictor_generate, predictor)
        try:
            return original_generate(*args, **kwargs)
        finally:
            predictor.generate = original_predictor_generate
            self._imz_talker_do_sample = previous

    talker.generate = types.MethodType(generate, talker)
    talker._imz_predictor_sampling_patched = True
    if logger is not None:
        logger.info("patched talker predictor sampling to follow talker do_sample instead of forcing stochastic decode")


def _patch_inputs_embeds_generation_device(module, module_name: str, logger=None):
    if getattr(module, "_xh_inputs_embeds_generation_device_patched", False):
        return

    original = getattr(module, "_maybe_initialize_input_ids_for_generation", None)
    if not callable(original):
        return

    def patched(self, inputs=None, bos_token_id=None, model_kwargs=None):
        generated_input_ids = None
        if inputs is None:
            generated_input_ids = _build_inputs_embeds_input_ids(model_kwargs)
        if generated_input_ids is not None:
            return generated_input_ids
        return original(inputs, bos_token_id, model_kwargs)

    module._maybe_initialize_input_ids_for_generation = types.MethodType(patched, module)
    module._xh_inputs_embeds_generation_device_patched = True
    if logger is not None:
        logger.info(f"patched inputs_embeds generation device for {module_name}")


def _resolve_runtime_execution_device(module) -> torch.device:
    hook = getattr(module, "_hf_hook", None)
    execution_device = getattr(hook, "execution_device", None)
    if execution_device is not None:
        device = torch.device(execution_device)
        if device.type != "meta":
            return device

    for parameter in module.parameters():
        if parameter.device.type != "meta":
            return parameter.device

    for buffer in module.buffers():
        if buffer.device.type != "meta":
            return buffer.device

    hf_device_map = getattr(module, "hf_device_map", None)
    if isinstance(hf_device_map, dict):
        for mapped_device in hf_device_map.values():
            if mapped_device in (None, "disk"):
                continue
            try:
                candidate = torch.device(mapped_device)
            except (TypeError, RuntimeError, ValueError):
                continue
            if candidate.type != "meta":
                return candidate

    raise RuntimeError(f"Cannot resolve a concrete runtime device for {module.__class__.__name__}")


def _patch_runtime_device_property(module, module_name: str, logger=None):
    if getattr(module, "_xh_runtime_device_property_patched", False):
        return

    patched_cls = type(
        f"{module.__class__.__name__}XHRuntimeDevicePatched",
        (module.__class__,),
        {"device": property(lambda self: _resolve_runtime_execution_device(self))},
    )
    module.__class__ = patched_cls
    module._xh_runtime_device_property_patched = True
    if logger is not None:
        logger.info(f"patched runtime device property for {module_name}")


def _create_hmonnx_session(onnx_path: Path) -> HMONNXInference:
    try:
        return HMONNXInference(str(onnx_path))
    except KeyError as exc:
        if "indices_or_sections" not in str(exc):
            raise

    cache_key = str(onnx_path.resolve())
    fixed_path = _HMONNX_RUNTIME_FIX_CACHE.get(cache_key)
    if fixed_path is None or not fixed_path.exists():
        import onnx
        from onnx import helper
        from onnx import numpy_helper

        model = onnx.load(str(onnx_path))
        initializer_map = {tensor.name: numpy_helper.to_array(tensor) for tensor in model.graph.initializer}
        value_shape_map = {}
        for value in list(model.graph.input) + list(model.graph.value_info) + list(model.graph.output):
            tensor_type = value.type.tensor_type
            dims = []
            for dim in tensor_type.shape.dim:
                if dim.HasField("dim_value"):
                    dims.append(int(dim.dim_value))
                else:
                    dims.append(None)
            value_shape_map[value.name] = dims

        def _infer_split_sections(node) -> list[int]:
            has_indices = any(attr.name == "indices_or_sections" for attr in node.attribute)
            if has_indices:
                indices_attr = next(attr for attr in node.attribute if attr.name == "indices_or_sections")
                return list(indices_attr.ints)

            split_attr = next((attr for attr in node.attribute if attr.name == "split"), None)
            if split_attr is not None:
                split_sections = list(split_attr.ints)
                if not split_sections and split_attr.i != 0:
                    split_sections = [int(split_attr.i)]
                if split_sections:
                    return split_sections

            if len(node.input) > 1 and node.input[1] in initializer_map:
                return initializer_map[node.input[1]].reshape(-1).astype("int64").tolist()

            axis_attr = next((attr for attr in node.attribute if attr.name == "axis"), None)
            num_outputs_attr = next((attr for attr in node.attribute if attr.name == "num_outputs"), None)
            input_shape = value_shape_map.get(node.input[0], [])
            if axis_attr is None or num_outputs_attr is None or not input_shape:
                return []

            axis = int(axis_attr.i)
            if axis < 0:
                axis += len(input_shape)
            if not (0 <= axis < len(input_shape)):
                return []

            axis_dim = input_shape[axis]
            num_outputs = int(num_outputs_attr.i)
            if axis_dim is None or num_outputs <= 0 or axis_dim % num_outputs != 0:
                return []
            return [axis_dim // num_outputs] * num_outputs

        patched = False
        rewritten_nodes = []
        for node in model.graph.node:
            if node.op_type != "Split":
                rewritten_nodes.append(node)
                continue

            split_sections = _infer_split_sections(node)
            if not split_sections:
                raise KeyError(f"indices_or_sections missing and cannot infer split sections from {onnx_path}")

            axis_attr = next((attr for attr in node.attribute if attr.name == "axis"), None)
            axis = int(axis_attr.i) if axis_attr is not None else 0

            if len(node.output) > 1:
                current_start = 0
                for index, output_name in enumerate(node.output):
                    section = int(split_sections[index])
                    starts_name = f"{node.name}_starts_{index}"
                    ends_name = f"{node.name}_ends_{index}"
                    axes_name = f"{node.name}_axes_{index}"
                    steps_name = f"{node.name}_steps_{index}"
                    model.graph.initializer.extend(
                        [
                            numpy_helper.from_array(np.asarray([current_start], dtype=np.int64), starts_name),
                            numpy_helper.from_array(np.asarray([current_start + section], dtype=np.int64), ends_name),
                            numpy_helper.from_array(np.asarray([axis], dtype=np.int64), axes_name),
                            numpy_helper.from_array(np.asarray([1], dtype=np.int64), steps_name),
                        ]
                    )
                    rewritten_nodes.append(
                        helper.make_node(
                            "Slice",
                            inputs=[node.input[0], starts_name, ends_name, axes_name, steps_name],
                            outputs=[output_name],
                            name=f"{node.name}_slice_{index}",
                        )
                    )
                    current_start += section
                patched = True
                continue

            has_indices = any(attr.name == "indices_or_sections" for attr in node.attribute)
            if not has_indices:
                node.attribute.extend([helper.make_attribute("indices_or_sections", split_sections)])
                patched = True
            rewritten_nodes.append(node)

        model.graph.ClearField("node")
        model.graph.node.extend(rewritten_nodes)

        if not patched:
            raise KeyError(f"No runtime-compatible graph fixes were applied for {onnx_path}")

        fixed_path = onnx_path.with_name(f"{onnx_path.stem}.runtimefix.onnx")
        onnx.save(model, str(fixed_path))
        _HMONNX_RUNTIME_FIX_CACHE[cache_key] = fixed_path

    return HMONNXInference(str(fixed_path))

from qwen_omni_utils import process_mm_info

def build_conversation(case: str, text_prompt: Optional[str] = None):
    image_path = str(SCRIPT_DIR / "data" / "cars.jpg")
    audio_path = str(SCRIPT_DIR / "data" / "cough.wav")

    if case == "text":
        return [
            {"role": "user", "content": [{"type": "text", "text": text_prompt or "请用一句话介绍你自己。"}]},
        ], False
    if case == "vision":
        return [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image_path},
                    {"type": "text", "text": text_prompt or "请描述这张图。"},
                ],
            },
        ], False
    if case == "audio":
        return [
            {
                "role": "user",
                "content": [
                    {"type": "audio", "audio": audio_path},
                    {"type": "text", "text": text_prompt or "请描述你听到了什么。"},
                ],
            },
        ], False
    return [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image_path},
                {"type": "audio", "audio": audio_path},
                {"type": "text", "text": text_prompt or "What can you see and hear? Answer in one short sentence."},
            ],
        },
    ], True


def save_json(file_path: Path, data: Dict[str, Any]):
    with open(file_path, "w", encoding="utf-8") as file_obj:
        json.dump(data, file_obj, indent=4, ensure_ascii=False)


def load_json(file_path: Path) -> Dict[str, Any]:
    with open(file_path, "r", encoding="utf-8") as file_obj:
        return json.load(file_obj)


def save_case_golden(
    golden_dir: Path,
    case_name: str,
    input_ids: torch.Tensor,
    output_ids: torch.Tensor,
    output_text: list[str],
    audio: Optional[torch.Tensor] = None,
    sample_rate: int = 24000,
    meta_overrides: Optional[Dict[str, Any]] = None,
):
    """Persist one validation case into standard golden artifacts.

    Output layout:
    - golden_<case>_ids.pt
    - optional golden_<case>.wav
    - golden_meta.json (incremental update)
    - golden_validation.json (single-case integrity check)
    """
    golden_dir.mkdir(exist_ok=True, parents=True)

    input_ids_cpu = input_ids.detach().cpu() if isinstance(input_ids, torch.Tensor) else torch.as_tensor(input_ids)
    output_ids_cpu = output_ids.detach().cpu() if isinstance(output_ids, torch.Tensor) else torch.as_tensor(output_ids)
    ids_path = golden_dir / f"golden_{case_name}_ids.pt"
    torch.save(
        {
            "input_ids": input_ids_cpu,
            "output_ids": output_ids_cpu,
        },
        ids_path,
    )

    result = {
        "text": output_text,
        "input_ids_shape": list(input_ids_cpu.shape),
    }

    if audio is not None:
        wav_path = golden_dir / f"golden_{case_name}.wav"
        sf.write(str(wav_path), audio.reshape(-1).detach().cpu().numpy(), samplerate=sample_rate)
        result["audio_file"] = wav_path.name

    meta_path = golden_dir / "golden_meta.json"
    if meta_path.exists() and meta_path.stat().st_size > 0:
        try:
            meta = load_json(meta_path)
        except Exception:
            meta = {}
    else:
        meta = {}

    if "create_time" not in meta:
        meta["create_time"] = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
    meta["last_update_time"] = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
    meta.setdefault("results", {})
    meta["results"][case_name] = result
    if meta_overrides:
        for key, value in meta_overrides.items():
            meta[key] = value

    save_json(meta_path, meta)

    validation = validate_golden_outputs(golden_dir, [case_name])
    save_json(golden_dir / "golden_validation.json", validation)

    return {
        "golden_dir": str(golden_dir),
        "ids_file": str(ids_path),
        "meta_file": str(meta_path),
        "case": case_name,
    }


def _latest_matching(root_dir: Path, pattern: str, required_key: Optional[Any] = None) -> Optional[Path]:
    candidates = sorted(root_dir.rglob(pattern), key=lambda item: item.stat().st_mtime, reverse=True)
    if required_key is None:
        return candidates[0] if candidates else None

    if isinstance(required_key, (list, tuple, set)):
        required_keys = tuple(required_key)
    else:
        required_keys = (required_key,)

    for candidate in candidates:
        try:
            meta = load_json(candidate)
        except Exception:
            continue
        if any(key in meta for key in required_keys):
            return candidate
    return None


def discover_artifacts(root_dir: Path) -> Dict[str, Dict[str, Any]]:
    discovered: Dict[str, Dict[str, Any]] = {}
    mapping = {
        "text": ("meta.json", "prefill_onnx"),
        "audio": ("meta_audio.json", "audio_encoder_onnx"),
        "vision": ("meta_vision.json", "vision_encoder_onnx"),
        "talker": ("meta_talker.json", "talker_prefill_onnx"),
        "talker_prediction": ("meta_talker_prediction.json", "talker_prediction_prefill_onnx"),
        "code2wav": ("meta_code2wav.json", "code2wav_hmonnx"),
    }
    for key, (pattern, required_key) in mapping.items():
        meta_path = _latest_matching(root_dir, pattern, required_key)
        if meta_path is None:
            continue
        meta = load_json(meta_path)
        meta["_meta_path"] = str(meta_path)
        meta["_root_dir"] = str(meta_path.parent)
        discovered[key] = meta
    return discovered


def _resolve_meta_path(meta: Dict[str, Any], key: str) -> Path:
    return Path(meta["_root_dir"]) / meta[key]


def _ensure_tensor(value: Any, device: torch.device, dtype: Optional[torch.dtype] = None):
    if value is None:
        raise ValueError("Cannot convert None to tensor")

    # Handle model output objects
    if hasattr(value, "last_hidden_state"):
        value = value.last_hidden_state
    elif hasattr(value, "hidden_states"):
        value = value.hidden_states

    if isinstance(value, torch.Tensor):
        tensor = value
    else:
        try:
            tensor = torch.as_tensor(value)
        except Exception as e:
            raise ValueError(f"Cannot convert {type(value)} to tensor: {e}")

    tensor = tensor.to(device)
    if dtype is not None and tensor.is_floating_point():
        tensor = tensor.to(dtype)
    return tensor


def _extract_primary_output(output: Any):
    if output is None:
        raise ValueError("HMONNX session returned None output")

    # Handle model output objects with .last_hidden_state or similar attributes
    if hasattr(output, "last_hidden_state"):
        return output.last_hidden_state
    if hasattr(output, "hidden_states"):
        return output.hidden_states

    if isinstance(output, (list, tuple)):
        if len(output) == 0:
            raise ValueError("HMONNX session returned empty output list/tuple")
        return output[0]
    return output


def _get_modal_token_id(model_config: Any, attr_name: str) -> int:
    if hasattr(model_config, attr_name):
        return int(getattr(model_config, attr_name))
    thinker_config = getattr(model_config, "thinker_config", None)
    if thinker_config is not None and hasattr(thinker_config, attr_name):
        return int(getattr(thinker_config, attr_name))
    raise AttributeError(f"Cannot resolve {attr_name} from model config")


def _extract_outputs(output: Any):
    if isinstance(output, (list, tuple)):
        return list(output)
    return [output]


def _build_dense_deepstack_tensors(
    inputs_embeds: torch.Tensor,
    image_mask: torch.Tensor,
    deepstack_outputs: list[Any],
) -> list[torch.Tensor]:
    dense_tensors = []
    for deepstack_output in deepstack_outputs:
        deepstack_tensor = _ensure_tensor(deepstack_output, torch.device("cpu"), torch.float16)
        dense_tensor = torch.zeros_like(inputs_embeds, dtype=torch.float16)
        dense_tensor[image_mask] = deepstack_tensor.to(dense_tensor.dtype)
        dense_tensors.append(dense_tensor)
    return dense_tensors


def _ensure_hm_pixel_values(inputs: Dict[str, Any]) -> Dict[str, Any]:
    if "hm_pixel_values" not in inputs and "pixel_values" in inputs:
        inputs["hm_pixel_values"] = inputs["pixel_values"]
    if "hm_pixel_values_videos" not in inputs and "pixel_values_videos" in inputs:
        inputs["hm_pixel_values_videos"] = inputs["pixel_values_videos"]
    return inputs


def _prepare_vision_hmonnx_input(pixel_values: torch.Tensor, expected_shape) -> torch.Tensor:
    vision_input = pixel_values
    if vision_input.ndim == 6 and vision_input.shape[1] == 1:
        vision_input = vision_input[:, 0]
    if vision_input.ndim != 5:
        raise ValueError(f"Unexpected vision input rank: {vision_input.shape}")

    batch, channels, frames, height, width = vision_input.shape
    target_batch, target_channels, target_frames, target_height, target_width = [int(v) for v in expected_shape]

    if channels != target_channels:
        raise ValueError(f"Unexpected vision channels: got {channels}, expected {target_channels}")
    if frames != target_frames:
        if frames == 1 and target_frames > 1:
            vision_input = vision_input.repeat(1, 1, target_frames, 1, 1)
            frames = target_frames
        else:
            raise ValueError(f"Unexpected vision frames: got {frames}, expected {target_frames}")

    if height != target_height or width != target_width:
        vision_input = vision_input.to(torch.float32)
        vision_input = vision_input.permute(0, 2, 1, 3, 4).reshape(batch * frames, channels, height, width)
        vision_input = torch.nn.functional.interpolate(
            vision_input,
            size=(target_height, target_width),
            mode="bilinear",
            align_corners=False,
        )
        vision_input = vision_input.reshape(batch, frames, channels, target_height, target_width).permute(0, 2, 1, 3, 4)

    if batch != target_batch:
        if batch == 1 and target_batch == 1:
            pass
        else:
            raise ValueError(f"Unexpected vision batch: got {batch}, expected {target_batch}")

    return vision_input.to(torch.float16)


def _make_cache_state(kv_cache_info: Dict[str, Any]):
    kv_shape = kv_cache_info["shape"]
    num_layers = kv_cache_info["num_decoder_layers"]
    return {
        "past_seq_length": 0,
        "past_key_caches": [CacheTensor(torch.zeros(kv_shape, dtype=torch.float16)) for _ in range(num_layers)],
        "past_value_caches": [CacheTensor(torch.zeros(kv_shape, dtype=torch.float16)) for _ in range(num_layers)],
    }

class _LengthOnlyPastKeyValues:
    def __init__(self, seq_length: int = 0):
        self.seq_length = int(seq_length)

    def get_seq_length(self, layer_idx: int = 0):
        return int(self.seq_length)


def _extract_hmonnx_logits_and_hidden_states(
    output: Any,
    device: torch.device,
    actual_seq_len: int,
    hidden_dtype: Optional[torch.dtype] = None,
):
    outputs = _extract_outputs(output)
    if len(outputs) < 2:
        raise RuntimeError(
            "HMONNX artifact uses legacy single-output contract; please re-export talker/talker_prediction artifacts"
        )

    logits = _ensure_tensor(outputs[0], device)
    hidden_states = _ensure_tensor(outputs[1], device, hidden_dtype)
    if logits.ndim == 2:
        logits = logits.unsqueeze(1)
    if hidden_states.ndim == 2:
        hidden_states = hidden_states.unsqueeze(1)
    return logits[:, :actual_seq_len, :], hidden_states[:, :actual_seq_len, :]

def _replace_code2wav(native_model, code2wav_hmonnx_path: Path, static_code_len: int, logger):
    session = _create_hmonnx_session(code2wav_hmonnx_path)
    native_model.code2wav.hmonnx = session
    native_model.code2wav.hmonnx_max_code_len = static_code_len

    def forward(self, codes):
        max_code_len = int(self.hmonnx_max_code_len)
        code_len = int(codes.shape[-1])
        if code_len > max_code_len:
            raise ValueError(f"code2wav hmonnx max code len is {max_code_len}, but got {code_len}")
        hmonnx_input = codes.to(torch.int32)
        if code_len < max_code_len:
            hmonnx_input = torch.nn.functional.pad(hmonnx_input, (0, max_code_len - code_len))
        wav = self.hmonnx.forward(hmonnx_input)
        wav = _extract_primary_output(wav)
        wav = _ensure_tensor(wav, codes.device)
        return wav[..., : code_len * self.total_upsample]

    def chunked_decode(self, codes, chunk_size=300, left_context_size=25):
        max_code_len = int(self.hmonnx_max_code_len)
        safe_chunk = min(chunk_size, max(1, max_code_len - left_context_size))
        wavs = []
        start_index = 0
        while start_index < codes.shape[-1]:
            end_index = min(start_index + safe_chunk, codes.shape[-1])
            context_size = left_context_size if start_index - left_context_size > 0 else start_index
            chunk_token_len = end_index - start_index
            if chunk_token_len + context_size > max_code_len:
                context_size = max(0, max_code_len - chunk_token_len)
            codes_chunk = codes[..., start_index - context_size : end_index]
            wav_chunk = self.forward(codes_chunk)
            wavs.append(wav_chunk[..., context_size * self.total_upsample :])
            start_index = end_index
        return torch.cat(wavs, dim=-1)

    native_model.code2wav.forward = types.MethodType(forward, native_model.code2wav)
    native_model.code2wav.chunked_decode = types.MethodType(chunked_decode, native_model.code2wav)
    logger.info(f"code2wav replaced with HMONNX: {code2wav_hmonnx_path}")


def _patch_multimodal_encoders(
    native_model, audio_hmonnx_path: Optional[Path], vision_hmonnx_path: Optional[Path], logger
):
    thinker = native_model.thinker
    thinker.forward = types.MethodType(Qwen3OmniMoeThinkerForConditionalGeneration_forward, thinker)

    if audio_hmonnx_path is not None:
        thinker._audio_hmonnx_session = _create_hmonnx_session(audio_hmonnx_path)

        def get_audio_features(self, input_features, feature_attention_mask=None, audio_feature_lengths=None):
            original_device = input_features.device
            if feature_attention_mask is not None:
                audio_feature_lengths = torch.sum(feature_attention_mask, dim=1)
                input_features = input_features.permute(0, 2, 1)[feature_attention_mask.bool()].permute(1, 0)

            feature_lens = (
                audio_feature_lengths if audio_feature_lengths is not None else feature_attention_mask.sum(-1)
            )
            aftercnn_lens = _get_feat_extract_output_lengths(feature_lens)
            n_window = self.audio_tower.n_window
            n_window_infer = self.audio_tower.n_window_infer

            chunk_num = torch.ceil(feature_lens / (n_window * 2)).long()
            chunk_lengths = torch.tensor(
                [n_window * 2] * int(chunk_num.sum().item()),
                dtype=torch.long,
                device=feature_lens.device,
            )
            tail_chunk_index = torch.nn.functional.pad(chunk_num, (1, 0), value=-1).cumsum(0)[1:]
            chunk_lengths[tail_chunk_index] = feature_lens % (n_window * 2)
            chunk_lengths[chunk_lengths == 0] = n_window * 2

            chunk_list = input_features.T.split(chunk_lengths.tolist(), dim=0)
            padded_feature = nn.utils.rnn.pad_sequence(chunk_list, batch_first=True).transpose(1, 2)
            feature_lens_after_cnn = _get_feat_extract_output_lengths(chunk_lengths)
            padded_mask_after_cnn = nn.utils.rnn.pad_sequence(
                [
                    torch.ones(length, dtype=torch.bool, device=padded_feature.device)
                    for length in feature_lens_after_cnn
                ],
                batch_first=True,
            )

            cu_chunk_lens = [0]
            window_aftercnn = padded_mask_after_cnn.shape[-1] * (n_window_infer // (n_window * 2))
            for cnn_len in aftercnn_lens:
                cu_chunk_lens += [window_aftercnn] * int(cnn_len.item() // window_aftercnn)
                remainder = int(cnn_len.item() % window_aftercnn)
                if remainder != 0:
                    cu_chunk_lens += [remainder]
            cu_seqlens = torch.tensor(cu_chunk_lens, device=aftercnn_lens.device).cumsum(-1, dtype=torch.int32)

            # Process one chunk at a time (HMONNX exported with batch_size=1)
            all_outputs = []
            for i in range(padded_feature.shape[0]):
                single_feature = padded_feature[i : i + 1].cpu().to(torch.float16)  # [1, mel, len]
                single_cu = torch.tensor([0, int(feature_lens_after_cnn[i])], dtype=torch.int32)
                out_i = self._audio_hmonnx_session.forward(single_feature, single_cu)
                out_i = _extract_primary_output(out_i)
                out_i = _ensure_tensor(out_i, original_device, torch.float16)
                out_i = out_i[: int(feature_lens_after_cnn[i])]
                all_outputs.append(out_i)
            output = torch.cat(all_outputs, dim=0) if len(all_outputs) > 1 else all_outputs[0]
            return output

        thinker.get_audio_features = types.MethodType(get_audio_features, thinker)
        logger.info(f"audio encoder replaced with HMONNX: {audio_hmonnx_path}")

    if vision_hmonnx_path is not None:
        thinker._vision_hmonnx_session = _create_hmonnx_session(vision_hmonnx_path)
        thinker._vision_hmonnx_input_shape = thinker._vision_hmonnx_session.inputs[0].shape

        def get_image_features(self, pixel_values, image_grid_thw=None):
            try:
                hmonnx_input = _prepare_vision_hmonnx_input(pixel_values, self._vision_hmonnx_input_shape)
                output = self._vision_hmonnx_session.forward(hmonnx_input.cpu())
                all_outputs = _extract_outputs(output)
                image_embeds = _ensure_tensor(all_outputs[0], pixel_values.device, torch.float16)
                deepstack_features = (
                    tuple(_ensure_tensor(ds, pixel_values.device, torch.float16) for ds in all_outputs[1:4])
                    if len(all_outputs) > 1
                    else ()
                )
                return image_embeds, deepstack_features
            except Exception as e:
                print(f"[ERROR] get_image_features failed: {e}")
                raise

        def get_video_features(self, pixel_values_videos, video_grid_thw=None):
            try:
                hmonnx_input = _prepare_vision_hmonnx_input(pixel_values_videos, self._vision_hmonnx_input_shape)
                output = self._vision_hmonnx_session.forward(hmonnx_input.cpu())
                all_outputs = _extract_outputs(output)
                video_embeds = _ensure_tensor(all_outputs[0], pixel_values_videos.device, torch.float16)
                deepstack_features = (
                    tuple(_ensure_tensor(ds, pixel_values_videos.device, torch.float16) for ds in all_outputs[1:4])
                    if len(all_outputs) > 1
                    else ()
                )
                return video_embeds, deepstack_features
            except Exception as e:
                print(f"[ERROR] get_video_features failed: {e}")
                raise

        thinker.get_image_features = types.MethodType(get_image_features, thinker)
        thinker.get_video_features = types.MethodType(get_video_features, thinker)
        logger.info(f"vision encoder replaced with HMONNX: {vision_hmonnx_path}")


def _patch_talker_projection(talker, projection_hmonnx_path: Path, static_seq_len: int, logger):
    """Replace talker.hidden_projection.forward and talker.text_projection.forward
    with the quantized projection HMONNX exported alongside the talker graph.

    The HMONNX takes a single thinker-hidden input and returns both heads'
    outputs, so we share one session and dispatch per-call. Seq len is handled
    by zero-padding up to the exported static length and slicing the result.
    """
    session = _create_hmonnx_session(projection_hmonnx_path)

    def _run_projection(x: torch.Tensor, out_idx: int) -> torch.Tensor:
        squeezed = False
        if x.ndim == 2:
            x = x.unsqueeze(0)
            squeezed = True
        if x.ndim != 3:
            raise ValueError(f"projection input must be 2D or 3D, got shape {tuple(x.shape)}")

        actual_seq = int(x.shape[1])
        if actual_seq > static_seq_len:
            raise ValueError(f"talker projection seq_len {actual_seq} exceeds exported static length {static_seq_len}")

        x_cpu = x.detach().cpu().to(torch.float16)
        pad_seq = static_seq_len - actual_seq
        if pad_seq > 0:
            pad = torch.zeros(x_cpu.shape[0], pad_seq, x_cpu.shape[2], dtype=x_cpu.dtype)
            x_cpu = torch.cat([x_cpu, pad], dim=1)

        outputs = _extract_outputs(session.forward(x_cpu))
        out = outputs[out_idx]
        out = _ensure_tensor(out, x.device, x.dtype)
        out = out[:, :actual_seq, :]
        if squeezed:
            out = out.squeeze(0)
        return out

    def hidden_forward(self, hidden_state):
        return _run_projection(hidden_state, 0)

    def text_forward(self, hidden_state):
        return _run_projection(hidden_state, 1)

    talker.hidden_projection.forward = types.MethodType(hidden_forward, talker.hidden_projection)
    talker.text_projection.forward = types.MethodType(text_forward, talker.text_projection)
    logger.info(f"talker projection replaced with HMONNX: {projection_hmonnx_path}")


def _patch_talker_shadow(native_model, talker_meta: Dict[str, Any], logger):
    prefill_session = _create_hmonnx_session(_resolve_meta_path(talker_meta, "talker_prefill_onnx"))
    decode_session = _create_hmonnx_session(_resolve_meta_path(talker_meta, "talker_decode_onnx"))
    state = _make_cache_state(talker_meta["talker_kv_cache"])
    static_prefill_len = int(talker_meta.get("talker_input_sequence_length", 0))
    thinker_hs = int(talker_meta.get("talker_thinker_hidden_size", 0)) or int(
        talker_meta.get("talker_projection_in_features", 0)
    )
    if thinker_hs <= 0:
        raise RuntimeError(
            "talker meta missing thinker hidden size (talker_thinker_hidden_size / talker_projection_in_features)"
        )
    guidance_state = {"segments": []}
    original_get_user_parts = native_model._get_talker_user_parts
    original_get_assistant_parts = native_model._get_talker_assistant_parts

    def user_parts_hook(self, im_start_index, segment_end_index, multimodal_mask, thinker_hidden, thinker_embed):
        user_talker_part = original_get_user_parts(
            im_start_index,
            segment_end_index,
            multimodal_mask,
            thinker_hidden,
            thinker_embed,
        )
        user_mm_mask = multimodal_mask[:, im_start_index:segment_end_index]
        user_source = thinker_embed[:, im_start_index:segment_end_index].clone()
        if user_mm_mask.any():
            user_source[user_mm_mask] = thinker_hidden[:, im_start_index:segment_end_index][user_mm_mask]
        guidance_state["segments"].append(
            {
                "hidden_state": user_source.detach(),
                "role_mask": (~user_mm_mask).unsqueeze(-1).to(user_talker_part.dtype).detach(),
                "bypass_embeds": torch.zeros_like(user_talker_part).detach(),
                "bypass_mask": torch.zeros(
                    *user_talker_part.shape[:2], 1, dtype=user_talker_part.dtype, device=user_talker_part.device
                ).detach(),
            }
        )
        return user_talker_part

    def assistant_parts_hook(
        self,
        im_start_index,
        segment_end_index,
        speaker_id,
        thinker_embed,
        tts_pad_embed,
        tts_bos_embed,
        tts_eos_embed,
    ):
        input_embeds, input_ids, trailing_text_hidden = original_get_assistant_parts(
            im_start_index,
            segment_end_index,
            speaker_id,
            thinker_embed,
            tts_pad_embed,
            tts_bos_embed,
            tts_eos_embed,
        )
        assistant_source = torch.zeros(
            input_embeds.shape[0],
            input_embeds.shape[1],
            thinker_embed.shape[-1],
            dtype=thinker_embed.dtype,
            device=thinker_embed.device,
        )
        assistant_role_mask = torch.ones(
            input_embeds.shape[0], input_embeds.shape[1], 1, dtype=input_embeds.dtype, device=input_embeds.device
        )
        assistant_bypass_embeds = input_embeds.clone()
        assistant_bypass_mask = torch.ones(
            input_embeds.shape[0], input_embeds.shape[1], 1, dtype=input_embeds.dtype, device=input_embeds.device
        )

        projected_prefix = min(3, max(segment_end_index - im_start_index, 0))
        if projected_prefix > 0:
            assistant_source[:, :projected_prefix, :] = thinker_embed[
                :, im_start_index : im_start_index + projected_prefix, :
            ]
            assistant_bypass_embeds[:, :projected_prefix, :] = 0
            assistant_bypass_mask[:, :projected_prefix, :] = 0

        guidance_state["segments"].append(
            {
                "hidden_state": assistant_source.detach(),
                "role_mask": assistant_role_mask.detach(),
                "bypass_embeds": assistant_bypass_embeds.detach(),
                "bypass_mask": assistant_bypass_mask.detach(),
            }
        )
        return input_embeds, input_ids, trailing_text_hidden

    native_model._get_talker_user_parts = types.MethodType(user_parts_hook, native_model)
    native_model._get_talker_assistant_parts = types.MethodType(assistant_parts_hook, native_model)

    def _pad_prefill_tensor(tensor: torch.Tensor, target_seq_len: int, fill_value: float = 0.0) -> torch.Tensor:
        tensor = tensor.detach().cpu().to(torch.float16)
        current_seq_len = int(tensor.shape[1])
        if current_seq_len >= target_seq_len:
            return tensor[:, :target_seq_len, ...]
        pad_shape = (tensor.shape[0], target_seq_len - current_seq_len, *tensor.shape[2:])
        pad = torch.full(pad_shape, fill_value, dtype=tensor.dtype)
        return torch.cat([tensor, pad], dim=1)

    def _build_dynamic_prefill_guidance(actual_seq_len: int, target_seq_len: int):
        segments = guidance_state["segments"]
        if not segments:
            return None
        hidden_state = torch.cat([segment["hidden_state"] for segment in segments], dim=1)
        role_mask = torch.cat([segment["role_mask"] for segment in segments], dim=1)
        bypass_embeds = torch.cat([segment["bypass_embeds"] for segment in segments], dim=1)
        bypass_mask = torch.cat([segment["bypass_mask"] for segment in segments], dim=1)
        if int(hidden_state.shape[1]) != int(actual_seq_len):
            return None
        hidden_state = _pad_prefill_tensor(hidden_state, target_seq_len)
        role_mask = _pad_prefill_tensor(role_mask, target_seq_len)
        bypass_embeds = _pad_prefill_tensor(bypass_embeds, target_seq_len)
        bypass_mask = _pad_prefill_tensor(bypass_mask, target_seq_len)
        return hidden_state, role_mask, bypass_embeds, bypass_mask

    def _call_fused(session, shadow_inputs_embeds, shadow_seq_len, actual_seq_len, require_dynamic_guidance: bool):
        batch = int(shadow_inputs_embeds.shape[0])
        guidance = _build_dynamic_prefill_guidance(actual_seq_len, shadow_seq_len)
        if guidance is not None:
            source, role_mask, bypass_embeds, bypass_mask = guidance
        else:
            if require_dynamic_guidance:
                guidance_state["segments"].clear()
                raise RuntimeError(
                    "talker HMONNX takeover failed to build dynamic projection guidance for prefill; "
                    "refusing to fall back to Python-side projection."
                )
            # Decode or unmatched prefill falls back to bypass path: feed the
            # pre-projected embeds the HF side produced via python projection.
            source = torch.zeros(batch, shadow_seq_len, thinker_hs, dtype=torch.float16)
            role_mask = torch.zeros(batch, shadow_seq_len, 1, dtype=torch.float16)
            bypass_embeds = shadow_inputs_embeds
            bypass_mask = torch.ones(batch, shadow_seq_len, 1, dtype=torch.float16)
        return session.forward(
            source,
            role_mask,
            bypass_embeds,
            bypass_mask,
            torch.tensor([state["past_seq_length"]], dtype=torch.int32),
            torch.tensor([shadow_seq_len], dtype=torch.int32),
            *state["past_key_caches"],
            *state["past_value_caches"],
        )

    def forward(self, *args, **kwargs):
        inputs_embeds = kwargs.get("inputs_embeds")
        if inputs_embeds is not None:
            seq_len = int(inputs_embeds.shape[1])
            if seq_len > 1:
                state["past_seq_length"] = 0
                state["past_key_caches"] = _make_cache_state(talker_meta["talker_kv_cache"])["past_key_caches"]
                state["past_value_caches"] = _make_cache_state(talker_meta["talker_kv_cache"])["past_value_caches"]
            session = prefill_session if seq_len > 1 else decode_session
            shadow_inputs_embeds = inputs_embeds.detach().cpu().to(torch.float16)
            shadow_seq_len = seq_len
            if seq_len > 1 and static_prefill_len > 0:
                if seq_len > static_prefill_len:
                    raise ValueError(
                        f"talker shadow seq_len {seq_len} exceeds exported static length {static_prefill_len}"
                    )
                if seq_len < static_prefill_len:
                    pad = torch.zeros(
                        shadow_inputs_embeds.shape[0],
                        static_prefill_len - seq_len,
                        shadow_inputs_embeds.shape[2],
                        dtype=shadow_inputs_embeds.dtype,
                    )
                    shadow_inputs_embeds = torch.cat([shadow_inputs_embeds, pad], dim=1)
                    shadow_seq_len = static_prefill_len

            generation_step = kwargs.get("generation_step")
            residual_codes = kwargs.get("residual_codes")
            if seq_len > 1:
                generation_step = -1
                residual_codes = None

            logits, hidden_states = _extract_hmonnx_logits_and_hidden_states(
                _call_fused(
                    session,
                    shadow_inputs_embeds,
                    shadow_seq_len,
                    seq_len,
                    require_dynamic_guidance=seq_len > 1,
                ),
                inputs_embeds.device,
                seq_len,
                hidden_dtype=inputs_embeds.dtype,
            )
            if seq_len > 1:
                guidance_state["segments"].clear()
            state["past_seq_length"] += seq_len

            return Qwen3OmniMoeTalkerOutputWithPast(
                logits=logits,
                aux_loss=None,
                past_key_values=_LengthOnlyPastKeyValues(state["past_seq_length"]),
                hidden_states=((hidden_states,), residual_codes),
                generation_step=int(generation_step) + 1,
            )

        raise RuntimeError("talker HMONNX takeover requires inputs_embeds")

    native_model.talker.forward = types.MethodType(forward, native_model.talker)
    # Skip custom kwarg validation in talker.generate()
    native_model.talker._validate_model_kwargs = types.MethodType(lambda self, model_kwargs: None, native_model.talker)
    logger.info("talker model inserted in HMONNX takeover mode")


def _patch_talker_prediction_shadow(native_model, predictor_meta: Dict[str, Any], logger):
    prefill_session = _create_hmonnx_session(_resolve_meta_path(predictor_meta, "talker_prediction_prefill_onnx"))
    decode_session = _create_hmonnx_session(_resolve_meta_path(predictor_meta, "talker_prediction_decode_onnx"))
    state = _make_cache_state(predictor_meta["talker_prediction_kv_cache"])
    static_prefill_len = int(predictor_meta.get("talker_prediction_input_sequence_length", 0))
    num_lm_heads = int(predictor_meta.get("lm_head_count", 15))
    original_forward = native_model.talker.code_predictor.forward

    def _call_fused(session, shadow_inputs_embeds, padded_seq_len, current_seq_len, head_mask):
        return session.forward(
            shadow_inputs_embeds,
            head_mask,
            torch.tensor([state["past_seq_length"]], dtype=torch.int32),
            torch.tensor([current_seq_len], dtype=torch.int32),
            *state["past_key_caches"],
            *state["past_value_caches"],
        )

    def forward(self, *args, **kwargs):
        inputs_embeds = kwargs.get("inputs_embeds")
        generation_step = kwargs.get("generation_steps", 0)
        if generation_step is None:
            generation_step = 0
        if inputs_embeds is None:
            input_ids = kwargs.get("input_ids")
            if input_ids is None:
                raise RuntimeError("talker prediction HMONNX takeover requires inputs_embeds or input_ids")
            embed_index = max(0, int(generation_step) - 1)
            inputs_embeds = self.model.get_input_embeddings()[embed_index](input_ids)
        if inputs_embeds is not None:
            seq_len = int(inputs_embeds.shape[1])
            current_seq_len = seq_len
            if seq_len > 1:
                state["past_seq_length"] = 0
                state["past_key_caches"] = _make_cache_state(predictor_meta["talker_prediction_kv_cache"])[
                    "past_key_caches"
                ]
                state["past_value_caches"] = _make_cache_state(predictor_meta["talker_prediction_kv_cache"])[
                    "past_value_caches"
                ]
            session = prefill_session if seq_len > 1 else decode_session
            shadow_inputs_embeds = inputs_embeds.detach().cpu().to(torch.float16)
            shadow_seq_len = seq_len
            if seq_len > 1 and static_prefill_len > 0:
                if seq_len > static_prefill_len:
                    raise ValueError(
                        f"talker prediction shadow seq_len {seq_len} exceeds exported static length {static_prefill_len}"
                    )
                if seq_len < static_prefill_len:
                    pad = torch.zeros(
                        shadow_inputs_embeds.shape[0],
                        static_prefill_len - seq_len,
                        shadow_inputs_embeds.shape[2],
                        dtype=shadow_inputs_embeds.dtype,
                    )
                    shadow_inputs_embeds = torch.cat([shadow_inputs_embeds, pad], dim=1)
                    shadow_seq_len = static_prefill_len

            # Build head mask to mirror HF predictor semantics:
            # prefill -> generation_steps = seq_len - 2, decode -> explicit generation_steps.
            batch = int(shadow_inputs_embeds.shape[0])
            if seq_len > 1:
                step = max(0, min(seq_len - 2, num_lm_heads - 1))
                head_mask = torch.zeros(batch, shadow_seq_len, num_lm_heads, 1, dtype=torch.float16)
                head_mask[:, :, step, 0] = 1.0
            else:
                generation_step = kwargs.get("generation_steps", 0)
                if generation_step is None:
                    generation_step = 0
                head_mask = torch.zeros(batch, 1, num_lm_heads, 1, dtype=torch.float16)
                step = int(generation_step) % num_lm_heads
                head_mask[0, 0, step, 0] = 1.0

            logits, hidden_states = _extract_hmonnx_logits_and_hidden_states(
                _call_fused(session, shadow_inputs_embeds, shadow_seq_len, current_seq_len, head_mask),
                inputs_embeds.device,
                seq_len,
                hidden_dtype=inputs_embeds.dtype,
            )
            state["past_seq_length"] += seq_len

            return Qwen3OmniMoeTalkerCodePredictorOutputWithPast(
                logits=logits,
                past_key_values=_LengthOnlyPastKeyValues(state["past_seq_length"]),
                hidden_states=(hidden_states,),
                generation_steps=step + 1,
            )

        raise RuntimeError("talker prediction HMONNX takeover requires inputs_embeds or input_ids")

    native_model.talker.code_predictor.forward = types.MethodType(forward, native_model.talker.code_predictor)
    logger.info("talker prediction inserted in HMONNX takeover mode")
    # Skip custom kwarg validation in code_predictor.model.generate()
    native_model.talker.code_predictor.model._validate_model_kwargs = types.MethodType(
        lambda self, model_kwargs: None, native_model.talker.code_predictor.model
    )


def apply_artifact_replacements(native_model, artifacts: Dict[str, Dict[str, Any]], logger):
    if "code2wav" in artifacts:
        _replace_code2wav(
            native_model,
            _resolve_meta_path(artifacts["code2wav"], "code2wav_hmonnx"),
            int(artifacts["code2wav"]["static_code_len"]),
            logger,
        )

    audio_hmonnx_path = None
    if "audio" in artifacts:
        audio_hmonnx_path = _resolve_meta_path(artifacts["audio"], "audio_encoder_onnx")

    vision_hmonnx_path = None
    if "vision" in artifacts:
        vision_hmonnx_path = _resolve_meta_path(artifacts["vision"], "vision_encoder_onnx")

    if audio_hmonnx_path is not None or vision_hmonnx_path is not None:
        _patch_multimodal_encoders(native_model, audio_hmonnx_path, vision_hmonnx_path, logger)

    if "talker" in artifacts:
        _patch_talker_shadow(native_model, artifacts["talker"], logger)

    if "talker_prediction" in artifacts:
        _patch_talker_prediction_shadow(native_model, artifacts["talker_prediction"], logger)


def run_dialogue_validation(
    model_path: str,
    work_dir: Path,
    logger,
    case: str,
    max_new_tokens: int = 64,
    device_map: str = "auto",
    max_memory: Optional[Dict[Any, str]] = None,
    artifacts: Optional[Dict[str, Dict[str, Any]]] = None,
    report_name: str = "dialogue_validation.json",
    output_prefix: str = "dialogue",
    talker_max_new_tokens: Optional[int] = None,
    save_golden: bool = False,
    golden_dir: Optional[Path] = None,
    validation_prompt: Optional[str] = None,
):
    device_map = _resolve_validation_device_map(device_map, logger)
    if device_map == "auto" and max_memory is None:
        max_memory = _build_safe_validation_max_memory(logger)

    load_kwargs = dict(
        torch_dtype=torch.float16,
        device_map=device_map,
        attn_implementation="eager",
        trust_remote_code=True,
    )
    if max_memory is not None:
        load_kwargs["max_memory"] = max_memory

    native_model = Qwen3OmniMoeForConditionalGeneration.from_pretrained(
        model_path,
        **load_kwargs,
    )
    native_model.eval()
    _force_eager_moe_implementation(native_model, logger)
    _patch_inputs_embeds_generation_device(native_model.talker, "talker", logger)
    _patch_inputs_embeds_generation_device(native_model.talker.code_predictor, "talker.code_predictor", logger)
    if hasattr(native_model, "code2wav"):
        _patch_runtime_device_property(native_model.code2wav, "code2wav", logger)
    processor = Qwen3OmniMoeProcessor.from_pretrained(model_path)

    if artifacts:
        apply_artifact_replacements(native_model, artifacts, logger)

    conversation, use_audio_in_video = build_conversation(case, text_prompt=validation_prompt)
    text = processor.apply_chat_template(conversation, add_generation_prompt=True, tokenize=False)
    audios, images, videos = process_mm_info(conversation, use_audio_in_video=use_audio_in_video)
    inputs = processor(
        text=text,
        audio=audios,
        images=images,
        videos=videos or None,
        return_tensors="pt",
        padding=True,
        seconds_per_chunk=2.0,
        position_id_per_seconds=13,
        use_audio_in_video=use_audio_in_video,
    )
    inputs = _ensure_hm_pixel_values(inputs)
    # When vision artifact replacement is active, monkey-patched thinker expects hm_* tensors.
    use_vision_hmonnx = artifacts is not None and "vision" in artifacts
    if not use_vision_hmonnx:
        # Pure HF path: keep only canonical keys to avoid unsupported kwargs in generate.
        inputs.pop("hm_pixel_values", None)
        inputs.pop("hm_pixel_values_videos", None)

    device = next(native_model.parameters()).device
    dtype = next(native_model.parameters()).dtype
    inputs = inputs.to(device).to(dtype)

    try:
        generate_kwargs = dict(
            **inputs,
            speaker="Ethan",
            thinker_return_dict_in_generate=True,
            use_audio_in_video=use_audio_in_video,
            max_new_tokens=max_new_tokens,
        )
        if talker_max_new_tokens is not None:
            generate_kwargs["talker_max_new_tokens"] = talker_max_new_tokens
        with torch.no_grad():
            text_ids, audio = native_model.generate(**generate_kwargs)
    except Exception as e:
        import traceback

        if logger is not None:
            logger.warning(f"dialogue validation generate failed: {e}")
            logger.warning(traceback.format_exc())
        raise

    sequences = text_ids.sequences if hasattr(text_ids, "sequences") else text_ids
    output_text = processor.batch_decode(
        sequences[:, inputs["input_ids"].shape[1] :],
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )

    report = {
        "create_time": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()),
        "case": case,
        "validation_prompt": validation_prompt,
        "max_new_tokens": max_new_tokens,
        "talker_max_new_tokens": talker_max_new_tokens,
        "output_text": output_text,
        "input_ids_shape": list(inputs["input_ids"].shape),
        "applied_artifacts": sorted(list(artifacts.keys())) if artifacts else [],
    }

    if audio is not None:
        wav_path = work_dir / f"{output_prefix}_{case}.wav"
        sf.write(str(wav_path), audio.reshape(-1).detach().cpu().numpy(), samplerate=24000)
        report["audio_file"] = str(wav_path.relative_to(work_dir))

    report_path = work_dir / report_name
    save_json(report_path, report)
    logger.info(f"dialogue validation report saved to {report_path}")

    if save_golden:
        target_golden_dir = Path(golden_dir) if golden_dir is not None else work_dir / "golden"
        golden_info = save_case_golden(
            target_golden_dir,
            case,
            inputs["input_ids"],
            sequences,
            output_text,
            audio=audio,
            meta_overrides={
                "source_work_dir": str(work_dir),
                "hmonnx_modules": sorted(list(artifacts.keys())) if artifacts else [],
            },
        )
        report["golden_dir"] = golden_info["golden_dir"]
        logger.info(f"dialogue golden saved to {target_golden_dir}")

    return report

def _run_audio_encoder_hmonnx(session: HMONNXInference, audio_tower, input_features, feature_attention_mask):
    audio_feature_lengths = torch.sum(feature_attention_mask, dim=1)
    input_features = input_features.permute(0, 2, 1)[feature_attention_mask.bool()].permute(1, 0)
    feature_lens = audio_feature_lengths
    aftercnn_lens = _get_feat_extract_output_lengths(feature_lens)
    n_window = audio_tower.n_window
    n_window_infer = audio_tower.n_window_infer
    chunk_num = torch.ceil(feature_lens / (n_window * 2)).long()
    chunk_lengths = torch.tensor(
        [n_window * 2] * int(chunk_num.sum().item()),
        dtype=torch.long,
        device=feature_lens.device,
    )
    tail_chunk_index = torch.nn.functional.pad(chunk_num, (1, 0), value=-1).cumsum(0)[1:]
    chunk_lengths[tail_chunk_index] = feature_lens % (n_window * 2)
    chunk_lengths[chunk_lengths == 0] = n_window * 2
    chunk_list = input_features.T.split(chunk_lengths.tolist(), dim=0)
    padded_feature = nn.utils.rnn.pad_sequence(chunk_list, batch_first=True).transpose(1, 2)
    feature_lens_after_cnn = _get_feat_extract_output_lengths(chunk_lengths)
    padded_mask_after_cnn = nn.utils.rnn.pad_sequence(
        [torch.ones(length, dtype=torch.bool, device=padded_feature.device) for length in feature_lens_after_cnn],
        batch_first=True,
    )
    cu_chunk_lens = [0]
    window_aftercnn = padded_mask_after_cnn.shape[-1] * (n_window_infer // (n_window * 2))
    for cnn_len in aftercnn_lens:
        cu_chunk_lens += [window_aftercnn] * int(cnn_len.item() // window_aftercnn)
        remainder = int(cnn_len.item() % window_aftercnn)
        if remainder != 0:
            cu_chunk_lens += [remainder]
    cu_seqlens = torch.tensor(cu_chunk_lens, device=aftercnn_lens.device).cumsum(-1, dtype=torch.int32)
    # Process one chunk at a time (HMONNX exported with batch_size=1)
    all_outputs = []
    for i in range(padded_feature.shape[0]):
        single_feature = padded_feature[i : i + 1].cpu().to(torch.float16)
        single_cu = torch.tensor([0, int(feature_lens_after_cnn[i])], dtype=torch.int32)
        out_i = session.forward(single_feature, single_cu)
        out_i = _extract_primary_output(out_i)
        out_i = _ensure_tensor(out_i, torch.device("cpu"), torch.float16)
        out_i = out_i[: int(feature_lens_after_cnn[i])]
        all_outputs.append(out_i)
    output = torch.cat(all_outputs, dim=0) if len(all_outputs) > 1 else all_outputs[0]
    return output

def run_text_hmonnx_chain_forward(
    model_path: str,
    text_meta: Dict[str, Any],
    logger,
    case: str = "multimodal",
    audio_meta: Optional[Dict[str, Any]] = None,
    vision_meta: Optional[Dict[str, Any]] = None,
    report_path: Optional[Path] = None,
    max_new_tokens: int = 256,
    device_map: str = "auto",
    save_golden: bool = False,
    golden_dir: Optional[Path] = None,
):
    _ensure_mistral_common_reasoning_effort()
    config_for_validation = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
    native_model = None
    processor = None
    tokenizer = None
    if audio_meta is not None or vision_meta is not None:
        processor = Qwen3OmniMoeProcessor.from_pretrained(model_path)
        tokenizer = processor.tokenizer
        if logger is not None and device_map != "cpu":
            logger.info(f"text chain validation keeps HF model on cpu regardless of requested device_map={device_map}")
        native_model = Qwen3OmniMoeForConditionalGeneration.from_pretrained(
            model_path,
            torch_dtype=torch.float16,
            device_map="cpu",
            attn_implementation="eager",
            trust_remote_code=True,
        )
        native_model.eval()
        _force_eager_moe_implementation(native_model, logger)
        config_for_validation = native_model.config
    else:
        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        if getattr(tokenizer, "chat_template", None) is None:
            chat_template_path = Path(model_path) / "chat_template.json"
            if chat_template_path.exists():
                chat_template_payload = load_json(chat_template_path)
                if isinstance(chat_template_payload, dict):
                    tokenizer.chat_template = chat_template_payload.get("chat_template")
                elif isinstance(chat_template_payload, str):
                    tokenizer.chat_template = chat_template_payload
        if logger is not None:
            logger.info("text-only chain validation skips full HF omni model loading and uses tokenizer-only inputs")

    conversation, use_audio_in_video = build_conversation(case)
    if processor is not None:
        text = processor.apply_chat_template(conversation, add_generation_prompt=True, tokenize=False)
    else:
        text = tokenizer.apply_chat_template(conversation, add_generation_prompt=True, tokenize=False)
    audios, images, videos = process_mm_info(conversation, use_audio_in_video=use_audio_in_video)
    if processor is not None:
        inputs = processor(
            text=text,
            audio=audios,
            images=images,
            videos=videos,
            return_tensors="pt",
            padding=True,
            seconds_per_chunk=2.0,
            position_id_per_seconds=13,
            use_audio_in_video=use_audio_in_video,
        )
    else:
        inputs = tokenizer(text=text, return_tensors="pt", padding=True)
    inputs = _ensure_hm_pixel_values(inputs)

    token_embedding_state_dict = torch.load(
        Path(text_meta["_root_dir"]) / text_meta["token_embedding_file"],
        map_location="cpu",
        weights_only=False,
    )
    if isinstance(token_embedding_state_dict, dict) and "weight" in token_embedding_state_dict:
        token_embedding = nn.Embedding(
            token_embedding_state_dict["weight"].shape[0],
            token_embedding_state_dict["weight"].shape[1],
        )
        token_embedding.load_state_dict(token_embedding_state_dict)
    else:
        raise RuntimeError("token embedding file does not contain a standard state_dict")

    inputs_embeds = token_embedding(inputs["input_ids"].cpu())
    prefill_token_length = int(inputs_embeds.shape[1])

    if audio_meta is not None and "input_features" in inputs and "feature_attention_mask" in inputs:
        if native_model is None:
            raise RuntimeError("audio validation requires a full HF omni model")
        audio_session = _create_hmonnx_session(_resolve_meta_path(audio_meta, "audio_encoder_onnx"))
        audio_features = _run_audio_encoder_hmonnx(
            audio_session,
            native_model.thinker.audio_tower,
            inputs["input_features"].cpu(),
            inputs["feature_attention_mask"].cpu(),
        )
        audio_mask = inputs["input_ids"].cpu() == _get_modal_token_id(config_for_validation, "audio_token_id")
        inputs_embeds[audio_mask] = audio_features.to(inputs_embeds.dtype)

    if vision_meta is not None and "hm_pixel_values" in inputs:
        vision_session = _create_hmonnx_session(_resolve_meta_path(vision_meta, "vision_encoder_onnx"))
        vision_hmonnx_input = _prepare_vision_hmonnx_input(
            inputs["hm_pixel_values"].cpu(), vision_session.inputs[0].shape
        )
        vision_output = vision_session.forward(vision_hmonnx_input.to(torch.float16))
        vision_outputs = _extract_outputs(vision_output)
        vision_embeds = _ensure_tensor(vision_outputs[0], torch.device("cpu"), torch.float16)
        image_mask = inputs["input_ids"].cpu() == _get_modal_token_id(config_for_validation, "image_token_id")
        inputs_embeds[image_mask] = vision_embeds.to(inputs_embeds.dtype)
        deepstack_tensors = _build_dense_deepstack_tensors(inputs_embeds, image_mask, vision_outputs[1:4])
    else:
        deepstack_tensors = [torch.zeros_like(inputs_embeds, dtype=torch.float16) for _ in range(3)]

    kv_cache_info = text_meta["kv_cache"]
    kv_shape = kv_cache_info["shape"]
    num_layers = kv_cache_info["num_decoder_layers"]
    past_key_caches = [CacheTensor(torch.zeros(kv_shape, dtype=torch.float16)) for _ in range(num_layers)]
    past_value_caches = [CacheTensor(torch.zeros(kv_shape, dtype=torch.float16)) for _ in range(num_layers)]

    prefill_session = _create_hmonnx_session(_resolve_meta_path(text_meta, "prefill_onnx"))
    decode_session = _create_hmonnx_session(_resolve_meta_path(text_meta, "decode_onnx"))

    input_sequence_length = int(text_meta["wrap_cfg"]["input_sequence_length"])
    if prefill_token_length > input_sequence_length:
        raise RuntimeError(
            f"text chain input length {prefill_token_length} exceeds exported input_sequence_length {input_sequence_length}"
        )
    if prefill_token_length < input_sequence_length:
        pad_embeds = torch.zeros(
            (inputs_embeds.shape[0], input_sequence_length - prefill_token_length, inputs_embeds.shape[2]),
            dtype=inputs_embeds.dtype,
        )
        inputs_embeds = torch.cat([inputs_embeds, pad_embeds], dim=1)
        deepstack_tensors = [
            torch.cat([tensor, torch.zeros_like(pad_embeds, dtype=torch.float16)], dim=1)
            for tensor in deepstack_tensors
        ]

    def _pad_position_ids(position_ids: torch.Tensor, actual_seq_len: int, target_seq_len: int) -> torch.Tensor:
        position_ids = position_ids.detach().cpu().to(torch.int32)[:actual_seq_len]
        if actual_seq_len >= target_seq_len:
            return position_ids
        return torch.cat([position_ids, torch.zeros(target_seq_len - actual_seq_len, dtype=torch.int32)], dim=0)

    def _build_prefill_position_ids():
        attention_mask = inputs.get("attention_mask")
        if attention_mask is None:
            attention_mask = torch.ones_like(inputs["input_ids"])
        attention_mask = attention_mask.cpu()
        audio_feature_lengths = None
        if "feature_attention_mask" in inputs:
            audio_feature_lengths = torch.sum(inputs["feature_attention_mask"].cpu(), dim=1)
        if native_model is not None:
            position_ids, rope_deltas = native_model.thinker.get_rope_index(
                inputs["input_ids"].cpu(),
                inputs.get("image_grid_thw"),
                inputs.get("video_grid_thw"),
                attention_mask,
                use_audio_in_video,
                audio_feature_lengths,
                inputs.get("video_second_per_grid"),
            )
            delta0 = (1 - attention_mask).sum(dim=-1).unsqueeze(1)
            rope_deltas = rope_deltas - delta0
        else:
            position_ids = attention_mask.to(torch.float32).cumsum(-1) - 1
            position_ids.masked_fill_(attention_mask == 0, 1)
            position_ids = position_ids.unsqueeze(0).expand(3, -1, -1)
            max_position_ids = position_ids.max(0, keepdim=False)[0].max(-1, keepdim=True)[0]
            rope_deltas = max_position_ids + 1 - torch.sum(attention_mask, dim=-1, keepdim=True)
        return (
            _pad_position_ids(position_ids[0, 0], prefill_token_length, input_sequence_length),
            _pad_position_ids(position_ids[1, 0], prefill_token_length, input_sequence_length),
            _pad_position_ids(position_ids[2, 0], prefill_token_length, input_sequence_length),
            rope_deltas.detach().cpu().to(torch.long),
        )

    def _build_decode_position_ids(decode_past_seq_length: torch.Tensor, rope_deltas: torch.Tensor):
        delta = int(decode_past_seq_length.item()) + int(rope_deltas.reshape(-1)[0].item())
        position_ids = torch.full((3, 1), delta, dtype=torch.int32)
        return position_ids[0], position_ids[1], position_ids[2]

    text_supports_position_ids = bool(text_meta.get("supports_multimodal_position_ids", False))
    rope_deltas = None
    if text_supports_position_ids:
        time_position_ids, height_position_ids, width_position_ids, rope_deltas = _build_prefill_position_ids()

    current_input_length = torch.tensor([prefill_token_length], dtype=torch.int32)
    past_seq_length = torch.tensor([0], dtype=torch.int32)
    zero_decode_deepstack = [torch.zeros((1, 1, inputs_embeds.shape[2]), dtype=torch.float16) for _ in range(3)]

    prefill_inputs = [
        inputs_embeds.to(torch.float16),
    ]
    if text_supports_position_ids:
        prefill_inputs.extend([time_position_ids, height_position_ids, width_position_ids])
    prefill_inputs.extend([
        past_seq_length,
        current_input_length,
    ])
    prefill_tensor_input_count = len(prefill_session.inputs) - (2 * num_layers)
    expected_base_inputs = 6 if text_supports_position_ids else 3
    text_supports_deepstack = prefill_tensor_input_count == expected_base_inputs + 3
    if text_supports_deepstack:
        prefill_inputs.extend(deepstack_tensors)

    # --- Prefill ---
    prefill_logits = prefill_session.forward(
        *prefill_inputs,
        *past_key_caches,
        *past_value_caches,
    )
    prefill_logits = _ensure_tensor(_extract_primary_output(prefill_logits), torch.device("cpu"), torch.float32)
    if prefill_logits.ndim == 2:
        prefill_logits = prefill_logits.unsqueeze(1)
    next_token = torch.argmax(prefill_logits[:, -1, :], dim=-1, keepdim=True)

    # Determine EOS token ids for stopping
    eos_token_id = tokenizer.eos_token_id
    if isinstance(eos_token_id, int):
        eos_token_ids = {eos_token_id}
    elif isinstance(eos_token_id, (list, tuple)):
        eos_token_ids = set(eos_token_id)
    else:
        eos_token_ids = set()
    cfg_eos = getattr(config_for_validation, "eos_token_id", None)
    if isinstance(cfg_eos, int):
        eos_token_ids.add(cfg_eos)
    elif isinstance(cfg_eos, (list, tuple)):
        eos_token_ids.update(cfg_eos)
    cfg_im_end = getattr(config_for_validation, "im_end_token_id", None)
    if isinstance(cfg_im_end, int):
        eos_token_ids.add(cfg_im_end)

    # --- Autoregressive decode loop ---
    generated_tokens = [next_token]  # first token from prefill
    decode_past_seq_length = current_input_length.clone()
    one_length = torch.ones_like(current_input_length)

    kv_max_seq = kv_shape[2] if len(kv_shape) > 2 else kv_shape[-1]

    for step in range(max_new_tokens - 1):
        token_id = int(next_token.item())
        if token_id in eos_token_ids:
            break
        # Check KV cache capacity
        if int(decode_past_seq_length.item()) + 1 > kv_max_seq:
            logger.warning(f"KV cache full at step {step + 1}, stopping decode")
            break

        decode_inputs = [
            token_embedding(next_token).to(torch.float16),
        ]
        if text_supports_position_ids:
            decode_inputs.extend(_build_decode_position_ids(decode_past_seq_length, rope_deltas))
        decode_inputs.extend([
            decode_past_seq_length,
            one_length,
        ])
        if text_supports_deepstack:
            decode_inputs.extend(zero_decode_deepstack)
        decode_logits = decode_session.forward(
            *decode_inputs,
            *past_key_caches,
            *past_value_caches,
        )
        decode_logits = _ensure_tensor(_extract_primary_output(decode_logits), torch.device("cpu"), torch.float32)
        if decode_logits.ndim == 2:
            decode_logits = decode_logits.unsqueeze(1)
        next_token = torch.argmax(decode_logits[:, -1, :], dim=-1, keepdim=True)
        generated_tokens.append(next_token)
        decode_past_seq_length = decode_past_seq_length + 1

    # Concat all generated token ids and decode to text
    all_token_ids = torch.cat(generated_tokens, dim=-1)  # [1, num_tokens]
    output_text = tokenizer.batch_decode(
        all_token_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )
    logger.info(f"generated {all_token_ids.shape[-1]} tokens: {output_text}")

    report = {
        "create_time": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()),
        "case": case,
        "prefill_logits_shape": list(prefill_logits.shape),
        "num_generated_tokens": int(all_token_ids.shape[-1]),
        "output_text": output_text,
        "used_audio_hmonnx": audio_meta is not None,
        "used_vision_hmonnx": vision_meta is not None,
        "used_deepstack": text_supports_deepstack,
    }
    if report_path is not None:
        save_json(report_path, report)
        logger.info(f"text HMONNX chain report saved to {report_path}")

    if save_golden:
        if golden_dir is not None:
            target_golden_dir = Path(golden_dir)
        elif report_path is not None:
            target_golden_dir = report_path.parent / "golden"
        else:
            target_golden_dir = Path("golden")
        save_case_golden(
            target_golden_dir,
            case,
            inputs["input_ids"],
            all_token_ids,
            output_text,
            audio=None,
            meta_overrides={
                "source_work_dir": str(report_path.parent)
                if report_path is not None
                else str(target_golden_dir.parent),
                "hmonnx_modules": ["text"],
            },
        )
        logger.info(f"text chain golden saved to {target_golden_dir}")

    return report

def validate_golden_outputs(golden_dir: Path, selected_cases):
    meta_file = golden_dir / "golden_meta.json"
    if not meta_file.exists() or meta_file.stat().st_size == 0:
        raise RuntimeError(f"missing or empty golden meta file: {meta_file}")

    meta = load_json(meta_file)
    missing = []
    for case_name in selected_cases:
        ids_file = golden_dir / f"golden_{case_name}_ids.pt"
        if not ids_file.exists() or ids_file.stat().st_size == 0:
            missing.append(str(ids_file))
        if case_name in ("audio", "multimodal"):
            audio_file = golden_dir / f"golden_{case_name}.wav"
            if not audio_file.exists() or audio_file.stat().st_size == 0:
                missing.append(str(audio_file))
        if case_name not in meta.get("results", {}):
            missing.append(f"golden_meta.json::results::{case_name}")

    if missing:
        raise RuntimeError("golden outputs incomplete: " + ", ".join(missing))

    return {
        "golden_dir": str(golden_dir),
        "validated_cases": selected_cases,
        "result_count": len(meta.get("results", {})),
    }
