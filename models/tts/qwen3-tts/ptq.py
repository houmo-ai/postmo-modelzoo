# Copyright (c) 2025 HOUMO AI
#
# File: ptq.py
# Description:
#   Qwen3-TTS PTQ quantization and model export tool.
#   Supports both CustomVoice and Base models via model_name/model_size.
#   Exports sub-models (talker, code_predictor, text_projection,
#   speech_tokenizer, stateful_decoder) and base-only frontend models
#   (speaker_encoder, speech_tokenizer_encoder) to HMONNX format.
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
import argparse
import json
import math
import shutil
import time
import types
from pathlib import Path
from typing import cast

import onnx
import soundfile as sf
import torch
import torch.nn as nn
from qwen_tts import Qwen3TTSModel
from qwen_tts.core.tokenizer_12hz.modeling_qwen3_tts_tokenizer_v2 import (
    Qwen3TTSTokenizerV2DecoderTransformerModel,
    Qwen3TTSTokenizerV2Model,
)
from tqdm import tqdm
from transformers.masking_utils import (
    create_causal_mask,
    create_sliding_window_causal_mask,
)
from transformers.modeling_outputs import BaseModelOutputWithPast
from transformers.models.mimi.modeling_mimi import MimiConv1d, MimiEuclideanCodebook

from xhquant.api import (
    Config,
    ConfigDict,
    HMONNXGoldenInference,
    PrecisionMode,
    convert_onnx_to_hmonnx,
    ptq_quantize,
    set_random_seed,
)
from xhquant.export.onnx.transforms import hmonnx_transforms
from xhquant.utils.time_profiler import time_profiler
from xh_model_zoo.api import get_root_logger, xhquant_llm_init
from xh_model_zoo.xh_llm.models.base_llm_model import LLMBaseModel
from xh_model_zoo.xh_llm.models.builder import MODELS
from xh_model_zoo.xh_llm.models.eval_model_type import EvalModelType
from xh_model_zoo.xh_llm.models.qwen3_tts import (
    XHQwen3TTSCodePredictor,
    XHQwen3TTSModel,
    XHQwen3TTSTalker,
)
from hmatc.utils.utils import get_model_configs, first_not_none

HOUMO_TARGET = os.getenv("HOUMO_TARGET")
assert HOUMO_TARGET in ["xh2"], f"Unsupported HOUMO_TARGET: {HOUMO_TARGET}"

script_dir = os.path.dirname(os.path.abspath(__file__))
HOUMO_EXAMPLES_PATH = os.getenv("HOUMO_EXAMPLES_PATH", os.path.abspath("../../../"))
DEFAULT_CONFIG_PATH = os.path.join(script_dir, "config.yaml")

SUPPORTED_MODELS = [
    "talker",
    "code_predictor",
    "text_projection",
    "speech_tokenizer",
    "stateful_decoder",
    "speaker_encoder",
    "speech_tokenizer_encoder",
]

BASE_ONLY_MODELS = {"speaker_encoder", "speech_tokenizer_encoder"}


def get_default_model_dir(model_config: dict) -> str:
    """Infer the local HF model directory name from config.yaml."""
    repo_ids = model_config.get("modelscope_repo", [])
    if repo_ids:
        return repo_ids[0].rsplit("/", maxsplit=1)[-1]
    model_name = model_config.get("model_name", "qwen3-tts")
    model_size = model_config.get("model_size", "0.6b-customvoice")
    return f"{model_name}-{model_size}"


def get_tts_model_type(model_name: str, model_size: str) -> str:
    """Map model_name/model_size to the generation/calibration model type."""
    model_key = str(model_name).strip().lower().replace("_", "-")
    normalized = str(model_size).strip().lower().replace("_", "-")

    known_model_types = {
        ("qwen3-tts", "0.6b-base"): "base",
        ("qwen3-tts", "0.6b-customvoice"): "custom_voice",
        ("qwen3-tts", "0.6b-custom-voice"): "custom_voice",
    }
    model_type = known_model_types.get((model_key, normalized))
    if model_type is not None:
        return model_type

    if model_key == "qwen3-tts" and normalized.endswith("-base"):
        return "base"
    if model_key == "qwen3-tts" and (
        normalized.endswith("-customvoice") or normalized.endswith("-custom-voice")
    ):
        return "custom_voice"

    raise ValueError(
        f"Cannot infer TTS model type from model_name={model_name!r}, "
        f"model_size={model_size!r}. Please add this model combination to "
        "get_tts_model_type()."
    )


def is_base_model(model_name: str, model_size: str) -> bool:
    return get_tts_model_type(model_name, model_size) == "base"


def parse_models_arg(models_str: str) -> list[str]:
    """
    解析 --models 参数，返回要量化的模型列表。

    Args:
        models_str: 逗号分隔的模型名称字符串，或 "all" 表示全部模型

    Returns:
        要量化的模型名称列表

    Raises:
        ValueError: 如果指定的模型名称不在支持列表中
    """
    if models_str.lower() == "all":
        return SUPPORTED_MODELS.copy()

    models = [m.strip().lower() for m in models_str.split(",")]
    invalid_models = [m for m in models if m not in SUPPORTED_MODELS]
    if invalid_models:
        raise ValueError(
            f"Invalid model(s): {invalid_models}. "
            f"Supported models are: {SUPPORTED_MODELS}"
        )
    return models


def export_golden(
    onnx_path: str | Path,
    inputs: tuple[torch.Tensor, ...],
    golden_dir: str | Path,
    exec_device: str,
    logger,
) -> None:
    """
    使用 HMONNXGoldenInference 对 ONNX/HMONNX 模型生成 golden 数据。

    Args:
        onnx_path: ONNX 或 HMONNX 模型文件路径
        inputs: 输入张量元组
        golden_dir: golden 数据输出目录
        exec_device: 执行设备 ("cuda" 或 "cpu")
        logger: 日志记录器
    """

    def _flatten_inputs(input_args):
        flat_inputs = []
        for input_arg in input_args:
            if isinstance(input_arg, (list, tuple)):
                flat_inputs.extend(input_arg)
            else:
                flat_inputs.append(input_arg)
        return flat_inputs

    def _align_dtype(input_args, float_dtype=torch.float16):
        aligned_inputs = []
        for input_arg in input_args:
            if type(input_arg) is torch.Tensor and input_arg.is_floating_point():
                input_arg = input_arg.to(float_dtype)
            aligned_inputs.append(input_arg)
        return aligned_inputs

    golden_dir = Path(golden_dir)
    golden_dir.mkdir(parents=True, exist_ok=True)
    inputs = _align_dtype(_flatten_inputs(inputs))

    logger.info(f"Generating golden data for {onnx_path}...")
    golden_model = HMONNXGoldenInference(str(onnx_path))
    golden_model.save_golden = True
    golden_model.exec_device = torch.device(
        exec_device if torch.cuda.is_available() else "cpu"
    )
    golden_model.golden_dir = str(golden_dir)

    with torch.no_grad():
        golden_model.forward(*inputs)

    logger.info(f"Golden data saved to {golden_dir}")


def copy_embedding_file(
    src_path: Path,
    output_dir: str,
    dst_name: str,
    logger,
) -> None:
    """
    复制 embedding 文件到 output_dir 根目录并重命名。

    Args:
        src_path: 源文件路径
        output_dir: 目标输出目录
        dst_name: 目标文件名
        logger: 日志记录器
    """
    if src_path.exists():
        dst = Path(output_dir) / dst_name
        shutil.copy2(src_path, dst)
        logger.info(f"Copied embedding file: {src_path} -> {dst}")


def copy_golden_files(
    src_dir: Path,
    output_dir: str,
    dst_subdir: str,
    logger,
) -> None:
    """
    复制 golden 数据文件到 output_dir 下。

    Args:
        src_dir: 源目录（包含 .npy 文件）
        output_dir: 目标输出目录
        dst_subdir: 目标子目录名（通常为 "step_0"）
        logger: 日志记录器
    """
    if not src_dir.exists():
        logger.warning(f"Golden source directory not found: {src_dir}")
        return

    dst_dir = Path(output_dir) / dst_subdir
    dst_dir.mkdir(exist_ok=True, parents=True)

    for src_file in src_dir.glob("*.npy"):
        dst_file = dst_dir / src_file.name
        shutil.copy2(src_file, dst_file)

    logger.info(f"Golden files copied to {dst_dir}")


def generate_hmquant_filename(
    model_name: str,
    model_size: str,
    quant_type: str,
    sub_model_name: str,
    length_info: str = None,
    phase: str = None,
) -> str:
    """
    生成 hmquant 格式的文件名。

    格式: hmquant_<model_name>-<model_size>_<quant_type>_<length_info>_<sub_model_name>_<phase>.onnx

    Args:
        model_name: 模型名称，如 "qwen3-tts"
        model_size: 模型大小，如 "0.6B"
        quant_type: 量化类型，如 "int8"
        sub_model_name: 子模型名称，如 "talker", "code_predictor"
        length_info: 长度信息，如 "seq256", "seq1"（可选）
        phase: 阶段，如 "prefill", "decode"（可选）

    Returns:
        生成的文件名
    """
    parts = [f"hmquant_{model_name}-{model_size}_{quant_type}"]
    if length_info:
        parts.append(length_info)
    parts.append(sub_model_name)
    if phase:
        parts.append(phase)
    return "_".join(parts) + ".onnx"


def copy_onnx_model(
    src_dir: Path,
    output_dir: str,
    dst_subdir: str,
    hmquant_filename: str,
    logger,
) -> None:
    """
    复制 ONNX 模型目录到 output_dir 下，并重命名模型文件。

    Args:
        src_dir: 源目录（包含 onnx 文件和 external_data）
        output_dir: 目标输出目录
        dst_subdir: 目标子目录名
        hmquant_filename: 目标 onnx 文件名（hmquant 格式）
        logger: 日志记录器
    """
    if not src_dir.exists():
        logger.warning(f"Source directory not found: {src_dir}")
        return

    dst_dir = Path(output_dir) / dst_subdir
    dst_dir.mkdir(exist_ok=True, parents=True)

    # 复制所有文件，并将 onnx 文件重命名为 hmquant 格式
    for src_file in src_dir.iterdir():
        if src_file.is_file():
            if src_file.suffix == ".onnx":
                # 重命名 onnx 为 hmquant 格式
                dst_file = dst_dir / hmquant_filename
            else:
                dst_file = dst_dir / src_file.name
            shutil.copy2(src_file, dst_file)
            logger.info(f"Copied: {src_file} -> {dst_file}")
        elif src_file.is_dir():
            dst_sub = dst_dir / src_file.name
            if dst_sub.exists():
                shutil.rmtree(dst_sub)
            shutil.copytree(src_file, dst_sub)
            logger.info(f"Copied directory: {src_file} -> {dst_sub}")

    logger.info(f"ONNX model copied to {dst_dir}")


def copy_hmonnx_model(
    src_dir: Path,
    output_dir: str,
    dst_subdir: str,
    hmquant_filename: str,
    logger,
) -> None:
    """
    复制 hmonnx 模型到 output_dir 下。

    Args:
        src_dir: 源目录（hmonnx）
        output_dir: 目标输出目录
        dst_subdir: 目标子目录名
        hmquant_filename: 目标 onnx 文件名（hmquant 格式）
        logger: 日志记录器
    """
    if not src_dir.exists():
        logger.warning(f"Source directory not found: {src_dir}")
        return

    dst_dir = Path(output_dir) / dst_subdir
    dst_dir.mkdir(exist_ok=True, parents=True)

    # 复制所有文件，并将 onnx 文件重命名为 hmquant 格式
    for src_file in src_dir.iterdir():
        if src_file.is_file():
            if src_file.suffix == ".onnx":
                # 重命名 onnx 为 hmquant 格式
                dst_file = dst_dir / hmquant_filename
            else:
                dst_file = dst_dir / src_file.name
            shutil.copy2(src_file, dst_file)
            logger.info(f"Copied: {src_file} -> {dst_file}")
        elif src_file.is_dir():
            dst_sub = dst_dir / src_file.name
            if dst_sub.exists():
                shutil.rmtree(dst_sub)
            shutil.copytree(src_file, dst_sub)
            logger.info(f"Copied directory: {src_file} -> {dst_sub}")

    logger.info(f"HMONNX model copied to {dst_dir}")


class SpeechTokenizerEncodeWrapper(nn.Module):
    """Static export wrapper for Qwen3-TTS base speech tokenizer encoder."""

    def __init__(self, tokenizer_model: Qwen3TTSTokenizerV2Model):
        super().__init__()
        self.encoder = tokenizer_model.encoder.encoder
        self.encoder_transformer = tokenizer_model.encoder.encoder_transformer
        self.downsample = tokenizer_model.encoder.downsample
        self.quantizer = tokenizer_model.encoder.quantizer
        self.valid_num_quantizers = int(tokenizer_model.encoder_valid_num_quantizers)
        self.encode_downsample_rate = int(tokenizer_model.encode_downsample_rate)
        self.num_quantizers = int(tokenizer_model.encoder.config.num_quantizers)

    def forward(self, input_values: torch.Tensor, padding_mask: torch.Tensor):
        hidden = self.encoder(input_values.unsqueeze(1), padding_cache=None)
        hidden = hidden.transpose(1, 2)

        seq_len = hidden.shape[1]
        device = hidden.device
        cache_position = torch.arange(seq_len, device=device, dtype=torch.long)
        position_ids = cache_position.unsqueeze(0)
        q_idx = torch.arange(seq_len, device=device, dtype=torch.long).unsqueeze(1)
        kv_idx = torch.arange(seq_len, device=device, dtype=torch.long).unsqueeze(0)
        causal_mask = (
            torch.where(kv_idx <= q_idx, 0.0, -10000.0).unsqueeze(0).unsqueeze(0)
        )

        for layer in self.encoder_transformer.layers:
            hidden = layer(
                hidden,
                attention_mask=causal_mask,
                position_ids=position_ids,
                past_key_values=None,
                output_attentions=False,
                use_cache=False,
                cache_position=cache_position,
            )[0]

        hidden = hidden.transpose(1, 2)
        hidden = self.downsample(hidden, padding_cache=None)
        audio_codes = self.quantizer.encode(hidden, self.num_quantizers)
        audio_codes = audio_codes.transpose(0, 1)[:, : self.valid_num_quantizers]
        audio_codes = audio_codes.transpose(1, 2).to(torch.int32)
        valid_samples = padding_mask.to(torch.int64).sum(dim=1)
        valid_frames = (
            (valid_samples + self.encode_downsample_rate - 1)
            // self.encode_downsample_rate
        ).to(torch.int32)
        return audio_codes, valid_frames


class SpeakerEncoderWrapper(nn.Module):
    """Export wrapper for the base-model speaker encoder."""

    def __init__(self, speaker_encoder: nn.Module):
        super().__init__()
        self.speaker_encoder = speaker_encoder

    def forward(self, mels: torch.Tensor):
        return self.speaker_encoder(mels)


def _patch_mimi_conv1d_static_padding(module: nn.Module) -> None:
    """Use Python int padding for fixed-shape ONNX export."""

    def static_forward(self, hidden_states, padding_cache=None):
        if not self.causal and padding_cache is not None:
            raise ValueError(
                "`padding_cache` is not supported for non-causal convolutions."
            )
        if padding_cache is not None:
            layer_padding_cache = padding_cache.update(hidden_states, self.layer_idx)
            hidden_states = torch.cat([layer_padding_cache, hidden_states], dim=2)
            return self.conv(hidden_states)

        length = int(hidden_states.shape[-1])
        kernel_size = (self.conv.kernel_size[0] - 1) * self.conv.dilation[0] + 1
        stride = self.conv.stride[0]
        padding_total = kernel_size - stride
        n_frames = math.ceil((length - kernel_size + padding_total) / stride + 1) - 1
        ideal_length = n_frames * stride + kernel_size - padding_total
        extra_padding = int(ideal_length - length)

        if self.causal:
            paddings = (padding_total, extra_padding)
        else:
            padding_right = padding_total // 2
            padding_left = padding_total - padding_right
            paddings = (padding_left, padding_right + extra_padding)

        hidden_states = MimiConv1d._pad1d(hidden_states, paddings, mode=self.pad_mode)
        return self.conv(hidden_states)

    for submodule in module.modules():
        if isinstance(submodule, MimiConv1d):
            submodule.forward = types.MethodType(static_forward, submodule)


def _patch_mimi_codebook_matmul_distance(module: nn.Module) -> None:
    """Replace cdist with an ONNX-friendly squared-distance matmul."""

    def quantize_without_cdist(self, hidden_states):
        hidden_states = hidden_states.float()
        embed = self.embed.float()
        hidden_norm = hidden_states.square().sum(dim=-1, keepdim=True)
        embed_norm = embed.square().sum(dim=-1).unsqueeze(0)
        dists = (
            hidden_norm + embed_norm - 2.0 * hidden_states.matmul(embed.transpose(0, 1))
        )
        return dists.argmin(dim=-1)

    for submodule in module.modules():
        if isinstance(submodule, MimiEuclideanCodebook):
            submodule.quantize = types.MethodType(quantize_without_cdist, submodule)


# ---------------------------------------------------------------------------
# Common utilities shared across export functions
# ---------------------------------------------------------------------------


def _init_export_env(cfg, cfg_name: str, args: argparse.Namespace):
    """Common environment setup: work_dir, logger, config dump, random seed."""
    cfg.work_dir = str(Path(args.work_dir) / cfg_name)
    Path(cfg.work_dir).mkdir(exist_ok=True, parents=True)
    cfg.device = "cuda" if torch.cuda.is_available() else "cpu"
    cfg.dtype = "float16"
    cfg.debug = args.debug
    cfg.exec_device = "cuda" if torch.cuda.is_available() else "cpu"
    set_random_seed(args.seed)
    log_file = Path(cfg.work_dir) / f"{cfg_name}_debug.log"
    xhquant_llm_init(log_file, cfg.debug)
    config_file = Path(cfg.work_dir) / f"{cfg_name}.py"
    cfg.dump(config_file)
    cfg.config_file = config_file
    return get_root_logger()


def _flatten_inputs(inputs) -> list:
    """Flatten nested list/tuple inputs into a flat list for PTQ calibration."""
    flat = []
    for arg in inputs:
        if isinstance(arg, (list, tuple)):
            flat.extend(arg)
        else:
            flat.append(arg)
    return flat


def _run_generate(hf_model, args: argparse.Namespace):
    """Run TTS generation for calibration based on model_name/model_size."""
    text = "基于先进的存算一体技术和存储工艺，后摩智能致力于突破芯片的性能与功耗瓶颈，加速人工智能技术的普惠落地"
    if is_base_model(args.model_name, args.model_size):
        assert Path(args.ref_audio).exists(), (
            f"Reference audio not found: {args.ref_audio}\n"
            f"Please provide a valid path via --ref_audio"
        )
        return hf_model.generate_voice_clone(
            text=text,
            language="Chinese",
            ref_audio=args.ref_audio,
            ref_text=args.ref_text,
        )
    else:
        return hf_model.generate_custom_voice(
            text=text,
            language="Chinese",
            speaker="vivian",
        )


def _capture_feature_dim(hf_model, hook_target, args: argparse.Namespace) -> int:
    """Register a forward hook to capture feature_dim, then abort inference early."""
    feature_dim = None

    def _hook(self, hook_args, kwargs):
        nonlocal feature_dim
        inputs_embeds = kwargs.get("inputs_embeds", None)
        feature_dim = inputs_embeds.shape[-1] if inputs_embeds is not None else None
        raise RuntimeError("stop")

    handle = hook_target.register_forward_pre_hook(_hook, with_kwargs=True)
    try:
        _run_generate(hf_model, args)
    except RuntimeError:
        pass
    handle.remove()
    assert feature_dim is not None, "Failed to capture feature_dim"
    return feature_dim


def _run_ptq(
    xh_model, data_batch: dict, target_device: str, exec_device: str, logger
) -> list:
    """Frontend graph → quant graph → PTQ calibration. Returns flattened calib_data."""
    xh_model.convert_to_fronted_graph(data_batch)
    torch.cuda.empty_cache()
    xh_model.change_eval_type(eval_type=EvalModelType.FRONTEND)

    xh_model.convert_to_quant_graph(target_device)
    xh_model.change_eval_type(EvalModelType.CALIBRATION)
    xh_model.enable_calibration()

    calib_data = _flatten_inputs(xh_model.prepare_inputs(data_batch))
    with time_profiler() as t:
        ptq_quantize(
            xh_model.quanted_model,
            [calib_data],
            PrecisionMode.ALIGNED,
            [exec_device],
            auto_release_unused_parameters=True,
        )
        logger.info(f"PTQ Quantize time: {t():.04f} s")
    return calib_data


def xhmodel_export_onnx(
    xh_model: LLMBaseModel,
    data_batch,
    onnx_output_dir: str,
    cfg_name,
    logger,
):
    """Convert xhquant model graph to ONNX format."""
    xh_model.to("cpu")
    torch.cuda.empty_cache()
    xh_model.convert_to_export_graph(data_batch)

    torch.cuda.empty_cache()
    xh_model.change_eval_type(EvalModelType.EXPORTED)

    xh_model.to("cpu")
    torch.cuda.empty_cache()
    onnx_file = xh_model.to_export_onnx(data_batch, onnx_output_dir, cfg_name)[0]
    return onnx_file


def _resolve_tokenizer_dir(model_dir: str) -> str:
    speech_tokenizer_dir = Path(model_dir) / "speech_tokenizer"
    return str(
        speech_tokenizer_dir if speech_tokenizer_dir.exists() else Path(model_dir)
    )


def _export_static_onnx(
    model: nn.Module,
    dummy_inputs: tuple[torch.Tensor, ...],
    onnx_file: str | Path,
    input_names: list[str],
    output_names: list[str],
    opset: int,
    logger,
    use_dynamo: bool = True,
) -> None:
    export_kwargs = dict(
        model=model,
        args=dummy_inputs,
        f=str(onnx_file),
        input_names=input_names,
        output_names=output_names,
        opset_version=opset,
    )
    with torch.no_grad():
        if use_dynamo:
            try:
                torch.onnx.export(**export_kwargs, dynamo=True)
            except Exception as exc:
                logger.warning(
                    f"Dynamo ONNX export failed for {onnx_file}, retrying legacy tracer: {exc}"
                )
                torch.onnx.export(**export_kwargs, dynamo=False)
        else:
            torch.onnx.export(**export_kwargs, dynamo=False)

    onnx_model = onnx.load(str(onnx_file))
    onnx.save(onnx_model, str(onnx_file))


def _assert_base_only_export(args: argparse.Namespace, sub_model_name: str) -> None:
    if not is_base_model(args.model_name, args.model_size):
        raise ValueError(
            f"{sub_model_name} is only used by qwen3-tts base models, "
            f"but got model_name={args.model_name}, model_size={args.model_size}"
        )


def export_speech_tokenizer_encoder(args: argparse.Namespace) -> None:
    """Export base speech_tokenizer.encode: wav -> audio_codes + valid_frames."""
    _assert_base_only_export(args, "speech_tokenizer_encoder")
    cfg = Config(
        dict(
            model=dict(hf_model=args.model_dir),
            target_device="XH2a",
            hf_model_dir=args.model_dir,
        )
    )
    cfg_name = "qwen3_tts_speech_tokenizer_encoder"
    logger = _init_export_env(cfg, cfg_name, args)

    work_dir = Path(cfg.work_dir)
    exec_device = cfg.exec_device
    target_device = cfg.target_device
    tokenizer_dir = _resolve_tokenizer_dir(cfg.hf_model_dir)

    logger.info(f"Loading Qwen3TTSTokenizerV2Model from {tokenizer_dir}")
    tokenizer_model = (
        Qwen3TTSTokenizerV2Model.from_pretrained(tokenizer_dir).float().cpu().eval()
    )
    tokenizer_model.config._attn_implementation = "eager"
    tokenizer_model.encoder.config._attn_implementation = "eager"
    tokenizer_model.encoder.encoder_transformer.config._attn_implementation = "eager"
    _patch_mimi_conv1d_static_padding(tokenizer_model.encoder)
    _patch_mimi_codebook_matmul_distance(tokenizer_model.encoder.quantizer)

    wrapper = SpeechTokenizerEncodeWrapper(tokenizer_model).float().cpu().eval()
    input_values = torch.zeros(
        args.batch, args.frontend_audio_samples, dtype=torch.float32
    )
    padding_mask = torch.ones(
        args.batch, args.frontend_audio_samples, dtype=torch.int32
    )
    dummy_inputs = (input_values, padding_mask)

    onnx_dir = work_dir / "onnx"
    hmonnx_dir = work_dir / "hmonnx"
    onnx_dir.mkdir(exist_ok=True, parents=True)
    hmonnx_dir.mkdir(exist_ok=True, parents=True)
    onnx_file = onnx_dir / "speech_tokenizer_encode.onnx"
    hmonnx_file = hmonnx_dir / f"speech_tokenizer_encode_{target_device}.onnx"

    logger.info(f"Exporting speech_tokenizer.encode ONNX to {onnx_file}")
    _export_static_onnx(
        wrapper,
        dummy_inputs,
        onnx_file,
        ["input_values", "padding_mask"],
        ["audio_codes", "valid_frames"],
        args.frontend_opset,
        logger,
        use_dynamo=False,
    )

    logger.info(f"Converting speech_tokenizer.encode to HMONNX: {hmonnx_file}")
    convert_onnx_to_hmonnx(
        str(onnx_file),
        [x.cpu() for x in dummy_inputs],
        target_device,
        str(hmonnx_file),
    )

    if args.dump_golden:
        golden_dir = work_dir / "golden" / "speech_tokenizer_encode"
        export_golden(
            str(hmonnx_file),
            tuple(x.cpu() for x in dummy_inputs),
            golden_dir,
            exec_device,
            logger,
        )

    meta = {
        "create_time": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()),
        "hf_model": cfg.hf_model_dir,
        "tokenizer_dir": tokenizer_dir,
        "target_device": target_device,
        "speech_tokenizer_encode_onnx": str(onnx_file.relative_to(work_dir)),
        "speech_tokenizer_encode_hmonnx": str(hmonnx_file.relative_to(work_dir)),
        "batch_size": int(args.batch),
        "audio_samples": int(args.frontend_audio_samples),
        "input_sample_rate": int(tokenizer_model.input_sample_rate),
        "encode_downsample_rate": int(tokenizer_model.encode_downsample_rate),
        "encoder_valid_num_quantizers": int(
            tokenizer_model.encoder_valid_num_quantizers
        ),
        "interfaces": {
            "speech_tokenizer_encode": {
                "inputs": [
                    "input_values: float32[B, audio_samples]",
                    "padding_mask: int32[B, audio_samples]",
                ],
                "outputs": ["audio_codes: int32[B, T, 16]", "valid_frames: int32[B]"],
                "note": (
                    "Crop audio_codes by valid_frames. valid_frames = "
                    "ceil(valid_samples / encode_downsample_rate)."
                ),
            }
        },
    }
    if args.dump_golden:
        meta["speech_tokenizer_encode_golden_dir"] = str(
            golden_dir.relative_to(work_dir)
        )
    with open(work_dir / "meta.json", "w") as f:
        json.dump(meta, f, indent=4)

    if args.output_dir:
        Path(args.output_dir).mkdir(exist_ok=True, parents=True)
        hmonnx_filename = generate_hmquant_filename(
            model_name=args.model_name,
            model_size=args.model_size,
            quant_type=args.quant_type,
            sub_model_name="speech_tokenizer_encoder",
        )
        copy_hmonnx_model(
            hmonnx_dir,
            args.output_dir,
            "speech_tokenizer_encoder",
            hmonnx_filename,
            logger,
        )
        if args.dump_golden:
            copy_golden_files(
                golden_dir,
                args.output_dir,
                "speech_tokenizer_encoder/step_0",
                logger,
            )


def export_speaker_encoder(args: argparse.Namespace) -> None:
    """Export base speaker encoder: mel -> speaker_embedding."""
    _assert_base_only_export(args, "speaker_encoder")
    cfg = Config(
        dict(
            model=dict(hf_model=args.model_dir),
            target_device="XH2a",
            hf_model_dir=args.model_dir,
        )
    )
    cfg_name = "qwen3_tts_speaker_encoder"
    logger = _init_export_env(cfg, cfg_name, args)

    work_dir = Path(cfg.work_dir)
    exec_device = cfg.exec_device
    target_device = cfg.target_device
    logger.info(f"Loading Qwen3TTSModel from {cfg.hf_model_dir}")
    hf_model = Qwen3TTSModel.from_pretrained(
        cfg.hf_model_dir,
        device_map="cpu",
        dtype=torch.float32,
        attn_implementation="sdpa",
    )
    hf_model = cast(Qwen3TTSModel, hf_model)
    tts_model = hf_model.model.float().cpu().eval()
    if tts_model.speaker_encoder is None:
        raise ValueError(f"model does not have speaker_encoder: {cfg.hf_model_dir}")

    wrapper = SpeakerEncoderWrapper(tts_model.speaker_encoder).float().cpu().eval()
    mels = torch.zeros(
        args.batch,
        args.frontend_mel_frames,
        args.frontend_mel_dim,
        dtype=torch.float32,
    )
    dummy_inputs = (mels,)

    onnx_dir = work_dir / "onnx"
    hmonnx_dir = work_dir / "hmonnx"
    onnx_dir.mkdir(exist_ok=True, parents=True)
    hmonnx_dir.mkdir(exist_ok=True, parents=True)
    onnx_file = onnx_dir / "speaker_encoder.onnx"
    hmonnx_file = hmonnx_dir / f"speaker_encoder_{target_device}.onnx"

    logger.info(f"Exporting speaker_encoder ONNX to {onnx_file}")
    _export_static_onnx(
        wrapper,
        dummy_inputs,
        onnx_file,
        ["mels"],
        ["speaker_embedding"],
        args.frontend_opset,
        logger,
    )

    logger.info(f"Converting speaker_encoder to HMONNX: {hmonnx_file}")
    convert_onnx_to_hmonnx(
        str(onnx_file),
        [x.cpu() for x in dummy_inputs],
        target_device,
        str(hmonnx_file),
    )

    if args.dump_golden:
        golden_dir = work_dir / "golden" / "speaker_encoder"
        export_golden(
            str(hmonnx_file),
            tuple(x.cpu() for x in dummy_inputs),
            golden_dir,
            exec_device,
            logger,
        )

    meta = {
        "create_time": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()),
        "hf_model": cfg.hf_model_dir,
        "target_device": target_device,
        "speaker_encoder_onnx": str(onnx_file.relative_to(work_dir)),
        "speaker_encoder_hmonnx": str(hmonnx_file.relative_to(work_dir)),
        "batch_size": int(args.batch),
        "mel_frames": int(args.frontend_mel_frames),
        "mel_dim": int(args.frontend_mel_dim),
        "speaker_encoder_sample_rate": int(tts_model.speaker_encoder_sample_rate),
        "interfaces": {
            "speaker_encoder": {
                "inputs": ["mels: float32[B, mel_frames, mel_dim]"],
                "outputs": ["speaker_embedding: float32[B, 1024]"],
                "note": (
                    "mels should match "
                    "qwen_tts.core.models.modeling_qwen3_tts."
                    "mel_spectrogram(...).transpose(1, 2)."
                ),
            }
        },
    }
    if args.dump_golden:
        meta["speaker_encoder_golden_dir"] = str(golden_dir.relative_to(work_dir))
    with open(work_dir / "meta.json", "w") as f:
        json.dump(meta, f, indent=4)

    if args.output_dir:
        Path(args.output_dir).mkdir(exist_ok=True, parents=True)
        hmonnx_filename = generate_hmquant_filename(
            model_name=args.model_name,
            model_size=args.model_size,
            quant_type=args.quant_type,
            sub_model_name="speaker_encoder",
        )
        copy_hmonnx_model(
            hmonnx_dir,
            args.output_dir,
            "speaker_encoder",
            hmonnx_filename,
            logger,
        )
        if args.dump_golden:
            copy_golden_files(
                golden_dir, args.output_dir, "speaker_encoder/step_0", logger
            )


def export_talker(args: argparse.Namespace) -> None:
    """Export talker model: HF model -> TorchFX -> PTQ -> ONNX (prefill + decode)."""
    cfg = Config(
        dict(
            model=dict(
                type="XHQwen3TTSTalker",
                hf_model=args.model_dir,
                wrap_cfg=dict(
                    max_sequence_length=args.max_sequence_length,
                    input_sequence_length=args.input_sequence_length,
                    use_cache=True,
                    num_logits_to_keep=1,
                    kv_cache=dict(
                        cache_axis=2,
                    ),
                    enable_rope=True,
                ),
                quant_config=dict(),
                frontend_type="TorchFX",
                export_cfg=dict(
                    input_names=[
                        "inputs_embeds",
                        "past_seq_length",
                        "current_input_length",
                    ],
                    output_names=[
                        "logits",
                        "past_hidden",
                    ],
                ),
            ),
            target_device="XH2a",
            hf_model_dir=args.model_dir,
        )
    )
    cfg_name = "qwen3_tts_talker"
    logger = _init_export_env(cfg, cfg_name, args)

    dtype = getattr(torch, cfg.dtype)
    work_dir = cfg.work_dir
    exec_device = cfg.exec_device
    xh_model = MODELS.build(cfg.model)
    assert isinstance(xh_model, XHQwen3TTSTalker)

    meta_info = ConfigDict(
        dict(
            create_time=time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()),
            config=str(cfg.config_file.relative_to(cfg.work_dir)),
        )
    )
    meta_info["model_name"] = cfg_name
    meta_info["wrap_cfg"] = cfg.model.wrap_cfg.to_dict()
    meta_info.hf_model = cfg.hf_model_dir

    hf_model = xh_model.get_hf_model(device_map="cpu", dtype=torch.float16)
    assert isinstance(hf_model, XHQwen3TTSModel)
    hf_model = cast(XHQwen3TTSModel, hf_model)

    feature_dim = _capture_feature_dim(hf_model, hf_model.model.talker, args)

    token_embedding = hf_model.model.talker.get_input_embeddings()
    token_embedding_file = Path(cfg.work_dir) / "token_embedding.pt"
    torch.save(token_embedding.state_dict(), str(token_embedding_file))
    meta_info.token_embedding_file = str(token_embedding_file.relative_to(cfg.work_dir))

    text_embedding = hf_model.model.talker.get_text_embeddings()
    text_embedding_file = Path(cfg.work_dir) / "text_embedding.pt"
    torch.save(text_embedding.state_dict(), str(text_embedding_file))
    meta_info.text_embedding_file = str(text_embedding_file.relative_to(cfg.work_dir))

    xh_model.init_wrap_model(hf_model)

    xh_model.to(dtype=dtype)
    xh_model.change_eval_type(eval_type=EvalModelType.WRAPED)

    if xh_model.past_key_caches is not None and len(xh_model.past_key_caches) > 0:
        meta_info.use_cache = True
        meta_info.kv_cache_shape = xh_model.past_key_caches[0].shape
        meta_info.num_hidden_layers = len(xh_model.past_key_caches)

    del hf_model

    input_sequence_length = xh_model.wrap_cfg.input_sequence_length
    data_batch = {
        "inputs_embeds": torch.randn(1, input_sequence_length, feature_dim),
        "past_seq_length": 0,
    }
    calib_data = _run_ptq(xh_model, data_batch, cfg.target_device, exec_device, logger)

    meta_file = Path(work_dir) / "meta.json"
    xh_model.change_eval_type(EvalModelType.QUANTED_ALIGNED)
    prefill_onnx_dir = Path(cfg.work_dir) / "prefill_onnx"
    decode_onnx_dir = Path(cfg.work_dir) / "decode_onnx"
    prefill_onnx_dir.mkdir(exist_ok=True, parents=True)
    decode_onnx_dir.mkdir(exist_ok=True, parents=True)
    prefill_onnx_file = xhmodel_export_onnx(
        xh_model,
        data_batch,
        str(prefill_onnx_dir),
        f"{cfg_name}_prefill",
        logger,
    )

    if args.dump_golden:
        prefill_golden_dir = Path(cfg.work_dir) / "prefill_golden"
        export_golden(
            prefill_onnx_file,
            tuple(calib_data),
            prefill_golden_dir,
            exec_device,
            logger,
        )

    meta_info.prefill_onnx_file = str(Path(prefill_onnx_file).relative_to(cfg.work_dir))
    xh_model.release_exported_model()
    logger.info(f"Prefill model saved to {prefill_onnx_file}")

    xh_model.change_eval_type(EvalModelType.QUANTED_ALIGNED)
    xh_model.to(dtype)
    data_batch = {
        "inputs_embeds": torch.randn(1, 1, feature_dim),
        "past_seq_length": 0,
    }

    torch.cuda.empty_cache()
    xh_model.set_input_sequence_length(1)
    xh_model.to("cpu")

    decode_onnx_file = xhmodel_export_onnx(
        xh_model,
        data_batch,
        str(decode_onnx_dir),
        f"{cfg_name}_decode",
        logger,
    )

    decode_calib_inputs = _flatten_inputs(xh_model.prepare_inputs(data_batch))

    if args.dump_golden:
        decode_golden_dir = Path(cfg.work_dir) / "decode_golden"
        export_golden(
            decode_onnx_file,
            tuple(decode_calib_inputs),
            decode_golden_dir,
            exec_device,
            logger,
        )

    meta_info.decode_onnx_file = str(Path(decode_onnx_file).relative_to(cfg.work_dir))
    xh_model.release_exported_model()
    logger.info(f"Decode model saved to {decode_onnx_file}")
    with open(meta_file, "w") as f:
        json.dump(meta_info, f, indent=4)

    if args.output_dir:
        Path(args.output_dir).mkdir(exist_ok=True, parents=True)
        copy_embedding_file(
            Path(cfg.work_dir) / "token_embedding.pt",
            args.output_dir,
            "quant_embedding.pt",
            logger,
        )
        copy_embedding_file(
            Path(cfg.work_dir) / "text_embedding.pt",
            args.output_dir,
            "text_embedding.pt",
            logger,
        )
        prefill_filename = generate_hmquant_filename(
            model_name=args.model_name,
            model_size=args.model_size,
            quant_type=args.quant_type,
            sub_model_name="talker",
            length_info=f"{args.input_sequence_length}_{args.max_sequence_length}",
            phase="prefill",
        )
        decode_filename = generate_hmquant_filename(
            model_name=args.model_name,
            model_size=args.model_size,
            quant_type=args.quant_type,
            sub_model_name="talker",
            length_info=f"{args.input_sequence_length}_{args.max_sequence_length}",
            phase="decode",
        )
        copy_onnx_model(
            Path(cfg.work_dir) / "prefill_onnx",
            args.output_dir,
            "talker_prefill",
            prefill_filename,
            logger,
        )
        copy_onnx_model(
            Path(cfg.work_dir) / "decode_onnx",
            args.output_dir,
            "talker_decode",
            decode_filename,
            logger,
        )
        if args.dump_golden:
            copy_golden_files(
                Path(cfg.work_dir) / "prefill_golden",
                args.output_dir,
                "talker_prefill/step_0",
                logger,
            )
            copy_golden_files(
                Path(cfg.work_dir) / "decode_golden",
                args.output_dir,
                "talker_decode/step_0",
                logger,
            )


def export_code_predictor(args: argparse.Namespace) -> None:
    """Export code predictor model: HF model -> TorchFX -> PTQ -> ONNX (prefill + decode)."""
    cfg = Config(
        dict(
            model=dict(
                type="XHQwen3TTSCodePredictor",
                hf_model=args.model_dir,
                wrap_cfg=dict(
                    max_sequence_length=args.code_predictor_max_sequence_length,
                    input_sequence_length=args.code_predictor_input_sequence_length,
                    use_cache=True,
                    num_logits_to_keep=1,
                    kv_cache=dict(
                        cache_axis=2,
                    ),
                    enable_rope=True,
                ),
                quant_config=dict(),
                frontend_type="TorchFX",
                export_cfg=dict(
                    input_names=[
                        "inputs_embeds",
                        "past_seq_length",
                        "current_input_length",
                    ],
                    output_names=["logits"],
                ),
            ),
            target_device="XH2a",
            hf_model_dir=args.model_dir,
        )
    )
    cfg_name = "qwen3_tts_code_predictor"
    logger = _init_export_env(cfg, cfg_name, args)

    exec_device = cfg.exec_device
    dtype = getattr(torch, cfg.dtype)
    work_dir = cfg.work_dir
    xh_model = MODELS.build(cfg.model)
    assert isinstance(xh_model, XHQwen3TTSCodePredictor)

    meta_info = ConfigDict(
        dict(
            create_time=time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()),
            config=str(cfg.config_file.relative_to(cfg.work_dir)),
        )
    )
    meta_info["model_name"] = cfg_name
    meta_info["wrap_cfg"] = cfg.model.wrap_cfg.to_dict()
    meta_info.hf_model = cfg.hf_model_dir

    hf_model = xh_model.get_hf_model(device_map="cpu", dtype=torch.float16)
    assert isinstance(hf_model, XHQwen3TTSModel)
    hf_model = cast(XHQwen3TTSModel, hf_model)

    feature_dim = _capture_feature_dim(
        hf_model, hf_model.model.talker.code_predictor, args
    )

    token_embedding = hf_model.model.talker.code_predictor.get_input_embeddings()
    token_embedding_file = Path(cfg.work_dir) / "token_embedding.pt"
    torch.save(token_embedding.state_dict(), str(token_embedding_file))
    meta_info.token_embedding_file = str(token_embedding_file.relative_to(cfg.work_dir))

    xh_model.init_wrap_model(hf_model)
    xh_model.to(dtype=dtype)
    xh_model.change_eval_type(eval_type=EvalModelType.WRAPED)

    if xh_model.past_key_caches is not None and len(xh_model.past_key_caches) > 0:
        meta_info.use_cache = True
        meta_info.kv_cache_shape = xh_model.past_key_caches[0].shape
        meta_info.num_hidden_layers = len(xh_model.past_key_caches)

    del hf_model

    input_sequence_length = xh_model.wrap_cfg.input_sequence_length
    data_batch = {
        "inputs_embeds": torch.randn(1, input_sequence_length, feature_dim),
        "past_seq_length": 0,
        "generate_steps": 0,
    }
    calib_data = _run_ptq(xh_model, data_batch, cfg.target_device, exec_device, logger)

    meta_file = Path(work_dir) / "meta.json"
    xh_model.change_eval_type(EvalModelType.QUANTED_ALIGNED)
    prefill_onnx_dir = Path(cfg.work_dir) / "prefill_onnx"
    decode_onnx_dir = Path(cfg.work_dir) / "decode_onnx"
    prefill_onnx_dir.mkdir(exist_ok=True, parents=True)
    decode_onnx_dir.mkdir(exist_ok=True, parents=True)
    prefill_onnx_file = xhmodel_export_onnx(
        xh_model,
        data_batch,
        str(prefill_onnx_dir),
        f"{cfg_name}_prefill",
        logger,
    )

    if args.dump_golden:
        prefill_golden_dir = Path(cfg.work_dir) / "prefill_golden"
        export_golden(
            prefill_onnx_file,
            tuple(calib_data),
            prefill_golden_dir,
            exec_device,
            logger,
        )

    meta_info.prefill_onnx_file = str(Path(prefill_onnx_file).relative_to(cfg.work_dir))
    xh_model.release_exported_model()
    logger.info(f"Prefill model saved to {prefill_onnx_file}")

    xh_model.change_eval_type(EvalModelType.QUANTED_ALIGNED)
    xh_model.to(dtype)
    data_batch = {
        "inputs_embeds": torch.randn(1, 1, feature_dim),
        "past_seq_length": 0,
        "generate_steps": 0,
    }

    torch.cuda.empty_cache()
    xh_model.set_input_sequence_length(1)
    xh_model.to("cpu")
    decode_onnx_file = xhmodel_export_onnx(
        xh_model,
        data_batch,
        str(decode_onnx_dir),
        f"{cfg_name}_decode",
        logger,
    )

    decode_calib_inputs = _flatten_inputs(xh_model.prepare_inputs(data_batch))

    if args.dump_golden:
        decode_golden_dir = Path(cfg.work_dir) / "decode_golden"
        export_golden(
            decode_onnx_file,
            tuple(decode_calib_inputs),
            decode_golden_dir,
            exec_device,
            logger,
        )

    meta_info.decode_onnx_file = str(Path(decode_onnx_file).relative_to(cfg.work_dir))
    xh_model.release_exported_model()
    logger.info(f"Decode model saved to {decode_onnx_file}")
    with open(meta_file, "w") as f:
        json.dump(meta_info, f, indent=4)

    if args.output_dir:
        Path(args.output_dir).mkdir(exist_ok=True, parents=True)
        copy_embedding_file(
            Path(cfg.work_dir) / "token_embedding.pt",
            args.output_dir,
            "quant_embedding_code_predictor.pt",
            logger,
        )
        prefill_filename = generate_hmquant_filename(
            model_name=args.model_name,
            model_size=args.model_size,
            quant_type=args.quant_type,
            sub_model_name="code_predictor",
            phase="prefill",
        )
        decode_filename = generate_hmquant_filename(
            model_name=args.model_name,
            model_size=args.model_size,
            quant_type=args.quant_type,
            sub_model_name="code_predictor",
            phase="decode",
        )
        copy_onnx_model(
            Path(cfg.work_dir) / "prefill_onnx",
            args.output_dir,
            "code_predictor_prefill",
            prefill_filename,
            logger,
        )
        copy_onnx_model(
            Path(cfg.work_dir) / "decode_onnx",
            args.output_dir,
            "code_predictor_decode",
            decode_filename,
            logger,
        )
        if args.dump_golden:
            copy_golden_files(
                Path(cfg.work_dir) / "prefill_golden",
                args.output_dir,
                "code_predictor_prefill/step_0",
                logger,
            )
            copy_golden_files(
                Path(cfg.work_dir) / "decode_golden",
                args.output_dir,
                "code_predictor_decode/step_0",
                logger,
            )


def export_text_projection(args: argparse.Namespace) -> None:
    """Export text projection: HF model -> torch.onnx.export -> HMONNX (no PTQ needed)."""
    cfg = Config(
        dict(
            model=dict(hf_model=args.model_dir),
            target_device="XH2a",
            hf_model_dir=args.model_dir,
        )
    )
    cfg_name = "qwen3_tts_text_projection"
    logger = _init_export_env(cfg, cfg_name, args)

    device = cfg.exec_device
    dtype = getattr(torch, cfg.dtype)
    work_dir = cfg.work_dir
    model_dir = cfg.hf_model_dir
    hf_model = Qwen3TTSModel.from_pretrained(
        model_dir,
        device_map="cuda",
        dtype=torch.float32,
        attn_implementation="sdpa",
    )
    hf_model = cast(Qwen3TTSModel, hf_model)
    config_file = cfg.config_file
    meta_info = ConfigDict(
        dict(
            create_time=time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()),
            config=str(config_file.relative_to(cfg.work_dir)),
        )
    )
    target_device = cfg.target_device
    meta_info["model_name"] = cfg_name
    meta_info["act"] = type(hf_model.model.talker.text_projection.act_fn).__name__
    meta_info["target_device"] = target_device
    hf_model_dir = cfg.hf_model_dir
    meta_info.hf_model = hf_model_dir

    meta_file = Path(work_dir) / "meta.json"
    with open(meta_file, "w") as f:
        json.dump(meta_info, f, indent=4)

    feature_dim = 2048

    def _hook(module, inputs):
        nonlocal feature_dim
        feature_dim = inputs[0].shape[-1]
        return inputs

    text_projection_hook = (
        hf_model.model.talker.text_projection.register_forward_pre_hook(_hook)
    )

    wavs, sr = _run_generate(hf_model, args)
    out_file = Path(work_dir) / "output_tts.wav"
    sf.write(out_file, wavs[0], sr)
    logger.info(f"Audio saved to {out_file}")

    text_projection_hook.remove()
    example_input = torch.randn(1, 1, feature_dim, device=device, dtype=dtype)
    onnx_dir = Path(work_dir) / "onnx"
    onnx_dir.mkdir(exist_ok=True, parents=True)
    text_projection_onnx_file = Path(work_dir) / "onnx" / "text_projection.onnx"
    torch.onnx.export(
        hf_model.model.talker.text_projection.float().cpu(),
        (example_input.float().cpu(),),
        text_projection_onnx_file,
        input_names=["inputs_embeds"],
        output_names=["outputs"],
    )

    hmonnx_dir = Path(work_dir) / "hmonnx"
    hmonnx_dir.mkdir(exist_ok=True, parents=True)
    text_projection_hmonnx_file = str(
        hmonnx_dir / f"text_projection_{target_device}.onnx"
    )
    convert_onnx_to_hmonnx(
        text_projection_onnx_file,
        [example_input.float().cpu()],
        target_device,
        text_projection_hmonnx_file,
    )
    logger.info(
        f"Text projection exported: ONNX -> {text_projection_onnx_file}, HMONNX -> {text_projection_hmonnx_file}"
    )

    if args.dump_golden:
        golden_dir = Path(work_dir) / "golden"
        export_golden(
            text_projection_hmonnx_file,
            (example_input.float().cpu(),),
            golden_dir,
            device,
            logger,
        )

    meta_info["hmonnx"] = str(
        Path(text_projection_hmonnx_file).relative_to(Path(meta_file).parent)
    )
    with open(meta_file, "w") as f:
        json.dump(meta_info, f, indent=4)

    if args.output_dir:
        Path(args.output_dir).mkdir(exist_ok=True, parents=True)
        hmonnx_filename = generate_hmquant_filename(
            model_name=args.model_name,
            model_size=args.model_size,
            quant_type=args.quant_type,
            sub_model_name="text_projection",
        )
        copy_hmonnx_model(
            Path(cfg.work_dir) / "hmonnx",
            args.output_dir,
            "text_projection",
            hmonnx_filename,
            logger,
        )
        if args.dump_golden:
            copy_golden_files(
                Path(cfg.work_dir) / "golden",
                args.output_dir,
                "step_0",
                logger,
            )


class XHQwen3TTSTokenizerV2DecoderTransformerModel(
    Qwen3TTSTokenizerV2DecoderTransformerModel
):
    """Export-friendly transformer: pre-computes attention masks and RoPE for static ONNX export."""

    def _setup(self, chunk_size=300):
        self.chunk_size = chunk_size
        cache_position = torch.arange(0, self.chunk_size)
        attention_mask = None
        inputs_embeds = None
        past_key_values = None
        position_ids = cache_position.unsqueeze(0)
        inputs_embeds = torch.randn(1, self.chunk_size, self.config.hidden_size)
        mask_kwargs = {
            "config": self.config,
            "input_embeds": inputs_embeds,
            "attention_mask": attention_mask,
            "cache_position": cache_position,
            "past_key_values": past_key_values,
            "position_ids": position_ids,
        }
        full_attention = create_causal_mask(**mask_kwargs)
        self.register_buffer("full_attention", full_attention, persistent=False)

        if self.has_sliding_layers:
            sliding_attention = create_sliding_window_causal_mask(**mask_kwargs)
            self.register_buffer(
                "sliding_attention", sliding_attention, persistent=False
            )

        hidden_states = torch.randn(
            1, self.chunk_size, self.config.hidden_size, dtype=torch.float16
        )
        if torch.cuda.is_available():
            hidden_states = hidden_states.cuda()
        cos, sin = self.rotary_emb(hidden_states, position_ids.to(hidden_states.device))
        cos = cos.cpu()
        sin = sin.cpu()
        self.register_buffer("cos_cached", cos, persistent=False)
        self.register_buffer("sin_cached", sin, persistent=False)

    def forward(
        self,
        inputs_embeds=None,
    ) -> BaseModelOutputWithPast:

        inputs_embeds = self.input_proj(inputs_embeds)
        cache_position = None
        position_ids = None

        if cache_position is None:
            cache_position = torch.arange(
                0, inputs_embeds.shape[1], device=inputs_embeds.device
            )

        if position_ids is None:
            position_ids = cache_position.unsqueeze(0)

        causal_mask_mapping = {
            "full_attention": self.full_attention,
            "sliding_attention": (
                self.sliding_attention if self.has_sliding_layers else None
            ),
        }
        hidden_states = inputs_embeds

        position_embeddings = (self.cos_cached, self.sin_cached)
        for decoder_layer in self.layers[: self.config.num_hidden_layers]:
            hidden_states = decoder_layer(
                hidden_states,
                attention_mask=causal_mask_mapping[decoder_layer.attention_type],
                position_ids=position_ids,
                cache_position=cache_position,
                position_embeddings=position_embeddings,
            )

        hidden_states = self.norm(hidden_states)
        hidden_states = self.output_proj(hidden_states)
        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
        )


def export_speech_tokenizer(args: argparse.Namespace) -> None:
    """Export speech tokenizer (codes -> wav): HF model -> torch.onnx.export -> HMONNX."""
    cfg = Config(
        dict(
            model=dict(hf_model=args.model_dir),
            target_device="XH2a",
            hf_model_dir=args.model_dir,
        )
    )
    cfg_name = "qwen3_tts_speech_tokenizer"
    logger = _init_export_env(cfg, cfg_name, args)

    device = cfg.exec_device
    dtype = getattr(torch, cfg.dtype)
    work_dir = cfg.work_dir
    model_dir = cfg.hf_model_dir
    hf_model = Qwen3TTSModel.from_pretrained(
        model_dir,
        device_map="cuda",
        dtype=torch.float32,
        attn_implementation="sdpa",
    )
    hf_model = cast(Qwen3TTSModel, hf_model)
    config_file = cfg.config_file
    meta_info = ConfigDict(
        dict(
            create_time=time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()),
            config=str(config_file.relative_to(cfg.work_dir)),
        )
    )
    target_device = cfg.target_device
    meta_info["model_name"] = cfg_name
    meta_info["target_device"] = target_device
    hf_model_dir = cfg.hf_model_dir
    meta_info.hf_model = hf_model_dir

    meta_file = Path(work_dir) / "meta.json"
    with open(meta_file, "w") as f:
        json.dump(meta_info, f, indent=4)

    input_shape = None
    input_dtype = None

    def _hook(module, inputs):
        nonlocal input_shape
        nonlocal input_dtype
        input_shape = list(inputs[0].shape)
        input_dtype = inputs[0].dtype
        return inputs

    decode_hook = (
        hf_model.model.speech_tokenizer.model.decoder.register_forward_pre_hook(_hook)
    )
    pre_transformer = hf_model.model.speech_tokenizer.model.decoder.pre_transformer
    pre_transformer.config._attn_implementation = "eager"
    wavs, sr = _run_generate(hf_model, args)
    out_file = Path(work_dir) / "output_tts.wav"
    sf.write(out_file, wavs[0], sr)
    logger.info(f"Audio saved to {out_file}")

    decode_hook.remove()

    chunk_size = 300

    gt_shapes = {}
    for seq_len_in in tqdm(range(1, 301), desc="decoder padding shape validation"):
        dummy_input_shape = list(input_shape)
        dummy_input_shape[-1] = seq_len_in
        dummy_input = torch.randint(
            0, 100, dummy_input_shape, device=device, dtype=input_dtype
        )

        wav_out = hf_model.model.speech_tokenizer.model.decoder(dummy_input)
        actual_output_shape = list(wav_out.shape)
        input_shape_str = f"{'_'.join(map(str, dummy_input_shape))}"
        gt_shapes[input_shape_str] = actual_output_shape

        del dummy_input, wav_out
        torch.cuda.empty_cache()
    json.dump(
        gt_shapes,
        open(str(Path(work_dir) / "decode_padding_shapes.json"), "w"),
        indent=2,
    )
    meta_info["decode_padding_shapes"] = str(Path("decode_padding_shapes.json"))

    input_shape[-1] = chunk_size
    example_input = torch.randint(0, 100, input_shape, device=device, dtype=input_dtype)
    onnx_dir = Path(work_dir) / "onnx"
    onnx_dir.mkdir(exist_ok=True, parents=True)

    speech_tokenizer_onnx_file = str(Path(work_dir) / "onnx" / "speech_tokenizer.onnx")
    pre_transformer = hf_model.model.speech_tokenizer.model.decoder.pre_transformer
    pre_transformer.__class__ = XHQwen3TTSTokenizerV2DecoderTransformerModel
    pre_transformer = cast(
        XHQwen3TTSTokenizerV2DecoderTransformerModel, pre_transformer
    )
    pre_transformer._setup()
    torch.onnx.export(
        hf_model.model.speech_tokenizer.model.decoder.float().cpu(),
        example_input.to(torch.int32).cpu(),
        speech_tokenizer_onnx_file,
        input_names=["codes"],
        output_names=["wav"],
        dynamo=True,
    )

    meta_file = Path(work_dir) / "meta.json"
    with open(meta_file, "w") as f:
        json.dump(meta_info, f, indent=4)

    logger.info(f"speech_tokenizer exported to {speech_tokenizer_onnx_file}")

    # Convert unsupported ops (SplitToSequence/SequenceAt) for HMONNX compatibility
    onnx_model = onnx.load(speech_tokenizer_onnx_file)
    hmonnx_transforms(onnx_model)
    onnx.save(onnx_model, speech_tokenizer_onnx_file)

    hmonnx_dir = Path(work_dir) / "hmonnx"
    hmonnx_dir.mkdir(exist_ok=True, parents=True)
    speech_tokenizer_hmonnx_file = str(
        hmonnx_dir / f"speech_tokenizer_{target_device}.onnx"
    )
    convert_onnx_to_hmonnx(
        speech_tokenizer_onnx_file,
        [example_input.to(torch.int32).cpu()],
        target_device,
        speech_tokenizer_hmonnx_file,
    )
    logger.info(f"speech_tokenizer HMONNX saved to {speech_tokenizer_hmonnx_file}")

    if args.dump_golden:
        golden_dir = Path(work_dir) / "golden"
        export_golden(
            speech_tokenizer_hmonnx_file,
            (example_input.to(torch.int32).cpu(),),
            golden_dir,
            device,
            logger,
        )

    meta_info["hmonnx"] = str(Path("hmonnx") / f"speech_tokenizer_{target_device}.onnx")
    with open(meta_file, "w") as f:
        json.dump(meta_info, f, indent=4)

    if args.output_dir:
        Path(args.output_dir).mkdir(exist_ok=True, parents=True)
        hmonnx_filename = generate_hmquant_filename(
            model_name=args.model_name,
            model_size=args.model_size,
            quant_type=args.quant_type,
            sub_model_name="speech_tokenizer",
        )
        copy_hmonnx_model(
            Path(cfg.work_dir) / "hmonnx",
            args.output_dir,
            "speech_tokenizer",
            hmonnx_filename,
            logger,
        )
        decode_padding_shapes_src = Path(cfg.work_dir) / "decode_padding_shapes.json"
        if decode_padding_shapes_src.exists():
            dst = Path(args.output_dir) / "decode_padding_shapes.json"
            shutil.copy2(decode_padding_shapes_src, dst)
            logger.info(f"Copied decode_padding_shapes.json -> {dst}")
        if args.dump_golden:
            copy_golden_files(
                Path(cfg.work_dir) / "golden",
                args.output_dir,
                "step_0",
                logger,
            )


# ============================================================
# Stateful Decoder Export (for streaming inference)
# ============================================================


class StatefulDecoderPart1PreConv(torch.nn.Module):
    def __init__(self, decoder_model, chunk_size: int):
        super().__init__()
        self.quantizer = decoder_model.quantizer
        self.pre_conv = decoder_model.pre_conv
        self.chunk_size = int(chunk_size)
        self.pre_conv_history_window = 2

    def forward(self, audio_codes: torch.Tensor, pre_conv_history: torch.Tensor):
        codes = audio_codes.to(torch.long).transpose(1, 2)
        quantized = self.quantizer.decode(codes)
        quant_full = torch.cat([pre_conv_history, quantized], dim=-1)
        hidden_all = self.pre_conv(quant_full)
        hidden = hidden_all[:, :, -self.chunk_size :].transpose(1, 2)
        next_pre_conv_hist = quant_full[:, :, -self.pre_conv_history_window :]
        return hidden, next_pre_conv_hist


class StatefulDecoderFixedKVStack:
    def __init__(self, keys, values, window_size: int):
        self.key_cache = keys
        self.value_cache = values
        self.window_size = int(window_size)

    def update(self, key_states, value_states, layer_idx, cache_kwargs=None):
        k_combined = torch.cat([self.key_cache[layer_idx], key_states], dim=2)
        v_combined = torch.cat([self.value_cache[layer_idx], value_states], dim=2)
        self.key_cache[layer_idx] = k_combined[:, :, -self.window_size :, :]
        self.value_cache[layer_idx] = v_combined[:, :, -self.window_size :, :]
        return self.key_cache[layer_idx], self.value_cache[layer_idx]

    def get_seq_length(self, layer_idx=0):
        return self.window_size

    def __len__(self):
        return len(self.key_cache)


class StatefulDecoderPart2Transformer(torch.nn.Module):
    def __init__(self, decoder_model, chunk_size: int):
        super().__init__()
        self.trans = decoder_model.pre_transformer
        self.num_layers = int(self.trans.config.num_hidden_layers)
        self.window_size = int(getattr(self.trans.config, "sliding_window", 72) or 72)
        self.chunk_size = int(chunk_size)

    def forward(self, hidden, kv_valid_len: torch.Tensor, *past_kv_flat):
        device = hidden.device
        keys_in = list(past_kv_flat[: self.num_layers])
        values_in = list(past_kv_flat[self.num_layers :])
        kv_stack = StatefulDecoderFixedKVStack(keys_in, values_in, self.window_size)

        past_len = kv_valid_len.to(torch.long).reshape(1)
        hidden = self.trans.input_proj(hidden)
        frame_idx = torch.arange(self.chunk_size, device=device, dtype=torch.long)
        position_ids = (past_len + frame_idx).unsqueeze(0)
        position_embeddings = self.trans.rotary_emb(hidden, position_ids)

        query_pos = (past_len + frame_idx).unsqueeze(1)
        key_idx = torch.arange(
            self.window_size, device=device, dtype=torch.long
        ).unsqueeze(0)
        key_pos = past_len + self.chunk_size - self.window_size + key_idx
        mask_cond = (
            (key_pos >= 0)
            & (key_pos <= query_pos)
            & (key_pos > query_pos - self.window_size)
        )
        attention_mask = torch.where(mask_cond, 0.0, -10000.0).unsqueeze(0).unsqueeze(0)

        for layer in self.trans.layers:
            layer_out = layer(
                hidden,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_values=kv_stack,
                use_cache=True,
                position_embeddings=position_embeddings,
            )
            hidden = layer_out[0] if isinstance(layer_out, (tuple, list)) else layer_out

        hidden = self.trans.norm(hidden)
        new_hidden = self.trans.output_proj(hidden).transpose(1, 2)
        return (new_hidden,) + tuple(kv_stack.key_cache) + tuple(kv_stack.value_cache)


class StatefulDecoderPart3Upsample(torch.nn.Module):
    def __init__(self, decoder_model, chunk_size: int):
        super().__init__()
        self.upsample = decoder_model.upsample
        self.decoder = decoder_model.decoder
        self.samples_per_frame = int(decoder_model.total_upsample)
        self.lookahead_frames = 4
        self.conv_history_window = 4
        self.chunk_size = int(chunk_size)

    def forward(
        self,
        new_hidden: torch.Tensor,
        latent_buffer: torch.Tensor,
        conv_history: torch.Tensor,
        is_last: torch.Tensor,
        kv_valid_len: torch.Tensor,
        valid_frames: torch.Tensor,
    ):
        device = new_hidden.device
        accumulated = torch.cat([latent_buffer, new_hidden], dim=-1)

        has_history = (kv_valid_len.to(torch.float32) > 0).to(torch.float32).view(1)
        latent_valid = has_history * float(self.lookahead_frames)
        valid_frames_f = valid_frames.to(torch.float32).view(1)
        total_valid = latent_valid + valid_frames_f
        lookahead = torch.full_like(total_valid, float(self.lookahead_frames))
        is_last_f = is_last.to(torch.float32).view(1)
        num_finalize_f = is_last_f * total_valid + (1.0 - is_last_f) * torch.clamp(
            total_valid - lookahead, min=0.0
        )
        num_finalize = num_finalize_f.to(torch.long)
        num_finalize_idx = num_finalize[0]

        curr = torch.cat([conv_history, accumulated], dim=-1)
        for blocks in self.upsample:
            for block in blocks:
                curr = block(curr)
        for block in self.decoder:
            curr = block(curr)
        wav = curr.squeeze(1).clamp(min=-1, max=1)

        start_samples_idx = self.conv_history_window * self.samples_per_frame
        valid_samples = (num_finalize * self.samples_per_frame).view(1)
        final_wav = wav[:, start_samples_idx:]

        next_latent_buf = accumulated[:, :, -self.lookahead_frames :]
        batch, channels = accumulated.size(0), accumulated.size(1)
        indices = torch.arange(
            self.conv_history_window, device=device, dtype=torch.long
        )
        latent_pad = (1.0 - has_history).to(torch.long).view(1) * self.lookahead_frames
        target_indices = (
            latent_pad + (num_finalize_idx - self.conv_history_window) + indices
        )
        gather_indices = (
            torch.clamp(target_indices, min=0)
            .unsqueeze(0)
            .unsqueeze(0)
            .expand(batch, channels, -1)
        )
        next_conv_hist = torch.gather(accumulated, 2, gather_indices)
        return final_wav, valid_samples, next_latent_buf, next_conv_hist


class StatefulDecoderCombined(torch.nn.Module):
    def __init__(self, decoder_model, chunk_size: int = 12):
        super().__init__()
        self.chunk_size = int(chunk_size)
        self.part1 = StatefulDecoderPart1PreConv(decoder_model, self.chunk_size)
        self.part2 = StatefulDecoderPart2Transformer(decoder_model, self.chunk_size)
        self.part3 = StatefulDecoderPart3Upsample(decoder_model, self.chunk_size)
        self.num_layers = self.part2.num_layers
        self.kv_cache_window = self.part2.window_size
        self.samples_per_frame = self.part3.samples_per_frame

    def forward(
        self,
        audio_codes: torch.Tensor,
        pre_conv_history: torch.Tensor,
        latent_buffer: torch.Tensor,
        conv_history: torch.Tensor,
        is_last: torch.Tensor,
        kv_valid_len: torch.Tensor,
        valid_frames: torch.Tensor,
        *past_kv_flat,
    ):
        hidden, next_pre_conv_hist = self.part1(audio_codes, pre_conv_history)
        trans_outputs = self.part2(hidden, kv_valid_len, *past_kv_flat)
        new_hidden = trans_outputs[0]
        next_kv_flat = trans_outputs[1:]
        final_wav, valid_samples, next_latent_buf, next_conv_hist = self.part3(
            new_hidden, latent_buffer, conv_history, is_last, kv_valid_len, valid_frames
        )
        return (
            final_wav,
            valid_samples,
            next_pre_conv_hist,
            next_latent_buf,
            next_conv_hist,
            *next_kv_flat,
        )


def export_stateful_decoder(args: argparse.Namespace) -> None:
    """导出 stateful decoder 用于流式推理"""
    cfg_name = "qwen3_tts_stateful_decoder"
    chunk_size = getattr(args, "stateful_decoder_chunk_size", 12)
    head_dim = 64

    work_dir = Path(args.work_dir) / cfg_name
    work_dir.mkdir(exist_ok=True, parents=True)
    log_file = work_dir / f"{cfg_name}_debug.log"

    target_device = "XH2a"
    exec_device = "cuda" if torch.cuda.is_available() else "cpu"

    set_random_seed(args.seed)
    xhquant_llm_init(log_file, args.debug)
    logger = get_root_logger()

    # 加载 speech tokenizer 模型
    model_dir = args.model_dir
    speech_tokenizer_dir = Path(model_dir) / "speech_tokenizer"
    tokenizer_dir = str(
        speech_tokenizer_dir if speech_tokenizer_dir.exists() else model_dir
    )

    logger.info(f"Loading Qwen3TTSTokenizerV2Model from {tokenizer_dir}")
    model = Qwen3TTSTokenizerV2Model.from_pretrained(tokenizer_dir).float().cpu().eval()

    if hasattr(model.config, "decoder_config"):
        model.config.decoder_config._attn_implementation = "eager"
        model.config.decoder_config.head_dim = head_dim
    if hasattr(model.decoder.pre_transformer, "config"):
        model.decoder.pre_transformer.config._attn_implementation = "eager"
        model.decoder.pre_transformer.config.head_dim = head_dim

    wrapper = (
        StatefulDecoderCombined(model.decoder, chunk_size=chunk_size)
        .float()
        .cpu()
        .eval()
    )
    cfg = model.decoder.config
    num_layers = int(wrapper.num_layers)
    num_heads = int(
        getattr(cfg, "num_key_value_heads", getattr(cfg, "num_attention_heads", 16))
    )

    logger.info(
        f"StatefulDecoder: num_layers={num_layers}, num_heads={num_heads}, "
        f"head_dim={head_dim}, kv_window={wrapper.kv_cache_window}, chunk_size={chunk_size}"
    )

    # Dummy inputs
    batch = 1
    audio_codes = torch.zeros(batch, chunk_size, 16, dtype=torch.int32)
    pre_conv_history = torch.zeros(batch, 512, 2, dtype=torch.float32)
    latent_buffer = torch.zeros(batch, 1024, 4, dtype=torch.float32)
    conv_history = torch.zeros(batch, 1024, 4, dtype=torch.float32)
    is_last = torch.tensor([0.0], dtype=torch.float32)
    kv_valid_len_tensor = torch.tensor([0], dtype=torch.int32)
    valid_frames_tensor = torch.tensor([chunk_size], dtype=torch.int32)
    kv = [
        torch.zeros(
            batch, num_heads, wrapper.kv_cache_window, head_dim, dtype=torch.float32
        )
        for _ in range(num_layers * 2)
    ]
    dummy_inputs = (
        audio_codes,
        pre_conv_history,
        latent_buffer,
        conv_history,
        is_last,
        kv_valid_len_tensor,
        valid_frames_tensor,
        *kv,
    )

    input_names = [
        "audio_codes",
        "pre_conv_history",
        "latent_buffer",
        "conv_history",
        "is_last",
        "kv_valid_len",
        "valid_frames",
    ]
    output_names = [
        "final_wav",
        "valid_samples",
        "next_pre_conv_history",
        "next_latent_buffer",
        "next_conv_history",
    ]
    input_names.extend([f"past_key_{i}" for i in range(num_layers)])
    input_names.extend([f"past_value_{i}" for i in range(num_layers)])
    output_names.extend([f"next_key_{i}" for i in range(num_layers)])
    output_names.extend([f"next_value_{i}" for i in range(num_layers)])

    # Export ONNX
    onnx_dir = work_dir / "onnx"
    onnx_dir.mkdir(exist_ok=True, parents=True)
    onnx_file = onnx_dir / "qwen3_tts_decoder_stateful_static.onnx"

    logger.info("Exporting stateful decoder to ONNX...")
    with torch.no_grad():
        torch.onnx.export(
            model=wrapper,
            args=dummy_inputs,
            f=str(onnx_file),
            input_names=input_names,
            output_names=output_names,
            opset_version=18,
            dynamo=True,
        )
    logger.info(f"ONNX exported to {onnx_file}")

    # Convert unsupported ops for HMONNX compatibility
    onnx_model = onnx.load(str(onnx_file))
    hmonnx_transforms(onnx_model)
    onnx_model = onnx.load(str(onnx_file))

    # Convert to HMONNX
    hmonnx_dir = work_dir / "hmonnx"
    hmonnx_dir.mkdir(exist_ok=True, parents=True)
    hmonnx_file = hmonnx_dir / f"stateful_decoder_{target_device}.onnx"
    convert_onnx_to_hmonnx(
        str(onnx_file),
        [x.cpu() for x in dummy_inputs],
        target_device,
        str(hmonnx_file),
    )
    logger.info(f"HMONNX saved to {hmonnx_file}")

    # Fix name conflicts between initializers and node outputs
    hmonnx_model = onnx.load(str(hmonnx_file))
    init_names = {i.name for i in hmonnx_model.graph.initializer}
    nodes = list(hmonnx_model.graph.node)
    for conflict_idx, node in enumerate(nodes):
        for out_name in node.output:
            if out_name in init_names:
                new_name = f"{out_name}_const"
                for init in hmonnx_model.graph.initializer:
                    if init.name == out_name:
                        init.name = new_name
                        break
                for i, n in enumerate(nodes):
                    if i >= conflict_idx:
                        break
                    for j, inp in enumerate(n.input):
                        if inp == out_name:
                            n.input[j] = new_name
    onnx.save(hmonnx_model, str(hmonnx_file))

    if args.dump_golden:
        golden_dir = work_dir / "golden"
        export_golden(
            str(hmonnx_file),
            tuple(x.cpu() for x in dummy_inputs),
            golden_dir,
            exec_device,
            logger,
        )

    # Save meta.json
    meta = {
        "create_time": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()),
        "hf_model": model_dir,
        "tokenizer_dir": tokenizer_dir,
        "target_device": target_device,
        "stateful_onnx": str(onnx_file.relative_to(work_dir)),
        "stateful_hmonnx": str(hmonnx_file.relative_to(work_dir)),
        "stateful_static_buffers": True,
        "stateful_num_layers": num_layers,
        "stateful_num_heads": num_heads,
        "stateful_head_dim": head_dim,
        "stateful_kv_cache_window": int(wrapper.kv_cache_window),
        "stateful_chunk_size": chunk_size,
        "stateful_samples_per_frame": int(wrapper.samples_per_frame),
        "stateful_initial_output_skip_frames": int(wrapper.part3.lookahead_frames),
        "input_names": input_names,
        "output_names": output_names,
    }
    meta_file = work_dir / "meta.json"
    with open(meta_file, "w") as f:
        json.dump(meta, f, indent=4)
    logger.info(f"Meta saved to {meta_file}")

    if args.output_dir:
        Path(args.output_dir).mkdir(exist_ok=True, parents=True)
        hmonnx_filename = generate_hmquant_filename(
            model_name=args.model_name,
            model_size=args.model_size,
            quant_type=args.quant_type,
            sub_model_name="stateful_decoder",
        )
        copy_hmonnx_model(
            hmonnx_dir,
            args.output_dir,
            "stateful_decoder",
            hmonnx_filename,
            logger,
        )
        if args.dump_golden:
            copy_golden_files(
                work_dir / "golden",
                args.output_dir,
                "stateful_decoder/step_0",
                logger,
            )


def get_args() -> argparse.Namespace:
    """Parse commandline."""
    # fmt: off
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--config", dest="config_path", type=str, default=DEFAULT_CONFIG_PATH, help="path to config.yaml")
    parser.add_argument("--model_dir", type=str, default=None, help="HF model dir (auto-resolved from model_name/model_size if not set)")
    parser.add_argument("--work_dir", type=str, default="./work_dirs", help="Base working directory (model tag appended automatically)")
    parser.add_argument("--output_dir", type=str, default=os.path.join("output", HOUMO_TARGET, "hmquant"), help="输出目录，用于存放最终的量化产物。结构类似 output/xh2/hmquant")
    parser.add_argument("--model_name", type=str, default=None, help="模型名称")
    parser.add_argument("--model_size", type=str, default=None, help="模型大小")
    parser.add_argument("--batch", type=int, default=None, help="batch size")
    parser.add_argument("--quant_type", type=str, default=None, help="量化类型，如 w8a16h1_sefp")
    parser.add_argument("--max_sequence_length", type=int, default=2048, help="最大序列长度")
    parser.add_argument("--input_sequence_length", type=int, default=256, help="prefill_length")
    parser.add_argument("--code_predictor_max_sequence_length", type=int, default=18)
    parser.add_argument("--code_predictor_input_sequence_length", type=int, default=2)
    parser.add_argument("--stateful_decoder_chunk_size", type=int, default=12, help="stateful decoder chunk size (frames)")
    parser.add_argument("--frontend_audio_samples", type=int, default=101760, help="speech_tokenizer_encoder export audio samples")
    parser.add_argument("--frontend_mel_frames", type=int, default=400, help="speaker_encoder export mel frames")
    parser.add_argument("--frontend_mel_dim", type=int, default=128, help="speaker_encoder export mel dim")
    parser.add_argument("--frontend_opset", type=int, default=18, help="ONNX opset for base-only frontend models")
    parser.add_argument("--ref_audio", type=str, default=f"{HOUMO_EXAMPLES_PATH}/data/audio/clone_1.wav", help="Reference audio for base model voice clone")
    parser.add_argument("--ref_text", type=str, default="甚至出现交易几乎停滞的情况。", help="Reference text for base model voice clone")
    parser.add_argument("--models", type=str, default="all",
                        help=f"指定要量化的模型，支持逗号分隔的多个模型名称: {SUPPORTED_MODELS}，或 'all' 表示全部模型")
    parser.add_argument("--dump_golden", action="store_true", help="是否生成 golden 数据用于验证量化模型推理结果")
    parser.add_argument("--debug", action="store_true", help="debug mode")
    parser.add_argument("--seed", type=int, default=1024)
    # fmt: on
    args = parser.parse_args()

    # 从 config.yaml 读取默认配置
    default_model_size, default_model_name, model_configs = get_model_configs(
        args.config_path
    )
    args.model_name = first_not_none(args.model_name, default_model_name)
    args.model_size = first_not_none(args.model_size, default_model_size)
    model_config = model_configs.get(args.model_name, {}).get(args.model_size, {})
    if not model_config:
        available_sizes = sorted(model_configs.get(args.model_name, {}).keys())
        raise ValueError(
            f"Unsupported model config: model_name={args.model_name}, "
            f"model_size={args.model_size}. Available model_size values: {available_sizes}"
        )

    args.model_dir = first_not_none(args.model_dir, get_default_model_dir(model_config))
    safe_model_tag = f"{args.model_name}_{args.model_size}".replace("/", "_")
    args.work_dir = f"{args.work_dir}_{safe_model_tag}"

    args.quant_type = first_not_none(
        args.quant_type, model_config.get("quant_type", "w8a16h1_sefp")
    )
    args.batch = first_not_none(args.batch, model_config.get("batch", 1))
    return args


if __name__ == "__main__":
    args = get_args()

    # 解析要量化的模型列表
    requested_all = args.models.lower() == "all"
    models_to_export = parse_models_arg(args.models)
    if not is_base_model(args.model_name, args.model_size) and requested_all:
        models_to_export = [m for m in models_to_export if m not in BASE_ONLY_MODELS]

    # 根据配置选择性地导出模型
    if "speech_tokenizer_encoder" in models_to_export:
        export_speech_tokenizer_encoder(args)
    if "speaker_encoder" in models_to_export:
        export_speaker_encoder(args)
    if "code_predictor" in models_to_export:
        export_code_predictor(args)
    if "talker" in models_to_export:
        export_talker(args)
    if "speech_tokenizer" in models_to_export:
        export_speech_tokenizer(args)
    if "text_projection" in models_to_export:
        export_text_projection(args)
    if "stateful_decoder" in models_to_export:
        export_stateful_decoder(args)

    # 非 debug 模式下清理 work_dir
    if not args.debug:
        work_dir = Path(args.work_dir)
        if work_dir.exists():
            shutil.rmtree(work_dir)
            print(f"Cleaned up work_dir: {work_dir}")
