# Copyright (c) 2025 HOUMO AI
#
# File: demo_base.py
# Description:
#   Qwen3-TTS-0.6B-Base tcim_lite inference demo (voice clone mode).
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
import math
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Generator, List, Tuple, Union
import logging

logging.getLogger("qwen_tts").setLevel(logging.ERROR)

import transformers

transformers.logging.set_verbosity_error()

import numpy as np
import soundfile as sf
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio
from tqdm import tqdm
import transformers

transformers.logging.set_verbosity_error()

from transformers.generation.logits_process import (
    LogitsProcessorList,
    TemperatureLogitsWarper,
    TopKLogitsWarper,
    TopPLogitsWarper,
    RepetitionPenaltyLogitsProcessor,
    MinNewTokensLengthLogitsProcessor,
    SuppressTokensLogitsProcessor,
)

import tcim_lite as tcim
from hmatc.python.get_hm_devices import get_hm_devices
from hmatc.utils.utils import first_not_none, get_model_configs
from loguru import logger
from perf_utils import (
    PerfKey,
    PerfTracker,
    log_base_oneshot_perf,
    log_base_streaming_perf,
)
from qwen3_tts_runtime import (
    CodeChunkBuffer,
    Qwen3TTSCodePredictorInference,
    Qwen3TTSSpeechTokenizerInference,
    Qwen3TTSStatefulDecoderInference,
    Qwen3TTSTalkerInference,
    Qwen3TTSTextProjectionInference,
    StatefulDecoderState,
    StreamingPlaybackGapTracker,
    build_assistant_text,
    elapsed_ms,
    elapsed_s,
    get_default_hf_model_dir,
    get_hmm_path,
    get_module_input_names,
    get_module_output_names,
    pad_or_trim_mels,
    to_numpy,
    to_torch_dtype,
    tokenize_texts,
)

HOUMO_TARGET = os.getenv("HOUMO_TARGET")
assert HOUMO_TARGET in ["xh2"], f"Unsupported HOUMO_TARGET: {HOUMO_TARGET}"

HOUMO_EXAMPLES_PATH = os.getenv("HOUMO_EXAMPLES_PATH", os.path.abspath("../../../"))
DEFAULT_CONFIG_PATH = os.path.join(os.path.dirname(__file__), "config.yaml")


class Qwen3TTSSpeechTokenizerEncoderInference(nn.Module):
    """Qwen3 TTS Speech Tokenizer Encoder 推理，用于 Base voice clone 参考音频编码"""

    def __init__(self, hmm_file: str, ndevice: int = 1) -> None:
        super().__init__()
        self.ndevice = ndevice

        dev_manager = tcim.runtime.DevManager(get_hm_devices(ndevice), "Xh2HalBackend")
        weight_manager = tcim.runtime.WeightManager(dev_manager)
        option = tcim.runtime.Option(weight_manager)

        self.model = tcim.runtime.load(hmm_file, option=option)
        self.input_names = get_module_input_names(self.model)
        self.output_names = get_module_output_names(self.model)

        input_info = self.model.get_input_info(self.input_names[0])
        mask_info = self.model.get_input_info(self.input_names[1])
        self.net_input_samples = int(input_info.shape[1])
        self.input_dtype = input_info.dtype
        self.mask_dtype = mask_info.dtype
        self.sample_rate = 24000

        logger.info("SpeechTokenizerEncoder loaded")

    def _prepare_chunk(self, wav_chunk: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        valid_samples = int(wav_chunk.shape[0])
        if valid_samples > self.net_input_samples:
            raise ValueError(
                f"wav chunk length {valid_samples} exceeds HMM input length "
                f"{self.net_input_samples}"
            )

        input_values = np.zeros((1, self.net_input_samples), dtype=self.input_dtype)
        padding_mask = np.zeros((1, self.net_input_samples), dtype=self.mask_dtype)
        if valid_samples > 0:
            input_values[0, :valid_samples] = wav_chunk.astype(
                self.input_dtype, copy=False
            )
            padding_mask[0, :valid_samples] = 1
        return input_values, padding_mask

    def encode(self, audio: Union[np.ndarray, torch.Tensor], sr: int) -> torch.Tensor:
        """编码参考音频，返回 [T, num_code_groups] 的 long tensor。"""
        if int(sr) != self.sample_rate:
            raise ValueError(
                f"Speech tokenizer encoder expects {self.sample_rate}Hz audio, got {sr}. "
                "Load and resample reference audio before encoding."
            )
        if isinstance(audio, torch.Tensor):
            if audio.dim() > 1:
                audio = audio.mean(dim=0)
            audio = audio.detach().cpu().to(torch.float32).contiguous().numpy()
        else:
            if audio.ndim > 1:
                audio = audio.mean(axis=1)
            audio = audio.astype(np.float32, copy=False)

        if audio.shape[0] == 0:
            raise ValueError("Reference audio is empty after loading/resampling.")

        if audio.shape[0] > self.net_input_samples:
            logger.warning(
                f"Reference audio has {audio.shape[0]} samples; truncating to "
                f"{self.net_input_samples} for static speech_tokenizer encoder HMM"
            )

        valid_samples = min(int(audio.shape[0]), self.net_input_samples)
        input_values, padding_mask = self._prepare_chunk(audio[:valid_samples])

        self.model.set_input(self.input_names[0], input_values)
        self.model.set_input(self.input_names[1], padding_mask)
        self.model.run()
        self.model.sync()

        audio_codes = self.model.get_dev_output(self.output_names[0]).to_host().numpy()
        valid_frames = self.model.get_dev_output(self.output_names[1]).to_host().numpy()

        hmonnx_valid_frames = int(valid_frames.reshape(-1)[0])
        used_valid_frames = min(int(audio_codes.shape[1]), hmonnx_valid_frames)

        if used_valid_frames <= 0:
            raise ValueError("Speech tokenizer encoder produced no valid frames.")
        return torch.from_numpy(audio_codes[0, :used_valid_frames]).to(torch.long)


def build_ref_text(text: str) -> str:
    """构建 reference text 格式"""
    return f"<|im_start|>assistant\n{text}<|im_end|>\n"


@dataclass
class BasePromptContext:
    # Talker prefill 的初始上下文，包含 role/language/speaker/ref text/ref codec 等条件。
    talker_input_embed: torch.Tensor
    # streaming-style prompt 中未放入 prefill 的文本 hidden，后续每步 decode 逐步叠加。
    trailing_text_hidden: torch.Tensor
    # 参考音频编码得到的 codec frames，oneshot 解码和 streaming decoder 预热都会使用。
    ref_code: torch.Tensor
    # 文本条件耗尽后补齐 Talker decode 输入的 pad embedding。
    tts_pad_embed: torch.Tensor


class Qwen3TTSSpeakerEncoderInference(nn.Module):
    """Qwen3 TTS Speaker Encoder 推理，用于 Base voice clone speaker embedding"""

    def __init__(self, hmm_file: str, ndevice: int = 1) -> None:
        super().__init__()
        self.ndevice = ndevice

        dev_manager = tcim.runtime.DevManager(get_hm_devices(ndevice), "Xh2HalBackend")
        weight_manager = tcim.runtime.WeightManager(dev_manager)
        option = tcim.runtime.Option(weight_manager)

        self.model = tcim.runtime.load(hmm_file, option=option)
        self.input_names = get_module_input_names(self.model)
        self.output_names = get_module_output_names(self.model)

        input_info = self.model.get_input_info(self.input_names[0])
        self.net_input_frames = int(input_info.shape[1])
        self.num_mels = int(input_info.shape[2])
        self.input_dtype = input_info.dtype
        self.input_torch_dtype = to_torch_dtype(self.input_dtype)
        self.sample_rate = 24000

        logger.info(
            "SpeakerEncoder loaded: "
            f"input_names={self.input_names}, output_names={self.output_names}, "
            f"input_frames={self.net_input_frames}, num_mels={self.num_mels}, "
            f"input_dtype={self.input_dtype}"
        )

    def prepare_mels(
        self, audio: Union[np.ndarray, torch.Tensor], sr: int
    ) -> torch.Tensor:
        """按 demo_hmonnx_base.build_prompt 的 speaker mel 逻辑构造 [1, T, 128]。"""
        if int(sr) != self.sample_rate:
            raise ValueError(
                f"Speaker encoder only supports {self.sample_rate}Hz audio, got {sr}."
            )
        if isinstance(audio, torch.Tensor):
            if audio.dim() > 1:
                audio = audio.mean(dim=0)
            audio_tensor = audio.detach().cpu().to(torch.float32).contiguous()
        else:
            if audio.ndim > 1:
                audio = audio.mean(axis=1)
            audio_tensor = torch.from_numpy(audio.astype(np.float32, copy=False))

        from qwen_tts.core.models.modeling_qwen3_tts import mel_spectrogram

        mels = mel_spectrogram(
            audio_tensor.unsqueeze(0),
            n_fft=1024,
            num_mels=self.num_mels,
            sampling_rate=self.sample_rate,
            hop_size=256,
            win_size=1024,
            fmin=0,
            fmax=12000,
        ).transpose(1, 2)
        return pad_or_trim_mels(mels, self.net_input_frames, self.num_mels)

    def forward(self, audio: Union[np.ndarray, torch.Tensor], sr: int) -> torch.Tensor:
        """从参考音频中提取 speaker embedding，返回 [1024] float32。"""
        mels = self.prepare_mels(audio, sr)
        mels = mels.to(self.input_torch_dtype)

        self.model.set_input(self.input_names[0], to_numpy(mels))
        self.model.run()
        self.model.sync()

        speaker_embedding = (
            self.model.get_dev_output(self.output_names[0]).to_host().numpy()
        )
        speaker_embedding_tensor = torch.from_numpy(speaker_embedding[0]).to(
            torch.float32
        )

        return speaker_embedding_tensor


class Qwen3TTSInference:
    """Qwen3 TTS 推理主类"""

    def __init__(
        self,
        hf_model: str,
        text_projection: dict,
        code_predictor: dict,
        talker: dict,
        speech_tokenizer: dict,
        speech_tokenizer_encoder: dict,
        speaker_encoder: dict,
        ndevice: int = 1,
    ) -> None:
        self.hf_model_dir = hf_model
        self.ndevice = ndevice

        # 加载模型组件
        self._load_model_components(hf_model)

        # 初始化推理组件
        self.text_projection: Qwen3TTSTextProjectionInference = (
            Qwen3TTSTextProjectionInference(
                hmm_file=text_projection["hmm_file"],
                ndevice=ndevice,
            )
        )

        self.code_predictor: Qwen3TTSCodePredictorInference = (
            Qwen3TTSCodePredictorInference(
                prefill_hmm=code_predictor["prefill_hmm"],
                decode_hmm=code_predictor["decode_hmm"],
                token_embedding=code_predictor["token_embedding"],
                ndevice=ndevice,
            )
        )

        self.talker = Qwen3TTSTalkerInference(
            prefill_hmm=talker["prefill_hmm"],
            decode_hmm=talker["decode_hmm"],
            token_embedding=talker["token_embedding"],
            text_embedding=talker["text_embedding"],
            ndevice=ndevice,
        )

        self.speech_tokenizer = Qwen3TTSSpeechTokenizerInference(
            hmm_file=speech_tokenizer["hmm_file"],
            decode_padding_shapes=speech_tokenizer["decode_padding_shapes"],
            ndevice=ndevice,
        )

        self.speech_tokenizer_encoder = Qwen3TTSSpeechTokenizerEncoderInference(
            hmm_file=speech_tokenizer_encoder["hmm_file"],
            ndevice=ndevice,
        )

        self.speaker_encoder = Qwen3TTSSpeakerEncoderInference(
            hmm_file=speaker_encoder["hmm_file"],
            ndevice=ndevice,
        )
        if (
            self.speaker_encoder.sample_rate
            != self.speech_tokenizer_encoder.sample_rate
        ):
            raise ValueError(
                "Reference frontend sample-rate mismatch: "
                f"speech_tokenizer_encoder={self.speech_tokenizer_encoder.sample_rate}, "
                f"speaker_encoder={self.speaker_encoder.sample_rate}"
            )
        self.speaker_encoder_sample_rate = self.speaker_encoder.sample_rate

        logger.info("Qwen3TTSInference initialized successfully")

    def _load_model_components(self, model_path: str):
        """从 HF 模型目录加载配置和 processor"""
        from transformers import AutoConfig, AutoProcessor

        # 注册 qwen3_tts 配置和模型类型
        from qwen_tts.core.models.modeling_qwen3_tts import Qwen3TTSConfig
        from qwen_tts.core.models.processing_qwen3_tts import Qwen3TTSProcessor

        AutoConfig.register("qwen3_tts", Qwen3TTSConfig)
        AutoProcessor.register(Qwen3TTSConfig, Qwen3TTSProcessor)

        # 加载配置
        config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
        self.config = config
        self.talker_config = config.talker_config

        # 加载 processor
        self.processor = AutoProcessor.from_pretrained(
            model_path, trust_remote_code=True
        )

        # 提取必要的配置信息
        self.tts_bos_token_id = config.tts_bos_token_id
        self.tts_eos_token_id = config.tts_eos_token_id
        self.tts_pad_token_id = config.tts_pad_token_id

        # Talker 配置
        self.codec_eos_token_id = self.talker_config.codec_eos_token_id
        self.codec_bos_id = self.talker_config.codec_bos_id
        self.codec_pad_id = self.talker_config.codec_pad_id
        self.codec_think_id = self.talker_config.codec_think_id
        self.codec_nothink_id = self.talker_config.codec_nothink_id
        self.codec_think_bos_id = self.talker_config.codec_think_bos_id
        self.codec_think_eos_id = self.talker_config.codec_think_eos_id
        self.codec_language_id = self.talker_config.codec_language_id
        self.num_code_groups = self.talker_config.num_code_groups
        self.speaker_encoder_sample_rate = config.speaker_encoder_config.sample_rate

    def _get_language_id(self, language: str):
        """获取语言 ID"""
        if language.lower() == "auto":
            return None
        if language.lower() not in self.codec_language_id:
            raise NotImplementedError(f"Language {language} not implemented")
        return self.codec_language_id[language.lower()]

    def _load_reference_audio(self, wav_path: str) -> Tuple[torch.Tensor, int]:
        """按 demo_hmonnx_base.build_prompt 的方式加载并重采样参考音频。"""
        wav, sr = torchaudio.load(wav_path)
        wav = wav.to(torch.float32)
        if wav.shape[0] > 1:
            wav = wav.mean(dim=0, keepdim=True)
        target_sr = self.speech_tokenizer_encoder.sample_rate
        if int(sr) != target_sr:
            wav = torchaudio.functional.resample(wav, int(sr), target_sr)
        wav = wav.squeeze(0).contiguous()

        wav_peak = float(wav.abs().max().item()) if wav.numel() > 0 else 0.0
        wav_tail_rms = (
            float(wav[-min(wav.numel(), target_sr // 2) :].pow(2).mean().sqrt().item())
            if wav.numel() > 0
            else 0.0
        )
        logger.info(
            "Reference audio frontend input: "
            f"path={wav_path}, samples={wav.numel()}, sr={target_sr}, "
            f"duration={wav.numel() / max(target_sr, 1):.4f}s, "
            f"peak={wav_peak:.6f}, tail_0p5s_rms={wav_tail_rms:.6f}"
        )
        return wav, target_sr

    def extract_speaker_embedding(
        self, audio: Union[np.ndarray, torch.Tensor], sr: int
    ) -> torch.Tensor:
        """从参考音频中提取说话人 embedding。"""
        return self.speaker_encoder(audio, sr).cpu()

    def _validate_ref_code(self, ref_code: torch.Tensor) -> None:
        """校验参考音频 codec token 是否能被 talker/code_predictor embedding 接受。"""
        if ref_code.dim() != 2:
            raise ValueError(
                f"ref_code must be 2D [T, C], got shape={tuple(ref_code.shape)}"
            )
        if ref_code.shape[1] < self.num_code_groups:
            raise ValueError(
                f"ref_code has {ref_code.shape[1]} code groups, "
                f"but talker expects {self.num_code_groups}"
            )

        ref_code = ref_code[:, : self.num_code_groups]
        embedding_limits = [self.talker.get_input_embeddings().num_embeddings]
        embedding_limits.extend(
            emb.num_embeddings for emb in self.code_predictor.token_embedding
        )
        embedding_limits = embedding_limits[: self.num_code_groups]

        per_group_min = ref_code.amin(dim=0).detach().cpu().tolist()
        per_group_max = ref_code.amax(dim=0).detach().cpu().tolist()
        logger.info(
            "Reference codec token ranges: "
            f"shape={tuple(ref_code.shape)}, dtype={ref_code.dtype}, "
            f"per_group_min={per_group_min}, per_group_max={per_group_max}, "
            f"embedding_limits={embedding_limits}, "
            f"head_frames={ref_code[: min(3, ref_code.shape[0])].detach().cpu().tolist()}, "
            f"tail_frames={ref_code[-min(3, ref_code.shape[0]) :].detach().cpu().tolist()}"
        )

        invalid_messages = []
        for group_idx, limit in enumerate(embedding_limits):
            group_values = ref_code[:, group_idx]
            invalid = (group_values < 0) | (group_values >= limit)
            if invalid.any():
                bad_values = group_values[invalid][:8].detach().cpu().tolist()
                invalid_messages.append(
                    f"group={group_idx}, limit=[0,{limit}), "
                    f"min={int(group_values.min())}, max={int(group_values.max())}, "
                    f"bad_values={bad_values}"
                )
        if invalid_messages:
            raise ValueError(
                "speech_tokenizer encoder produced codec ids outside embedding ranges: "
                + "; ".join(invalid_messages)
            )

    def encode_reference_audio(
        self, ref_audio_path: str
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """编码参考音频，返回 (ref_code, speaker_embedding)。"""
        if not Path(ref_audio_path).exists():
            raise FileNotFoundError(f"Reference audio not found: {ref_audio_path}")

        # Base 模型没有固定 speaker 条件。voice clone 的参考发音上下文来自
        # ref_code，音色条件来自 speaker_embedding。
        ref_perf = PerfTracker()

        ref_perf.start(PerfKey.PREP_REF_AUDIO_LOAD)
        wav, sr = self._load_reference_audio(ref_audio_path)
        ref_perf.stop(PerfKey.PREP_REF_AUDIO_LOAD, count=1)

        ref_perf.start(PerfKey.PREP_REF_SPEECH_TOKENIZER_ENCODE)
        ref_code = self.speech_tokenizer_encoder.encode(wav, sr=sr).cpu()
        ref_perf.stop(PerfKey.PREP_REF_SPEECH_TOKENIZER_ENCODE, count=1)

        ref_perf.start(PerfKey.PREP_REF_SPEAKER_EMBED)
        speaker_embedding = self.extract_speaker_embedding(wav, sr)
        ref_perf.stop(PerfKey.PREP_REF_SPEAKER_EMBED, count=1)
        self._last_reference_audio_perf, self._last_reference_audio_perf_count = (
            ref_perf.snapshot()
        )
        return ref_code, speaker_embedding

    def _generate_icl_prompt(
        self,
        text_id: torch.Tensor,
        ref_id: torch.Tensor,
        ref_code: torch.Tensor,
        tts_pad_embed: torch.Tensor,
        tts_eos_embed: torch.Tensor,
        non_streaming_mode: bool,
    ):
        """生成 Base 模型 ICL prompt（参考文本 + 参考音频 codes）。"""
        # Base prompt 先把参考文本和目标文本拼在文本侧，再用 TextProjection
        # 对齐到 Talker hidden space。
        text_embed = self.text_projection(
            self.talker.get_text_embeddings()(torch.cat([ref_id, text_id], dim=-1))
        )
        text_embed = torch.cat([text_embed, tts_eos_embed], dim=1)

        # 参考音频 codec 的第一组 codebook 使用 Talker token embedding，
        # 其余 codebook 使用 CodePredictor token embedding；同一帧各组求和后
        # 作为参考音频 codec 条件。
        codec_embed = []
        for i in range(self.num_code_groups):
            if i == 0:
                codec_embed.append(self.talker.get_input_embeddings()(ref_code[:, :1]))
            else:
                codec_embed.append(
                    self.code_predictor.token_embedding[i - 1](ref_code[:, i : i + 1])
                )
        codec_embed = torch.cat(codec_embed, dim=1).sum(1).unsqueeze(0)
        codec_embed = torch.cat(
            [
                self.talker.get_input_embeddings()(
                    torch.tensor([[self.codec_bos_id]], dtype=torch.long)
                ),
                codec_embed,
            ],
            dim=1,
        )

        text_lens = text_embed.shape[1]
        codec_lens = codec_embed.shape[1]
        # non-streaming 把完整文本条件放入 prefill；streaming 只放入能与
        # 参考 codec 对齐的前段，剩余文本 hidden 通过 trailing_text_hidden
        # 在 Talker decode 阶段逐步注入。
        if non_streaming_mode:
            logger.info("Using non-streaming mode for talker input embedding")
            icl_input_embed = text_embed + self.talker.get_input_embeddings()(
                torch.tensor([[self.codec_pad_id] * text_lens], dtype=torch.long)
            )
            icl_input_embed = torch.cat(
                [icl_input_embed, codec_embed + tts_pad_embed], dim=1
            )
            return icl_input_embed, tts_pad_embed
        logger.info("Using streaming mode for talker input embedding")
        if text_lens > codec_lens:
            return text_embed[:, :codec_lens] + codec_embed, text_embed[:, codec_lens:]
        text_embed = torch.cat(
            [text_embed] + [tts_pad_embed] * (codec_lens - text_lens), dim=1
        )
        return text_embed + codec_embed, tts_pad_embed

    def _run_talker_prefill(
        self,
        talker_input_embed: torch.Tensor,
        past_seq_length: int = 0,
    ) -> tuple:
        """执行 Talker Prefill 推理（分块调用 talker.prefill）

        Returns:
            (logits, past_hidden, num_chunks)
        """
        prefill_length = self.talker.prefill_length
        seq_length = talker_input_embed.shape[1]
        prefill_loop_round = math.ceil(seq_length / prefill_length)

        # Padding 到 prefill_length 的整数倍
        padding_len = prefill_loop_round * prefill_length - seq_length
        if padding_len > 0:
            padding_embeds = self.talker.get_input_embeddings()(
                torch.zeros(1, padding_len, dtype=torch.long)
            )
            talker_input_embed = torch.cat([talker_input_embed, padding_embeds], dim=1)

        self.talker.reset_cache_inputs()

        for round_idx in range(prefill_loop_round):
            start = round_idx * prefill_length
            if round_idx == prefill_loop_round - 1:
                current_length = seq_length - start
            else:
                current_length = prefill_length

            chunk_embeds = talker_input_embed[:, start : start + prefill_length, :]
            valid_length_data = np.array([past_seq_length + start]).astype("int32")
            current_length_data = np.array([current_length]).astype("int32")
            is_last_chunk = round_idx == prefill_loop_round - 1

            logits_np, past_hidden_np = self.talker.prefill(
                to_numpy(chunk_embeds),
                valid_length_data,
                current_length_data,
                fetch_output=is_last_chunk,
            )

        return (
            torch.from_numpy(logits_np),
            torch.from_numpy(past_hidden_np),
            prefill_loop_round,
        )

    def _run_talker_decode(
        self,
        inputs_embeds: torch.Tensor,
        past_seq_length: int,
    ) -> tuple:
        """执行 Talker 单步 Decode 推理"""
        valid_length_data = np.array([past_seq_length]).astype("int32")

        logits_np, past_hidden_np = self.talker.decode(
            to_numpy(inputs_embeds),
            valid_length_data,
        )
        return torch.from_numpy(logits_np), torch.from_numpy(past_hidden_np)

    def _sample_next_token(
        self,
        logits_processor: LogitsProcessorList,
        input_ids: torch.Tensor,
        logits: torch.Tensor,
        do_sample: bool,
    ) -> torch.Tensor:
        """Apply logits processors and sample or greedily select the next token."""
        scores = logits_processor(input_ids, logits.to(dtype=torch.float32))
        if do_sample:
            probs = F.softmax(scores, dim=-1)
            return torch.multinomial(probs, num_samples=1).squeeze(-1)
        return torch.argmax(scores, dim=-1)

    def _run_code_predictor_generate(
        self,
        inputs_embeds: torch.Tensor,
        max_new_tokens: int,
        logits_processor: LogitsProcessorList,
        do_sample: bool = True,
    ) -> tuple:
        """执行 Code Predictor 生成

        Returns:
            (input_ids, cp_prefill_time, cp_decode_time, cp_prefill_count, cp_decode_count)
        """

        prefill_length = self.code_predictor.prefill_length
        seq_length = inputs_embeds.shape[1]
        prefill_loop_round = math.ceil(seq_length / prefill_length)

        # Padding 到 prefill_length 的整数倍
        padding_len = prefill_loop_round * prefill_length - seq_length
        if padding_len > 0:
            embed_layer = self.code_predictor.token_embedding[0]
            padding_embeds = embed_layer(torch.zeros(1, padding_len, dtype=torch.long))
            inputs_embeds = torch.cat([inputs_embeds, padding_embeds], dim=1)

        # ========== Prefill ==========
        self.code_predictor.reset_cache_inputs()

        t_pf = time.time()
        for round_idx in range(prefill_loop_round):
            start = round_idx * prefill_length
            if round_idx == prefill_loop_round - 1:
                current_length = seq_length - start
            else:
                current_length = prefill_length

            chunk_embeds = inputs_embeds[:, start : start + prefill_length, :]
            valid_length_data = np.array([start]).astype("int32")
            current_length_data = np.array([current_length]).astype("int32")
            generation_steps_data = np.array([0]).astype("int32")
            is_last_chunk = round_idx == prefill_loop_round - 1

            output = self.code_predictor.prefill(
                to_numpy(chunk_embeds),
                valid_length_data,
                current_length_data,
                generation_steps_data,
                fetch_output=is_last_chunk,
            )
        cp_prefill_time = elapsed_s(t_pf)

        # Prefill 最后一轮的输出即为 logits
        cp_sampling_time = 0.0
        cp_sampling_count = 0
        logits = torch.from_numpy(output)

        # 采样第一个 token
        t_sample = time.time()
        next_token_logits = logits[:, -1, :].to(dtype=torch.float32)
        input_ids = torch.empty((1, 0), dtype=torch.long)
        next_token = self._sample_next_token(
            logits_processor, input_ids, next_token_logits, do_sample
        )
        cp_sampling_time += elapsed_s(t_sample)
        cp_sampling_count += 1

        input_ids = torch.cat([input_ids, next_token[:, None]], dim=-1)

        # ========== Decode ==========
        context_length = seq_length
        cp_decode_time = 0.0
        decode_embeds = []

        for step in range(max_new_tokens - 1):
            next_embed = self.code_predictor.token_embedding[step](next_token)
            if next_embed.dim() == 2:
                next_embed = next_embed.unsqueeze(1)
            decode_embeds.append(next_embed)

            valid_length_data = np.array([context_length]).astype("int32")
            generation_steps_data = np.array([step + 1]).astype("int32")

            t_dc = time.time()
            output = self.code_predictor.decode(
                to_numpy(next_embed),
                valid_length_data,
                generation_steps_data,
            )
            cp_decode_time += elapsed_s(t_dc)

            t_sample = time.time()
            next_token_logits = torch.from_numpy(output)[:, -1, :].to(
                dtype=torch.float32
            )
            next_token = self._sample_next_token(
                logits_processor, input_ids, next_token_logits, do_sample
            )
            cp_sampling_time += elapsed_s(t_sample)
            cp_sampling_count += 1

            input_ids = torch.cat([input_ids, next_token[:, None]], dim=-1)
            context_length += 1

        return (
            input_ids,
            decode_embeds,
            cp_prefill_time,
            cp_decode_time,
            cp_sampling_time,
            cp_sampling_count,
            prefill_loop_round,
            max_new_tokens - 1,
        )

    def init_logits_processors(
        self,
        min_new_tokens: int = 2,
        top_k: int = 50,
        top_p: float = 1.0,
        temperature: float = 0.9,
        repetition_penalty: float = 1.05,
        subtalker_top_k: int = 50,
        subtalker_top_p: float = 1.0,
        subtalker_temperature: float = 0.9,
    ) -> Tuple[LogitsProcessorList, LogitsProcessorList]:
        """初始化 Talker / CodePredictor 采样 processor。"""
        logits_processor = LogitsProcessorList()
        if repetition_penalty is not None and repetition_penalty != 1.0:
            logits_processor.append(
                RepetitionPenaltyLogitsProcessor(
                    repetition_penalty,
                    prompt_ignore_length=0,
                )
            )
        if min_new_tokens is not None and min_new_tokens > 0:
            logits_processor.append(
                MinNewTokensLengthLogitsProcessor(
                    prompt_length_to_skip=0,
                    min_new_tokens=min_new_tokens,
                    eos_token_id=self.codec_eos_token_id,
                )
            )
        # 抑制 vocab 末尾的 tokens（除了 eos_token）
        vocab_size = self.talker_config.vocab_size
        suppress_tokens = [
            i
            for i in range(vocab_size - 1024, vocab_size)
            if i != self.codec_eos_token_id
        ]
        logits_processor.append(SuppressTokensLogitsProcessor(suppress_tokens))
        logits_processor.append(TemperatureLogitsWarper(temperature))
        if top_k is not None and top_k > 0:
            logits_processor.append(TopKLogitsWarper(top_k))
        if top_p is not None and top_p < 1.0:
            logits_processor.append(TopPLogitsWarper(top_p))

        # subtalker 的 logits_processor
        subtalker_logits_processor = LogitsProcessorList()
        subtalker_logits_processor.append(
            TemperatureLogitsWarper(subtalker_temperature)
        )
        if subtalker_top_k is not None and subtalker_top_k > 0:
            subtalker_logits_processor.append(TopKLogitsWarper(subtalker_top_k))
        if subtalker_top_p is not None and subtalker_top_p < 1.0:
            subtalker_logits_processor.append(TopPLogitsWarper(subtalker_top_p))
        return logits_processor, subtalker_logits_processor

    def _prepare_voice_clone_prompt(
        self,
        text: str,
        language: str,
        ref_audio: str,
        ref_text: str,
        non_streaming_mode: bool,
        perf: PerfTracker,
    ) -> BasePromptContext:
        """准备 Base voice clone 的 Talker prompt。"""
        ref_code, speaker_embed = self.encode_reference_audio(ref_audio)
        for key, seconds in getattr(self, "_last_reference_audio_perf", {}).items():
            perf.add(
                key,
                seconds,
                count=getattr(self, "_last_reference_audio_perf_count", {}).get(key, 0),
            )

        perf.start(PerfKey.PREP_TEXT_TOKENIZE)
        input_text = build_assistant_text(text)
        input_ids = tokenize_texts(self.processor, [input_text])
        input_id = input_ids[0]

        ref_text_formatted = build_ref_text(ref_text)
        ref_ids = tokenize_texts(self.processor, [ref_text_formatted])
        ref_id = ref_ids[0]

        language_id = self._get_language_id(language)
        perf.stop(PerfKey.PREP_TEXT_TOKENIZE, count=2)

        perf.start(PerfKey.PREP_SPECIAL_TOKEN_EMBED)
        # TTS special token 走文本 embedding + TextProjection，得到与 Talker
        # codec 条件同维度的 bos/eos/pad hidden。
        tts_bos_embed, tts_eos_embed, tts_pad_embed = self.text_projection(
            self.talker.get_text_embeddings()(
                torch.tensor(
                    [
                        [
                            self.tts_bos_token_id,
                            self.tts_eos_token_id,
                            self.tts_pad_token_id,
                        ]
                    ],
                    dtype=torch.long,
                )
            )
        ).chunk(3, dim=1)
        perf.stop(PerfKey.PREP_SPECIAL_TOKEN_EMBED, count=1)

        perf.start(PerfKey.PREP_CODEC_PROMPT_EMBED)
        # codec prompt 前缀携带 think/no-think、language 和 speaker 条件；
        # 它与文本 role embedding 一起构成 Base voice clone 的起始上下文。
        if language_id is None:
            codec_prefill_list = [
                [
                    self.codec_nothink_id,
                    self.codec_think_bos_id,
                    self.codec_think_eos_id,
                ]
            ]
        else:
            codec_prefill_list = [
                [
                    self.codec_think_id,
                    self.codec_think_bos_id,
                    language_id,
                    self.codec_think_eos_id,
                ]
            ]

        codec_input_embedding_0 = self.talker.get_input_embeddings()(
            torch.tensor(codec_prefill_list, dtype=torch.long)
        )
        codec_input_embedding_1 = self.talker.get_input_embeddings()(
            torch.tensor([[self.codec_pad_id, self.codec_bos_id]], dtype=torch.long)
        )

        if speaker_embed is None:
            codec_input_embedding = torch.cat(
                [codec_input_embedding_0, codec_input_embedding_1], dim=1
            )
        else:
            codec_input_embedding = torch.cat(
                [
                    codec_input_embedding_0,
                    speaker_embed.view(1, 1, -1),
                    codec_input_embedding_1,
                ],
                dim=1,
            )
        perf.stop(PerfKey.PREP_CODEC_PROMPT_EMBED, count=1)

        perf.start(PerfKey.PREP_TALKER_ROLE_EMBED)
        # input_id[:, :3] 对应 assistant role 开头 token，先走文本 embedding
        # 和 TextProjection。
        _talker_input_embed_role = self.text_projection(
            self.talker.get_text_embeddings()(input_id[:, :3])
        )

        _talker_input_embed = (
            torch.cat(
                (
                    tts_pad_embed.expand(-1, codec_input_embedding.shape[1] - 2, -1),
                    tts_bos_embed,
                ),
                dim=1,
            )
            + codec_input_embedding[:, :-1]
        )

        talker_input_embed = torch.cat(
            (_talker_input_embed_role, _talker_input_embed), dim=1
        )
        perf.stop(PerfKey.PREP_TALKER_ROLE_EMBED, count=1)

        perf.start(PerfKey.PREP_ICL_PROMPT_EMBED)
        # ICL prompt 注入参考文本和参考音频 codec，让 Base 模型在上下文中
        # 模仿参考音频的说话人和发音风格。
        icl_input_embed, trailing_text_hidden = self._generate_icl_prompt(
            text_id=input_id[:, 3:-5],
            ref_id=ref_id[:, 3:-2],
            ref_code=ref_code,
            tts_pad_embed=tts_pad_embed,
            tts_eos_embed=tts_eos_embed,
            non_streaming_mode=non_streaming_mode,
        )
        perf.stop(PerfKey.PREP_ICL_PROMPT_EMBED, count=1)

        perf.start(PerfKey.PREP_CONCAT)
        talker_input_embed = torch.cat([talker_input_embed, icl_input_embed], dim=1)
        perf.stop(PerfKey.PREP_CONCAT, count=1)

        if talker_input_embed.shape[1] >= self.talker.context_max_length - 1:
            raise ValueError(
                f"Base prompt is too long: prompt_len={talker_input_embed.shape[1]}, "
                f"context={self.talker.context_max_length}"
            )

        return BasePromptContext(
            talker_input_embed=talker_input_embed,
            trailing_text_hidden=trailing_text_hidden,
            ref_code=ref_code,
            tts_pad_embed=tts_pad_embed,
        )

    def _generate_codec_frames(
        self,
        prompt: BasePromptContext,
        logits_processors: Tuple[LogitsProcessorList, LogitsProcessorList],
        perf: PerfTracker,
        max_new_tokens: int = 4096,
        do_sample: bool = True,
        subtalker_dosample: bool = True,
        show_progress: bool = False,
    ) -> Generator[torch.Tensor, None, None]:
        """从准备好的 Base prompt 逐帧生成 codec ids。"""
        logits_processor, subtalker_logits_processor = logits_processors
        talker_input_embed = prompt.talker_input_embed
        trailing_text_hidden = prompt.trailing_text_hidden
        tts_pad_embed = prompt.tts_pad_embed

        perf.start(PerfKey.TALKER_PREFILL)
        logits, past_hidden, talker_prefill_chunks = self._run_talker_prefill(
            talker_input_embed, past_seq_length=0
        )
        perf.stop(PerfKey.TALKER_PREFILL, count=talker_prefill_chunks)

        first_token_logits = logits[:, -1, :].to(dtype=torch.float32)
        input_ids_for_processor = torch.empty((1, 0), dtype=torch.long)
        perf.start(PerfKey.TALKER_SAMPLING)
        next_token = self._sample_next_token(
            logits_processor,
            input_ids_for_processor,
            first_token_logits,
            do_sample,
        )
        perf.stop(PerfKey.TALKER_SAMPLING, count=1)

        generated_talker_input_ids = next_token[:, None]
        past_seq_length = talker_input_embed.shape[1]

        this_peer_finished = False
        max_new_tokens = max(
            0,
            min(max_new_tokens, self.talker.context_max_length - past_seq_length - 1),
        )
        with tqdm(
            total=max_new_tokens,
            desc="Generating tokens",
            disable=not show_progress,
        ) as pbar:
            while not this_peer_finished:
                step = past_seq_length - talker_input_embed.shape[1]

                if step >= max_new_tokens:
                    logger.info(f"Reached max_new_tokens at step {step}")
                    break

                last_id_hidden = self.talker.get_input_embeddings()(next_token)
                if last_id_hidden.dim() == 2:
                    last_id_hidden = last_id_hidden.unsqueeze(1)

                predictor_input_embeds = torch.cat((past_hidden, last_id_hidden), dim=1)
                (
                    predictor_tokens,
                    predictor_embeds,
                    cp_pf_t,
                    cp_dc_t,
                    cp_sampling_t,
                    cp_sampling_n,
                    cp_pf_n,
                    cp_dc_n,
                ) = self._run_code_predictor_generate(
                    predictor_input_embeds,
                    max_new_tokens=self.num_code_groups - 1,
                    logits_processor=subtalker_logits_processor,
                    do_sample=subtalker_dosample,
                )
                perf.add(PerfKey.CODE_PREDICTOR_PREFILL, cp_pf_t, count=cp_pf_n)
                perf.add(PerfKey.CODE_PREDICTOR_DECODE, cp_dc_t, count=cp_dc_n)
                perf.add(
                    PerfKey.CODE_PREDICTOR_SAMPLING,
                    cp_sampling_t,
                    count=cp_sampling_n,
                )

                codec_ids = torch.cat(
                    (next_token.unsqueeze(0), predictor_tokens), dim=-1
                )
                yield codec_ids

                codec_hiddens = torch.cat([last_id_hidden] + predictor_embeds, dim=1)
                inputs_embeds = codec_hiddens.sum(1, keepdim=True)

                generation_step = step
                if generation_step < trailing_text_hidden.shape[1]:
                    inputs_embeds = inputs_embeds + trailing_text_hidden[
                        :, generation_step
                    ].unsqueeze(1)
                else:
                    inputs_embeds = inputs_embeds + tts_pad_embed

                perf.start(PerfKey.TALKER_DECODE)
                logits, past_hidden = self._run_talker_decode(
                    inputs_embeds, past_seq_length
                )
                perf.stop(PerfKey.TALKER_DECODE, count=1)

                next_token_logits = logits[:, 0, :].to(dtype=torch.float32)
                perf.start(PerfKey.TALKER_SAMPLING)
                next_token = self._sample_next_token(
                    logits_processor,
                    generated_talker_input_ids,
                    next_token_logits,
                    do_sample,
                )
                perf.stop(PerfKey.TALKER_SAMPLING, count=1)

                generated_talker_input_ids = torch.cat(
                    [generated_talker_input_ids, next_token[:, None]],
                    dim=-1,
                )
                past_seq_length += 1

                pbar.update(1)

                if next_token.item() == self.codec_eos_token_id:
                    logger.info(f"Reached EOS token at step {step + 1}")
                    this_peer_finished = True

    def _collect_talker_codes(
        self, generated_codes: List[torch.Tensor]
    ) -> List[torch.Tensor]:
        """裁掉 EOS 及其后的无效 codec frames。"""
        if generated_codes:
            talker_codes = torch.stack(generated_codes, dim=1)

            first_codebook = talker_codes[:, :, 0]
            is_stop_token = first_codebook == self.codec_eos_token_id
            stop_indices = torch.argmax(is_stop_token.int(), dim=1)
            has_stop_token = is_stop_token.any(dim=1)
            effective_lengths = torch.where(
                has_stop_token, stop_indices, talker_codes.shape[1]
            )

            talker_codes_list = [
                talker_codes[i, :length] for i, length in enumerate(effective_lengths)
            ]
        else:
            talker_codes_list = []
        return talker_codes_list

    def _create_perf(self) -> PerfTracker:
        return PerfTracker()

    @torch.no_grad()
    def generate_voice_clone(
        self,
        text: str,
        language: str,
        ref_audio: str,
        ref_text: str,
        non_streaming_mode: bool = True,
        max_new_tokens: int = 4096,
        min_new_tokens: int = 2,
        do_sample: bool = True,
        top_k: int = 50,
        top_p: float = 1.0,
        temperature: float = 0.9,
        repetition_penalty: float = 1.05,
        subtalker_dosample: bool = True,
        subtalker_top_k: int = 50,
        subtalker_top_p: float = 1.0,
        subtalker_temperature: float = 0.9,
        show_progress: bool = True,
        **kwargs,
    ):
        """使用 Qwen3-TTS Base 模型进行非流式 voice clone 推理。"""
        logger.info(f"Starting voice clone generation for text: {text[:50]}")

        perf = self._create_perf()
        perf.start(PerfKey.PREP_LOGITS_PROCESSOR)
        logits_processors = self.init_logits_processors(
            min_new_tokens=min_new_tokens,
            top_k=top_k,
            top_p=top_p,
            temperature=temperature,
            repetition_penalty=repetition_penalty,
            subtalker_top_k=subtalker_top_k,
            subtalker_top_p=subtalker_top_p,
            subtalker_temperature=subtalker_temperature,
        )
        perf.stop(PerfKey.PREP_LOGITS_PROCESSOR, count=1)

        prompt = self._prepare_voice_clone_prompt(
            text=text,
            language=language,
            ref_audio=ref_audio,
            ref_text=ref_text,
            non_streaming_mode=non_streaming_mode,
            perf=perf,
        )

        generated_codes = list(
            self._generate_codec_frames(
                prompt=prompt,
                logits_processors=logits_processors,
                perf=perf,
                max_new_tokens=max_new_tokens,
                do_sample=do_sample,
                subtalker_dosample=subtalker_dosample,
                show_progress=show_progress,
            )
        )
        talker_codes_list = self._collect_talker_codes(generated_codes)

        if talker_codes_list:
            # SpeechTokenizer oneshot 解码时把参考 codec 也拼进去，保证目标音频
            # 开头有与 prompt 一致的 codec 上下文；解码后再裁掉参考音频 waveform。
            codes_for_decode = [
                torch.cat([prompt.ref_code.to(codes.device), codes], dim=0)
                for codes in talker_codes_list
            ]
            perf.start(PerfKey.SPEECH_TOKENIZER)
            wavs_all, sr = self.speech_tokenizer.decode(
                [{"audio_codes": c} for c in codes_for_decode]
            )
            perf.stop(PerfKey.SPEECH_TOKENIZER, count=1)

            wavs = []
            for i, wav in enumerate(wavs_all):
                ref_len = int(prompt.ref_code.shape[0])
                total_len = int(codes_for_decode[i].shape[0])
                cut = int(ref_len / max(total_len, 1) * wav.shape[0])
                wavs.append(wav[cut:])
        else:
            wavs, sr = [], 24000
            logger.warning("No codes generated, returning empty audio")

        perf_dict, perf_count = perf.snapshot()
        return wavs, sr, perf_dict, perf_count

    def _prime_stateful_decoder_with_ref_code(
        self,
        ref_code: torch.Tensor,
        stateful_decoder: Qwen3TTSStatefulDecoderInference,
        decoder_state: StatefulDecoderState,
        chunk_size: int,
        perf: PerfTracker,
    ) -> StatefulDecoderState:
        """用参考音频 codec 预热流式 decoder，并丢弃参考音频输出。"""
        ref_code = ref_code[:, : self.num_code_groups].to(torch.long)
        buffer = CodeChunkBuffer(chunk_size=chunk_size)

        for frame in ref_code:
            buffer.push(frame)
            chunk = buffer.flush()
            if chunk is not None:
                perf.start(PerfKey.STATEFUL_DECODER_REF_PRIME)
                _, decoder_state = stateful_decoder.decode(
                    to_numpy(chunk).astype(np.int32),
                    decoder_state,
                    is_final=False,
                )
                perf.stop(PerfKey.STATEFUL_DECODER_REF_PRIME, count=1)

        residual = buffer.finalize()
        if residual is not None:
            # StatefulDecoder 的非 final decode 期望固定 chunk_size；残余参考帧
            # 用最后一帧补齐，只用于更新 decoder state，输出会被丢弃。
            pad_count = chunk_size - residual.shape[0]
            pad_frame = residual[-1:].expand(pad_count, -1)
            padded = torch.cat([residual, pad_frame], dim=0)
            perf.start(PerfKey.STATEFUL_DECODER_REF_PRIME)
            _, decoder_state = stateful_decoder.decode(
                to_numpy(padded).astype(np.int32),
                decoder_state,
                is_final=False,
            )
            perf.stop(PerfKey.STATEFUL_DECODER_REF_PRIME, count=1)

        return decoder_state

    @torch.no_grad()
    def generate_voice_clone_streaming(
        self,
        text: str,
        language: str,
        ref_audio: str,
        ref_text: str,
        stateful_decoder: Qwen3TTSStatefulDecoderInference,
        chunk_size: int = 12,
        non_streaming_mode: bool = False,
        max_new_tokens: int = 4096,
        min_new_tokens: int = 2,
        do_sample: bool = True,
        top_k: int = 50,
        top_p: float = 1.0,
        temperature: float = 0.9,
        repetition_penalty: float = 1.05,
        subtalker_dosample: bool = True,
        subtalker_top_k: int = 50,
        subtalker_top_p: float = 1.0,
        subtalker_temperature: float = 0.9,
        **kwargs,
    ) -> Generator[Tuple[np.ndarray, int], None, None]:
        """使用 Qwen3-TTS Base 模型进行流式 voice clone 推理。"""
        logger.info(f"Starting streaming voice clone generation for text: {text[:50]}")

        perf = self._create_perf()
        perf.start(PerfKey.PREP_LOGITS_PROCESSOR)
        logits_processors = self.init_logits_processors(
            min_new_tokens=min_new_tokens,
            top_k=top_k,
            top_p=top_p,
            temperature=temperature,
            repetition_penalty=repetition_penalty,
            subtalker_top_k=subtalker_top_k,
            subtalker_top_p=subtalker_top_p,
            subtalker_temperature=subtalker_temperature,
        )
        perf.stop(PerfKey.PREP_LOGITS_PROCESSOR, count=1)

        prompt = self._prepare_voice_clone_prompt(
            text=text,
            language=language,
            ref_audio=ref_audio,
            ref_text=ref_text,
            non_streaming_mode=non_streaming_mode,
            perf=perf,
        )

        decoder_state = stateful_decoder.create_state()
        decoder_state = self._prime_stateful_decoder_with_ref_code(
            ref_code=prompt.ref_code,
            stateful_decoder=stateful_decoder,
            decoder_state=decoder_state,
            chunk_size=chunk_size,
            perf=perf,
        )

        # Base streaming 的首包延迟从 ref_code 预热完成后开始统计；
        # 参考音频处理和 decoder 预热耗时会进入总 inference / perf table。
        streaming_start_time = time.time()
        first_chunk_emitted = False
        self.last_streaming_first_chunk_latency_ms = None
        buffer = CodeChunkBuffer(chunk_size=chunk_size)

        for codec_ids in self._generate_codec_frames(
            prompt=prompt,
            logits_processors=logits_processors,
            perf=perf,
            max_new_tokens=max_new_tokens,
            do_sample=do_sample,
            subtalker_dosample=subtalker_dosample,
            show_progress=False,
        ):
            buffer.push(codec_ids.squeeze(0))
            chunk = buffer.flush()
            if chunk is not None:
                perf.start(PerfKey.STATEFUL_DECODER)
                audio, decoder_state = stateful_decoder.decode(
                    to_numpy(chunk).astype(np.int32),
                    decoder_state,
                    is_final=False,
                )
                perf.stop(PerfKey.STATEFUL_DECODER, count=1)
                if len(audio) > 0:
                    if not first_chunk_emitted:
                        self.last_streaming_first_chunk_latency_ms = elapsed_ms(
                            streaming_start_time
                        )
                        first_chunk_emitted = True
                    yield audio, 24000

        residual = buffer.finalize()
        if residual is not None:
            perf.start(PerfKey.STATEFUL_DECODER)
            audio, decoder_state = stateful_decoder.decode(
                to_numpy(residual).astype(np.int32),
                decoder_state,
                is_final=True,
            )
            perf.stop(PerfKey.STATEFUL_DECODER, count=1)
            if len(audio) > 0:
                if not first_chunk_emitted:
                    self.last_streaming_first_chunk_latency_ms = elapsed_ms(
                        streaming_start_time
                    )
                    first_chunk_emitted = True
                yield audio, 24000
        else:
            perf.start(PerfKey.STATEFUL_DECODER)
            audio, decoder_state = stateful_decoder.decode(
                np.zeros((0, 16), dtype=np.int32),
                decoder_state,
                is_final=True,
            )
            perf.stop(PerfKey.STATEFUL_DECODER, count=1)
            if len(audio) > 0:
                if not first_chunk_emitted:
                    self.last_streaming_first_chunk_latency_ms = elapsed_ms(
                        streaming_start_time
                    )
                    first_chunk_emitted = True
                yield audio, 24000

        self.last_streaming_perf, self.last_streaming_perf_count = perf.snapshot()


def main(args: argparse.Namespace) -> None:
    torch.manual_seed(args.seed)

    xh_model = Qwen3TTSInference(
        hf_model=args.hf_model_dir,
        text_projection=dict(
            hmm_file=args.text_projection_hmm,
        ),
        code_predictor=dict(
            prefill_hmm=args.code_predictor_prefill_hmm,
            decode_hmm=args.code_predictor_decode_hmm,
            token_embedding=args.code_predictor_token_embedding,
        ),
        talker=dict(
            prefill_hmm=args.talker_prefill_hmm,
            decode_hmm=args.talker_decode_hmm,
            token_embedding=args.talker_token_embedding,
            text_embedding=args.talker_text_embedding,
        ),
        speech_tokenizer=dict(
            hmm_file=args.speech_tokenizer_hmm,
            decode_padding_shapes=args.speech_tokenizer_decode_padding_shapes,
        ),
        speech_tokenizer_encoder=dict(
            hmm_file=args.speech_tokenizer_encoder_hmm,
        ),
        speaker_encoder=dict(
            hmm_file=args.speaker_encoder_hmm,
        ),
        ndevice=args.ndevice,
    )

    out_file = Path(args.output_wav)
    out_file.parent.mkdir(parents=True, exist_ok=True)
    if out_file.exists():
        out_file.unlink()

    if args.mode == "streaming":
        stateful_decoder = Qwen3TTSStatefulDecoderInference(
            hmm_file=args.stateful_decoder_hmm,
            chunk_size=args.chunk_size,
            ndevice=args.ndevice,
        )
        chunk_count = 0
        first_chunk_latency_ms = None
        output_sr = None
        all_chunks = []
        playback_gap = StreamingPlaybackGapTracker()
        start_time = time.time()
        for audio_chunk, sr in xh_model.generate_voice_clone_streaming(
            text=args.text,
            language=args.language,
            ref_audio=args.ref_audio,
            ref_text=args.ref_text,
            stateful_decoder=stateful_decoder,
            chunk_size=args.chunk_size,
        ):
            chunk_emit_time = time.perf_counter()
            chunk_count += 1
            if first_chunk_latency_ms is None:
                first_chunk_latency_ms = getattr(
                    xh_model, "last_streaming_first_chunk_latency_ms", None
                )
                if first_chunk_latency_ms is None:
                    first_chunk_latency_ms = elapsed_ms(start_time)
                logger.info(
                    f"First audio chunk latency: {first_chunk_latency_ms:.1f}ms"
                )
            if output_sr is None:
                output_sr = sr
            elif sr != output_sr:
                raise ValueError(
                    f"Streaming sample rate changed from {output_sr} to {sr}."
                )

            all_chunks.append(audio_chunk)
            gap_ms = playback_gap.update(len(audio_chunk), sr, chunk_emit_time)
            chunk_audio_ms = len(audio_chunk) / sr * 1000
            if gap_ms is None:
                logger.info(
                    f"  Chunk {chunk_count}: {len(audio_chunk)} samples "
                    f"({chunk_audio_ms:.0f}ms audio)"
                )
            else:
                logger.info(
                    f"  Chunk {chunk_count}: {len(audio_chunk)} samples "
                    f"({chunk_audio_ms:.0f}ms audio) | "
                    f"playback_gap: {gap_ms:.1f}ms"
                )

        inference_time = time.time() - start_time

        if all_chunks:
            full_audio = np.concatenate(all_chunks)
            sf.write(out_file, full_audio, output_sr)
            audio_duration = len(full_audio) / output_sr
            rtf = inference_time / audio_duration
            first_chunk_ms = (
                first_chunk_latency_ms if first_chunk_latency_ms is not None else 0.0
            )
            logger.info(
                f"Audio saved to {out_file} | "
                f"duration: {audio_duration:.2f}s | "
                f"inference: {inference_time:.2f}s | "
                f"RTF: {rtf:.4f} | "
                f"chunks: {chunk_count} | "
                f"first_chunk_latency: {first_chunk_ms:.1f}ms | "
                f"playback_gap_chunks: {playback_gap.gap_chunks} | "
                f"max_playback_gap: {playback_gap.max_gap_ms:.1f}ms | "
                f"total_playback_gap: {playback_gap.total_gap_ms:.1f}ms"
            )
            perf = getattr(xh_model, "last_streaming_perf", {})
            perf_count = getattr(xh_model, "last_streaming_perf_count", {})
            if perf:
                log_base_streaming_perf(perf, perf_count, inference_time)
        else:
            logger.error("No audio generated")
        return

    start_time = time.time()
    wavs, sr, perf, perf_count = xh_model.generate_voice_clone(
        text=args.text,
        language=args.language,
        ref_audio=args.ref_audio,
        ref_text=args.ref_text,
        show_progress=not args.no_progress,
    )
    inference_time = time.time() - start_time

    if wavs:
        sf.write(out_file, wavs[0], sr)
        audio_duration = len(wavs[0]) / sr
        rtf = inference_time / audio_duration
        logger.info(
            f"Audio saved to {out_file} | "
            f"duration: {audio_duration:.2f}s | "
            f"inference: {inference_time:.2f}s | "
            f"RTF: {rtf:.4f}"
        )
        log_base_oneshot_perf(perf, perf_count, inference_time)
    else:
        logger.error("No audio generated")


def get_args() -> argparse.Namespace:
    """Parse commandline."""
    # fmt: off
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--config", dest="config_path", type=str, default=DEFAULT_CONFIG_PATH, help="path to config.yaml")
    parser.add_argument("--model_name", type=str, default=None, help="模型名称")
    parser.add_argument("--model_size", type=str, default=None, help="模型大小，demo_base.py 默认使用 0.6b-base")
    parser.add_argument("--hf_model_dir", type=str, default=None, help="ModelScope/HF 模型目录路径，仅加载 config/processor，不加载原始模型权重")
    parser.add_argument("--text_projection_hmm", type=str, default=None, help="Text Projection HMM 模型文件路径")
    parser.add_argument("--code_predictor_prefill_hmm", type=str, default=None, help="Code Predictor Prefill HMM 文件路径")
    parser.add_argument("--code_predictor_decode_hmm", type=str, default=None, help="Code Predictor Decode HMM 文件路径")
    parser.add_argument("--code_predictor_token_embedding", type=str, default=os.path.join("output", HOUMO_TARGET, "hmquant", "quant_embedding_code_predictor.pt"), help="Code Predictor Token Embedding 文件路径")
    parser.add_argument("--talker_prefill_hmm", type=str, default=None, help="Talker Prefill HMM 文件路径")
    parser.add_argument("--talker_decode_hmm", type=str, default=None, help="Talker Decode HMM 文件路径")
    parser.add_argument("--talker_token_embedding", type=str, default=os.path.join("output", HOUMO_TARGET, "hmquant", "quant_embedding.pt"), help="Talker Token Embedding 文件路径")
    parser.add_argument("--talker_text_embedding", type=str, default=os.path.join("output", HOUMO_TARGET, "hmquant", "text_embedding.pt"), help="Talker Text Embedding 文件路径")
    parser.add_argument("--speech_tokenizer_hmm", type=str, default=None, help="Speech Tokenizer HMM 模型文件路径")
    parser.add_argument("--speech_tokenizer_decode_padding_shapes", type=str, default=os.path.join("output", HOUMO_TARGET, "hmquant", "decode_padding_shapes.json"), help="Speech Tokenizer decode_padding_shapes JSON 文件路径")
    parser.add_argument("--speech_tokenizer_encoder_hmm", type=str, default=None, help="Speech Tokenizer Encoder HMM 文件路径（Base voice clone 参考音频编码）")
    parser.add_argument("--speaker_encoder_hmm", type=str, default=None, help="Speaker Encoder HMM 文件路径（Base voice clone speaker embedding）")
    parser.add_argument("--stateful_decoder_hmm", type=str, default=None, help="Stateful Decoder HMM 文件路径（streaming 模式使用）")
    parser.add_argument("--ndevice", type=int, default=1, help="设备数量")
    parser.add_argument("--output_wav", type=str, default="./output_voice_clone.wav", help="输出 wav 文件路径")
    parser.add_argument("--text", type=str, default="基于先进的存算一体技术和存储工艺，后摩智能致力于突破芯片的性能与功耗瓶颈，加速人工智能技术的普惠落地。", help="待合成的文本，若不指定则使用默认文本")
    parser.add_argument("--language", type=str, default="Chinese", choices=["auto", "Chinese", "English", "Japanese", "Korean", "French", "German", "Spanish", "Italian", "Portuguese", "Russian"], help="语言，默认为 Chinese")
    parser.add_argument("--ref_audio", type=str, default=f"{HOUMO_EXAMPLES_PATH}/data/audio/clone_1.wav", help="参考音频文件路径（用于 Base 模型 voice clone）")
    parser.add_argument("--ref_text", type=str, default="甚至出现交易几乎停滞的情况。", help="参考音频对应的文本内容")
    parser.add_argument("--seed", type=int, default=1024, help="随机种子")
    parser.add_argument("--mode", type=str, default="oneshot", choices=["oneshot", "streaming"], help="推理模式: oneshot=全量生成; streaming=流式生成")
    parser.add_argument("--chunk_size", type=int, default=12, help="流式模式下每个解码块的 codec 帧数")
    parser.add_argument("--no_progress", action="store_true", help="关闭 oneshot 模式下的 tqdm 生成进度条，便于性能评测")
    # fmt: on

    args = parser.parse_args()
    _default_model_size, default_model_name, model_configs = get_model_configs(
        args.config_path
    )
    args.model_name = first_not_none(args.model_name, default_model_name)
    args.model_size = first_not_none(args.model_size, "0.6b-base")
    model_config = model_configs.get(args.model_name, {}).get(args.model_size, {})
    if not model_config:
        available_sizes = sorted(model_configs.get(args.model_name, {}).keys())
        raise ValueError(
            f"Unsupported model config: model_name={args.model_name}, "
            f"model_size={args.model_size}. Available model_size values: {available_sizes}"
        )
    if "base" not in args.model_size.lower():
        raise ValueError(
            f"demo_base.py only supports Base model_size, got {args.model_size}"
        )

    args.hf_model_dir = first_not_none(
        args.hf_model_dir,
        get_default_hf_model_dir(model_config, default_model_size="0.6b-base"),
    )
    args.text_projection_hmm = first_not_none(
        args.text_projection_hmm,
        get_hmm_path(args.model_name, args.model_size, "text_projection"),
    )
    args.code_predictor_prefill_hmm = first_not_none(
        args.code_predictor_prefill_hmm,
        get_hmm_path(args.model_name, args.model_size, "code_predictor_prefill"),
    )
    args.code_predictor_decode_hmm = first_not_none(
        args.code_predictor_decode_hmm,
        get_hmm_path(args.model_name, args.model_size, "code_predictor_decode"),
    )
    args.talker_prefill_hmm = first_not_none(
        args.talker_prefill_hmm,
        get_hmm_path(args.model_name, args.model_size, "talker_prefill"),
    )
    args.talker_decode_hmm = first_not_none(
        args.talker_decode_hmm,
        get_hmm_path(args.model_name, args.model_size, "talker_decode"),
    )
    args.speech_tokenizer_hmm = first_not_none(
        args.speech_tokenizer_hmm,
        get_hmm_path(args.model_name, args.model_size, "speech_tokenizer"),
    )
    args.speech_tokenizer_encoder_hmm = first_not_none(
        args.speech_tokenizer_encoder_hmm,
        get_hmm_path(args.model_name, args.model_size, "speech_tokenizer_encoder"),
    )
    args.speaker_encoder_hmm = first_not_none(
        args.speaker_encoder_hmm,
        get_hmm_path(args.model_name, args.model_size, "speaker_encoder"),
    )
    args.stateful_decoder_hmm = first_not_none(
        args.stateful_decoder_hmm,
        get_hmm_path(args.model_name, args.model_size, "stateful_decoder"),
    )

    if args.mode == "streaming" and not args.stateful_decoder_hmm:
        raise RuntimeError("--stateful_decoder_hmm is required for streaming mode")

    return args


if __name__ == "__main__":
    args = get_args()
    main(args)
