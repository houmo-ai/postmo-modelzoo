# Copyright (c) 2025 HOUMO AI
#
# File: demo.py
# Description:
#   Qwen3-TTS-0.6B-CustomVoice tcim_lite inference demo.
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
from pathlib import Path
from typing import Dict, Generator, Optional, Tuple
import logging

logging.getLogger("qwen_tts").setLevel(logging.ERROR)

import numpy as np
import soundfile as sf
import torch
import torch.nn.functional as F
from torch import Tensor
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

from hmatc.utils.utils import first_not_none, get_model_configs
from loguru import logger
from perf_utils import PerfKey, PerfTracker, log_oneshot_perf, log_streaming_perf
from qwen3_tts_runtime import (
    CodeChunkBuffer,
    Qwen3TTSCodePredictorInference,
    Qwen3TTSSpeechTokenizerInference,
    Qwen3TTSStatefulDecoderInference,
    Qwen3TTSTalkerInference,
    Qwen3TTSTextProjectionInference,
    build_assistant_text,
    get_default_hf_model_dir,
    get_hmm_path,
    StreamingPlaybackGapTracker,
    to_numpy,
    tokenize_texts,
)

HOUMO_TARGET = os.getenv("HOUMO_TARGET")
assert HOUMO_TARGET in ["xh2"], f"Unsupported HOUMO_TARGET: {HOUMO_TARGET}"
DEFAULT_CONFIG_PATH = os.path.join(os.path.dirname(__file__), "config.yaml")


class Qwen3TTSCodecGenerator:
    """Qwen3 TTS Codec Token 生成器

    负责 Talker + CodePredictor + TextProjection 核心生成组件，
    音频解码器（speech_tokenizer / stateful_decoder）由调用方外部创建并传入。

    设计上把“生成 codec token”和“把 codec token 解码为音频”拆开：
    - oneshot 模式：先生成完整 audio_codes，再交给 SpeechTokenizer。
    - streaming 模式：每攒满一段 audio_codes，就交给 StatefulDecoder 输出音频 chunk。
    """

    def __init__(
        self,
        hf_model: str,
        text_projection: dict,
        code_predictor: dict,
        talker: dict,
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

        self._talker_prefill_valid_length_buf = np.empty((1,), dtype=np.int32)
        self._talker_prefill_current_length_buf = np.empty((1,), dtype=np.int32)
        self._talker_decode_valid_length_buf = np.empty((1,), dtype=np.int32)
        self._cp_prefill_valid_length_buf = np.empty((1,), dtype=np.int32)
        self._cp_prefill_current_length_buf = np.empty((1,), dtype=np.int32)
        self._cp_prefill_generation_steps_buf = np.empty((1,), dtype=np.int32)
        self._cp_decode_valid_length_buf = np.empty((1,), dtype=np.int32)
        self._cp_decode_generation_steps_buf = np.empty((1,), dtype=np.int32)

        self._init_static_embeddings()

        logger.info("Qwen3TTSCodecGenerator initialized successfully")

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
        self.spk_id = self.talker_config.spk_id
        self.spk_is_dialect = self.talker_config.spk_is_dialect
        self.num_code_groups = self.talker_config.num_code_groups

    def _init_static_embeddings(self) -> None:
        """预计算请求间不变的 embedding，减少首包路径中的重复小算子。"""
        input_embeddings = self.talker.get_input_embeddings()

        tts_special_ids = torch.tensor(
            [[self.tts_bos_token_id, self.tts_eos_token_id, self.tts_pad_token_id]],
            dtype=torch.long,
        )
        self.tts_bos_embed, self.tts_eos_embed, self.tts_pad_embed = (
            self.text_projection(
                self.talker.get_text_embeddings()(tts_special_ids)
            ).chunk(3, dim=1)
        )

        self.codec_pad_bos_embed = input_embeddings(
            torch.tensor([[self.codec_pad_id, self.codec_bos_id]], dtype=torch.long)
        )
        self.codec_bos_embed = input_embeddings(
            torch.tensor([[self.codec_bos_id]], dtype=torch.long)
        )
        self._codec_pad_embed_cache: Dict[int, Tensor] = {}
        self._codec_prefill_embed_cache: Dict[Optional[int], Tensor] = {}

        self._speaker_embed_cache: Dict[str, Tensor] = {}
        for speaker_name, speaker_id in self.spk_id.items():
            self._speaker_embed_cache[speaker_name.lower()] = input_embeddings(
                torch.tensor(speaker_id, dtype=torch.long)
            )

    def _get_speaker_embedding(self, speaker: str):
        """获取说话人 embedding"""
        speaker_key = speaker.lower()
        if speaker_key not in self._speaker_embed_cache:
            raise NotImplementedError(f"Speaker {speaker} not implemented")
        return self._speaker_embed_cache[speaker_key]

    def _get_codec_pad_embedding(self, length: int) -> Tensor:
        """获取指定长度的 codec pad embedding，按长度缓存。"""
        if length not in self._codec_pad_embed_cache:
            self._codec_pad_embed_cache[length] = self.talker.get_input_embeddings()(
                torch.tensor([[self.codec_pad_id] * length], dtype=torch.long)
            )
        return self._codec_pad_embed_cache[length]

    def _get_codec_prefill_embedding(self, language_id: Optional[int]) -> Tensor:
        """按语言条件构造 codec prefill embedding。"""
        if language_id in self._codec_prefill_embed_cache:
            return self._codec_prefill_embed_cache[language_id]

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
        self._codec_prefill_embed_cache[
            language_id
        ] = self.talker.get_input_embeddings()(
            torch.tensor(codec_prefill_list, dtype=torch.long)
        )
        return self._codec_prefill_embed_cache[language_id]

    def _get_language_id(self, language: str, speaker: str = None):
        """获取语言 ID"""
        if language.lower() == "auto":
            language_id = None
        else:
            if language.lower() not in self.codec_language_id:
                raise NotImplementedError(f"Language {language} not implemented")
            language_id = self.codec_language_id[language.lower()]

        # 处理方言：如果 speaker 被标记为方言说话人，覆盖 language_id
        if (
            language.lower() in ["chinese", "auto"]
            and speaker is not None
            and speaker != ""
            and speaker.lower() in self.spk_is_dialect
            and self.spk_is_dialect[speaker.lower()]
        ):
            dialect = self.spk_is_dialect[speaker.lower()]
            language_id = self.codec_language_id[dialect]

        return language_id

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
        """预初始化并返回 Talker / CodePredictor logits processors。

        调用方可在正式生成前调用本函数，并将返回值传给 generate_streaming()
        的 logits_processors 参数，以便把 processor 初始化移出首包延迟路径。
        """
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

        # 抑制 vocab 末尾保留区域的 token，但允许 eos 作为停止信号。
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

        subtalker_logits_processor = LogitsProcessorList()
        subtalker_logits_processor.append(
            TemperatureLogitsWarper(subtalker_temperature)
        )
        if subtalker_top_k is not None and subtalker_top_k > 0:
            subtalker_logits_processor.append(TopKLogitsWarper(subtalker_top_k))
        if subtalker_top_p is not None and subtalker_top_p < 1.0:
            subtalker_logits_processor.append(TopPLogitsWarper(subtalker_top_p))

        return logits_processor, subtalker_logits_processor

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

        # HMM prefill 的输入长度是编译期固定的 prefill_length。
        # 实际 prompt 可能更长或不足一块，因此这里 pad 到整块后循环调用。
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
            self._talker_prefill_valid_length_buf[0] = past_seq_length + start
            self._talker_prefill_current_length_buf[0] = current_length
            is_last_chunk = round_idx == prefill_loop_round - 1

            logits_np, past_hidden_np = self.talker.prefill(
                to_numpy(chunk_embeds),
                self._talker_prefill_valid_length_buf,
                self._talker_prefill_current_length_buf,
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
        self._talker_decode_valid_length_buf[0] = past_seq_length

        logits_np, past_hidden_np = self.talker.decode(
            to_numpy(inputs_embeds),
            self._talker_decode_valid_length_buf,
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

        输入是当前 Talker hidden + 第一组 codec token embedding。函数内部先 prefill
        当前上下文，再逐步 decode 出剩余 codebook token，最终组成一个完整 codec frame。

        Returns:
            (input_ids, cp_prefill_time, cp_decode_time, cp_prefill_count, cp_decode_count)
        """

        cp_prepare_time = time.perf_counter()
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
        cp_prepare_time = time.perf_counter() - cp_prepare_time

        t_pf = time.perf_counter()
        for round_idx in range(prefill_loop_round):
            start = round_idx * prefill_length
            if round_idx == prefill_loop_round - 1:
                current_length = seq_length - start
            else:
                current_length = prefill_length

            chunk_embeds = inputs_embeds[:, start : start + prefill_length, :]
            self._cp_prefill_valid_length_buf[0] = start
            self._cp_prefill_current_length_buf[0] = current_length
            self._cp_prefill_generation_steps_buf[0] = 0
            is_last_chunk = round_idx == prefill_loop_round - 1

            output = self.code_predictor.prefill(
                to_numpy(chunk_embeds),
                self._cp_prefill_valid_length_buf,
                self._cp_prefill_current_length_buf,
                self._cp_prefill_generation_steps_buf,
                fetch_output=is_last_chunk,
            )
        cp_prefill_time = time.perf_counter() - t_pf

        # Prefill 最后一轮的输出即为 logits
        cp_sampling_time = 0.0
        cp_sampling_count = 0
        t_sample = time.perf_counter()
        logits = torch.from_numpy(output)

        # 采样第一个 token
        next_token_logits = logits[:, -1, :].to(dtype=torch.float32)
        input_ids = torch.empty((1, 0), dtype=torch.long)
        next_token = self._sample_next_token(
            logits_processor, input_ids, next_token_logits, do_sample
        )

        input_ids = torch.cat([input_ids, next_token[:, None]], dim=-1)
        cp_sampling_time += time.perf_counter() - t_sample
        cp_sampling_count += 1

        # ========== Decode ==========
        context_length = seq_length
        cp_decode_time = 0.0
        decode_embeds = []

        for step in range(max_new_tokens - 1):
            next_embed = self.code_predictor.token_embedding[step](next_token)
            if next_embed.dim() == 2:
                next_embed = next_embed.unsqueeze(1)
            decode_embeds.append(next_embed)

            self._cp_decode_valid_length_buf[0] = context_length
            self._cp_decode_generation_steps_buf[0] = step + 1

            t_dc = time.perf_counter()
            output = self.code_predictor.decode(
                to_numpy(next_embed),
                self._cp_decode_valid_length_buf,
                self._cp_decode_generation_steps_buf,
            )
            cp_decode_time += time.perf_counter() - t_dc

            t_sample = time.perf_counter()
            next_token_logits = torch.from_numpy(output)[:, -1, :].to(
                dtype=torch.float32
            )
            next_token = self._sample_next_token(
                logits_processor, input_ids, next_token_logits, do_sample
            )

            input_ids = torch.cat([input_ids, next_token[:, None]], dim=-1)
            context_length += 1
            cp_sampling_time += time.perf_counter() - t_sample
            cp_sampling_count += 1

        return (
            input_ids,
            decode_embeds,
            cp_prepare_time,
            cp_prefill_time,
            cp_decode_time,
            cp_sampling_time,
            cp_sampling_count,
            prefill_loop_round,
            max_new_tokens - 1,
        )

    def _generate_codec_frames(
        self,
        text: str,
        language: str,
        speaker: str,
        logits_processors: Tuple[LogitsProcessorList, LogitsProcessorList],
        perf: PerfTracker,
        non_streaming_mode: bool = True,
        max_new_tokens: int = 4096,
        do_sample: bool = True,
        subtalker_dosample: bool = True,
        show_progress: bool = False,
    ) -> Generator[torch.Tensor, None, None]:
        """生成 codec frames，供 oneshot 和 streaming 两种音频解码路径复用。"""
        perf.start(PerfKey.EMBEDDING_PREP)

        logits_processor, subtalker_logits_processor = logits_processors

        # 构建 HF processor 期望的 assistant 格式文本，然后转成 token ids。
        # 后续会把文本 token embedding 和 codec/speaker/language embedding 融合为 Talker 输入。
        input_text = build_assistant_text(text)
        input_ids = tokenize_texts(self.processor, [input_text])
        input_id = input_ids[0]

        speaker_embed = self._get_speaker_embedding(speaker)
        language_id = self._get_language_id(language, speaker)

        tts_bos_embed = self.tts_bos_embed
        tts_eos_embed = self.tts_eos_embed
        tts_pad_embed = self.tts_pad_embed

        codec_input_embedding_0 = self._get_codec_prefill_embedding(language_id)
        codec_input_embedding_1 = self.codec_pad_bos_embed

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

        # Talker 输入不是简单的文本 embedding。Qwen3-TTS 会把文本侧 hidden 与 codec 侧
        # embedding 相加/拼接，让同一个自回归模型同时感知文本进度和 codec 生成进度。
        # input_id[:, :3] 对应 chat/template 开头的 role token。
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

        # non_streaming_mode 会一次性把完整文本条件放入 prompt，适合 oneshot 生成。
        # streaming-style embedding 只把开头文本放入 prompt，剩余文本 hidden 在每步 decode
        # 时通过 trailing_text_hidden 逐步加入，减少首包等待。
        if non_streaming_mode:
            logger.info("Using non-streaming mode for talker input embedding")
            talker_input_embed = talker_input_embed[:, :-1]
            talker_input_embed = torch.cat(
                [
                    talker_input_embed,
                    torch.cat(
                        (
                            self.text_projection(
                                self.talker.get_text_embeddings()(input_id[:, 3:-5])
                            ),
                            tts_eos_embed,
                        ),
                        dim=1,
                    )
                    + self._get_codec_pad_embedding(input_id[:, 3:-5].shape[1] + 1),
                    tts_pad_embed + self.codec_bos_embed,
                ],
                dim=1,
            )
            trailing_text_hidden = tts_pad_embed
        else:
            logger.info("Using streaming mode for talker input embedding")
            talker_input_embed = torch.cat(
                [
                    talker_input_embed,
                    self.text_projection(
                        self.talker.get_text_embeddings()(input_id[:, 3:4])
                    )
                    + codec_input_embedding[:, -1:],
                ],
                dim=1,
            )
            trailing_text_hidden = torch.cat(
                (
                    self.text_projection(
                        self.talker.get_text_embeddings()(input_id[:, 4:-5])
                    ),
                    tts_eos_embed,
                ),
                dim=1,
            )

        perf.stop(PerfKey.EMBEDDING_PREP)

        perf.start(PerfKey.TALKER_PREFILL)
        logits, past_hidden, talker_prefill_chunks = self._run_talker_prefill(
            talker_input_embed, past_seq_length=0
        )
        perf.stop(PerfKey.TALKER_PREFILL, count=talker_prefill_chunks)

        # 获取第一个 token（使用 logits_processor）
        perf.start(PerfKey.TALKER_SAMPLING)
        first_token_logits = logits[:, -1, :].to(dtype=torch.float32)
        input_ids_for_processor = torch.empty((1, 0), dtype=torch.long)
        next_token = self._sample_next_token(
            logits_processor,
            input_ids_for_processor,
            first_token_logits,
            do_sample,
        )
        perf.stop(PerfKey.TALKER_SAMPLING, count=1)

        past_seq_length = talker_input_embed.shape[1]
        max_new_tokens = max(
            0,
            min(max_new_tokens, self.talker.context_max_length - past_seq_length - 1),
        )
        generated_talker_input_ids = torch.empty(
            (1, max_new_tokens + 1),
            dtype=torch.long,
        )
        generated_talker_input_ids[:, 0] = next_token
        generated_talker_input_len = 1

        with tqdm(
            total=max_new_tokens, desc="Generating tokens", disable=not show_progress
        ) as pbar:
            while True:
                step = past_seq_length - talker_input_embed.shape[1]

                if step >= max_new_tokens:
                    logger.info(f"Reached max_new_tokens at step {step}")
                    break

                perf.start(PerfKey.FRAME_PREPARE)
                last_id_hidden = self.talker.get_input_embeddings()(next_token)
                if last_id_hidden.dim() == 2:
                    last_id_hidden = last_id_hidden.unsqueeze(1)

                # 当前 step 的第一组 codec token 已由 Talker 给出；CodePredictor 补齐同一帧
                # 剩余 codebook token，形成 shape [1, num_code_groups] 的 codec_ids。
                predictor_input_embeds = torch.cat((past_hidden, last_id_hidden), dim=1)
                perf.stop(PerfKey.FRAME_PREPARE)

                (
                    predictor_tokens,
                    predictor_embeds,
                    cp_prepare_t,
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
                perf.add(PerfKey.CODE_PREDICTOR_PREPARE, cp_prepare_t)
                perf.add(PerfKey.CODE_PREDICTOR_PREFILL, cp_pf_t, count=cp_pf_n)
                perf.add(PerfKey.CODE_PREDICTOR_DECODE, cp_dc_t, count=cp_dc_n)
                perf.add(
                    PerfKey.CODE_PREDICTOR_SAMPLING,
                    cp_sampling_t,
                    count=cp_sampling_n,
                )

                perf.start(PerfKey.FRAME_PREPARE)
                codec_ids = torch.cat(
                    (next_token.unsqueeze(0), predictor_tokens), dim=-1
                )
                perf.stop(PerfKey.FRAME_PREPARE)
                yield codec_ids

                # Talker 下一步需要知道刚生成的完整 codec frame。各 codebook embedding
                # 求和后作为下一步 Talker decode 的 codec 条件。
                perf.start(PerfKey.FRAME_PREPARE)
                codec_hiddens = torch.cat([last_id_hidden] + predictor_embeds, dim=1)
                inputs_embeds = codec_hiddens.sum(1, keepdim=True)

                if step < trailing_text_hidden.shape[1]:
                    inputs_embeds = inputs_embeds + trailing_text_hidden[
                        :, step
                    ].unsqueeze(1)
                else:
                    inputs_embeds = inputs_embeds + tts_pad_embed
                perf.stop(PerfKey.FRAME_PREPARE)

                perf.start(PerfKey.TALKER_DECODE)
                logits, past_hidden = self._run_talker_decode(
                    inputs_embeds, past_seq_length
                )
                perf.stop(PerfKey.TALKER_DECODE, count=1)

                perf.start(PerfKey.TALKER_SAMPLING)
                next_token_logits = logits[:, 0, :].to(dtype=torch.float32)
                next_token = self._sample_next_token(
                    logits_processor,
                    generated_talker_input_ids[:, :generated_talker_input_len],
                    next_token_logits,
                    do_sample,
                )
                perf.stop(PerfKey.TALKER_SAMPLING, count=1)

                generated_talker_input_ids[:, generated_talker_input_len] = next_token
                generated_talker_input_len += 1
                past_seq_length += 1
                pbar.update(1)

                if next_token.item() == self.codec_eos_token_id:
                    logger.info(f"Reached EOS token at step {step + 1}")
                    break

    @torch.no_grad()
    def generate_custom_voice(
        self,
        text: str,
        language: str,
        speaker: str,
        speech_tokenizer: "Qwen3TTSSpeechTokenizerInference",
        logits_processors: Tuple[LogitsProcessorList, LogitsProcessorList],
        non_streaming_mode: bool = True,
        max_new_tokens: int = 4096,
        do_sample: bool = True,
        subtalker_dosample: bool = True,
        show_progress: bool = True,
        **kwargs,
    ):
        """
        生成自定义语音

        Args:
            text: 待合成的文本
            language: 语言
            speaker: 说话人
            speech_tokenizer: 音频解码器实例
            logits_processors: 预初始化的 Talker / CodePredictor logits processors
            non_streaming_mode: 是否使用非流式模式
            max_new_tokens: 最大生成 token 数
            do_sample: 是否使用采样
            subtalker_dosample: subtalker 是否采样
            show_progress: 是否显示 tqdm 生成进度条

        Returns:
            wavs: 音频波形列表
            sr: 采样率
        """
        logger.info(f"Starting TTS generation for text: {text[:50]}...")

        perf = PerfTracker()

        generated_codes = list(
            self._generate_codec_frames(
                text=text,
                language=language,
                speaker=speaker,
                logits_processors=logits_processors,
                perf=perf,
                non_streaming_mode=non_streaming_mode,
                max_new_tokens=max_new_tokens,
                do_sample=do_sample,
                subtalker_dosample=subtalker_dosample,
                show_progress=show_progress,
            )
        )

        # 生成循环中 EOS 可能已经进入第一组 codebook。送入 SpeechTokenizer 前需要裁掉
        # EOS 及其之后的无效帧，只保留真正的音频 codec frames。
        perf.start(PerfKey.POSTPROCESS)
        if generated_codes:
            talker_codes = torch.stack(generated_codes, dim=1)

            # 找到停止位置
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
        perf.stop(PerfKey.POSTPROCESS)

        # oneshot 模式在这里才开始音频解码，因此首包延迟等于完整 codec 生成 + 解码耗时。
        if talker_codes_list:
            perf.start(PerfKey.SPEECH_TOKENIZER)
            wavs, sr = speech_tokenizer.decode(
                [{"audio_codes": c} for c in talker_codes_list]
            )
            perf.stop(PerfKey.SPEECH_TOKENIZER, count=1)
        else:
            wavs, sr = [], 24000
            logger.warning("No codes generated, returning empty audio")

        perf_dict, perf_count = perf.snapshot()
        return wavs, sr, perf_dict, perf_count

    @torch.no_grad()
    def generate_streaming(
        self,
        text: str,
        language: str,
        speaker: str,
        stateful_decoder: "Qwen3TTSStatefulDecoderInference",
        logits_processors: Tuple[LogitsProcessorList, LogitsProcessorList],
        chunk_size: int = 12,
        non_streaming_mode: bool = False,
        max_new_tokens: int = 4096,
        do_sample: bool = True,
        subtalker_dosample: bool = True,
        **kwargs,
    ) -> Generator[Tuple[np.ndarray, int], None, None]:
        """流式生成语音，逐 chunk yield (audio_samples, sample_rate)

        生成过程中每攒满 chunk_size 帧 codec codes 就通过 stateful_decoder 解码并 yield 音频。
        non_streaming_mode 只控制 Talker 文本条件的组织方式；函数名中的 streaming
        指音频解码和输出按 chunk 流式进行。
        """
        logger.info(f"Starting streaming TTS for text: {text[:50]}...")
        # 提前创建并上传全零 decoder state；首包延迟从 state 准备完成后开始统计。
        decoder_state = stateful_decoder.create_state()
        streaming_start_time = time.perf_counter()
        first_chunk_emitted = False
        self.last_streaming_first_chunk_latency_ms = None

        perf = PerfTracker()

        # ===== Streaming Decode 循环 =====
        # codec frame 生成逻辑与 oneshot 共用，但这里不全部攒到最后。
        # 每生成一帧就写入 buffer，满 chunk_size 后立即调用 StatefulDecoder 并 yield 音频。
        buffer = CodeChunkBuffer(chunk_size=chunk_size)

        for codec_ids in self._generate_codec_frames(
            text=text,
            language=language,
            speaker=speaker,
            logits_processors=logits_processors,
            perf=perf,
            non_streaming_mode=non_streaming_mode,
            max_new_tokens=max_new_tokens,
            do_sample=do_sample,
            subtalker_dosample=subtalker_dosample,
            show_progress=False,
        ):
            buffer.push(codec_ids.squeeze(0))

            chunk = buffer.flush()
            if chunk is not None:
                chunk_np = to_numpy(chunk).astype(np.int32)
                perf.start(PerfKey.STATEFUL_DECODER)
                audio, decoder_state = stateful_decoder.decode(
                    chunk_np, decoder_state, is_final=False
                )
                perf.stop(PerfKey.STATEFUL_DECODER, count=1)
                if len(audio) > 0:
                    if not first_chunk_emitted:
                        self.last_streaming_first_chunk_latency_ms = (
                            time.perf_counter() - streaming_start_time
                        ) * 1000
                        first_chunk_emitted = True
                    yield audio, 24000

        # ===== Flush remaining frames =====
        # 生成结束时可能还剩不足 chunk_size 的 codec frames。final decode 会通知
        # StatefulDecoder 处理 padding、输出剩余音频，并清理内部 latent 缓冲。
        residual = buffer.finalize()
        if residual is not None:
            residual_np = to_numpy(residual).astype(np.int32)
            perf.start(PerfKey.STATEFUL_DECODER)
            audio, decoder_state = stateful_decoder.decode(
                residual_np, decoder_state, is_final=True
            )
            perf.stop(PerfKey.STATEFUL_DECODER, count=1)
            if len(audio) > 0:
                if not first_chunk_emitted:
                    self.last_streaming_first_chunk_latency_ms = (
                        time.perf_counter() - streaming_start_time
                    ) * 1000
                    first_chunk_emitted = True
                yield audio, 24000
        else:
            # 最后一块刚好满 chunk_size 时，发送 0 帧 final decode 以 flush latent audio。
            perf.start(PerfKey.STATEFUL_DECODER)
            audio, decoder_state = stateful_decoder.decode(
                np.zeros((0, 16), dtype=np.int32), decoder_state, is_final=True
            )
            perf.stop(PerfKey.STATEFUL_DECODER, count=1)
            if len(audio) > 0:
                if not first_chunk_emitted:
                    self.last_streaming_first_chunk_latency_ms = (
                        time.perf_counter() - streaming_start_time
                    ) * 1000
                    first_chunk_emitted = True
                yield audio, 24000

        # 保存性能数据
        self.last_streaming_perf, self.last_streaming_perf_count = perf.snapshot()


def main(args: argparse.Namespace) -> None:
    torch.manual_seed(args.seed)

    # 构建 HMM 模型配置
    hf_model_dir = args.hf_model_dir

    # 创建 Qwen3TTSCodecGenerator 实例
    codec_generator = Qwen3TTSCodecGenerator(
        hf_model=hf_model_dir,
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
        ndevice=args.ndevice,
    )

    if args.mode == "streaming":
        # ===== 流式模式 =====
        stateful_decoder = Qwen3TTSStatefulDecoderInference(
            hmm_file=args.stateful_decoder_hmm,
            chunk_size=args.chunk_size,
            ndevice=args.ndevice,
        )
        # 预初始化采样 processor
        logits_processors = codec_generator.init_logits_processors()

        first_chunk_latency_ms = None
        chunk_count = 0
        output_sr = None
        all_chunks = []
        playback_gap = StreamingPlaybackGapTracker()
        out_file = Path(args.output_wav)
        out_file.parent.mkdir(parents=True, exist_ok=True)
        if out_file.exists():
            out_file.unlink()
        start_time = time.perf_counter()

        for audio_chunk, sr in codec_generator.generate_streaming(
            text=args.text,
            language=args.language,
            speaker=args.speaker,
            stateful_decoder=stateful_decoder,
            chunk_size=args.chunk_size,
            logits_processors=logits_processors,
        ):
            chunk_emit_time = time.perf_counter()
            chunk_count += 1
            if first_chunk_latency_ms is None:
                first_chunk_latency_ms = getattr(
                    codec_generator,
                    "last_streaming_first_chunk_latency_ms",
                    None,
                )
                if first_chunk_latency_ms is None:
                    first_chunk_latency_ms = (time.perf_counter() - start_time) * 1000
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

        inference_time = time.perf_counter() - start_time

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

            # 流式性能分析
            perf = getattr(codec_generator, "last_streaming_perf", {})
            perf_count = getattr(codec_generator, "last_streaming_perf_count", {})
            if perf:
                log_streaming_perf(perf, perf_count, inference_time)
        else:
            logger.error("No audio generated")

    else:
        # ===== Oneshot 模式 =====
        speech_tokenizer = Qwen3TTSSpeechTokenizerInference(
            hmm_file=args.speech_tokenizer_hmm,
            decode_padding_shapes=args.speech_tokenizer_decode_padding_shapes,
            ndevice=args.ndevice,
        )
        # 预初始化采样 processor，避免 oneshot 推理计时中构造 Python processor 对象。
        logits_processors = codec_generator.init_logits_processors()

        start_time = time.perf_counter()
        wavs, sr, perf, perf_count = codec_generator.generate_custom_voice(
            text=args.text,
            language=args.language,
            speaker=args.speaker,
            speech_tokenizer=speech_tokenizer,
            logits_processors=logits_processors,
            show_progress=not args.no_progress,
        )
        inference_time = time.perf_counter() - start_time

        out_file = Path(args.output_wav)
        out_file.parent.mkdir(parents=True, exist_ok=True)
        if out_file.exists():
            out_file.unlink()

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
        else:
            logger.error("No audio generated")

        log_oneshot_perf(perf, perf_count, inference_time)


def get_args() -> argparse.Namespace:
    """Parse commandline."""
    # fmt: off
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--config", dest="config_path", type=str, default=DEFAULT_CONFIG_PATH, help="path to config.yaml")
    parser.add_argument("--model_name", type=str, default=None, help="模型名称")
    parser.add_argument("--model_size", type=str, default=None, help="模型大小")
    parser.add_argument("--hf_model_dir", type=str, default=None, help="Modelscope 模型目录路径")
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
    parser.add_argument("--stateful_decoder_hmm", type=str, default=None, help="Stateful Decoder HMM 文件路径 (streaming 模式必需)")
    parser.add_argument("--ndevice", type=int, default=1, help="设备数量")
    parser.add_argument("--output_wav", type=str, default="./output_custom_voice.wav", help="输出 wav 文件路径，默认保存到当前目录下的 output_custom_voice.wav")
    parser.add_argument("--text", type=str, default="基于先进的存算一体技术和存储工艺，后摩智能致力于突破芯片的性能与功耗瓶颈，加速人工智能技术的普惠落地。", help="待合成的文本，若不指定则使用默认文本")
    parser.add_argument("--language", type=str, default="Chinese", choices=["auto", "Chinese", "English", "Japanese", "Korean", "French", "German", "Spanish", "Italian", "Portuguese", "Russian"], help="语言，默认为 Chinese")
    parser.add_argument("--speaker", type=str, default="vivian", choices=["vivian", "serena", "uncle_fu", "ryan", "aiden", "ono_anna", "sohee", "eric", "dylan"], help="说话人，默认为 vivian")
    parser.add_argument("--seed", type=int, default=1024, help="随机种子")
    parser.add_argument("--mode", type=str, default="oneshot", choices=["oneshot", "streaming"], help="推理模式: oneshot=全量生成; streaming=流式生成")
    parser.add_argument("--chunk_size", type=int, default=12, help="流式模式下每个解码块的 codec 帧数 (默认 12, 即 1 秒 @12Hz)")
    parser.add_argument("--no_progress", action="store_true", help="关闭 oneshot 模式下的 tqdm 生成进度条，便于性能评测")
    # fmt: on
    args = parser.parse_args()

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

    args.hf_model_dir = first_not_none(
        args.hf_model_dir,
        get_default_hf_model_dir(model_config),
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
