# Copyright (c) 2026 HOUMO AI
#
# File: qwen3_tts.py
# Description:
#   Qwen3-TTS inference Engine implementation.
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

"""Qwen3-TTS inference orchestration over the Process and Module layers.

The Engine drives the full synthesis pipeline for one request:

1. Talker prefill over the projected prompt embeddings.
2. First codec-group-0 token sampling from the Talker logits.
3. A decode loop that, for every frame, runs the Code Predictor prefill/decode
   sub-loop to complete the remaining codec groups, then advances the Talker by
   one step.
4. Audio decode: oneshot yields the full waveform after codec generation;
   streaming yields per-chunk waveforms through the stateful decoder.

Qwen3-TTS reproduces the original demo's stochastic sampling built from
``transformers`` logits processors followed by multinomial sampling. That
sampler lives here (rather than the shared GreedySampler) because deviating
from it would change generated audio.
"""

import torch

from ..core import HoumoEngine
from ..core.types import Stage
from ..module.qwen3_tts import Qwen3TtsModule
from ..perf import PerfTracker
from ..process.qwen3_tts import Qwen3TtsGenerationState, Qwen3TtsProcess
from ..sampling.tts import Qwen3TtsSamplingParams


_FRAME_PREPARE_SCOPE = "tts.frame_prepare"


class Qwen3TtsEngine(HoumoEngine):
    """Qwen3-TTS Talker + Code Predictor orchestration and audio decode."""

    def __init__(
        self,
        hf_model_dir,
        text_projection_path,
        talker_prefill_path,
        talker_decode_path,
        code_predictor_prefill_path,
        code_predictor_decode_path,
        talker_token_embedding_path,
        talker_text_embedding_path,
        code_predictor_token_embedding_path,
        *,
        mode: str = "oneshot",
        speech_tokenizer_path=None,
        stateful_decoder_path=None,
        decode_padding_shapes_path=None,
        chunk_size: int = 12,
        ndevice: int = 1,
        batch: int = 1,
        sampling_params: Qwen3TtsSamplingParams | None = None,
        perf: bool = False,
    ):
        super().__init__(batch=batch)
        if self.batch != 1:
            raise ValueError("Qwen3TtsEngine only supports batch=1")
        if mode not in ("oneshot", "streaming"):
            raise ValueError(f"unsupported mode: {mode}")
        self.mode = mode
        self.chunk_size = int(chunk_size)
        self.perf = PerfTracker.create(perf)
        self.sampling_params = sampling_params or Qwen3TtsSamplingParams()

        with self.perf.scope("tts.init"):
            self.module = Qwen3TtsModule(
                text_projection_path,
                talker_prefill_path,
                talker_decode_path,
                code_predictor_prefill_path,
                code_predictor_decode_path,
                mode=mode,
                speech_tokenizer_path=speech_tokenizer_path,
                stateful_decoder_path=stateful_decoder_path,
                decode_padding_shapes_path=decode_padding_shapes_path,
                chunk_size=self.chunk_size,
                ndevice=ndevice,
                perf=self.perf,
            )
            self.process = Qwen3TtsProcess(
                hf_model_dir,
                talker_token_embedding_path,
                talker_text_embedding_path,
                code_predictor_token_embedding_path,
                perf=self.perf,
            )
            # Project the static special-token embeddings once (device op owned
            # by the Module, mirrors _init_static_embeddings in the demo).
            projected = self.module.run_text_projection(self.process.special_text_embeds())
            self.process.set_special_embeds(projected)

        self._build_logits_processors()
        self.state = Qwen3TtsGenerationState()

    # ------------------------------------------------------------------
    # Sampling (transformers logits processors + multinomial)
    # ------------------------------------------------------------------
    def _build_logits_processors(self) -> None:
        import torch.nn.functional as functional
        from transformers.generation.logits_process import (
            LogitsProcessorList,
            MinNewTokensLengthLogitsProcessor,
            RepetitionPenaltyLogitsProcessor,
            SuppressTokensLogitsProcessor,
            TemperatureLogitsWarper,
            TopKLogitsWarper,
            TopPLogitsWarper,
        )

        self._softmax = functional.softmax
        params = self.sampling_params

        talker = LogitsProcessorList()
        if params.repetition_penalty is not None and params.repetition_penalty != 1.0:
            talker.append(RepetitionPenaltyLogitsProcessor(params.repetition_penalty, prompt_ignore_length=0))
        if params.min_new_tokens is not None and params.min_new_tokens > 0:
            talker.append(
                MinNewTokensLengthLogitsProcessor(
                    prompt_length_to_skip=0,
                    min_new_tokens=params.min_new_tokens,
                    eos_token_id=self.process.codec_eos_token_id,
                )
            )
        vocab_size = self.process.vocab_size
        suppress_tokens = [i for i in range(vocab_size - 1024, vocab_size) if i != self.process.codec_eos_token_id]
        talker.append(SuppressTokensLogitsProcessor(suppress_tokens))
        talker.append(TemperatureLogitsWarper(params.temperature))
        if params.top_k is not None and params.top_k > 0:
            talker.append(TopKLogitsWarper(params.top_k))
        if params.top_p is not None and params.top_p < 1.0:
            talker.append(TopPLogitsWarper(params.top_p))

        subtalker = LogitsProcessorList()
        subtalker.append(TemperatureLogitsWarper(params.subtalker_temperature))
        if params.subtalker_top_k is not None and params.subtalker_top_k > 0:
            subtalker.append(TopKLogitsWarper(params.subtalker_top_k))
        if params.subtalker_top_p is not None and params.subtalker_top_p < 1.0:
            subtalker.append(TopPLogitsWarper(params.subtalker_top_p))

        self._talker_logits_processor = talker
        self._subtalker_logits_processor = subtalker

    def _sample(self, logits_processor, input_ids, logits, do_sample):
        import torch

        scores = logits_processor(input_ids, logits.to(dtype=torch.float32))
        if do_sample:
            probs = self._softmax(scores, dim=-1)
            return torch.multinomial(probs, num_samples=1).squeeze(-1)
        return torch.argmax(scores, dim=-1)

    # ------------------------------------------------------------------
    # Stage helpers
    # ------------------------------------------------------------------
    @property
    def talker_context_max_length(self) -> int:
        return self.module.talker_context_max_length

    def clear_session(self) -> None:
        self.state = Qwen3TtsGenerationState()

    def _talker_prefill(self, talker_input_embed):
        import torch

        logits = past_hidden = None
        with self.perf.scope("tts.talker.prefill"):
            prefill_length = self.module.talker_prefill_length
            padded, seq_length, rounds = self.process.pad_talker_prefill(talker_input_embed, prefill_length)
            self.module.reset_talker_cache()
            for round_idx in range(rounds):
                start = round_idx * prefill_length
                current_length = seq_length - start if round_idx == rounds - 1 else prefill_length
                chunk = padded[:, start : start + prefill_length, :]
                inputs = self.process.prepare_talker_prefill_chunk(chunk, start, current_length)
                self.module.set_input(Stage.PREFILL, inputs)
                self.module.run(Stage.PREFILL)
                if round_idx == rounds - 1:
                    outputs = self.module.get_output(Stage.PREFILL)
                    logits = torch.from_numpy(outputs.tensors[0])
                    past_hidden = torch.from_numpy(outputs.tensors[1])
        return logits, past_hidden, seq_length

    def _talker_decode(self, inputs_embeds, past_seq_length: int):
        import torch

        with self.perf.scope("tts.talker.decode"):
            inputs = self.process.prepare_talker_decode(inputs_embeds, past_seq_length)
            self.module.set_input(Stage.DECODE, inputs)
            self.module.run(Stage.DECODE)
            outputs = self.module.get_output(Stage.DECODE)
        return torch.from_numpy(outputs.tensors[0]), torch.from_numpy(outputs.tensors[1])

    def _code_predictor_generate(self, inputs_embeds, max_new_tokens):
        """Complete one codec frame's remaining groups. Returns (tokens, embeds)."""
        import torch

        with self.perf.scope("tts.code_predictor"):
            with self.perf.scope("tts.code_predictor.prepare"):
                prefill_length = self.module.cp_prefill_length
                inputs_embeds, seq_length, rounds = self.process.prepare_code_predictor_prefill(
                    inputs_embeds, prefill_length
                )
                self.module.reset_code_predictor_cache()

            output = None
            with self.perf.scope("tts.code_predictor.prefill"):
                for round_idx in range(rounds):
                    start = round_idx * prefill_length
                    current_length = seq_length - start if round_idx == rounds - 1 else prefill_length
                    chunk = inputs_embeds[:, start : start + prefill_length, :]
                    inputs = self.process.prepare_code_predictor_prefill_chunk(chunk, start, current_length, 0)
                    self.module.set_input(Stage.CODE_PREDICTOR_PREFILL, inputs)
                    self.module.run(Stage.CODE_PREDICTOR_PREFILL)
                    if round_idx == rounds - 1:
                        out = self.module.get_output(Stage.CODE_PREDICTOR_PREFILL)
                        output = out.tensors[0]

            with self.perf.scope("tts.code_predictor.sampling"):
                logits = torch.from_numpy(output)
                next_token_logits = logits[:, -1, :].to(dtype=torch.float32)
                input_ids = torch.empty((1, 0), dtype=torch.long)
                next_token = self._sample(
                    self._subtalker_logits_processor,
                    input_ids,
                    next_token_logits,
                    self.sampling_params.subtalker_do_sample,
                )
                input_ids = torch.cat([input_ids, next_token[:, None]], dim=-1)

            context_length = seq_length
            decode_embeds = []
            for step in range(max_new_tokens - 1):
                next_embed = self.process.code_predictor_decode_embed(step, next_token)
                decode_embeds.append(next_embed)
                with self.perf.scope("tts.code_predictor.decode"):
                    inputs = self.process.prepare_code_predictor_decode(next_embed, context_length, step + 1)
                    self.module.set_input(Stage.CODE_PREDICTOR_DECODE, inputs)
                    self.module.run(Stage.CODE_PREDICTOR_DECODE)
                    out = self.module.get_output(Stage.CODE_PREDICTOR_DECODE)
                with self.perf.scope("tts.code_predictor.sampling"):
                    next_token_logits = torch.from_numpy(out.tensors[0])[:, -1, :].to(dtype=torch.float32)
                    next_token = self._sample(
                        self._subtalker_logits_processor,
                        input_ids,
                        next_token_logits,
                        self.sampling_params.subtalker_do_sample,
                    )
                    input_ids = torch.cat([input_ids, next_token[:, None]], dim=-1)
                context_length += 1
        return input_ids, decode_embeds

    # ------------------------------------------------------------------
    # Codec frame generation (shared by oneshot and streaming)
    # ------------------------------------------------------------------
    def _prepare_talker_input(self, request):
        role_hidden = self.module.run_text_projection(request.role_text_embed)
        body_hidden = head_hidden = trailing_hidden = None
        if request.non_streaming_mode:
            body_hidden = self.module.run_text_projection(request.body_text_embed)
        else:
            head_hidden = self.module.run_text_projection(request.head_text_embed)
            trailing_hidden = self.module.run_text_projection(request.trailing_text_embed)
        return self.process.build_talker_input(
            request,
            role_hidden,
            body_hidden=body_hidden,
            head_hidden=head_hidden,
            trailing_hidden=trailing_hidden,
        )

    def _generate_codec_frames(self, talker_input_embed, trailing_text_hidden):
        import torch

        logits, past_hidden, seq_length = self._talker_prefill(talker_input_embed)

        with self.perf.scope("tts.talker.sampling"):
            first_token_logits = logits[:, -1, :].to(dtype=torch.float32)
            input_ids_for_processor = torch.empty((1, 0), dtype=torch.long)
            next_token = self._sample(
                self._talker_logits_processor,
                input_ids_for_processor,
                first_token_logits,
                self.sampling_params.do_sample,
            )

        past_seq_length = talker_input_embed.shape[1]
        max_new_tokens = max(
            0,
            min(
                self._request_max_new_tokens,
                self.talker_context_max_length - past_seq_length - 1,
            ),
        )
        generated_talker_input_ids = torch.empty((1, max_new_tokens + 1), dtype=torch.long)
        generated_talker_input_ids[:, 0] = next_token
        generated_talker_input_len = 1

        while True:
            step = past_seq_length - talker_input_embed.shape[1]
            if step >= max_new_tokens:
                break

            with self.perf.scope(_FRAME_PREPARE_SCOPE):
                last_id_hidden = self.process.talker_token_embed(next_token)
                predictor_input_embeds = torch.cat((past_hidden, last_id_hidden), dim=1)
            predictor_tokens, predictor_embeds = self._code_predictor_generate(
                predictor_input_embeds, self.process.num_code_groups - 1
            )
            with self.perf.scope(_FRAME_PREPARE_SCOPE):
                codec_ids = torch.cat((next_token.unsqueeze(0), predictor_tokens), dim=-1)
            yield codec_ids

            with self.perf.scope(_FRAME_PREPARE_SCOPE):
                inputs_embeds = self.process.prepare_talker_decode_input(
                    last_id_hidden, predictor_embeds, step, trailing_text_hidden
                )
            logits, past_hidden = self._talker_decode(inputs_embeds, past_seq_length)

            with self.perf.scope("tts.talker.sampling"):
                next_token_logits = logits[:, 0, :].to(dtype=torch.float32)
                next_token = self._sample(
                    self._talker_logits_processor,
                    generated_talker_input_ids[:, :generated_talker_input_len],
                    next_token_logits,
                    self.sampling_params.do_sample,
                )
            generated_talker_input_ids[:, generated_talker_input_len] = next_token
            generated_talker_input_len += 1
            past_seq_length += 1

            if next_token.item() == self.process.codec_eos_token_id:
                break

    # ------------------------------------------------------------------
    # Public entry point
    # ------------------------------------------------------------------
    @torch.inference_mode()
    def generate(
        self,
        text: str,
        *,
        language: str = "Chinese",
        speaker: str = "vivian",
        sampling_params: Qwen3TtsSamplingParams | None = None,
        max_new_tokens: int = 4096,
        non_streaming_mode: bool | None = None,
        **kwargs,
    ):
        del kwargs
        if not text:
            raise ValueError("text must not be empty")
        if max_new_tokens is not None and max_new_tokens <= 0:
            raise ValueError("max_new_tokens must be greater than zero")
        if sampling_params is not None:
            self.sampling_params = sampling_params
            self._build_logits_processors()

        # Oneshot uses non-streaming talker input; streaming uses streaming-style
        # talker input by default. Callers may override explicitly.
        if non_streaming_mode is None:
            non_streaming_mode = self.mode != "streaming"

        self.perf.reset(preserve_prefixes=("tts.init",))
        self.clear_session()
        self._request_max_new_tokens = max_new_tokens

        e2e_active = True
        self.perf.start("tts.e2e")
        try:
            with self.perf.scope("tts.embedding"):
                request = self.process.preprocess(text, language, speaker, non_streaming_mode=non_streaming_mode)
                talker_input_embed, trailing_text_hidden = self._prepare_talker_input(request)
            if self.mode == "streaming":
                yield from self._generate_streaming(talker_input_embed, trailing_text_hidden)
            else:
                yield from self._generate_oneshot(talker_input_embed, trailing_text_hidden)
        finally:
            if e2e_active:
                self.perf.end("tts.e2e")

    def _generate_oneshot(self, talker_input_embed, trailing_text_hidden):
        for codec_ids in self._generate_codec_frames(talker_input_embed, trailing_text_hidden):
            self.state.codec_frames.append(codec_ids)

        codes_list = self.process.postprocess(self.state, final=True)
        if not codes_list:
            self.perf.set_metrics("tts", output_tokens=0)
            return

        wavs, sr = self.module.run_speech_tokenizer(codes_list)
        self.perf.set_metrics("tts", output_tokens=len(self.state.codec_frames))
        for wav in wavs:
            yield wav, sr

    def _generate_streaming(self, talker_input_embed, trailing_text_hidden):
        import numpy as np

        self.module.create_decoder_state()
        buffer = _CodeChunkBuffer(self.chunk_size)

        for codec_ids in self._generate_codec_frames(talker_input_embed, trailing_text_hidden):
            self.state.codec_frames.append(codec_ids)
            buffer.push(codec_ids.squeeze(0))
            chunk = buffer.flush()
            if chunk is not None:
                chunk_np = chunk.detach().cpu().numpy().astype(np.int32)
                audio = self.module.run_stateful_decoder(chunk_np, is_final=False)
                if len(audio) > 0:
                    yield audio, 24000

        residual = buffer.finalize()
        if residual is not None:
            residual_np = residual.detach().cpu().numpy().astype(np.int32)
            audio = self.module.run_stateful_decoder(residual_np, is_final=True)
        else:
            audio = self.module.run_stateful_decoder(np.zeros((0, 16), dtype=np.int32), is_final=True)
        if len(audio) > 0:
            yield audio, 24000

        self.perf.set_metrics("tts", output_tokens=len(self.state.codec_frames))


class _CodeChunkBuffer:
    """Frame buffer that emits fixed-size codec chunks."""

    def __init__(self, chunk_size: int = 12):
        self.chunk_size = chunk_size
        self._frames = []

    def push(self, codec_ids):
        self._frames.append(codec_ids.view(16))

    def flush(self):
        import torch

        if len(self._frames) >= self.chunk_size:
            chunk = torch.stack(self._frames[: self.chunk_size])
            self._frames = self._frames[self.chunk_size :]
            return chunk
        return None

    def finalize(self):
        import torch

        if self._frames:
            chunk = torch.stack(self._frames)
            self._frames = []
            return chunk
        return None


__all__ = ["Qwen3TtsEngine"]
