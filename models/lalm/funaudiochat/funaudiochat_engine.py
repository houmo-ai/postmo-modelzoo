# Copyright (c) 2026 HOUMO AI
#
# File: funaudiochat_engine.py
# Description:
#   End-to-end Fun-Audio-Chat inference orchestration.
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

"""Engine orchestration for Fun-Audio-Chat inference pipelines."""

from __future__ import annotations

import numpy as np
import torch

from houmo_engine import HoumoEngine
from houmo_engine.core.types import Stage
from houmo_engine.perf import PerfTracker
from houmo_engine.sampling import GreedySampler, GreedySamplingParams

from funaudiochat_module import FunAudioChatModule
from funaudiochat_process import (
    AUDIO_BOS_ID,
    AUDIO_EOS_ID,
    AUDIO_TOKENS_PER_SECOND,
    FLOW_CFG_RATE,
    FLOW_STEPS,
    GROUP_SIZE,
    SAMPLE_RATE,
    SAMPLES_PER_TOKEN,
    TEXT_AUDIO_BOS_ID,
    TEXT_AUDIO_EOS_ID,
    TOKEN_MEL_RATIO,
    FunAudioChatProcess,
)
from funaudiochat_types import (
    AudioResult,
    FunAudioChatPaths,
    FunAudioChatRequest,
    FunAudioChatState,
    PerformanceResult,
    SpeechResult,
    TextResult,
    TurnResult,
    VadResult,
)


class FunAudioChatEngine(HoumoEngine):
    """Coordinate preprocessing, HMM execution, sampling, and postprocessing."""

    def __init__(
        self,
        paths: FunAudioChatPaths,
        *,
        stage: str = "s2t",
        device: int = 0,
        ndevice: int = 1,
        batch: int = 1,
        temperature: float = 0.6,
        top_k: int = 20,
        top_p: float = 0.95,
        repetition_penalty: float = 1.2,
        force_audio_bos: bool = True,
        token_hop: int = 125,
        token_overlap: int = 3,
        fade_ms: float = 5.0,
        seed: int = 42,
        perf: bool = False,
    ):
        super().__init__(batch=batch)
        if self.batch != 1:
            raise ValueError("FunAudioChatEngine only supports batch=1")
        if stage not in ("s2t", "s2s", "e2e"):
            raise ValueError(f"unsupported stage: {stage}")
        self.stage = stage
        self.perf = PerfTracker.create(perf)
        self.text_sampler = GreedySampler(GreedySamplingParams())
        self.temperature = float(temperature)
        self.top_k = int(top_k)
        self.top_p = float(top_p)
        self.repetition_penalty = float(repetition_penalty)
        self.force_audio_bos = bool(force_audio_bos)
        self.token_hop = int(token_hop)
        self.token_overlap = int(token_overlap)
        self.fade_ms = float(fade_ms)
        self.seed = int(seed)
        load_s2s = stage in ("s2s", "e2e")
        with self.perf.scope("lalm.init"):
            self.module = FunAudioChatModule(
                paths,
                device=device,
                ndevice=ndevice,
                load_s2s=load_s2s,
                load_token2wav=load_s2s,
                load_vad=stage == "e2e",
                perf=self.perf,
            )
            self.process = FunAudioChatProcess(
                paths.tokenizer_dir,
                paths.embedding_path,
                paths.audio_embedding_path,
                paths.pre_matching_path,
                paths.flow_input_embedding_path,
                paths.speaker_info_path,
                paths.config_path,
                paths.cmvn_path,
                prefill_length=self.module.prefill_length,
                hidden_size=self.module.hidden_size,
                audio_encoder_shapes=self.module.audio_encoder_shapes,
                flow_token_capacity=self.module.flow_token_capacity,
                flow_mel_capacity=self.module.flow_mel_capacity,
                hift_mel_capacity=self.module.hift_mel_capacity,
                load_s2s=load_s2s,
                load_vad=stage == "e2e",
                perf=self.perf,
            )
        self.state = FunAudioChatState()

    def _audio_encode(self, request):
        with self.perf.scope("lalm.audio_encoder"):
            inputs = self.process.prepare_audio_encode(request)
            self.module.set_input(Stage.ENCODE, inputs)
            self.module.run(Stage.ENCODE)
            return self.module.get_output(Stage.ENCODE)

    def _language_prefill(self, request):
        with self.perf.scope("lalm.prefill"):
            inputs = self.process.language_prefill_inputs(request)
            self.module.set_input(Stage.PREFILL, inputs)
            self.module.run(Stage.PREFILL)
            return self.module.get_output(Stage.PREFILL)

    def _language_decode(self, token: int, audio_features=None):
        with self.perf.scope("lalm.decode"):
            inputs = self.process.language_decode_inputs(token, self.state.context_length, audio_features)
            text_embedding = inputs.metadata["text_embedding"]
            self.module.set_input(Stage.DECODE, inputs)
            self.module.run(Stage.DECODE)
            outputs = self.module.get_output(Stage.DECODE)
        self.state.context_length += 1
        return outputs, text_embedding

    def _sample_speech(self, logits) -> int:
        scores = logits.reshape(-1, logits.shape[-1])[-1].astype(np.float64, copy=True)
        if self.temperature <= 0:
            raise ValueError("--temperature must be greater than zero")
        scores /= self.temperature
        if self.repetition_penalty != 1.0:
            for token in set(self.state.generated_speech_tokens):
                if 0 <= token < scores.size:
                    scores[token] = scores[token] * self.repetition_penalty if scores[token] < 0 else scores[token] / self.repetition_penalty
        scores -= np.max(scores)
        probabilities = np.exp(scores)
        probabilities /= probabilities.sum()
        if 0 < self.top_k < probabilities.size:
            indices = np.argpartition(probabilities, -self.top_k)[-self.top_k:]
            filtered = np.zeros_like(probabilities)
            filtered[indices] = probabilities[indices]
            probabilities = filtered / filtered.sum()
        if 0 < self.top_p < 1.0:
            order = np.argsort(probabilities)[::-1]
            cumulative = np.cumsum(probabilities[order])
            remove = cumulative > self.top_p
            remove[0] = False
            probabilities[order[remove]] = 0.0
            probabilities /= probabilities.sum()
        return int(self.speech_rng.choice(probabilities.size, p=probabilities))

    def _crq_decode_step(self, hidden, previous_embedding) -> tuple[int, np.ndarray]:
        if self.state.crq_past_length >= self.module.crq_context_length:
            raise ValueError("audio decoder CRQ context is full")
        inputs = self.process.crq_decode_inputs(hidden, previous_embedding, self.state.crq_past_length, self.module.crq_context_length)
        token = self._sample_speech(self.module.run_crq_decode(inputs).tensors[0])
        self.state.generated_speech_tokens.append(token)
        self.state.crq_past_length += 1
        return token, self.process.audio_embedding[token]

    def _warmup_crq(self, condition) -> None:
        expanded = self.process.pre_matching(condition).reshape(1, condition.shape[1] * GROUP_SIZE, self.module.hidden_size)
        previous = self.process.audio_embedding[AUDIO_BOS_ID]
        inputs, valid_length = self.process.crq_prefill_inputs(expanded, previous, self.module.crq_context_length)
        token = self._sample_speech(self.module.run_crq_prefill(inputs).tensors[0][:, :valid_length, :])
        previous = self.process.audio_embedding[token]
        self.state.crq_past_length = valid_length
        for sub_position in range(1, GROUP_SIZE):
            index = condition.shape[1] * GROUP_SIZE - (GROUP_SIZE - sub_position)
            _, previous = self._crq_decode_step(expanded[:, index : index + 1, :], previous)
        self.previous_audio_embedding = previous

    def _speech_group(self, condition) -> list[int]:
        expanded = self.process.pre_matching(condition).reshape(1, GROUP_SIZE, self.module.hidden_size)
        values = []
        for index in range(GROUP_SIZE):
            token, self.previous_audio_embedding = self._crq_decode_step(expanded[:, index : index + 1, :], self.previous_audio_embedding)
            values.append(token)
        return values

    def _run_s2t(self, audio, system_prompt, max_new_tokens) -> TextResult:
        self.module.clear_language_session()
        self.state = FunAudioChatState()
        request = self.process.preprocess(FunAudioChatRequest("s2t", audio, system_prompt))
        audio_outputs = self._audio_encode(request)
        prefill = self.process.prepare_language_prefill(request, audio_outputs)
        if prefill.prompt_length >= self.module.prefill_length:
            raise ValueError("prompt leaves no context for generated tokens")
        outputs = self._language_prefill(prefill)
        token = self.text_sampler.sample(outputs.tensors[0][0, prefill.prompt_length - 1])
        self.state.generated_ids = [token]
        self.state.context_length = prefill.prompt_length
        limit = min(max_new_tokens, self.module.prefill_length - prefill.prompt_length)
        while token not in self.process.stop_token_ids and len(self.state.generated_ids) < limit and self.state.context_length < self.module.prefill_length:
            outputs, _ = self._language_decode(token)
            token = self.text_sampler.sample(outputs.tensors[0][0, -1])
            self.state.generated_ids.append(token)
        text = self.process.postprocess(self.state, final=True)
        self.perf.set_metrics("lalm", audio_length_s=request.audio_length_s, input_tokens=prefill.prompt_length, output_tokens=len(self.state.generated_ids), decode_tokens=max(0, len(self.state.generated_ids) - 1))
        return TextResult(text, prefill.prompt_length, len(self.state.generated_ids))

    def _run_s2s(self, audio, system_prompt, max_new_tokens) -> SpeechResult:
        self.module.clear_language_session()
        self.module.clear_crq_session()
        self.state = FunAudioChatState()
        self.speech_rng = np.random.default_rng(self.seed)
        request = self.process.preprocess(FunAudioChatRequest("s2s", audio, system_prompt))
        audio_outputs = self._audio_encode(request)
        prefill = self.process.prepare_language_prefill(request, audio_outputs)
        outputs = self._language_prefill(prefill)
        logits, hidden = outputs.tensors
        self._warmup_crq(hidden[:, : prefill.prompt_length, :] + prefill.original_text_embeds[:, : prefill.prompt_length, :])
        token = TEXT_AUDIO_BOS_ID if self.force_audio_bos else self.text_sampler.sample(logits[0, prefill.prompt_length - 1])
        self.state.generated_ids = [token]
        self.state.generate_speech = token == TEXT_AUDIO_BOS_ID
        self.state.context_length = prefill.prompt_length
        limit = min(max_new_tokens, self.module.prefill_length - prefill.prompt_length)
        while token not in self.process.stop_token_ids and len(self.state.generated_ids) < limit and self.state.context_length < self.module.prefill_length:
            tower = None
            if self.state.generate_speech and self.state.speech_ids:
                tower = self.module.run_audio_tower(self.process.audio_tower_inputs(self.state.speech_ids[-GROUP_SIZE:])).tensors[0]
            outputs, text_embedding = self._language_decode(token, tower)
            text_logits, last_hidden = outputs.tensors
            speech_group = self._speech_group(last_hidden + text_embedding) if self.state.generate_speech else None
            next_token = self.text_sampler.sample(text_logits[0, -1])
            if next_token == TEXT_AUDIO_BOS_ID:
                self.state.generate_speech = True
            if speech_group is not None:
                if AUDIO_EOS_ID in speech_group:
                    speech_group = [AUDIO_EOS_ID] * GROUP_SIZE
                    next_token = TEXT_AUDIO_EOS_ID
                    self.state.generate_speech = False
                self.state.speech_ids.extend(speech_group)
            token = next_token
            self.state.generated_ids.append(token)
        text = self.process.postprocess(self.state, final=True)
        self.perf.set_metrics("lalm", audio_length_s=request.audio_length_s, input_tokens=prefill.prompt_length, output_tokens=len(self.state.generated_ids), decode_tokens=max(0, len(self.state.generated_ids) - 1), speech_tokens=len(self.state.speech_ids))
        return SpeechResult(text, list(self.state.speech_ids), prefill.prompt_length, len(self.state.generated_ids))

    def _flow(self, tokens: list[int], generator: torch.Generator):
        encoder, normalized, mask, x, mel_length = self.process.prepare_flow(tokens, generator)
        hidden = torch.from_numpy(self.module.run_flow_encoder(encoder).tensors[0]).repeat_interleave(TOKEN_MEL_RATIO, dim=1)
        speaker = torch.from_numpy(self.module.run_flow_spk(normalized).tensors[0])
        mu = hidden.transpose(1, 2).contiguous().to(torch.float16)
        t_span = torch.linspace(0, 1, FLOW_STEPS + 1, dtype=torch.float16)
        t_span = 1 - torch.cos(t_span * 0.5 * torch.pi)
        t = t_span[0]
        dt = t_span[1] - t_span[0]
        for step in range(1, len(t_span)):
            values = self.process.prepare_flow_decoder(x, mask, mu, speaker, t)
            derivative = torch.from_numpy(self.module.run_flow_decoder(values).tensors[0])
            x = x + dt * ((1.0 + FLOW_CFG_RATE) * derivative[0:1] - FLOW_CFG_RATE * derivative[1:2])
            t = t + dt
            if step < len(t_span) - 1:
                dt = t_span[step + 1] - t
        return x[:, :, :mel_length].to(torch.float32), mel_length

    def _vocoder(self, mel, mel_length):
        padded = torch.nn.functional.pad(mel, (0, self.module.hift_mel_capacity - mel.shape[2]))
        source = self.module.run_hift_part1(padded).tensors[0]
        output = self.module.run_hift_part2(self.process.stft(source)).tensors[0]
        return torch.from_numpy(output).to(torch.float32)[:, : 480 * mel_length]

    def _token2wav(self, speech_ids: list[int]):
        tokens = [token for token in speech_ids if 0 <= token < AUDIO_BOS_ID]
        if not tokens:
            raise ValueError("no valid CosyVoice codec tokens were generated")
        if self.token_hop <= 0:
            raise ValueError("--token_hop must be greater than zero")
        if not 0 <= self.token_overlap < self.token_hop:
            raise ValueError("--token_overlap must be >= 0 and smaller than --token_hop")
        if self.token_hop + self.token_overlap > self.process.max_chunk_tokens:
            raise ValueError(f"token hop + overlap must not exceed {self.process.max_chunk_tokens}")
        generator = torch.Generator().manual_seed(self.seed)
        chunks = []
        start = 0
        while start < len(tokens):
            chunk_start = max(0, start - self.token_overlap)
            chunk_end = min(start + self.token_hop, len(tokens))
            overlap = start - chunk_start
            mel, length = self._flow(tokens[chunk_start:chunk_end], generator)
            waveform = self._vocoder(mel, length)
            if overlap:
                waveform = waveform[:, overlap * SAMPLES_PER_TOKEN :]
            chunks.append(self.process.fade(waveform, self.fade_ms))
            start = chunk_end
        return torch.cat(chunks, dim=1)

    def _run_vad(self, audio) -> VadResult:
        request = self.process.prepare_vad(audio)
        scores = self.module.run_vad(request.features).tensors[0]
        segments, stats = self.process.vad_postprocess(scores, request.waveform)
        return VadResult(request.waveform, request.sample_rate, segments, stats)

    def _split_vad_segment(self, segment, start_ms: int, end_ms: int, sample_rate: int):
        """Split a VAD segment into independent turns accepted by the audio encoder HMM."""
        speech_capacity = int(self.module.audio_encoder_shapes["speech_ids"][1])
        group_size = int(self.process.processor.audio_group_size)
        usable_capacity = speech_capacity - speech_capacity % group_size
        max_samples = min(
            self.process.static_audio_samples,
            int(usable_capacity * sample_rate / AUDIO_TOKENS_PER_SECOND),
        )
        if max_samples <= 0:
            raise ValueError("audio encoder speech capacity must be greater than zero")
        if len(segment) <= max_samples:
            return [(segment, start_ms, end_ms)]

        base_sample = int(start_ms * sample_rate / 1000)
        chunks = []
        for offset in range(0, len(segment), max_samples):
            chunk = segment[offset : offset + max_samples]
            chunk_start = base_sample + offset
            chunk_end = chunk_start + len(chunk)
            chunks.append(
                (
                    chunk,
                    int(chunk_start * 1000 / sample_rate),
                    int(chunk_end * 1000 / sample_rate),
                )
            )
        return chunks

    def print_perf(self) -> None:
        self.perf.print_summary()

    def _run_timed(self, path: str, function, *args):
        self.perf.start(path)
        try:
            result = function(*args)
        finally:
            self.perf.end(path)
        return result

    def _run_token2wav(self, speech_ids: list[int]):
        waveform = self._run_timed("lalm.e2e_token2wav", self._token2wav, speech_ids)
        self.perf.set_metrics(
            "lalm",
            output_audio_length_s=waveform.shape[1] / SAMPLE_RATE,
        )
        return waveform

    def generate(self, request, *, system_prompt: str, max_new_tokens: int = 2048):
        if max_new_tokens <= 0:
            raise ValueError("max_new_tokens must be greater than zero")
        self.perf.reset(preserve_prefixes=("lalm.init",))
        if self.stage == "s2t":
            result = self._run_timed("lalm.e2e_s2t", self._run_s2t, request, system_prompt, max_new_tokens)
            yield result
            return
        if self.stage == "s2s":
            result = self._run_timed("lalm.e2e_s2s", self._run_s2s, request, system_prompt, max_new_tokens)
            waveform = self._run_token2wav(result.speech_ids)
            yield result
            yield AudioResult(waveform, SAMPLE_RATE, result.speech_ids)
            return

        vad = self._run_vad(request)
        vad_perf = PerformanceResult("E2E VAD", self.perf.summary()) if self.perf.enabled else None
        self.perf.reset()
        yield vad
        if vad_perf is not None:
            yield vad_perf
        if not vad.segments:
            raise RuntimeError("VAD did not detect any speech segments")
        turn_index = 0
        for start_ms, end_ms in vad.segments:
            start = int(start_ms * vad.sample_rate / 1000)
            end = int(end_ms * vad.sample_rate / 1000)
            segment = vad.waveform[start:end]
            for chunk, chunk_start_ms, chunk_end_ms in self._split_vad_segment(
                segment, start_ms, end_ms, vad.sample_rate
            ):
                result = self._run_timed("lalm.e2e_s2s", self._run_s2s, chunk, system_prompt, max_new_tokens)
                response = self._run_token2wav(result.speech_ids)
                turn_perf = PerformanceResult(f"E2E Turn {turn_index}", self.perf.summary()) if self.perf.enabled else None
                self.perf.reset()
                yield TurnResult(turn_index, chunk_start_ms, chunk_end_ms, chunk, result.text, result.speech_ids, response, SAMPLE_RATE)
                if turn_perf is not None:
                    yield turn_perf
                turn_index += 1


__all__ = ["FunAudioChatEngine"]
