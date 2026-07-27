# Copyright (c) 2026 HOUMO AI
#
# File: qwen3_asr.py
# Description:
#   Qwen3-ASR inference Engine implementation.
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

from ..core import HoumoEngine
from ..core.types import GenerationState, Stage
from ..module.qwen3_asr import Qwen3AsrModule
from ..perf import PerfTracker
from ..process.qwen3_asr import Qwen3AsrProcess
from ..sampling import GreedySampler, GreedySamplingParams


class Qwen3AsrEngine(HoumoEngine):
    """Qwen3-ASR inference orchestration over process and module layers."""

    def __init__(
        self,
        encode_path,
        prefill_path,
        decode_path,
        embedding_path,
        processor_path,
        *,
        ndevice: int = 1,
        batch: int = 1,
        sampling_params: GreedySamplingParams | None = None,
        perf: bool = False,
    ):
        super().__init__(batch=batch)
        if self.batch != 1:
            raise ValueError("Qwen3AsrEngine only supports batch=1")
        self.perf = PerfTracker.create(perf)
        self.sampler = GreedySampler(sampling_params)
        with self.perf.scope("asr.init"):
            self.module = Qwen3AsrModule(
                encode_path,
                prefill_path,
                decode_path,
                ndevice=ndevice,
                perf=self.perf,
            )
            self.process = Qwen3AsrProcess(
                processor_path,
                embedding_path,
                self.module.embedding_size,
                self.module.encode_feature_length,
                self.module.prefill_length,
                perf=self.perf,
            )
        eos = self.process.tokenizer.eos_token_id
        self.stop_token_ids = set(eos if isinstance(eos, list) else [eos])
        self.state = GenerationState()

    @property
    def context_max_length(self) -> int:
        return self.module.context_max_length

    def clear_session(self) -> None:
        self.state = GenerationState()
        self.module.clear_session()

    def _encode(self, request, chunk):
        with self.perf.scope("asr.encode"):
            inputs = self.process.prepare_encode(chunk)
            self.module.set_input(Stage.ENCODE, inputs)
            self.module.run(Stage.ENCODE)
            outputs = self.module.get_output(Stage.ENCODE)
            return self.process.merge_encode(request, chunk, outputs)

    def _prefill(self, request) -> int:
        with self.perf.scope("asr.prefill"):
            if request.current_length >= self.context_max_length:
                raise ValueError("input exceeds model context length")
            inputs = self.process.prepare_prefill(request, self.state)
            self.module.set_input(Stage.PREFILL, inputs)
            self.module.run(Stage.PREFILL)
            logits = self.module.get_output(Stage.PREFILL).tensors[0]
            token = self.sampler.sample(logits)
        self.state.context_length += request.current_length
        return token

    def _decode(self, token: int) -> int:
        with self.perf.scope("asr.decode"):
            inputs = self.process.prepare_decode(token, self.state)
            self.module.set_input(Stage.DECODE, inputs)
            self.module.run(Stage.DECODE)
            logits = self.module.get_output(Stage.DECODE).tensors[0]
            token = self.sampler.sample(logits, previous_tokens=self.state.generated_ids)
        self.state.context_length += 1
        return token

    def generate(
        self,
        audio,
        *,
        sampling_params: GreedySamplingParams | None = None,
        max_new_tokens: int | None = None,
        system_prompt: str | None = None,
    ):
        if audio is None:
            raise ValueError("audio must not be empty")
        if max_new_tokens is not None and max_new_tokens <= 0:
            raise ValueError("max_new_tokens must be greater than zero")
        if system_prompt is None:
            system_prompt = "You are a helpful assistant."
        if sampling_params is not None:
            self.sampler = GreedySampler(sampling_params)

        self.perf.reset(preserve_prefixes=("asr.init",))
        ttft_active = True
        e2e_active = True
        total_input_tokens = 0
        total_prefill_tokens = 0
        total_output_tokens = 0
        total_decode_tokens = 0
        chunk_count = 0
        self.perf.start("asr.e2e")
        self.perf.start("asr.ttft")
        try:
            request = self.process.preprocess(audio, system_prompt)
            self.perf.set_audio_length(request.audio_length_s)
            for chunk in request.feature_chunks:
                self.state = GenerationState()
                chunk_count += 1
                prefill_request = self._encode(request, chunk)
                token = self._prefill(prefill_request)
                self.state.generated_ids.append(token)
                total_input_tokens += int(request.input_ids.numel())
                total_prefill_tokens += prefill_request.current_length
                total_output_tokens += 1
                if chunk.chunk_index == 0:
                    self.perf.end("asr.ttft")
                    ttft_active = False

                token_limit = max_new_tokens or self.context_max_length
                while token not in self.stop_token_ids:
                    if len(self.state.generated_ids) >= token_limit:
                        break
                    if self.state.context_length >= self.context_max_length:
                        break
                    token = self._decode(token)
                    total_output_tokens += 1
                    total_decode_tokens += 1
                    if token in self.stop_token_ids:
                        break
                    self.state.generated_ids.append(token)
                    delta = self.process.postprocess(self.state)
                    if delta:
                        self.perf.end("asr.e2e")
                        e2e_active = False
                        yield delta
                        self.perf.start("asr.e2e")
                        e2e_active = True

                remainder = self.process.postprocess(self.state, final=True)
                if remainder:
                    self.perf.end("asr.e2e")
                    e2e_active = False
                    yield remainder
                    self.perf.start("asr.e2e")
                    e2e_active = True

            self.perf.set_metrics(
                "asr",
                input_tokens=total_input_tokens,
                prefill_tokens=total_prefill_tokens,
                output_tokens=total_output_tokens,
                decode_tokens=total_decode_tokens,
                chunk_count=chunk_count,
            )
        finally:
            if ttft_active:
                self.perf.end("asr.ttft")
            if e2e_active:
                self.perf.end("asr.e2e")


__all__ = ["Qwen3AsrEngine"]
