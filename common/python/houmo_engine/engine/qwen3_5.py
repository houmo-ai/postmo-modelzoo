# Copyright (c) 2026 HOUMO AI
#
# File: qwen3_5.py
# Description:
#   Qwen3.5 inference Engine implementation.
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
from ..module.qwen3_5 import Qwen35Module
from ..perf import PerfTracker
from ..process.qwen3_5 import Qwen35Process
from ..sampling import GreedySampler, GreedySamplingParams


class Qwen35Engine(HoumoEngine):
    """Qwen3.5 inference orchestration over process and module layers."""

    def __init__(
        self,
        prefill_path,
        decode_path,
        embedding_path,
        tokenizer_path,
        *,
        vision_path=None,
        ndevice: int = 1,
        batch: int = 1,
        max_size_h: int = 896,
        max_size_w: int = 896,
        patch_size: int = 16,
        sampling_params: GreedySamplingParams | None = None,
        perf: bool = False,
    ):
        super().__init__(batch=batch)
        if self.batch != 1:
            raise ValueError("Qwen35Engine only supports batch=1")
        self.perf = PerfTracker.create(perf)
        self.sampler = GreedySampler(sampling_params)
        self.module = Qwen35Module(
            prefill_path,
            decode_path,
            vision_path=vision_path,
            ndevice=ndevice,
            perf=self.perf,
        )
        self.process = Qwen35Process(
            tokenizer_path,
            embedding_path,
            self.module.embedding_size,
            max_size_h=max_size_h,
            max_size_w=max_size_w,
            patch_size=patch_size,
            perf=self.perf,
        )
        eos = self.process.tokenizer.eos_token_id
        self.stop_token_ids = set(eos if isinstance(eos, list) else [eos])
        self.state = GenerationState()

    @property
    def context_length(self) -> int:
        return self.state.context_length

    @property
    def context_max_length(self) -> int:
        return self.module.context_max_length

    @property
    def prefill_length(self) -> int:
        return self.module.prefill_length

    @property
    def embedding_size(self) -> int:
        return self.module.embedding_size

    @property
    def tokenizer(self):
        return self.process.tokenizer

    def clear_session(self) -> None:
        self.state = GenerationState()
        self.module.clear_session()

    def _vision(self, request) -> None:
        if not request.uses_vision:
            return
        with self.perf.scope("llm.vision"):
            outputs = self.module.run_vision(request.vision_values)
            self.process.merge_vision(request, outputs, self.state)

    def _prefill(self, request):
        with self.perf.scope("llm.prefill"):
            input_length = int(request.input_ids.shape[1])
            if self.state.context_length + input_length >= self.context_max_length:
                raise ValueError("input exceeds model context length")
            logits = None
            for start in range(0, input_length, self.prefill_length):
                inputs = self.process.prepare_prefill_chunk(
                    request,
                    self.state,
                    start,
                    self.prefill_length,
                    self.embedding_size,
                )
                self.module.set_input(Stage.PREFILL, inputs)
                self.module.run(Stage.PREFILL)
                logits = self.module.get_output(Stage.PREFILL).tensors[0]
            token = self.sampler.sample(logits)
        self.state.context_length += input_length
        return token, input_length

    def _decode(self, token: int) -> int:
        with self.perf.scope("llm.decode"):
            inputs = self.process.prepare_decode(token, self.state)
            self.module.set_input(Stage.DECODE, inputs)
            self.module.run(Stage.DECODE)
            logits = self.module.get_output(Stage.DECODE).tensors[0]
            token = self.sampler.sample(logits, previous_tokens=self.state.generated_ids)
        self.state.context_length += 1
        return token

    def generate(
        self,
        prompt: str,
        *,
        images=None,
        sampling_params: GreedySamplingParams | None = None,
        max_new_tokens: int | None = None,
        keep_history: bool = False,
        system_prompt: str | None = None,
        **kwargs,
    ):
        del kwargs
        if not prompt:
            raise ValueError("prompt must not be empty")
        if max_new_tokens is not None and max_new_tokens <= 0:
            raise ValueError("max_new_tokens must be greater than zero")
        images = self.process.normalize_images(images)
        if system_prompt is None:
            system_prompt = "介绍一下这些图片" if images else "You are a helpful assistant."
        self.perf.reset(preserve_prefixes=("llm.init",))
        if not keep_history:
            self.clear_session()
        if sampling_params is not None:
            self.sampler = GreedySampler(sampling_params)
        # RoPE delta describes the current request's multimodal layout, not
        # the retained session cache. A vision request will set it again in
        # Process.merge_vision(); a text-only continuation must keep it None.
        self.state.rope_deltas = None
        self.state.generated_ids = []
        self.state.emitted_text = ""
        decode_tokens = 0
        ttft_active = True
        e2e_active = True
        self.perf.start("llm.e2e")
        self.perf.start("llm.ttft")
        try:
            request = self.process.preprocess(prompt, images, system_prompt)
            self._vision(request)
            token, input_tokens = self._prefill(request)
            self.state.generated_ids.append(token)
            self.perf.end("llm.ttft")
            ttft_active = False
            delta = self.process.postprocess(self.state)
            if delta:
                self.perf.end("llm.e2e")
                e2e_active = False
                yield delta
                self.perf.start("llm.e2e")
                e2e_active = True
            while token not in self.stop_token_ids:
                if max_new_tokens is not None and len(self.state.generated_ids) >= max_new_tokens:
                    break
                if self.state.context_length >= self.context_max_length:
                    break
                token = self._decode(token)
                decode_tokens += 1
                if token in self.stop_token_ids:
                    break
                self.state.generated_ids.append(token)
                delta = self.process.postprocess(self.state)
                if delta:
                    self.perf.end("llm.e2e")
                    e2e_active = False
                    yield delta
                    self.perf.start("llm.e2e")
                    e2e_active = True
            remainder = self.process.postprocess(self.state, final=True)
            if remainder:
                self.perf.end("llm.e2e")
                e2e_active = False
                yield remainder
                self.perf.start("llm.e2e")
                e2e_active = True
            self.perf.set_metrics(
                "llm",
                input_tokens=input_tokens,
                output_tokens=1 + decode_tokens,
                decode_tokens=decode_tokens,
                num_images=len(images) if images else 0,
            )
        finally:
            if ttft_active:
                self.perf.end("llm.ttft")
            if e2e_active:
                self.perf.end("llm.e2e")
