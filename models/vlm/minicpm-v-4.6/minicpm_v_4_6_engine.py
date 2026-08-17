# Copyright (c) 2026 HOUMO AI
#
# File: minicpm_v_4_6_engine.py
# Description:
#   MiniCPM-V 4.6 inference Engine implementation.
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

import numpy as np

from houmo_engine import HoumoEngine
from houmo_engine.core.types import GenerationState, Stage
from houmo_engine.perf import PerfTracker

from minicpm_v_4_6_module import MiniCPMV46Module
from minicpm_v_4_6_process import MiniCPMV46Process

E2E_METRIC = "llm.e2e"
TTFT_METRIC = "llm.ttft"


class MiniCPMV46Engine(HoumoEngine):
    """MiniCPM-V 4.6 inference orchestration."""

    def __init__(
        self,
        prefill_path,
        decode_path,
        embedding_path,
        tokenizer_path,
        *,
        vision_path=None,
        downsample_mode: str = "16x",
        max_slice_nums: int = 36,
        ndevice: int = 1,
        batch: int = 1,
        do_sample: bool = True,
        temperature: float = 0.7,
        seed: int | None = None,
        perf: bool = False,
    ):
        super().__init__(batch=batch)
        if self.batch != 1:
            raise ValueError("MiniCPMV46Engine only supports batch=1")
        if do_sample and temperature <= 0:
            raise ValueError("temperature must be greater than zero")
        self.perf = PerfTracker.create(perf)
        self.do_sample = do_sample
        self.temperature = temperature
        self.rng = np.random.default_rng(seed)
        self.module = MiniCPMV46Module(
            prefill_path,
            decode_path,
            vision_path=vision_path,
            ndevice=ndevice,
            perf=self.perf,
        )
        self.process = MiniCPMV46Process(
            tokenizer_path,
            embedding_path,
            self.module.embedding_size,
            downsample_mode=downsample_mode,
            max_slice_nums=max_slice_nums,
            perf=self.perf,
        )
        generation_config = self.process.tokenizer.init_kwargs.get(
            "generation_config", {}
        )
        configured_eos = generation_config.get("eos_token_id")
        if configured_eos is None:
            configured_eos = [248044, self.process.tokenizer.eos_token_id]
        elif isinstance(configured_eos, int):
            configured_eos = [configured_eos]
        self.stop_token_ids = {int(token_id) for token_id in configured_eos}
        self.stop_token_ids.update({248044, 248046})
        self.state = GenerationState()

    def clear_session(self) -> None:
        self.state = GenerationState()
        self.module.clear_session()

    def _sample_token(self, logits: np.ndarray) -> int:
        scores = np.asarray(logits).reshape(-1, logits.shape[-1])[-1].astype(np.float64)
        if not self.do_sample:
            return int(scores.argmax())
        scores /= self.temperature
        scores -= scores.max()
        probabilities = np.exp(scores)
        probabilities /= probabilities.sum()
        return int(self.rng.choice(probabilities.size, p=probabilities))

    def _vision(self, request) -> None:
        if not request.uses_vision:
            return
        outputs = []
        with self.perf.scope("llm.vision"):
            for inputs in request.vision_inputs:
                self.module.set_input(Stage.VISION, inputs)
                self.module.run(Stage.VISION)
                outputs.append(self.module.get_output(Stage.VISION))
        self.process.merge_vision(request, outputs)

    def _prefill(self, request) -> int:
        input_length = int(request.input_ids.shape[1])
        logits = None
        with self.perf.scope("llm.prefill"):
            for start in range(0, input_length, self.module.prefill_length):
                inputs = self.process.prepare_prefill_chunk(
                    request,
                    start,
                    self.module.prefill_length,
                    self.module.embedding_size,
                )
                self.module.set_input(Stage.PREFILL, inputs)
                self.module.run(Stage.PREFILL)
                logits = self.module.get_output(Stage.PREFILL).tensors[0]
            token = self._sample_token(logits)
        self.state.context_length = input_length
        return token

    def _decode(self, token: int) -> int:
        with self.perf.scope("llm.decode"):
            inputs = self.process.prepare_decode(token, self.state.context_length)
            self.module.set_input(Stage.DECODE, inputs)
            self.module.run(Stage.DECODE)
            logits = self.module.get_output(Stage.DECODE).tensors[0]
            token = self._sample_token(logits)
        self.state.context_length += 1
        return token

    def _validate_request(self, prompt, images, max_new_tokens):
        if not prompt:
            raise ValueError("prompt must not be empty")
        if max_new_tokens <= 0:
            raise ValueError("max_new_tokens must be greater than zero")
        images = self.process.normalize_images(images)
        if images and self.module.vision is None:
            raise RuntimeError("image input requires a vision model")
        return images

    def generate(
        self,
        prompt: str,
        *,
        images=None,
        max_new_tokens: int = 512,
        system_prompt: str | None = None,
        **kwargs,
    ):
        del kwargs
        images = self._validate_request(prompt, images, max_new_tokens)

        self.perf.reset(preserve_prefixes=("llm.init",))
        self.clear_session()
        decode_tokens = 0
        ttft_active = True
        e2e_active = True
        self.perf.start(E2E_METRIC)
        self.perf.start(TTFT_METRIC)
        try:
            request = self.process.preprocess(prompt, images, system_prompt)
            input_length = int(request.input_ids.shape[1])
            if input_length >= self.module.context_max_length:
                raise ValueError(
                    f"input length {input_length} exceeds context "
                    f"{self.module.context_max_length}"
                )
            self._vision(request)
            token = self._prefill(request)
            self.perf.end(TTFT_METRIC)
            ttft_active = False

            for _ in range(max_new_tokens):
                if token in self.stop_token_ids:
                    break
                self.state.generated_ids.append(token)
                delta = self.process.postprocess(self.state)
                if delta:
                    self.perf.end(E2E_METRIC)
                    e2e_active = False
                    yield delta
                    self.perf.start(E2E_METRIC)
                    e2e_active = True
                if len(self.state.generated_ids) >= max_new_tokens:
                    break
                token = self._decode(token)
                decode_tokens += 1
                if self.state.context_length >= self.module.context_max_length:
                    break

            remainder = self.process.postprocess(self.state, final=True)
            if remainder:
                self.perf.end(E2E_METRIC)
                e2e_active = False
                yield remainder
                self.perf.start(E2E_METRIC)
                e2e_active = True
            self.perf.set_metrics(
                "llm",
                input_tokens=input_length,
                output_tokens=len(self.state.generated_ids),
                decode_tokens=decode_tokens,
                num_images=len(images),
            )
        finally:
            if ttft_active:
                self.perf.end(TTFT_METRIC)
            if e2e_active:
                self.perf.end(E2E_METRIC)


__all__ = ["MiniCPMV46Engine"]
