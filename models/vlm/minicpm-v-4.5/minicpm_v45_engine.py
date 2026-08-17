# Copyright (c) 2026 HOUMO AI
#
# File: minicpm_v45_engine.py
# Description:
#   Generation orchestration for MiniCPM-V 4.5.
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

from __future__ import annotations

import numpy as np
import torch

from houmo_engine import HoumoEngine
from houmo_engine.core.types import Stage
from houmo_engine.perf import PerfTracker

from minicpm_v45_module import MiniCPMV45Module
from minicpm_v45_process import MiniCPMV45Process
from minicpm_v45_types import MiniCPMV45Paths, MiniCPMV45Request, MiniCPMV45State, PrefillRequest

E2E_METRIC = "llm.e2e"
TTFT_METRIC = "llm.ttft"


class MiniCPMV45Engine(HoumoEngine):
    def __init__(self, paths: MiniCPMV45Paths, *, max_slice_nums: int = 9, ndevice: int = 1,
                 batch: int = 1, do_sample: bool = False, temperature: float = 0.7,
                 seed: int | None = 42, perf: bool = True):
        super().__init__(batch=batch)
        if batch != 1:
            raise ValueError("MiniCPMV45Engine only supports batch=1")
        self.perf = PerfTracker.create(perf)
        self.do_sample = do_sample
        self.temperature = float(temperature)
        self.rng = np.random.default_rng(seed)
        with self.perf.scope("llm.init"):
            self.module = MiniCPMV45Module(paths, ndevice=ndevice, perf=self.perf)
            self.process = MiniCPMV45Process(paths.tokenizer_dir, paths.embedding_path,
                                              prefill_length=self.module.prefill_length,
                                              embedding_dim=self.module.embedding_dim,
                                              max_slice_nums=max_slice_nums,
                                              video_group_capacity=self.module.vision_group_capacity,
                                              perf=self.perf)
        self.state = MiniCPMV45State()

    def _sample(self, logits) -> int:
        scores = np.asarray(logits).reshape(-1, logits.shape[-1])[-1].astype(np.float64)
        if not self.do_sample:
            return int(scores.argmax())
        if self.temperature <= 0:
            raise ValueError("temperature must be greater than zero")
        scores = scores / self.temperature
        scores -= scores.max()
        probabilities = np.exp(scores)
        probabilities /= probabilities.sum()
        return int(self.rng.choice(probabilities.size, p=probabilities))

    def _vision(self, request):
        outputs = []
        for inputs in self.process.prepare_vision(request):
            profile = inputs.metadata.get("profile", "vision_1x")
            with self.perf.scope(f"llm.{profile}"):
                self.module.set_input(Stage.VISION, inputs)
                self.module.run(Stage.VISION)
                outputs.append(self.module.get_output(Stage.VISION).tensors[0])
        return torch.cat(outputs, dim=1)

    def _prefill(self, token_embeds):
        logits = None
        with self.perf.scope("llm.prefill"):
            for start in range(0, token_embeds.shape[1], self.module.prefill_length):
                inputs = self.process.prepare_prefill(PrefillRequest(token_embeds), start)
                self.module.set_input(Stage.PREFILL, inputs)
                self.module.run(Stage.PREFILL)
                logits = self.module.get_output(Stage.PREFILL).tensors[0]
        return logits

    def _decode(self, token: int, position: int):
        with self.perf.scope("llm.decode"):
            inputs = self.process.prepare_decode(token, position)
            self.module.set_input(Stage.DECODE, inputs)
            self.module.run(Stage.DECODE)
            return self.module.get_output(Stage.DECODE).tensors[0]

    def _prepare_generation(self, request: MiniCPMV45Request):
        prepared = self.process.preprocess(request)
        if prepared.input_length >= self.module.context_max_length:
            raise ValueError(f"input length {prepared.input_length} exceeds context {self.module.context_max_length}")
        embeds = prepared.token_embeds
        if prepared.image_count:
            vision = self._vision(prepared).squeeze(0)
            embeds = self.process.merge_vision(prepared, vision)
        self.state.input_length = prepared.input_length
        self.state.context_length = prepared.input_length
        self.state.image_count = prepared.image_count
        return self._sample(self._prefill(embeds))

    def _generate_tokens(self, token: int, max_new_tokens: int):
        for _ in range(max_new_tokens):
            if token in self.process.eos_token_ids:
                break
            self.state.generated_ids.append(token)
            delta = self.process.postprocess(self.state)
            if delta:
                yield delta
            if len(self.state.generated_ids) >= max_new_tokens:
                break
            token = self._sample(self._decode(token, self.state.context_length))
            self.state.context_length += 1
            self.state.decode_tokens += 1
            if self.state.context_length >= self.module.context_max_length:
                break

    def generate(self, request: MiniCPMV45Request | str, *, images=None, videos=None,
                 video_fps: float = 3.0,
                 max_new_tokens: int = 512, system_prompt: str | None = None):
        if max_new_tokens <= 0:
            raise ValueError("max_new_tokens must be greater than zero")
        if isinstance(request, str):
            request = MiniCPMV45Request(
                request, list(images or []), list(videos or []), system_prompt, video_fps
            )
        if (request.images or request.videos) and self.module.vision is None:
            raise RuntimeError("image or video input requires a vision model")
        if request.videos and not self.module.supports_video:
            raise RuntimeError("video input requires the visual_6x HMM model")
        self.perf.reset(preserve_prefixes=("llm.init",))
        self.module.clear_session()
        self.state = MiniCPMV45State()
        self.perf.start(E2E_METRIC)
        self.perf.start(TTFT_METRIC)
        e2e_active = ttft_active = True
        try:
            token = self._prepare_generation(request)
            self.perf.end(TTFT_METRIC)
            ttft_active = False
            for delta in self._generate_tokens(token, max_new_tokens):
                self.perf.end(E2E_METRIC)
                e2e_active = False
                yield delta
                self.perf.start(E2E_METRIC)
                e2e_active = True
            delta = self.process.postprocess(self.state, final=True)
            if delta:
                self.perf.end(E2E_METRIC)
                e2e_active = False
                yield delta
                self.perf.start(E2E_METRIC)
                e2e_active = True
            self.perf.set_metrics("llm", input_tokens=self.state.input_length,
                                  output_tokens=len(self.state.generated_ids),
                                  decode_tokens=self.state.decode_tokens,
                                  num_images=self.state.image_count)
        finally:
            if ttft_active:
                self.perf.end(TTFT_METRIC)
            if e2e_active:
                self.perf.end(E2E_METRIC)

    def print_perf(self) -> None:
        self.perf.print_summary()


__all__ = ["MiniCPMV45Engine"]
