# Copyright (c) 2026 HOUMO AI
#
# File: qwen35_engine.py
# Description:
#   Qwen3.5 Text-only request orchestration, sampling, and streaming.
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

"""Qwen3.5 Text-only Engine: request loop, sampling, and streaming."""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

from postmo_engine.backend import TcimBackend
from postmo_engine.core import (
    CapabilityAccess,
    EngineCapabilities,
    EngineRequest,
    OutputChunk,
    PostMoEngine,
    RequestResult,
    StopReason,
)
from postmo_engine.module import Qwen35Module
from postmo_engine.perf import PerfTracker
from postmo_engine.process import Qwen35Process
from postmo_engine.sampling import GreedySampler

_LLM_E2E_SCOPE = "llm.e2e"
_LLM_TTFT_SCOPE = "llm.ttft"

_TEXT_CAPABILITIES = EngineCapabilities(
    features={
        "text": CapabilityAccess.AVAILABLE,
        "chunked_prefill": CapabilityAccess.AVAILABLE,
        "greedy_sampling": CapabilityAccess.AVAILABLE,
        "vision": CapabilityAccess.PLANNED,
        "prefix_cache": CapabilityAccess.PLANNED,
        "keep_history": CapabilityAccess.PLANNED,
        "random_sampling": CapabilityAccess.PLANNED,
        "continuous_batching": CapabilityAccess.PLANNED,
        "paged_attention": CapabilityAccess.BLOCKED,
    },
    reasons={
        "vision": "Qwen3.5 Text-only v1 does not load visual graphs",
        "prefix_cache": "keep_history is not supported in v1",
        "keep_history": "each request resets the Module session",
        "random_sampling": "v1 only supports greedy sampling",
        "continuous_batching": "v1 only supports batch=1 sequential requests",
        "paged_attention": "PagedAttention is not supported by the compiled graph",
    },
)


class Qwen35Engine(PostMoEngine):
    """Orchestrate one Qwen3.5 text request without owning Runtime tensors."""

    def __init__(
        self,
        process: Qwen35Process,
        module: Qwen35Module,
        *,
        sampler: GreedySampler | None = None,
        perf: PerfTracker | None = None,
    ) -> None:
        self.process = process
        self.module = module
        self.sampler = sampler or GreedySampler()
        if perf is not None:
            self.perf = perf
        elif getattr(module, "perf", None) is not None:
            self.perf = module.perf
        else:
            self.perf = PerfTracker.create(enabled=False)
        self.last_result: RequestResult | None = None

    @classmethod
    def from_model_dir(
        cls,
        model_dir: str | Path,
        *,
        perf: bool | PerfTracker = False,
        aggregate_parents: bool = False,
    ) -> "Qwen35Engine":
        root = Path(model_dir)
        if not root.is_dir():
            raise FileNotFoundError(f"model directory does not exist: {root}")
        tracker = (
            perf
            if isinstance(perf, PerfTracker)
            else PerfTracker.create(
                enabled=bool(perf),
                aggregate_parents=aggregate_parents,
            )
        )
        backend = TcimBackend(perf=tracker)
        prefill_models = tuple(root.glob("*_prefill.hmm"))
        decode_models = tuple(root.glob("*_decode.hmm"))
        if len(prefill_models) != 1 or len(decode_models) != 1:
            raise ValueError("model directory must contain one Prefill and one Decode HMM")
        prefill = prefill_models[0]
        decode = decode_models[0]
        tokenizer = root / "hmquant" / "hf_config"
        embedding = root / "hmquant" / "quant_embedding.pt"
        module = Qwen35Module(backend, prefill, decode, perf=tracker)
        process = Qwen35Process(tokenizer, embedding, module.embedding_size)
        return cls(process, module, perf=tracker)

    @property
    def capabilities(self) -> EngineCapabilities:
        return _TEXT_CAPABILITIES

    def clear_session(self) -> None:
        self.module.clear_session()

    def generate(self, request: EngineRequest) -> Iterator[OutputChunk]:
        if not isinstance(request, EngineRequest):
            raise TypeError("request must be an EngineRequest")
        self.capabilities.require("text")
        self.last_result = None
        sampled_tokens: list[int] = []
        visible_tokens: list[int] = []
        emitted_token_count = 0
        emitted_text = ""
        sequence_no = 0
        submitted_decode_tokens = 0
        stop_reason = StopReason.MAX_NEW_TOKENS
        self.perf.start(_LLM_E2E_SCOPE)
        self.perf.start(_LLM_TTFT_SCOPE)
        ttft_open = True
        e2e_open = True
        completed = False

        def pause_e2e() -> None:
            nonlocal e2e_open
            if e2e_open:
                self.perf.end(_LLM_E2E_SCOPE)
                e2e_open = False

        def resume_e2e() -> None:
            nonlocal e2e_open
            if not e2e_open:
                self.perf.start(_LLM_E2E_SCOPE)
                e2e_open = True

        try:
            self.clear_session()
            prepared = self.process.preprocess(request.prompt)
            prefill_inputs = self.process.build_prefill_inputs(prepared)
            prefill_outputs = self.module.prefill(prefill_inputs)
            logits = self.process.process_prefill_outputs(prefill_outputs)
            token = self.sampler.sample(logits).token_id
            sampled_tokens.append(token)
            self.perf.end(_LLM_TTFT_SCOPE)
            ttft_open = False
            if token in self.process.eos_token_ids:
                stop_reason = StopReason.EOS
            else:
                visible_tokens.append(token)
                delta = self.process.decode_text(tuple(visible_tokens), emitted_text)
                if delta:
                    chunk_tokens = tuple(visible_tokens[emitted_token_count:])
                    emitted_token_count = len(visible_tokens)
                    emitted_text += delta
                    sequence_no += 1
                    pause_e2e()
                    yield OutputChunk(
                        request_id=request.request_id,
                        sequence_no=sequence_no,
                        text_delta=delta,
                        token_ids=chunk_tokens,
                        is_final=False,
                    )
                    resume_e2e()
                while stop_reason is not StopReason.EOS and len(sampled_tokens) < request.max_new_tokens:
                    if self.module.remaining_context <= 0:
                        stop_reason = StopReason.CONTEXT_CAPACITY
                        break
                    with self.perf.scope("llm.decode"):
                        decode_inputs = self.process.build_decode_inputs(token)
                        decode_outputs = self.module.decode(decode_inputs)
                        submitted_decode_tokens += 1
                        logits = self.process.process_decode_outputs(decode_outputs)
                        token = self.sampler.sample(logits).token_id
                    sampled_tokens.append(token)
                    if token in self.process.eos_token_ids:
                        stop_reason = StopReason.EOS
                        break
                    visible_tokens.append(token)
                    delta = self.process.decode_text(tuple(visible_tokens), emitted_text)
                    if delta:
                        chunk_tokens = tuple(visible_tokens[emitted_token_count:])
                        emitted_token_count = len(visible_tokens)
                        emitted_text += delta
                        sequence_no += 1
                        pause_e2e()
                        yield OutputChunk(
                            request_id=request.request_id,
                            sequence_no=sequence_no,
                            text_delta=delta,
                            token_ids=chunk_tokens,
                            is_final=False,
                        )
                        resume_e2e()
            remainder = self.process.decode_text(
                tuple(visible_tokens),
                emitted_text,
                final=True,
            )
            if remainder:
                chunk_tokens = tuple(visible_tokens[emitted_token_count:])
                emitted_token_count = len(visible_tokens)
                sequence_no += 1
                pause_e2e()
                yield OutputChunk(
                    request_id=request.request_id,
                    sequence_no=sequence_no,
                    text_delta=remainder,
                    token_ids=chunk_tokens,
                    is_final=False,
                )
                resume_e2e()
            sequence_no += 1
            self.last_result = RequestResult(
                request_id=request.request_id,
                stop_reason=stop_reason,
                input_tokens=prefill_inputs.input_length,
                sampled_tokens=len(sampled_tokens),
                visible_tokens=len(visible_tokens),
                submitted_decode_tokens=submitted_decode_tokens,
                output_chunks=sequence_no,
            )
            self.perf.set_metrics(
                "llm",
                input_tokens=prefill_inputs.input_length,
                output_tokens=len(sampled_tokens),
                decode_tokens=submitted_decode_tokens,
            )
            completed = True
            pause_e2e()
            yield OutputChunk(
                request_id=request.request_id,
                sequence_no=sequence_no,
                text_delta="",
                token_ids=(),
                is_final=True,
                stop_reason=stop_reason,
            )
        finally:
            if ttft_open:
                self.perf.end(_LLM_TTFT_SCOPE)
            if e2e_open:
                self.perf.end(_LLM_E2E_SCOPE)
            if not completed:
                self.clear_session()
