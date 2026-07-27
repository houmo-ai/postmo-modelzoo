# Copyright (c) 2026 HOUMO AI
#
# File: qwen3_6_mtp.py
# Description:
#   Qwen3.6 MTP inference Engine implementation.
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

from dataclasses import dataclass

import numpy as np

from ..core import HoumoEngine
from ..core.types import Stage
from ..module.qwen3_6_mtp import Qwen36MtpModule
from ..perf import PerfTracker
from ..process.qwen3_6_mtp import (
    Qwen36MtpGenerationState,
    Qwen36MtpProcess,
)
from ..sampling import GreedySampler, GreedySamplingParams


@dataclass
class _VerifyResult:
    draft_tokens: list[int]
    accepted_count: int
    next_token: int
    next_hidden: np.ndarray


class Qwen36MtpEngine(HoumoEngine):
    """Qwen3.6 deterministic MTP speculative generation engine."""

    def __init__(
        self,
        prefill_path,
        prefill_mtp_path,
        decode_mtp_path,
        decode_verify_path,
        embedding_path,
        tokenizer_path,
        *,
        ndevice: int = 1,
        batch: int = 1,
        sampling_params: GreedySamplingParams | None = None,
        perf: bool = False,
        debug: bool = False,
    ):
        super().__init__(batch=batch)
        if self.batch != 1:
            raise ValueError("Qwen36MtpEngine only supports batch=1")
        self.perf = PerfTracker.create(perf)
        self.sampler = GreedySampler(sampling_params)
        with self.perf.scope("llm_mtp.init"):
            self.module = Qwen36MtpModule(
                prefill_path,
                prefill_mtp_path,
                decode_mtp_path,
                decode_verify_path,
                ndevice=ndevice,
                perf=self.perf,
                debug=debug,
            )
            self.process = Qwen36MtpProcess(
                tokenizer_path,
                embedding_path,
                self.module.embedding_size,
                perf=self.perf,
            )
        self.stop_token_ids = self.process.stop_token_ids
        self.state = Qwen36MtpGenerationState()

    @property
    def context_max_length(self) -> int:
        return self.module.context_max_length

    def clear_session(self) -> None:
        self.state = Qwen36MtpGenerationState()
        self.module.clear_session()

    def _prefill(self, request) -> int:
        input_length = int(request.input_ids.size)
        if input_length >= self.context_max_length:
            raise ValueError(
                f"prompt too long: {input_length} >= {self.context_max_length}"
            )
        pending_hidden = None
        last_logits = None
        last_hidden = None
        for start in range(0, input_length, self.module.prefill_length):
            with self.perf.scope("llm_mtp.prefill"):
                inputs = self.process.prepare_prefill_chunk(
                    request,
                    self.state,
                    start,
                    self.module.prefill_length,
                    self.module.embedding_size,
                )
                self.module.set_input(Stage.PREFILL, inputs)
                self.module.run(Stage.PREFILL)
                outputs = self.module.get_output(Stage.PREFILL)
                logits, hidden = outputs.tensors
                current_length = int(inputs.metadata["current_length"])
                chunk_ids = request.input_ids[start : start + current_length]
                last_logits = logits
                last_hidden = hidden[:, current_length - 1 : current_length, :].copy()

            hidden_parts = []
            token_parts = []
            if pending_hidden is not None:
                hidden_parts.append(pending_hidden)
                token_parts.append(chunk_ids[:1])
            if current_length > 1:
                hidden_parts.append(hidden[:, : current_length - 1, :])
                token_parts.append(chunk_ids[1:current_length])
            if hidden_parts:
                mtp_inputs = self.process.prepare_mtp_prefill_chunk(
                    np.concatenate(hidden_parts, axis=1),
                    np.concatenate(token_parts),
                    self.state,
                    self.module.mtp_prefill_length,
                )
                with self.perf.scope("llm_mtp.mtp_prefill"):
                    self.module.set_input(Stage.MTP_PREFILL, mtp_inputs)
                    self.module.run(Stage.MTP_PREFILL)
                    self.module.get_output(Stage.MTP_PREFILL)
                self.state.mtp_context_length += int(
                    mtp_inputs.metadata["current_length"]
                )
            pending_hidden = last_hidden

        if last_logits is None or last_hidden is None:
            raise RuntimeError("empty prompt is not supported")
        self.module.prepare_verify_from_prefill()
        token = self.sampler.sample(last_logits[:, -1:, :])
        self.state.context_length = input_length
        self.state.pending_token = token
        self.state.draft_anchor_hidden = last_hidden
        return token

    def _draft(self) -> list[int]:
        if self.state.pending_token is None or self.state.draft_anchor_hidden is None:
            raise RuntimeError("draft requires a pending token and anchor hidden state")
        drafts = []
        token = self.state.pending_token
        hidden = self.state.draft_anchor_hidden
        with self.perf.scope("llm_mtp.draft"):
            for offset in range(self.module.draft_block_size):
                inputs = self.process.prepare_draft(
                    hidden,
                    token,
                    self.state.mtp_context_length + offset,
                )
                self.module.set_input(Stage.DRAFT, inputs)
                self.module.run(Stage.DRAFT)
                logits, hidden = self.module.get_output(Stage.DRAFT).tensors
                token = self.sampler.sample(
                    logits,
                    previous_tokens=[*self.state.generated_ids, *drafts],
                )
                drafts.append(token)
        return drafts

    def _verify(self, draft_tokens: list[int]) -> _VerifyResult:
        if self.state.pending_token is None:
            raise RuntimeError("verify requires a pending token")
        verify_tokens = [self.state.pending_token, *draft_tokens]
        inputs = self.process.prepare_verify(
            verify_tokens,
            self.state,
            self.module.verify_length,
        )
        with self.perf.scope("llm_mtp.verify"):
            self.module.set_input(Stage.VERIFY, inputs)
            self.module.run(Stage.VERIFY)
            logits, hidden = self.module.get_output(Stage.VERIFY).tensors

        accepted_count = 0
        for index, draft_token in enumerate(draft_tokens):
            predicted = self.sampler.sample(
                logits[:, index : index + 1, :],
                previous_tokens=[*self.state.generated_ids, *draft_tokens[:index]],
            )
            if predicted != draft_token:
                next_token = predicted
                break
            accepted_count += 1
        else:
            next_token = self.sampler.sample(
                logits[:, -1:, :],
                previous_tokens=[*self.state.generated_ids, *draft_tokens],
            )
        next_hidden = hidden[:, accepted_count : accepted_count + 1, :].copy()
        return _VerifyResult(
            draft_tokens=draft_tokens,
            accepted_count=accepted_count,
            next_token=next_token,
            next_hidden=next_hidden,
        )

    def generate(
        self,
        prompt: str,
        *,
        sampling_params: GreedySamplingParams | None = None,
        max_new_tokens: int | None = None,
        keep_history: bool = False,
        system_prompt: str | None = None,
    ):
        if not prompt:
            raise ValueError("prompt must not be empty")
        if keep_history:
            raise NotImplementedError("Qwen3.6 MTP history is not implemented")
        if max_new_tokens is not None and max_new_tokens <= 0:
            raise ValueError("max_new_tokens must be greater than zero")
        if system_prompt is None:
            system_prompt = "You are a helpful assistant."
        if sampling_params is not None:
            self.sampler = GreedySampler(sampling_params)

        self.perf.reset(preserve_prefixes=("llm_mtp.init",))
        self.clear_session()
        decode_tokens = 0
        draft_tokens_total = 0
        accepted_draft_tokens = 0
        speculative_rounds = 0
        output_tokens = 0
        ttft_active = True
        e2e_active = True
        self.perf.start("llm_mtp.e2e")
        self.perf.start("llm_mtp.ttft")
        try:
            request = self.process.preprocess(prompt, system_prompt)
            token = self._prefill(request)
            initial_mtp_prefill_tokens = self.state.mtp_context_length
            output_tokens = 1
            if token in self.stop_token_ids:
                self.state.finish_reason = "stop"
                self.state.pending_token = None
            else:
                self.state.generated_ids.append(token)
            self.perf.end("llm_mtp.ttft")
            ttft_active = False

            delta = self.process.postprocess(self.state)
            if delta:
                self.perf.end("llm_mtp.e2e")
                e2e_active = False
                yield delta
                self.perf.start("llm_mtp.e2e")
                e2e_active = True

            while self.state.pending_token is not None:
                if max_new_tokens is not None and output_tokens >= max_new_tokens:
                    self.state.finish_reason = "length"
                    break
                if (
                    self.state.context_length + self.module.verify_length
                    > self.context_max_length
                ):
                    self.state.finish_reason = "context"
                    break

                speculative_rounds += 1
                drafts = self._draft()
                draft_tokens_total += len(drafts)
                result = self._verify(drafts)
                accepted_draft_tokens += result.accepted_count

                remaining = (
                    None
                    if max_new_tokens is None
                    else max_new_tokens - output_tokens
                )
                candidates = [
                    *result.draft_tokens[: result.accepted_count],
                    result.next_token,
                ]
                visible = candidates if remaining is None else candidates[:remaining]
                stop_index = next(
                    (
                        index
                        for index, candidate in enumerate(visible)
                        if candidate in self.stop_token_ids
                    ),
                    None,
                )
                if stop_index is not None:
                    visible = visible[: stop_index + 1]

                visible_accepted = min(result.accepted_count, len(visible))
                accepted_steps = 1 + visible_accepted
                self.module.commit_verify_cache(accepted_steps)
                self.state.context_length += accepted_steps
                self.state.mtp_context_length += accepted_steps

                continues = (
                    len(visible) > result.accepted_count
                    and visible[-1] == result.next_token
                    and result.next_token not in self.stop_token_ids
                )
                self.state.pending_token = result.next_token if continues else None
                self.state.draft_anchor_hidden = (
                    result.next_hidden if continues else None
                )

                for candidate in visible:
                    output_tokens += 1
                    decode_tokens += 1
                    if candidate in self.stop_token_ids:
                        self.state.finish_reason = "stop"
                        break
                    self.state.generated_ids.append(candidate)
                    delta = self.process.postprocess(self.state)
                    if delta:
                        self.perf.end("llm_mtp.e2e")
                        e2e_active = False
                        yield delta
                        self.perf.start("llm_mtp.e2e")
                        e2e_active = True

                if self.state.finish_reason == "stop":
                    break
                if remaining is not None and len(visible) >= remaining:
                    self.state.finish_reason = "length"
                    self.state.pending_token = None
                    self.state.draft_anchor_hidden = None
                    break

            remainder = self.process.postprocess(self.state, final=True)
            if remainder:
                self.perf.end("llm_mtp.e2e")
                e2e_active = False
                yield remainder
                self.perf.start("llm_mtp.e2e")
                e2e_active = True
            self.perf.set_metrics(
                "llm_mtp",
                input_tokens=int(request.input_ids.size),
                output_tokens=output_tokens,
                decode_tokens=decode_tokens,
                mtp_prefill_tokens=initial_mtp_prefill_tokens,
                speculative_rounds=speculative_rounds,
                draft_tokens=draft_tokens_total,
                verify_tokens=speculative_rounds * self.module.verify_length,
                accepted_draft_tokens=accepted_draft_tokens,
                drafts_per_round=self.module.draft_block_size,
            )
        finally:
            if ttft_active:
                self.perf.end("llm_mtp.ttft")
            if e2e_active:
                self.perf.end("llm_mtp.e2e")


__all__ = ["Qwen36MtpEngine"]
