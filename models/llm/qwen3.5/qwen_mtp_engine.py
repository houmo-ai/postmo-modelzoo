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

import numpy as np

from houmo_engine import HoumoEngine
from houmo_engine.core.types import Stage
from houmo_engine.perf import PerfTracker
from houmo_engine.sampling import GreedySampler, GreedySamplingParams

from qwen_module import Qwen36MtpModule
from qwen_process import Qwen36MtpProcess
from qwen_types import Qwen36MtpGenerationState, VerifyResult

E2E_METRIC = "llm_mtp.e2e"
TTFT_METRIC = "llm_mtp.ttft"


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

    def _verify(self, draft_tokens: list[int]) -> VerifyResult:
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
        return VerifyResult(
            draft_tokens=draft_tokens,
            accepted_count=accepted_count,
            next_token=next_token,
            next_hidden=next_hidden,
        )

    def _emit(self, text: str):
        if not text:
            return
        self.perf.end(E2E_METRIC)
        try:
            yield text
        finally:
            self.perf.start(E2E_METRIC)

    def _stop_speculation(self, output_tokens: int, max_new_tokens: int | None) -> bool:
        if max_new_tokens is not None and output_tokens >= max_new_tokens:
            self.state.finish_reason = "length"
            return True
        if self.state.context_length + self.module.verify_length > self.context_max_length:
            self.state.finish_reason = "context"
            return True
        return False

    def _visible_tokens(self, result: VerifyResult, remaining: int | None) -> list[int]:
        candidates = [*result.draft_tokens[: result.accepted_count], result.next_token]
        visible = candidates if remaining is None else candidates[:remaining]
        stop_index = next((index for index, token in enumerate(visible) if token in self.stop_token_ids), None)
        return visible if stop_index is None else visible[: stop_index + 1]

    def _commit_verify_result(self, result: VerifyResult, visible: list[int]) -> None:
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
        self.state.draft_anchor_hidden = result.next_hidden if continues else None

    def _consume_visible_tokens(self, visible: list[int], counters: dict[str, int]):
        for token in visible:
            counters["consumed"] += 1
            if token in self.stop_token_ids:
                self.state.finish_reason = "stop"
                break
            self.state.generated_ids.append(token)
            yield from self._emit(self.process.postprocess(self.state))

    def _finish_length_limited_round(self, visible: list[int], remaining: int | None) -> bool:
        if remaining is None or len(visible) < remaining:
            return False
        self.state.finish_reason = "length"
        self.state.pending_token = None
        self.state.draft_anchor_hidden = None
        return True

    def _run_speculative_loop(self, max_new_tokens: int | None, counters: dict[str, int]):
        while self.state.pending_token is not None:
            if self._stop_speculation(counters["output_tokens"], max_new_tokens):
                break
            counters["speculative_rounds"] += 1
            drafts = self._draft()
            counters["draft_tokens_total"] += len(drafts)
            result = self._verify(drafts)
            counters["accepted_draft_tokens"] += result.accepted_count
            remaining = None if max_new_tokens is None else max_new_tokens - counters["output_tokens"]
            visible = self._visible_tokens(result, remaining)
            self._commit_verify_result(result, visible)
            yield from self._consume_visible_tokens(visible, counters)
            counters["output_tokens"] += counters["consumed"]
            counters["decode_tokens"] += counters["consumed"]
            counters["consumed"] = 0
            if self.state.finish_reason == "stop":
                break
            if self._finish_length_limited_round(visible, remaining):
                break

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
        self.perf.start(E2E_METRIC)
        self.perf.start(TTFT_METRIC)
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
            self.perf.end(TTFT_METRIC)
            ttft_active = False

            yield from self._emit(self.process.postprocess(self.state))
            counters = {
                "consumed": 0,
                "output_tokens": output_tokens,
                "decode_tokens": decode_tokens,
                "draft_tokens_total": draft_tokens_total,
                "accepted_draft_tokens": accepted_draft_tokens,
                "speculative_rounds": speculative_rounds,
            }
            yield from self._run_speculative_loop(max_new_tokens, counters)
            output_tokens = counters["output_tokens"]
            decode_tokens = counters["decode_tokens"]
            draft_tokens_total = counters["draft_tokens_total"]
            accepted_draft_tokens = counters["accepted_draft_tokens"]
            speculative_rounds = counters["speculative_rounds"]
            yield from self._emit(self.process.postprocess(self.state, final=True))
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
                self.perf.end(TTFT_METRIC)
            if e2e_active:
                self.perf.end(E2E_METRIC)


__all__ = ["Qwen36MtpEngine"]
