# Copyright (c) 2026 HOUMO AI
#
# File: runner.py
# Description:
#   Fixed-length Processor, Module, and Backend performance runner.
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

"""Fixed-length Processor -> Module performance runner."""

from pathlib import Path
from typing import Any

from postmo_engine.backend import TcimBackend
from postmo_engine.module import Qwen35Module
from postmo_engine.perf import PerfTracker
from postmo_engine.process import Qwen35Process

from .config import PerfCase
from .input import generate_token_ids
from .progress import ProgressReporter
from .result import CaseResult, LoopResult, add_initialization_scopes, average_reports


class PerfRunner:
    def __init__(
        self,
        case: PerfCase,
        *,
        backend: Any | None = None,
        process: Any | None = None,
        module: Any | None = None,
    ) -> None:
        self.case = case
        self.prefill_path = str(case.prefill or (case.model_dir / "*_prefill.hmm"))
        self.decode_path = str(case.decode or (case.model_dir / "*_decode.hmm"))
        if (process is None) != (module is None):
            raise ValueError("process and module must be provided together")
        if process is not None:
            self.perf = getattr(module, "perf", None) or PerfTracker.create(
                enabled=True,
                aggregate_parents=True,
            )
            self.backend = backend
            self.process, self.module = process, module
        else:
            self.perf = (
                backend.perf
                if backend is not None
                else PerfTracker.create(enabled=True, aggregate_parents=True)
            )
            self.backend = backend if backend is not None else TcimBackend(perf=self.perf)
            self.process, self.module = self._create_model(case.model_dir)
        self.perf.aggregate_parents = True
        self.initialization_report = self.perf.summary()
        self.perf.reset()
        if case.input_tokens + case.output_tokens > self.module.context_max_length:
            raise ValueError("input_tokens + output_tokens exceed model context capacity")
        excluded = set(self.process.eos_token_ids)
        excluded.update(int(value) for value in getattr(self.process.tokenizer, "all_special_ids", ()) or ())
        self.input_ids = generate_token_ids(
            case.input_tokens,
            self.process.embedding_weight.shape[0],
            seed=case.seed,
            excluded_ids=excluded,
        )
        self.decode_ids = generate_token_ids(
            case.output_tokens,
            self.process.embedding_weight.shape[0],
            seed=case.seed + 1,
            excluded_ids=excluded,
        )

    @property
    def prefill_chunk_count(self) -> int:
        return (
            self.case.input_tokens + self.module.prefill_length - 1
        ) // self.module.prefill_length

    def _create_model(self, model_dir: Path):
        root = Path(model_dir)
        if self.case.prefill is not None:
            prefill = self.case.prefill
            decode = self.case.decode
        else:
            prefill_models = tuple(root.glob("*_prefill.hmm"))
            decode_models = tuple(root.glob("*_decode.hmm"))
            if len(prefill_models) != 1 or len(decode_models) != 1:
                raise ValueError("model directory must contain one Prefill and one Decode HMM")
            prefill = prefill_models[0]
            decode = decode_models[0]
        self.prefill_path = str(prefill)
        self.decode_path = str(decode)
        manager = self.backend.create_weight_manager()
        module = Qwen35Module(self.backend, prefill, decode, weight_manager=manager, perf=self.perf)
        process = Qwen35Process(
            root / "hmquant" / "hf_config",
            self.case.embedding or root / "hmquant" / "quant_embedding.pt",
            module.embedding_size,
        )
        return process, module

    def _run_once(self, progress: ProgressReporter | None = None, label: str = ""):
        self.module.clear_session()
        prefill_inputs = self.process.build_prefill_inputs_from_token_ids(self.input_ids)
        if progress is not None:
            progress.phase(f"{label} | Prefill chunks")
        with self.perf.scope("postmo.e2e"):
            with self.perf.scope("postmo.prefill"):
                if progress is None:
                    self.module.prefill(prefill_inputs)
                else:
                    self.module.prefill(
                        prefill_inputs,
                        progress_callback=lambda current, total: progress.update(current),
                    )
            if progress is not None:
                progress.phase(f"{label} | Decode")
                progress.reset_total(self.case.output_tokens)
            with self.perf.scope("postmo.decode"):
                for index, token_id in enumerate(self.decode_ids, start=1):
                    self.module.decode(self.process.build_decode_inputs(int(token_id)))
                    if progress is not None:
                        progress.update(index)
        self.perf.set_metrics(
            "llm",
            input_tokens=self.case.input_tokens,
            output_tokens=self.case.output_tokens,
            decode_tokens=self.case.output_tokens,
        )
        return self.perf.summary()

    def run(self, *, progress: bool | None = None) -> CaseResult:
        reporter = ProgressReporter(enabled=progress)
        try:
            for _ in range(self.case.warmup):
                index = _ + 1
                reporter.begin(
                    f"Warmup {index}/{self.case.warmup} | Prefill chunks",
                    self.prefill_chunk_count,
                )
                self._run_once(reporter, f"Warmup {index}/{self.case.warmup}")
                reporter.finish(newline=False)
                self.perf.reset()
            loops = []
            reports = []
            for index in range(1, self.case.loop + 1):
                reporter.begin(
                    f"Loop {index}/{self.case.loop} | Prefill chunks",
                    self.prefill_chunk_count,
                )
                report = average_reports(
                    [self._run_once(reporter, f"Loop {index}/{self.case.loop}")]
                )
                reporter.finish(newline=False)
                loops.append(LoopResult(index, report))
                reports.append(report)
                self.perf.reset()
        finally:
            reporter.close()
        average = average_reports(reports)
        add_initialization_scopes(average, self.initialization_report)
        return CaseResult(
            model_name=self.case.model_name,
            input_tokens=self.case.input_tokens,
            output_tokens=self.case.output_tokens,
            warmup=self.case.warmup,
            loops=tuple(loops),
            average=average,
            prefill_path=self.prefill_path,
            decode_path=self.decode_path,
            embedding_path=str(
                self.case.embedding
                or (self.case.model_dir / "hmquant" / "quant_embedding.pt")
            ),
            visual_path=str(self.case.visual or ""),
            devices=self.case.devices,
            batch=self.case.batch,
            lazy_mode=self.case.lazy_mode,
            skip_perf=self.case.skip_perf,
            monitor_interval=self.case.monitor_interval,
            perf_case_index=self.case.perf_case_index,
            perf_case_total=self.case.perf_case_total,
        )
