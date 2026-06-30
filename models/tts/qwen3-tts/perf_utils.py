# Copyright (c) 2025 HOUMO AI
#
# File: perf_utils.py
# Description:
#   Qwen3-TTS demo performance logging utilities.
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

import time
from collections import defaultdict
from typing import Dict, Iterable, Tuple

from loguru import logger


class PerfKey:
    PREP_LOGITS_PROCESSOR = "prep_logits_processor"
    PREP_REF_AUDIO_LOAD = "prep_ref_audio_load"
    PREP_REF_SPEECH_TOKENIZER_ENCODE = "prep_ref_speech_tokenizer_encode"
    PREP_REF_SPEAKER_EMBED = "prep_ref_speaker_embed"
    PREP_TEXT_TOKENIZE = "prep_text_tokenize"
    PREP_SPECIAL_TOKEN_EMBED = "prep_special_token_embed"
    PREP_CODEC_PROMPT_EMBED = "prep_codec_prompt_embed"
    PREP_TALKER_ROLE_EMBED = "prep_talker_role_embed"
    PREP_ICL_PROMPT_EMBED = "prep_icl_prompt_embed"
    PREP_CONCAT = "prep_concat"

    EMBEDDING_PREP = "embedding_prep"
    FRAME_PREPARE = "frame_prepare"
    TALKER_PREFILL = "talker_prefill"
    TALKER_DECODE = "talker_decode"
    TALKER_SAMPLING = "talker_sampling"
    CODE_PREDICTOR_PREPARE = "code_predictor_prepare"
    CODE_PREDICTOR_PREFILL = "code_predictor_prefill"
    CODE_PREDICTOR_DECODE = "code_predictor_decode"
    CODE_PREDICTOR_SAMPLING = "code_predictor_sampling"
    POSTPROCESS = "postprocess"
    SPEECH_TOKENIZER = "speech_tokenizer"
    STATEFUL_DECODER = "stateful_decoder"
    STATEFUL_DECODER_REF_PRIME = "stateful_decoder_ref_prime"


TALKER_KEYS = [
    PerfKey.TALKER_PREFILL,
    PerfKey.TALKER_DECODE,
    PerfKey.TALKER_SAMPLING,
]

CUSTOMVOICE_CODE_PREDICTOR_KEYS = [
    PerfKey.CODE_PREDICTOR_PREPARE,
    PerfKey.CODE_PREDICTOR_PREFILL,
    PerfKey.CODE_PREDICTOR_DECODE,
    PerfKey.CODE_PREDICTOR_SAMPLING,
]

BASE_CODE_PREDICTOR_KEYS = [
    PerfKey.CODE_PREDICTOR_PREFILL,
    PerfKey.CODE_PREDICTOR_DECODE,
    PerfKey.CODE_PREDICTOR_SAMPLING,
]

BASE_REFERENCE_AUDIO_KEYS = [
    PerfKey.PREP_REF_AUDIO_LOAD,
    PerfKey.PREP_REF_SPEECH_TOKENIZER_ENCODE,
    PerfKey.PREP_REF_SPEAKER_EMBED,
]

BASE_PROMPT_PREPARE_KEYS = [
    PerfKey.PREP_TEXT_TOKENIZE,
    PerfKey.PREP_SPECIAL_TOKEN_EMBED,
    PerfKey.PREP_CODEC_PROMPT_EMBED,
    PerfKey.PREP_TALKER_ROLE_EMBED,
    PerfKey.PREP_ICL_PROMPT_EMBED,
    PerfKey.PREP_CONCAT,
]

BASE_AUDIO_DECODER_KEYS = [
    PerfKey.SPEECH_TOKENIZER,
    PerfKey.STATEFUL_DECODER_REF_PRIME,
    PerfKey.STATEFUL_DECODER,
]


class PerfTracker:
    """Small helper for named performance timing and counters."""

    def __init__(self) -> None:
        self.times = defaultdict(float)
        self.counts = defaultdict(int)
        self._active = {}

    def start(self, key: str) -> None:
        if key in self._active:
            raise RuntimeError(f"Perf event already started: {key}")
        self._active[key] = time.perf_counter()

    def stop(self, key: str, count: int = 0) -> float:
        start_time = self._active.pop(key, None)
        if start_time is None:
            raise RuntimeError(f"Perf event was not started: {key}")
        elapsed = time.perf_counter() - start_time
        self.add(key, elapsed, count=count)
        return elapsed

    def add(self, key: str, seconds: float, count: int = 0) -> None:
        self.times[key] += seconds
        if count:
            self.counts[key] += count

    def snapshot(self) -> Tuple[Dict[str, float], Dict[str, int]]:
        if self._active:
            raise RuntimeError(f"Unclosed perf events: {list(self._active)}")
        return dict(self.times), dict(self.counts)


def _pct(t: float, total: float) -> float:
    return t / total * 100 if total > 0 else 0


def _avg_ms(t: float, n: int) -> float:
    return t / n * 1000 if n > 0 else 0


def _sum_perf(perf: dict, keys: Iterable[str]) -> float:
    return sum(perf.get(key, 0.0) for key in keys)


def _log_header(title: str) -> None:
    logger.info("=" * 72)
    logger.info(title)
    logger.info(
        f"  {'Component':<28} {'Time(s)':>9} {'Pct':>6} {'Count':>6} {'Avg(ms)':>8}"
    )
    logger.info(f"  {'-' * 62}")


def _log_footer(inference_time: float) -> None:
    logger.info(f"  {'-' * 62}")
    logger.info(f"  {'total':<28} {inference_time:>9.2f} 100.0%")
    logger.info("=" * 72)


def _log_row(
    label: str,
    seconds: float,
    inference_time: float,
    count: int | None = None,
    indent: int = 2,
) -> None:
    prefix = " " * indent
    name_width = 30 - indent
    if count is None:
        logger.info(
            f"{prefix}{label:<{name_width}} {seconds:>9.2f} "
            f"{_pct(seconds, inference_time):>5.1f}%"
        )
        return
    logger.info(
        f"{prefix}{label:<{name_width}} {seconds:>9.2f} "
        f"{_pct(seconds, inference_time):>5.1f}% {count:>6} "
        f"{_avg_ms(seconds, count):>8.2f}"
    )


def _log_key_row(
    perf: dict,
    perf_count: dict,
    key: str,
    label: str,
    inference_time: float,
    indent: int = 4,
    show_count: bool = True,
) -> None:
    count = perf_count.get(key, 0) if show_count else None
    _log_row(
        label,
        perf.get(key, 0.0),
        inference_time,
        count=count,
        indent=indent,
    )


def _log_group(label: str, seconds: float, inference_time: float) -> None:
    _log_row(label, seconds, inference_time, indent=2)


def _log_other(perf: dict, inference_time: float) -> None:
    _log_row("other", inference_time - sum(perf.values()), inference_time, indent=2)


def _log_talker(perf: dict, perf_count: dict, inference_time: float) -> None:
    _log_group("talker", _sum_perf(perf, TALKER_KEYS), inference_time)
    for key, label in [
        (PerfKey.TALKER_PREFILL, "prefill"),
        (PerfKey.TALKER_DECODE, "decode"),
        (PerfKey.TALKER_SAMPLING, "sampling"),
    ]:
        _log_key_row(perf, perf_count, key, label, inference_time)


def _log_code_predictor(
    perf: dict,
    perf_count: dict,
    inference_time: float,
    include_prepare: bool,
) -> None:
    keys = CUSTOMVOICE_CODE_PREDICTOR_KEYS if include_prepare else BASE_CODE_PREDICTOR_KEYS
    _log_group("code_predictor", _sum_perf(perf, keys), inference_time)
    rows = []
    if include_prepare:
        rows.append((PerfKey.CODE_PREDICTOR_PREPARE, "prepare", False))
    rows.extend(
        [
            (PerfKey.CODE_PREDICTOR_PREFILL, "prefill", True),
            (PerfKey.CODE_PREDICTOR_DECODE, "decode", True),
            (PerfKey.CODE_PREDICTOR_SAMPLING, "sampling", True),
        ]
    )
    for key, label, show_count in rows:
        _log_key_row(
            perf,
            perf_count,
            key,
            label,
            inference_time,
            show_count=show_count,
        )


def _log_customvoice_perf(
    title: str,
    perf: dict,
    perf_count: dict,
    inference_time: float,
    streaming: bool,
) -> None:
    _log_header(title)
    logger.info(
        f"  {'embedding_prep':<28} {perf.get(PerfKey.EMBEDDING_PREP, 0.0):>9.2f} "
        f"{_pct(perf.get(PerfKey.EMBEDDING_PREP, 0.0), inference_time):>5.1f}%"
    )
    _log_talker(perf, perf_count, inference_time)
    logger.info(
        f"  {'frame_prepare':<28} {perf.get(PerfKey.FRAME_PREPARE, 0.0):>9.2f} "
        f"{_pct(perf.get(PerfKey.FRAME_PREPARE, 0.0), inference_time):>5.1f}%"
    )
    _log_code_predictor(perf, perf_count, inference_time, include_prepare=True)
    if streaming:
        _log_key_row(
            perf,
            perf_count,
            PerfKey.STATEFUL_DECODER,
            "stateful_decoder",
            inference_time,
            indent=2,
        )
    else:
        _log_key_row(
            perf,
            perf_count,
            PerfKey.POSTPROCESS,
            "postprocess",
            inference_time,
            indent=2,
            show_count=False,
        )
        _log_key_row(
            perf,
            perf_count,
            PerfKey.SPEECH_TOKENIZER,
            "speech_tokenizer",
            inference_time,
            indent=2,
        )
    _log_other(perf, inference_time)
    _log_footer(inference_time)


def log_oneshot_perf(perf: dict, perf_count: dict, inference_time: float) -> None:
    """打印 oneshot 模式的性能分析"""
    _log_customvoice_perf(
        "Performance Breakdown:", perf, perf_count, inference_time, streaming=False
    )


def log_streaming_perf(perf: dict, perf_count: dict, inference_time: float) -> None:
    """打印 streaming 模式的性能分析"""
    _log_customvoice_perf(
        "Streaming Performance Breakdown:",
        perf,
        perf_count,
        inference_time,
        streaming=True,
    )


def _log_base_perf(
    title: str, perf: dict, perf_count: dict, inference_time: float
) -> None:
    """打印 Base 模型性能分析。

    Base perf 字典内部以秒为单位存储，与 CustomVoice logger 保持一致。
    oneshot 和 streaming 共用同一套指标，便于横向比较。
    """
    prep_keys = [
        PerfKey.PREP_LOGITS_PROCESSOR,
        *BASE_REFERENCE_AUDIO_KEYS,
        *BASE_PROMPT_PREPARE_KEYS,
    ]
    reference_audio_total = _sum_perf(perf, BASE_REFERENCE_AUDIO_KEYS)
    prompt_prepare_total = _sum_perf(perf, BASE_PROMPT_PREPARE_KEYS)

    _log_header(title)
    _log_group("preparation", _sum_perf(perf, prep_keys), inference_time)
    _log_key_row(
        perf,
        perf_count,
        PerfKey.PREP_LOGITS_PROCESSOR,
        "logits_processor",
        inference_time,
        show_count=False,
    )
    _log_row("reference_audio", reference_audio_total, inference_time, indent=4)
    for key, label in [
        (PerfKey.PREP_REF_AUDIO_LOAD, "audio_load"),
        (PerfKey.PREP_REF_SPEECH_TOKENIZER_ENCODE, "speech_tokenizer_encode"),
        (PerfKey.PREP_REF_SPEAKER_EMBED, "speaker_embed"),
    ]:
        _log_key_row(perf, perf_count, key, label, inference_time, indent=6)
    prompt_prepare_count = sum(
        perf_count.get(key, 0) for key in BASE_PROMPT_PREPARE_KEYS
    )
    _log_row(
        "prompt_prepare",
        prompt_prepare_total,
        inference_time,
        count=prompt_prepare_count,
        indent=4,
    )
    _log_talker(perf, perf_count, inference_time)
    _log_code_predictor(perf, perf_count, inference_time, include_prepare=False)
    _log_group("audio_decoder", _sum_perf(perf, BASE_AUDIO_DECODER_KEYS), inference_time)
    _log_key_row(
        perf,
        perf_count,
        PerfKey.SPEECH_TOKENIZER,
        "speech_tokenizer",
        inference_time,
    )
    for key, label in [
        (PerfKey.STATEFUL_DECODER_REF_PRIME, "stateful_ref_prime"),
        (PerfKey.STATEFUL_DECODER, "stateful_generated"),
    ]:
        _log_key_row(perf, perf_count, key, label, inference_time)
    _log_other(perf, inference_time)
    _log_footer(inference_time)


def log_base_oneshot_perf(
    perf: dict, perf_count: dict, inference_time: float
) -> None:
    """打印 Base oneshot 模式性能分析"""
    _log_base_perf("Performance Breakdown:", perf, perf_count, inference_time)


def log_base_streaming_perf(
    perf: dict, perf_count: dict, inference_time: float
) -> None:
    """打印 Base streaming 模式性能分析"""
    _log_base_perf(
        "Streaming Performance Breakdown:", perf, perf_count, inference_time
    )
