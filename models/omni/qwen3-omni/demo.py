#!/usr/bin/python3
# -*- coding: utf-8 -*-
# Copyright 2026 HOUMO AI
#
# File: demo.py
# Description:
#   Qwen3-Omni xh2 tcim_lite HMM inference pipeline.
#   Full omni chain: image + audio + text -> text + speech.
#   text_projection runs on-chip as its own standalone HMM engine
#   (qwen3-omni_text_projection.hmm), precomputing the projected text/codec sums
#   fed to the talker via bypass_embeds.
#
#   The talker uses its in-graph (fused) projection, matching the working
#   demo_hmonnx_full.py: the user segment and the assistant 3-token prefix are
#   projected by the talker's baked text/hidden projection (bypass_mask=0); the
#   codec tokens + first text token are supplied precomputed through bypass
#   (bypass_mask=1). This reproduces ptq.py's capture contract and produces
#   correct speech.
#
# SPDX-License-Identifier: Apache-2.0

import os

os.environ.setdefault("TCIM_LOG_LEVEL", "1")

import argparse
import json
import re
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import soundfile as sf
import torch
import torch.nn.functional as F
from loguru import logger

np.random.seed(1234)

import tcim_lite

# Local processor fork (same as demo_hmonnx_full.py): pins image/video frames to a
# fixed square so the vision token count is constant and matches the statically
# exported visual HMM engine. The resolution is auto-aligned to the loaded visual
# graph at runtime (processor.set_vision_resolution), so a re-export at a different
# resolution needs no manual edit here.
from local_processor.processing_qwen3_omni_moe import (
    Qwen3OmniMoeProcessor,
)
from transformers.models.qwen3_omni_moe.modeling_qwen3_omni_moe import (
    _get_feat_extract_output_lengths,
    Qwen3OmniMoePreTrainedModelForConditionalGeneration,
)

from qwen_omni_utils import process_mm_info

from hmatc.utils.utils import first_not_none, get_model_configs

# ═══════════════════════════════════════════════════════════════════════════════
# Constants
# ═══════════════════════════════════════════════════════════════════════════════
HOUMO_TARGET = os.getenv("HOUMO_TARGET")
assert HOUMO_TARGET in ["xh2"], f"Unsupported HOUMO_TARGET: {HOUMO_TARGET}"

SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_CONFIG_PATH = str(SCRIPT_DIR / "config.yaml")
DEFAULT_MODEL_DIR = str(SCRIPT_DIR / "Qwen3-Omni-30B-A3B-Instruct")
DEFAULT_HMQUANT_DIR = SCRIPT_DIR / "output" / HOUMO_TARGET / "hmquant"
DEFAULT_SAMPLE_DIR = SCRIPT_DIR / "sample_data"

# Example modes. image/audio/prompt are written inline per example (see main()),
# so they are not CLI args; new functional examples are added to this map.
EXAMPLES_MODE = {
    0: "omni",  # image + audio + text -> text + speech
    1: "conv",  # multi-turn speech conversation -> text + speech (context kept)
    2: "vlm",  # image + text -> text + speech
    3: "music",  # music audio + text -> text + speech (talker_segment on)
    4: "asr",  # speech audio -> text (speech recognition)
    5: "translate",  # english speech audio -> chinese text (speech translation)
}

# Post-processing defaults (overridable via CLI; see get_args).
DEFAULT_SPEAKER = "ethan"  # talker voice (see SPEAKER_ID_MAP)

IMAGE_TOKEN_ID = 151655
AUDIO_TOKEN_ID = 151675
VIDEO_TOKEN_ID = 151656
SYSTEM_TOKEN_ID = 8948
USER_TOKEN_ID = 872
ASSISTANT_TOKEN_ID = 77091
TTS_BOS_TOKEN_ID = 151672
TTS_EOS_TOKEN_ID = 151673
TTS_PAD_TOKEN_ID = 151671
CODEC_BOS_ID = 2149
CODEC_PAD_ID = 2148
CODEC_EOS_TOKEN_ID = 2150
CODEC_NOTHINK_ID = 2155
CODEC_THINK_BOS_ID = 2156
CODEC_THINK_EOS_ID = 2157
NUM_CODE_GROUPS = 16
TALKER_VOCAB_SIZE = 3072
# HF suppresses the top-1024 codec-vocab tokens (special tokens) except codec_eos,
# applied in BOTH greedy and sample modes. The primary codec token must come from
# [0, 2048) ∪ {codec_eos}; otherwise it can land on a special token -> silence/garbage.
TALKER_SUPPRESS_TOKENS = [
    i
    for i in range(TALKER_VOCAB_SIZE - 1024, TALKER_VOCAB_SIZE)
    if i != CODEC_EOS_TOKEN_ID
]
SPEAKER_ID_MAP = {"chelsie": 2301, "ethan": 2302, "aiden": 2303}
THINKER_HIDDEN_SIZE = 2048
TALKER_HIDDEN_SIZE = 1024
THINKER_NUM_LAYERS = 48
TALKER_NUM_LAYERS = 20
PREDICTOR_NUM_LAYERS = 5
CODE2WAV_UPSAMPLE = 8 * 5 * 4 * 3 * 2 * 2  # 1920
TALKER_STATIC_SEQ = 256
IM_START_TOKEN_ID = 151644
IM_END_TOKEN_ID = 151645
EOS_TOKEN_IDS = {151645, 151643}
SAMPLE_RATE = 24000


# ═══════════════════════════════════════════════════════════════════════════════
# Perf timing helpers (self-contained, cosyvoice3-style)
# ═══════════════════════════════════════════════════════════════════════════════
def _fmt_ms(seconds: float, digits: int = 3) -> str:
    return f"{seconds * 1000.0:.{digits}f} ms"


def _fmt_s(seconds: float, digits: int = 3) -> str:
    return f"{seconds:.{digits}f} s"


def _fmt_toks_per_s(tokens: float, seconds: float, digits: int = 2) -> str:
    if seconds <= 0:
        return "inf tokens/s"
    return f"{(tokens / seconds):.{digits}f} tokens/s"


def _format_perf_report(perf: Dict[str, float]) -> str:
    """Compact multi-line perf report. Keys are seconds unless noted."""
    L = []
    L.append("=" * 68)
    L.append("              Qwen3-Omni HMM Inference Performance")
    L.append("=" * 68)
    L.append(
        f"  Input Length: {int(perf.get('input_tokens', 0)):>6} tokens"
        f"  (audio={int(perf.get('audio_tokens', 0))}, image={int(perf.get('image_tokens', 0))})"
    )
    L.append(
        f"  Output Length: {int(perf.get('thinker_decode_tokens', 0)) + 1:>6} text tokens"
    )
    L.append("-" * 68)
    L.append("Encoders:")
    if perf.get("audio_encoder_s"):
        L.append(f"  Audio Encoder:  {_fmt_ms(perf['audio_encoder_s'])}")
    if perf.get("visual_encoder_s"):
        L.append(f"  Visual Encoder: {_fmt_ms(perf['visual_encoder_s'])}")
    L.append("Thinker (text LLM):")
    if perf.get("thinker_prefill_s"):
        L.append(
            f"  Prefill: {_fmt_ms(perf['thinker_prefill_s'])} | "
            f"Speed: {_fmt_toks_per_s(perf.get('input_tokens', 0), perf['thinker_prefill_s'])}"
        )
    if perf.get("thinker_ttft_s") is not None:
        L.append(f"  TTFT (Time To First Token): {_fmt_ms(perf['thinker_ttft_s'])}")
    dt = int(perf.get("thinker_decode_tokens", 0))
    if perf.get("thinker_decode_s") and dt > 0:
        L.append(
            f"  Decode: {_fmt_ms(perf['thinker_decode_s'])} | "
            f"TPOT: {_fmt_ms(perf['thinker_decode_s'] / dt)}/token | "
            f"{_fmt_toks_per_s(dt, perf['thinker_decode_s'])}"
        )
    if self_has(perf, "talker_s"):
        L.append("Talker (speech codec):")
        L.append(
            f"  Talker Generate: {_fmt_ms(perf['talker_s'])} | "
            f"{int(perf.get('talker_codes', 0))} codec steps"
        )
        if perf.get("code2wav_s"):
            L.append(f"  Code2Wav: {_fmt_ms(perf['code2wav_s'])}")
        audio_s = perf.get("audio_out_s", 0.0)
        if audio_s > 0:
            speech_s = perf.get(
                "speech_total_s", perf.get("talker_s", 0) + perf.get("code2wav_s", 0)
            )
            rtf = speech_s / audio_s if audio_s > 0 else 0.0
            L.append(
                f"  Generated Audio: {_fmt_s(audio_s)} | RTF: {rtf:.4f} "
                f"({(1.0 / rtf):.2f}x real-time)"
                if rtf > 0
                else ""
            )
    L.append("-" * 68)
    L.append(f"  E2E Latency (End-to-End): {_fmt_s(perf.get('e2e_total_s', 0.0))}")
    L.append("=" * 68)
    return "\n".join(x for x in L if x)


def self_has(perf: Dict[str, float], key: str) -> bool:
    return perf.get(key) is not None and perf.get(key) > 0


# ═══════════════════════════════════════════════════════════════════════════════
# Utility Functions
# ═══════════════════════════════════════════════════════════════════════════════
def get_input_infos(engine):
    infos = {}
    for idx in range(engine.get_num_inputs()):
        name = engine.get_input_name(idx)
        infos[name] = engine.get_input_info(name)
    return infos


def get_output_infos(engine):
    infos = {}
    for idx in range(engine.get_num_outputs()):
        name = engine.get_output_name(idx)
        infos[name] = engine.get_output_info(name)
    return infos


def load_text_embedding(hmquant_dir: Path) -> torch.Tensor:
    data = torch.load(
        hmquant_dir / "quant_embedding.pt", map_location="cpu", weights_only=False
    )
    if isinstance(data, dict) and "weight" in data:
        return data["weight"].to(torch.float16)
    return data.to(torch.float16)


def load_talker_embedding(hmquant_dir: Path) -> torch.Tensor:
    data = torch.load(
        hmquant_dir / "quant_embedding_talker.pt",
        map_location="cpu",
        weights_only=False,
    )
    if isinstance(data, torch.Tensor):
        return data.to(torch.float16)
    return data.weight.data.to(torch.float16)


def load_codec_embeddings(hmquant_dir: Path) -> List[torch.Tensor]:
    data = torch.load(
        hmquant_dir / "quant_embedding_codec.pt", map_location="cpu", weights_only=False
    )
    codec_list = data["codec_embeddings"]
    return [
        (
            e.to(torch.float16)
            if isinstance(e, torch.Tensor)
            else e.weight.data.to(torch.float16)
        )
        for e in codec_list
    ]


def run_projection_engine(
    engine, x: torch.Tensor, static: int = TALKER_STATIC_SEQ
) -> torch.Tensor:
    """Run a standalone projection HMM engine. x [1, T, 2048] -> [1, T, 1024].

    The engine is static ("source" [1, static, 2048] -> "output" [1, static, 1024]),
    so chunk by `static`, zero-pad the tail, run, then trim back. Returns CPU fp16.
    Replaces the previous onnxruntime FP16 CPU path with on-chip tcim_lite execution.
    """
    x_np = x.detach().cpu().to(torch.float16).numpy()
    batch, total, dim = x_np.shape
    in_name = engine.get_input_name(0)
    out_name = engine.get_output_name(0)
    outs = []
    for start in range(0, total, static):
        end = min(start + static, total)
        seg_len = end - start
        padded = np.zeros((batch, static, dim), dtype=np.float16)
        padded[:, :seg_len, :] = x_np[:, start:end, :]
        engine.set_input(in_name, np.ascontiguousarray(padded))
        engine.run()
        engine.sync()
        out = engine.get_output(out_name).numpy()
        outs.append(torch.from_numpy(out[:, :seg_len, :]).to(torch.float16))
    return torch.cat(outs, dim=1)


class ThinkerSamplingManager:
    """Thinker-only sampling path aligned with workspace reference behavior."""

    def __init__(
        self,
        do_sample: bool = False,
        temperature: float = 1.0,
        top_k: int = 50,
        top_p: float = 1.0,
        repetition_penalty: float = 1.0,
        min_tokens_to_keep: int = 1,
    ):
        self.do_sample = do_sample
        self.temperature = temperature
        self.top_k = top_k
        self.top_p = top_p
        self.repetition_penalty = repetition_penalty
        self.min_tokens_to_keep = min_tokens_to_keep

    def _apply_repetition_penalty(
        self, logits: np.ndarray, previous_tokens: Optional[List[int]] = None
    ) -> np.ndarray:
        if self.repetition_penalty == 1.0 or not previous_tokens:
            return logits
        adjusted = logits.copy()
        for token_id in set(previous_tokens):
            if 0 <= token_id < adjusted.shape[-1]:
                if adjusted[token_id] < 0:
                    adjusted[token_id] *= self.repetition_penalty
                else:
                    adjusted[token_id] /= self.repetition_penalty
        return adjusted

    def _apply_top_k(self, scores: np.ndarray) -> np.ndarray:
        if self.top_k is None or self.top_k <= 0:
            return scores
        top_k = min(self.top_k, len(scores))
        if top_k <= 0:
            return scores
        top_indices = np.argpartition(scores, -top_k)[-top_k:]
        masked = np.zeros_like(scores)
        masked[top_indices] = scores[top_indices]
        return masked

    def _apply_top_p(self, scores: np.ndarray) -> np.ndarray:
        if self.top_p >= 1.0:
            return scores
        sorted_indices = np.argsort(scores)[::-1]
        sorted_scores = scores[sorted_indices]
        cumulative = np.cumsum(sorted_scores)
        cutoff_indices = np.where(cumulative >= self.top_p)[0]
        if len(cutoff_indices) > 0:
            cutoff_index = cutoff_indices[0]
            if cutoff_index < self.min_tokens_to_keep - 1:
                cutoff_index = self.min_tokens_to_keep - 1
            selected_indices = sorted_indices[: cutoff_index + 1]
        else:
            selected_indices = sorted_indices
        masked = np.zeros_like(scores)
        masked[selected_indices] = scores[selected_indices]
        return masked

    def _process_logits(
        self, logits: np.ndarray, previous_tokens: Optional[List[int]] = None
    ) -> np.ndarray:
        processed = logits.copy()
        processed = self._apply_repetition_penalty(processed, previous_tokens)
        processed = self._apply_top_k(processed)
        processed = self._apply_top_p(processed)
        if self.temperature > 0 and self.temperature != 1.0:
            processed = processed / self.temperature
        return processed

    def sample(
        self, logits_np: np.ndarray, generated_ids: Optional[List[int]] = None
    ) -> int:
        logits_2d = logits_np.reshape(-1, logits_np.shape[-1])
        last = logits_2d[-1].astype(np.float64)

        if not self.do_sample:
            return int(np.argmax(last))

        processed = self._process_logits(last, generated_ids)
        if not np.any(processed):
            return int(np.argmax(last))
        return int(np.argmax(processed))


def greedy_or_sample(
    logits_np: np.ndarray,
    do_sample: bool,
    temperature: float = 1.0,
    top_k: int = 50,
    repetition_penalty: float = 1.0,
    generated_ids: list = None,
    suppress_tokens: Optional[List[int]] = None,
) -> int:
    """Talker primary-codec sampling, matching HF's processor order:
    RepetitionPenalty -> SuppressTokens -> (argmax | Temperature -> TopK -> multinomial).
    SuppressTokens and RepetitionPenalty apply in BOTH greedy and sample modes,
    exactly as HF's LogitsProcessorList does.
    """
    logits_2d = logits_np.reshape(-1, logits_np.shape[-1])
    last = logits_2d[-1].astype(np.float64)
    # 1) Repetition penalty (HF: RepetitionPenaltyLogitsProcessor)
    if repetition_penalty != 1.0 and generated_ids:
        for token_id in set(generated_ids):
            if 0 <= token_id < len(last):
                if last[token_id] > 0:
                    last[token_id] /= repetition_penalty
                else:
                    last[token_id] *= repetition_penalty
    # 2) Suppress special tokens (HF: SuppressTokensLogitsProcessor) — both modes
    if suppress_tokens:
        for tid in suppress_tokens:
            if 0 <= tid < len(last):
                last[tid] = -float("inf")
    if not do_sample:
        return int(np.argmax(last))
    # 3) Temperature -> TopK -> softmax -> multinomial
    if temperature > 0 and temperature != 1.0:
        last = last / temperature
    if top_k > 0 and top_k < len(last):
        indices_to_remove = np.argsort(last)[:-top_k]
        last[indices_to_remove] = -float("inf")
    finite_mask = np.isfinite(last)
    if not np.any(finite_mask):
        return int(np.argmax(logits_2d[-1]))
    exp_l = np.exp(last - np.max(last[finite_mask]))
    exp_l[~finite_mask] = 0.0
    probs_sum = np.sum(exp_l)
    if not np.isfinite(probs_sum) or probs_sum <= 0:
        return int(np.argmax(last))
    probs = exp_l / probs_sum
    return int(np.random.choice(len(probs), p=probs))


# ═══════════════════════════════════════════════════════════════════════════════
# Text sanitization for TTS (talker)
# ═══════════════════════════════════════════════════════════════════════════════
_MD_LINK_RE = re.compile(
    r"!?\[([^\]]*)\]\([^)]*\)"
)  # [text](url) / ![alt](src) -> text
_MD_INLINE_RE = re.compile(r"[*#`_~|>]+")  # emphasis/headers/code/rules
_WS_RE = re.compile(r"[ \t\r\f\v]+")  # collapse horizontal whitespace
_NL_RE = re.compile(r"\s*\n\s*")  # newline run -> single pause


def sanitize_text_for_talker(text: str) -> str:
    """Strip non-punctuation markup/escape characters that corrupt TTS speech.

    Removes markdown emphasis (``*``/``**``), headers (``#``), bullets, code
    ticks, table pipes, blockquote markers, and underscores; unwraps
    ``[text](url)`` links to their visible text; and converts newlines into a
    light comma pause so the talker reads list items as a continuous sentence
    instead of synthesizing the literal escape characters. Sentence punctuation
    (CJK and ASCII) is left untouched so prosody is preserved.

    Standalone and idempotent — applied to the thinker's decoded text right
    before it is re-tokenized/embedded for the talker.
    """
    if not text:
        return text
    s = _MD_LINK_RE.sub(r"\1", text)
    # Drop list-item bullet markers at line starts ("- ", "* ", "1. ")
    s = re.sub(r"(?m)^[ \t]*(?:[-*+]|\d+[.)])[ \t]+", "", s)
    s = _MD_INLINE_RE.sub("", s)
    s = _NL_RE.sub("，", s)  # newline -> Chinese comma (a short pause)
    s = _WS_RE.sub(" ", s)
    # Tidy up punctuation runs the substitutions may have produced
    s = re.sub(r"，{2,}", "，", s)
    s = re.sub(r"，\s*([，。！？、；：])", r"\1", s)
    s = re.sub(r"([。！？])，", r"\1", s)
    return s.strip(" ，")


# Sentence/clause boundary splitters for talker segmentation. Each regex matches
# a run of non-delimiter chars plus its trailing delimiter(s), so iterating with
# finditer yields atomic units whose concatenation reproduces the input exactly
# (no characters dropped) — the reliability invariant the segmenter must hold.
_SENT_UNIT_RE = re.compile(r"[^。！？!?；;…\n]*(?:[。！？!?；;…\n]+|$)")
_CLAUSE_UNIT_RE = re.compile(r"[^，、,：:]*(?:[，、,：:]+|$)")

# Length metric for TTS segmentation, mirroring minicpmo's count_mixed_text
# "Total": CJK chars + English words + standalone number tokens. Whitespace and
# punctuation don't count toward the budget (they cost ~no codec steps), so a
# segment's metric tracks its real synthesis cost rather than raw char count.
_CJK_RE = re.compile(r"[一-鿿]")
_EN_WORD_RE = re.compile(r"[A-Za-z]+(?:[0-9]*[A-Za-z]*)*")
_NUM_RE = re.compile(r"\b\d+\b")


def tts_text_length(s: str) -> int:
    """Mixed CN/EN length: CJK chars + EN words + number tokens (minicpmo-style)."""
    cjk = len(_CJK_RE.findall(s))
    en = len(_EN_WORD_RE.findall(_CJK_RE.sub("", s)))
    num = len(_NUM_RE.findall(s))
    return cjk + en + num


def _hard_split(s: str, max_chars: int) -> List[str]:
    """Last-resort split for a delimiter-less run longer than max_chars (e.g. a
    long unpunctuated CJK string, which has no spaces to break on). Split on
    whitespace first; any still-oversized piece is sliced at fixed width. Loses
    no characters."""
    out: List[str] = []
    for tok in re.split(r"(\s+)", s):
        if not tok:
            continue
        if tts_text_length(tok) <= max_chars:
            out.append(tok)
            continue
        # No usable boundary: slice by raw character count. CJK length≈chars, so
        # a width of max_chars keeps each slice near the budget.
        step = max(1, max_chars)
        for i in range(0, len(tok), step):
            out.append(tok[i : i + step])
    return out


def _atomic_units(text: str, max_chars: int) -> List[str]:
    """Break text into atoms each <= max_chars wherever the text is divisible,
    keeping every character. Sentences within budget stay whole; over-budget
    sentences are broken on clause delimiters; clauses still over budget fall back
    to `_hard_split`. The outer packer recombines atoms up toward the band, so
    these atoms are intentionally minimal rather than pre-packed.
    Concatenation of the returned list == `text`."""
    units: List[str] = []
    for su in (m.group(0) for m in _SENT_UNIT_RE.finditer(text)):
        if not su:
            continue
        if tts_text_length(su) <= max_chars:
            units.append(su)
            continue
        for cu in (m.group(0) for m in _CLAUSE_UNIT_RE.finditer(su)):
            if not cu:
                continue
            if tts_text_length(cu) <= max_chars:
                units.append(cu)
            else:
                units.extend(_hard_split(cu, max_chars))
    return units


def split_text_into_segments(
    text: str, max_chars: int = 60, min_chars: int = 30
) -> List[str]:
    """Split sanitized talker text into TTS-sized segments inside a length band.

    The on-chip talker codec stream degenerates into unintelligible audio past a
    few hundred decode steps (confirmed via whisper: long answers stay clean for
    ~4-6 sentences then garble). Synthesizing each segment as its own short
    utterance keeps every talker call inside the proven-reliable regime; the
    per-call KV-cache reset isolates them.

    Modeled on minicpmo's split_text_for_tts, but with an explicit length band so
    we neither degenerate (too long) nor waste a full prefill+decode on a tiny
    fragment (too short):
      - `max_chars` is the HARD bound (the reliability limit). `_atomic_units`
        makes every atom <= max_chars wherever the text is divisible, and the
        greedy packer never lets a segment exceed it.
      - `min_chars` is a SOFT target (efficiency). Greedy packing fills toward
        max_chars, so segments naturally sit well above min_chars; only an
        unavoidable trailing remainder can fall short, and we fold it back when
        that stays within max_chars.
    Length is the mixed CN/EN metric (`tts_text_length`: CN chars + EN words +
    numbers), so punctuation/whitespace don't burn budget. Invariant: the
    concatenation of segments preserves every non-whitespace character of the
    input (no dropped words/sentences) — exercised by test_split_stress.py.
    """
    if not text or not text.strip():
        return []
    if max_chars < 1:
        max_chars = 1
    min_chars = max(0, min(min_chars, max_chars))

    units = _atomic_units(text, max_chars)
    segments: List[str] = []
    cur = ""
    for u in units:
        # Pure greedy by the HARD bound: flush whenever appending would exceed
        # max_chars. Since every atom is <= max_chars, each emitted segment is too.
        if cur and tts_text_length(cur) + tts_text_length(u) > max_chars:
            segments.append(cur)
            cur = u
        else:
            cur += u
    if cur.strip():
        segments.append(cur)

    # Soft min-length cleanup: fold a stranded short tail into its predecessor,
    # but only if the merge still respects max_chars (a short tail is the lesser
    # evil vs. an over-long, degeneration-prone segment).
    if len(segments) >= 2 and tts_text_length(segments[-1]) < min_chars:
        if tts_text_length(segments[-2]) + tts_text_length(segments[-1]) <= max_chars:
            segments[-2] = segments[-2] + segments[-1]
            segments.pop()

    segments = [s for s in segments if s.strip()]
    return segments or [text]


# ═══════════════════════════════════════════════════════════════════════════════
# Main Pipeline Class
# ═══════════════════════════════════════════════════════════════════════════════
class Qwen3OmniHmmPipeline:

    def __init__(self, args):
        self.args = args
        self.model_dir = Path(args.model_dir).resolve()
        self.hmquant_dir = Path(args.embedding_path).resolve()
        self.enable_audio_generation = bool(args.enable_audio_generation)

        logger.info("Loading processor and embeddings...")
        self.processor = Qwen3OmniMoeProcessor.from_pretrained(str(self.model_dir))
        self.tokenizer = self.processor.tokenizer
        self.vision_start_token_id = int(
            self.tokenizer.convert_tokens_to_ids(self.processor.vision_bos_token)
        )
        self.vision_end_token_id = int(
            self.tokenizer.convert_tokens_to_ids(self.processor.vision_eos_token)
        )
        self.audio_start_token_id = int(
            self.tokenizer.convert_tokens_to_ids(self.processor.audio_bos_token)
        )
        self.audio_end_token_id = int(
            self.tokenizer.convert_tokens_to_ids(self.processor.audio_eos_token)
        )

        self.text_embedding = load_text_embedding(self.hmquant_dir)
        self.talker_embedding = load_talker_embedding(self.hmquant_dir)
        self.codec_embeddings = load_codec_embeddings(self.hmquant_dir)
        logger.info(f"  text_embedding: {self.text_embedding.shape}")
        logger.info(f"  talker_embedding: {self.talker_embedding.shape}")
        logger.info(
            f"  codec_embeddings: {len(self.codec_embeddings)}x{self.codec_embeddings[0].shape}"
        )

        self.thinker_sampling = ThinkerSamplingManager(
            do_sample=bool(args.thinker_do_sample),
            temperature=float(args.thinker_temperature),
            top_k=int(args.thinker_top_k),
            top_p=float(args.thinker_top_p),
            repetition_penalty=float(args.thinker_repetition_penalty),
        )

        self._load_engines()
        self._init_engine_meta()
        self._align_vision_resolution()

    def _align_vision_resolution(self):
        """Auto-align the processor to the exported visual graph's resolution.

        Read the spatial side length from the visual engine's input shape
        ([b, c, t, H, W]) and inject it into the local processor, so the image/
        video token count stays constant and matches the statically-exported
        visual HMM engine. A re-export at a different resolution needs no manual
        edit. Mirrors demo_hmonnx_full.py's set_vision_resolution call.
        """
        try:
            vis_h, vis_w = int(self.visual_shape[-2]), int(self.visual_shape[-1])
            if vis_h == vis_w and vis_h > 0:
                self.processor.set_vision_resolution(vis_h)
                logger.info(
                    f"  Vision resolution aligned to exported graph: {vis_h}x{vis_w}"
                )
            else:
                logger.warning(
                    f"  Visual graph input {self.visual_shape} is non-square; keeping processor default."
                )
        except Exception as e:
            logger.warning(
                f"  Could not auto-align vision resolution from graph ({e}); keeping processor default."
            )

    # ── Engine Loading ────────────────────────────────────────────────────────

    def _load_engines(self):
        args = self.args
        ndevice = args.ndevice

        if ndevice == 1:
            if args.run_device_num == 1:
                wm = tcim_lite.runtime.WeightManager(args.device_id)
                self.llm_wm = wm
                self.mm_wm = wm
                self.talker_wm = wm
            elif args.run_device_num == 2:
                self.llm_wm = tcim_lite.runtime.WeightManager(0)
                wm = tcim_lite.runtime.WeightManager(1)
                self.mm_wm = wm
                self.talker_wm = wm
            else:
                raise ValueError(
                    f"Unsupported run_device_num={args.run_device_num} for ndevice=1"
                )
        elif ndevice == 2:
            llm_dm = tcim_lite.runtime.DevManager([0, 1], "Xh2HalBackend")
            talker_dm = tcim_lite.runtime.DevManager([0], "Xh2HalBackend")
            mm_dm = tcim_lite.runtime.DevManager([1], "Xh2HalBackend")
            self.llm_wm = tcim_lite.runtime.WeightManager(llm_dm)
            self.talker_wm = tcim_lite.runtime.WeightManager(talker_dm)
            self.mm_wm = tcim_lite.runtime.WeightManager(mm_dm)
        else:
            raise ValueError(f"Unsupported ndevice={ndevice}")

        # When running on 2 devices with merged .hmms, the thinker prefill/decode
        # engines use the .hmms variant; everything else stays .hmm.
        def _llm_path(path):
            if ndevice > 1:
                return str(Path(path).with_suffix(".hmms"))
            return path

        logger.info("Loading HMM engines...")
        self.visual_engine = self._load_engine(args.visual_path, self.mm_wm, "visual")
        self.audio_engine = self._load_engine(args.audio_path, self.mm_wm, "audio")
        self.prefill_engine = self._load_engine(
            _llm_path(args.prefill_path), self.llm_wm, "prefill"
        )
        self.decode_engine = self._load_engine(
            _llm_path(args.decode_path),
            self.llm_wm,
            "decode",
            use_dummy_cache=True,
            num_layers=THINKER_NUM_LAYERS,
        )

        if self.enable_audio_generation:
            self.talker_prefill_engine = self._load_engine(
                args.talker_prefill_path, self.talker_wm, "talker_prefill"
            )
            self.talker_decode_engine = self._load_engine(
                args.talker_decode_path,
                self.talker_wm,
                "talker_decode",
                use_dummy_cache=True,
                num_layers=TALKER_NUM_LAYERS,
            )
            self.pred_prefill_engine = self._load_engine(
                args.talker_prediction_prefill_path, self.talker_wm, "pred_prefill"
            )
            self.pred_decode_engine = self._load_engine(
                args.talker_prediction_decode_path,
                self.talker_wm,
                "pred_decode",
                use_dummy_cache=True,
                num_layers=PREDICTOR_NUM_LAYERS,
            )
            self.code2wav_engine = self._load_engine(
                args.code2wav_path, self.talker_wm, "code2wav"
            )
            # Standalone text_projection HMM engine (on-chip). Precomputes the
            # projected text/codec sums fed to the talker via bypass. The talker's
            # user/assistant-prefix segments use its own in-graph (fused)
            # projection. Static "source" [1,S,2048] -> "output" [1,S,1024].
            self.text_proj_engine = self._load_engine(
                args.text_projection_path, self.talker_wm, "text_projection"
            )
            self.text_proj_static_seq = int(
                self.text_proj_engine.get_input_info(
                    self.text_proj_engine.get_input_name(0)
                ).shape[1]
            )

        # Bind KV caches: prefill -> decode share same buffers
        self._bind_cache(self.prefill_engine, self.decode_engine)
        if self.enable_audio_generation:
            self._bind_cache(self.talker_prefill_engine, self.talker_decode_engine)
            self._bind_cache(self.pred_prefill_engine, self.pred_decode_engine)

        logger.info("All HMM engines loaded.")

    def _load_engine(
        self, path, weight_manager, tag, use_dummy_cache=False, num_layers=0
    ):
        option = tcim_lite.runtime.Option(weight_manager)
        if use_dummy_cache and num_layers > 0:
            dummy_names = []
            for i in range(num_layers):
                dummy_names.append(f"model_layers_{i}_self_attn_kcache_input")
                dummy_names.append(f"model_layers_{i}_self_attn_vcache_input")
            option.set_dummy_tensors(dummy_names)
        try:
            engine = tcim_lite.runtime.load(path, option)
            logger.info(f"  Loaded [{tag}]: {path}")
            return engine
        except Exception as exc:
            raise RuntimeError(f"Failed to load HMM ({tag}): {path}") from exc

    def _bind_cache(self, prefill_engine, decode_engine):
        prefill_names = [
            prefill_engine.get_input_name(i)
            for i in range(prefill_engine.get_num_inputs())
        ]
        decode_names = [
            decode_engine.get_input_name(i)
            for i in range(decode_engine.get_num_inputs())
        ]
        for name in decode_names:
            if "cache" in name and name in prefill_names:
                decode_engine.set_input(name, prefill_engine.get_input(name))

    def _init_engine_meta(self):
        """Read engine input/output shapes for runtime use."""
        # Text LLM prefill/decode
        self.prefill_input_infos = get_input_infos(self.prefill_engine)
        self.prefill_input_names = [
            self.prefill_engine.get_input_name(i)
            for i in range(self.prefill_engine.get_num_inputs())
        ]
        self.decode_input_infos = get_input_infos(self.decode_engine)
        self.decode_input_names = [
            self.decode_engine.get_input_name(i)
            for i in range(self.decode_engine.get_num_inputs())
        ]

        # Get prefill static seq len from first input shape
        self.prefill_seq_len = int(
            self.prefill_input_infos[self.prefill_input_names[0]].shape[1]
        )
        # Get KV cache max length from decode cache shape
        decode_cache_names = [n for n in self.decode_input_names if "cache" in n]
        if decode_cache_names:
            self.kv_max_length = int(
                self.decode_input_infos[decode_cache_names[0]].shape[2]
            )
        else:
            self.kv_max_length = 4096

        # Visual engine shape
        vis_name = self.visual_engine.get_input_name(0)
        self.visual_shape = tuple(
            int(d) for d in self.visual_engine.get_input_info(vis_name).shape
        )

        # Audio engine
        self.audio_input_names = [
            self.audio_engine.get_input_name(i)
            for i in range(self.audio_engine.get_num_inputs())
        ]
        audio_info = self.audio_engine.get_input_info(self.audio_input_names[0])
        self.audio_feature_shape = tuple(int(d) for d in audio_info.shape)

        if self.enable_audio_generation:
            # Talker prefill/decode
            self.talker_prefill_names = [
                self.talker_prefill_engine.get_input_name(i)
                for i in range(self.talker_prefill_engine.get_num_inputs())
            ]
            self.talker_prefill_infos = get_input_infos(self.talker_prefill_engine)
            self.talker_decode_names = [
                self.talker_decode_engine.get_input_name(i)
                for i in range(self.talker_decode_engine.get_num_inputs())
            ]
            self.talker_decode_infos = get_input_infos(self.talker_decode_engine)
            tp0_info = self.talker_prefill_infos[self.talker_prefill_names[0]]
            self.talker_prefill_seq_len = int(tp0_info.shape[1])

            # Predictor prefill/decode
            self.pred_prefill_names = [
                self.pred_prefill_engine.get_input_name(i)
                for i in range(self.pred_prefill_engine.get_num_inputs())
            ]
            self.pred_prefill_infos = get_input_infos(self.pred_prefill_engine)
            self.pred_decode_names = [
                self.pred_decode_engine.get_input_name(i)
                for i in range(self.pred_decode_engine.get_num_inputs())
            ]
            self.pred_decode_infos = get_input_infos(self.pred_decode_engine)

            # Code2wav
            c2w_name = self.code2wav_engine.get_input_name(0)
            c2w_info = self.code2wav_engine.get_input_info(c2w_name)
            self.code2wav_static_len = int(c2w_info.shape[2])

        logger.info(
            f"  prefill_seq_len={self.prefill_seq_len}, kv_max={self.kv_max_length}"
        )

    # ── Audio Encoder ─────────────────────────────────────────────────────────

    def run_audio_encoder(
        self, input_features: torch.Tensor, feature_attention_mask: torch.Tensor
    ) -> torch.Tensor:
        audio_feature_lengths = feature_attention_mask.sum(dim=1).long()
        compact_features = input_features.permute(0, 2, 1)[
            feature_attention_mask.bool()
        ].permute(1, 0)
        aftercnn_lens = _get_feat_extract_output_lengths(audio_feature_lengths)

        n_window = 50
        chunk_num = torch.ceil(audio_feature_lengths.float() / (n_window * 2)).long()
        chunk_lengths = torch.tensor(
            [n_window * 2] * int(chunk_num.sum().item()), dtype=torch.long
        )
        tail_idx = F.pad(chunk_num, (1, 0), value=-1).cumsum(0)[1:]
        chunk_lengths[tail_idx] = audio_feature_lengths % (n_window * 2)
        chunk_lengths[chunk_lengths == 0] = n_window * 2

        chunk_list = compact_features.T.split(chunk_lengths.tolist(), dim=0)
        padded_feature = torch.nn.utils.rnn.pad_sequence(
            chunk_list, batch_first=True
        ).transpose(1, 2)
        feature_lens_after_cnn = _get_feat_extract_output_lengths(chunk_lengths)

        all_outputs = []
        for i in range(padded_feature.shape[0]):
            sf_chunk = padded_feature[i : i + 1].to(torch.float16)
            sc = torch.tensor([0, int(feature_lens_after_cnn[i])], dtype=torch.int32)
            # Pad to static audio shape
            eb, em, ea = self.audio_feature_shape
            engine_in = np.zeros((eb, em, ea), dtype=np.float16)
            chunk_np = sf_chunk.numpy()
            engine_in[:, :, : chunk_np.shape[2]] = chunk_np
            self.audio_engine.set_input(
                self.audio_input_names[0], np.ascontiguousarray(engine_in)
            )
            self.audio_engine.set_input(
                self.audio_input_names[1], np.ascontiguousarray(sc.numpy())
            )
            self.audio_engine.run()
            self.audio_engine.sync()
            out_name = self.audio_engine.get_output_name(0)
            out_np = self.audio_engine.get_output(out_name).numpy()
            out_t = torch.from_numpy(out_np).to(torch.float16)
            all_outputs.append(out_t[: int(feature_lens_after_cnn[i])])

        merged = []
        chunk_offset = 0
        for si, scc in enumerate(chunk_num.tolist()):
            sco = all_outputs[chunk_offset : chunk_offset + scc]
            sample_out = torch.cat(sco, dim=0) if len(sco) > 1 else sco[0]
            el = int(aftercnn_lens[si])
            merged.append(sample_out[:el])
            chunk_offset += scc
        return torch.cat(merged, dim=0) if len(merged) > 1 else merged[0]

    # ── Visual Encoder ────────────────────────────────────────────────────────

    def run_visual_encoder(self, pixel_values: torch.Tensor):
        pv = pixel_values.to(torch.float16)
        if pv.ndim == 4:
            pv = pv.unsqueeze(2)
        expected = self.visual_shape
        if list(pv.shape) != list(expected):
            b, c, t, h, w = expected
            pv = (
                F.interpolate(
                    pv.reshape(-1, pv.shape[-2], pv.shape[-1]).unsqueeze(1).float(),
                    size=(h, w),
                    mode="bilinear",
                    align_corners=False,
                )
                .to(torch.float16)
                .reshape(expected)
            )
        vis_name = self.visual_engine.get_input_name(0)
        self.visual_engine.set_input(vis_name, np.ascontiguousarray(pv.numpy()))
        self.visual_engine.run()
        self.visual_engine.sync()
        outputs = []
        for i in range(self.visual_engine.get_num_outputs()):
            oname = self.visual_engine.get_output_name(i)
            outputs.append(
                torch.from_numpy(self.visual_engine.get_output(oname).numpy()).to(
                    torch.float16
                )
            )
        vision_embeds = outputs[0]
        if vision_embeds.ndim == 3 and vision_embeds.shape[0] == 1:
            vision_embeds = vision_embeds.squeeze(0)
        ds_list = []
        for o in outputs[1:4]:
            if o.ndim == 3 and o.shape[0] == 1:
                o = o.squeeze(0)
            ds_list.append(o)
        while len(ds_list) < 3:
            ds_list.append(torch.zeros_like(vision_embeds))
        return vision_embeds, ds_list

    # ── Text LLM (Thinker) ────────────────────────────────────────────────────

    def run_text_prefill(
        self, inputs_embeds, deepstack_tensors, position_ids_3d, actual_seq_len
    ):
        """Chunked prefill. Returns (first_token, valid_length, all_hidden)."""
        # Zero the thinker KV cache first. The prefill engine has uninitialized
        # cache inputs at process start; without this the chunked prefill attends
        # into residual device memory, making the generated text nondeterministic
        # across runs (observed: 155 vs 97 tokens for identical greedy input),
        # which then cascades into divergent codecs/audio.
        self._reset_thinker_cache()
        static_seq = self.prefill_seq_len
        num_chunks = (actual_seq_len + static_seq - 1) // static_seq
        past_seq = 0
        last_logits = None
        all_hidden = []

        for ci in range(num_chunks):
            start = ci * static_seq
            end = min(start + static_seq, actual_seq_len)
            cl = end - start

            # Pad embeds to static_seq
            chunk_embed = np.zeros(
                (1, static_seq, THINKER_HIDDEN_SIZE), dtype=np.float16
            )
            chunk_embed[:, :cl, :] = inputs_embeds[:, start:end, :].detach().numpy()

            # Pad position ids
            def pad_pos(pos, s, e, target):
                p = pos[s:e].to(torch.int32).numpy()
                out = np.zeros(target, dtype=np.int32)
                out[: len(p)] = p
                return out

            time_pos = pad_pos(position_ids_3d[0], start, end, static_seq)
            height_pos = pad_pos(position_ids_3d[1], start, end, static_seq)
            width_pos = pad_pos(position_ids_3d[2], start, end, static_seq)

            # Pad deepstack
            chunk_ds = []
            for ds in deepstack_tensors:
                ds_pad = np.zeros(
                    (1, static_seq, THINKER_HIDDEN_SIZE), dtype=np.float16
                )
                ds_pad[:, :cl, :] = ds[:, start:end, :].detach().numpy()
                chunk_ds.append(ds_pad)

            past_arr = np.array([past_seq], dtype=np.int32)
            cl_arr = np.array([cl], dtype=np.int32)

            # Set inputs by index order matching exported model:
            # [0] input_embeds, [1] time_pos, [2] height_pos, [3] width_pos,
            # [4] valid_length, [5] current_length, [6-8] deepstack
            names = self.prefill_input_names
            self.prefill_engine.set_input(names[0], np.ascontiguousarray(chunk_embed))
            self.prefill_engine.set_input(names[1], np.ascontiguousarray(time_pos))
            self.prefill_engine.set_input(names[2], np.ascontiguousarray(height_pos))
            self.prefill_engine.set_input(names[3], np.ascontiguousarray(width_pos))
            self.prefill_engine.set_input(names[4], past_arr)
            self.prefill_engine.set_input(names[5], cl_arr)
            for di in range(3):
                if 6 + di < len(names):
                    self.prefill_engine.set_input(
                        names[6 + di], np.ascontiguousarray(chunk_ds[di])
                    )

            self.prefill_engine.run()
            self.prefill_engine.sync()

            # Read outputs: [0]=logits, [1]=hidden_states
            logits_np = self.prefill_engine.get_output(
                self.prefill_engine.get_output_name(0)
            ).numpy()
            last_logits = logits_np
            if self.prefill_engine.get_num_outputs() > 1:
                hidden_np = self.prefill_engine.get_output(
                    self.prefill_engine.get_output_name(1)
                ).numpy()
                h_t = torch.from_numpy(hidden_np).to(torch.float16)
                if h_t.ndim == 2:
                    h_t = h_t.unsqueeze(0)
                all_hidden.append(h_t[:, :cl, :])

            past_seq += cl

        # Sample first token
        logits_t = torch.from_numpy(last_logits).float()
        if logits_t.ndim == 2:
            logits_t = logits_t.unsqueeze(0)
        first_token = self.thinker_sampling.sample(
            logits_t[:, -1:, :].numpy(), generated_ids=[]
        )
        return first_token, past_seq, all_hidden

    def run_text_decode(
        self, first_token, past_seq, all_hidden, position_ids_3d, rope_deltas
    ):
        """Autoregressive decode loop. Returns (generated_ids, hidden_states)."""
        generated = [first_token]
        valid_length = past_seq

        for step in range(self.args.max_new_tokens - 1):
            token_id = generated[-1]
            if token_id in EOS_TOKEN_IDS:
                break
            if valid_length + 1 > self.kv_max_length:
                logger.warning("KV cache full")
                break

            # Embed token
            token_embed = self.text_embedding[token_id].unsqueeze(0).unsqueeze(0)
            token_np = np.ascontiguousarray(token_embed.detach().numpy())

            # Decode position: valid_length + rope_delta
            delta = int(rope_deltas.reshape(-1)[0].item())
            pos_val = valid_length + delta
            pos_np = np.array([pos_val], dtype=np.int32)

            # Zero deepstack for decode
            zero_ds = np.zeros((1, 1, THINKER_HIDDEN_SIZE), dtype=np.float16)

            names = self.decode_input_names
            self.decode_engine.set_input(names[0], token_np)
            self.decode_engine.set_input(names[1], pos_np)
            self.decode_engine.set_input(names[2], pos_np)
            self.decode_engine.set_input(names[3], pos_np)
            self.decode_engine.set_input(
                names[4], np.array([valid_length], dtype=np.int32)
            )
            self.decode_engine.set_input(names[5], np.array([1], dtype=np.int32))
            for di in range(3):
                if 6 + di < len(names):
                    self.decode_engine.set_input(names[6 + di], zero_ds.copy())

            self.decode_engine.run()
            self.decode_engine.sync()

            logits_np = self.decode_engine.get_output(
                self.decode_engine.get_output_name(0)
            ).numpy()
            if self.decode_engine.get_num_outputs() > 1:
                hidden_np = self.decode_engine.get_output(
                    self.decode_engine.get_output_name(1)
                ).numpy()
                h_t = torch.from_numpy(hidden_np).to(torch.float16)
                if h_t.ndim == 2:
                    h_t = h_t.unsqueeze(0)
                all_hidden.append(h_t)

            next_token = self.thinker_sampling.sample(
                logits_np, generated_ids=generated
            )
            generated.append(next_token)
            valid_length += 1

        hidden_states = torch.cat(all_hidden, dim=1) if all_hidden else None
        return generated, hidden_states

    # ── Projection helpers (standalone HMM engines, on-chip) ──────────────────

    def text_proj(self, x: torch.Tensor) -> torch.Tensor:
        """text_projection [1,T,2048]->[1,T,1024] via standalone HMM engine."""
        return run_projection_engine(
            self.text_proj_engine, x, static=self.text_proj_static_seq
        )

    # ── Talker Prefill Context (source/role/bypass fusion) ─────────────────────

    def build_talker_prefill_context(
        self, input_ids, full_ids, thinker_embed, thinker_hidden, speaker_name
    ):
        """Build fused talker prefill inputs, reproducing ptq.py's capture of the
        native HF _get_talker_user_parts / _get_talker_assistant_parts.

        The talker graph fuses as a SELECT (never a sum):
            projected = role_mask*text_proj(source) + (1-role_mask)*hidden_proj(source)
            inputs_embeds = (1-bypass_mask)*projected + bypass_mask*bypass_embeds
        where text_proj/hidden_proj are the talker's own baked-in projection
        branches. The user segment + the assistant 3-token prefix go through the
        talker's in-graph projection (source, bypass_mask=0); the codec tokens +
        first text token are supplied precomputed via bypass (bypass_mask=1). Only
        the standalone text_projection HMM engine is needed (to precompute the
        bypass sums) — this matches the working demo_hmonnx_full.py and produces
        correct speech.

        Returns: (source, role_mask, bypass_embeds, bypass_mask, actual_len,
                  trailing_text_hidden, tts_pad_proj) — projections are 1024-dim.
        """
        thinker_embed = thinker_embed.cpu().to(torch.float16)
        thinker_hidden = thinker_hidden.cpu().to(torch.float16)
        prompt_ids = input_ids[0].cpu().to(torch.long)
        full_seq = full_ids[0].cpu().to(torch.long)
        speaker_id = SPEAKER_ID_MAP.get(speaker_name.lower(), SPEAKER_ID_MAP["ethan"])

        # Segment by <|im_start|>; the role token is the token right after it.
        im_start_pos = torch.nonzero(prompt_ids == IM_START_TOKEN_ID).flatten()
        im_start_pos = torch.cat(
            [im_start_pos, torch.tensor([full_seq.shape[0]], dtype=torch.long)]
        )
        mm_mask = (
            (full_seq == AUDIO_TOKEN_ID)
            | (full_seq == IMAGE_TOKEN_ID)
            | (full_seq == VIDEO_TOKEN_ID)
        )

        # tts specials = text_projection(thinker_embed[special]) -> PROJECTED 1024.
        tts_bos_p = self.text_proj(
            self.text_embedding[torch.tensor([TTS_BOS_TOKEN_ID])].unsqueeze(0)
        )
        tts_eos_p = self.text_proj(
            self.text_embedding[torch.tensor([TTS_EOS_TOKEN_ID])].unsqueeze(0)
        )
        tts_pad_p = self.text_proj(
            self.text_embedding[torch.tensor([TTS_PAD_TOKEN_ID])].unsqueeze(0)
        )

        source_segs, role_segs, bypass_segs, bypass_mask_segs = [], [], [], []
        trailing_text_hidden = None

        def emit(src, role, byp, bypm):
            """Append a fused talker prefill segment (source/role/bypass/bypass_mask)."""
            source_segs.append(src)
            role_segs.append(role)
            bypass_segs.append(byp)
            bypass_mask_segs.append(bypm)

        for idx in range(len(im_start_pos) - 1):
            st, en = int(im_start_pos[idx]), int(im_start_pos[idx + 1])
            role_token = int(prompt_ids[st + 1]) if st + 1 < prompt_ids.shape[0] else -1

            if role_token == SYSTEM_TOKEN_ID:
                continue

            if role_token == USER_TOKEN_ID:
                # source = thinker_embed; mm positions replaced by thinker_hidden.
                seg_embed = thinker_embed[:, st:en, :].clone()
                seg_hidden = thinker_hidden[:, st:en, :]
                seg_mm = mm_mask[st:en]
                seg_len = en - st
                if seg_mm.any():
                    seg_embed[:, seg_mm, :] = seg_hidden[:, seg_mm, :]
                emit(
                    seg_embed,
                    (~seg_mm).unsqueeze(0).unsqueeze(-1).to(torch.float16),
                    torch.zeros(1, seg_len, TALKER_HIDDEN_SIZE, dtype=torch.float16),
                    torch.zeros(1, seg_len, 1, dtype=torch.float16),
                )
                continue

            if role_token == ASSISTANT_TOKEN_ID and idx == len(im_start_pos) - 2:
                asst_embed = thinker_embed[:, st:en, :]
                asst_proj = self.text_proj(asst_embed)  # [1, L, 1024]
                codec_ids = [
                    CODEC_NOTHINK_ID,
                    CODEC_THINK_BOS_ID,
                    CODEC_THINK_EOS_ID,
                    speaker_id,
                    CODEC_PAD_ID,
                    CODEC_BOS_ID,
                ]
                codec_embeds = self.talker_embedding[torch.tensor(codec_ids)].unsqueeze(
                    0
                )

                # native assistant input_embeds = assistant_text + assistant_codec
                assistant_text = torch.cat(
                    [
                        asst_proj[:, :3, :],
                        tts_pad_p.expand(1, 4, -1),
                        tts_bos_p,
                        asst_proj[:, 3:4, :],
                    ],
                    dim=1,
                )
                assistant_codec = torch.cat(
                    [
                        torch.zeros(1, 3, TALKER_HIDDEN_SIZE, dtype=torch.float16),
                        codec_embeds,
                    ],
                    dim=1,
                )
                input_embeds = assistant_text + assistant_codec
                asst_len = int(input_embeds.shape[1])

                # ptq capture: first <=3 tokens use the talker projection
                # (source=thinker_embed, bypass_mask=0); the rest bypass.
                projected_prefix = min(3, max(en - st, 0))
                assistant_source = torch.zeros(
                    1, asst_len, THINKER_HIDDEN_SIZE, dtype=torch.float16
                )
                assistant_source[:, :projected_prefix, :] = asst_embed[
                    :, :projected_prefix, :
                ]
                assistant_role_mask = torch.ones(1, asst_len, 1, dtype=torch.float16)
                assistant_bypass = input_embeds.clone()
                assistant_bypass[:, :projected_prefix, :] = 0
                assistant_bypass_mask = torch.ones(1, asst_len, 1, dtype=torch.float16)
                assistant_bypass_mask[:, :projected_prefix, :] = 0

                emit(
                    assistant_source,
                    assistant_role_mask,
                    assistant_bypass,
                    assistant_bypass_mask,
                )
                trailing_text_hidden = torch.cat(
                    [asst_proj[:, 4:, :], tts_eos_p], dim=1
                )
                continue

            if role_token == ASSISTANT_TOKEN_ID:
                continue

        source = torch.cat(source_segs, dim=1)
        role_mask = torch.cat(role_segs, dim=1)
        bypass_embeds = torch.cat(bypass_segs, dim=1)
        bypass_mask = torch.cat(bypass_mask_segs, dim=1)
        actual_len = int(source.shape[1])
        logger.info(f"Talker prefill context: actual_len={actual_len}")
        return (
            source,
            role_mask,
            bypass_embeds,
            bypass_mask,
            actual_len,
            trailing_text_hidden,
            tts_pad_p,
        )

    # ── Talker + Predictor Generate ───────────────────────────────────────────

    def run_talker_generate(
        self,
        source,
        role_mask,
        bypass_embeds,
        bypass_mask,
        actual_len,
        trailing_text_hidden,
        tts_pad,
    ):
        """Run talker + predictor decode loop. Returns codec codes [1, 16, N].

        Prefill feeds the captured (source, role_mask, bypass_embeds, bypass_mask)
        through the talker graph's fusion (select):
            projected = role*text_proj(source) + (1-role)*hidden_proj(source)
            inputs    = (1-bypass_mask)*projected + bypass_mask*bypass_embeds
        The user segment + assistant 3-token prefix project in-graph (bypass_mask=0);
        codec tokens + first text token are precomputed and bypassed (bypass_mask=1).
        """
        talker_static = self.talker_prefill_seq_len
        num_pred_heads = NUM_CODE_GROUPS - 1  # 15

        # Zero the talker KV cache before every generation. On-chip tcim buffers
        # persist across runs (and start uninitialized on the first run), so without
        # this the talker is polluted by the previous segment's state -> silence /
        # nondeterministic output even under deterministic argmax. Mirrors minicpmo's
        # per-segment TTS cache reset and our own _reset_pred_cache.
        self._reset_talker_cache()

        # ── Talker Prefill (chunked) ──
        num_chunks = (actual_len + talker_static - 1) // talker_static
        past_seq_val = 0
        talker_logits = None
        talker_hidden = None

        for ci in range(num_chunks):
            start = ci * talker_static
            end = min(start + talker_static, actual_len)
            cl = end - start

            padded_source = np.zeros(
                (1, talker_static, THINKER_HIDDEN_SIZE), dtype=np.float16
            )
            padded_role = np.zeros((1, talker_static, 1), dtype=np.float16)
            padded_bypass = np.zeros(
                (1, talker_static, TALKER_HIDDEN_SIZE), dtype=np.float16
            )
            padded_bypass_mask = np.zeros((1, talker_static, 1), dtype=np.float16)
            padded_source[:, :cl, :] = source[:, start:end, :].detach().numpy()
            padded_role[:, :cl, :] = role_mask[:, start:end, :].detach().numpy()
            padded_bypass[:, :cl, :] = bypass_embeds[:, start:end, :].detach().numpy()
            padded_bypass_mask[:, :cl, :] = (
                bypass_mask[:, start:end, :].detach().numpy()
            )
            past_arr = np.array([past_seq_val], dtype=np.int32)
            cl_arr = np.array([cl], dtype=np.int32)

            names = self.talker_prefill_names
            self.talker_prefill_engine.set_input(
                names[0], np.ascontiguousarray(padded_source)
            )
            self.talker_prefill_engine.set_input(
                names[1], np.ascontiguousarray(padded_role)
            )
            self.talker_prefill_engine.set_input(
                names[2], np.ascontiguousarray(padded_bypass)
            )
            self.talker_prefill_engine.set_input(
                names[3], np.ascontiguousarray(padded_bypass_mask)
            )
            self.talker_prefill_engine.set_input(names[4], past_arr)
            self.talker_prefill_engine.set_input(names[5], cl_arr)
            self.talker_prefill_engine.run()
            self.talker_prefill_engine.sync()

            talker_logits = torch.from_numpy(
                self.talker_prefill_engine.get_output(
                    self.talker_prefill_engine.get_output_name(0)
                ).numpy()
            )
            if self.talker_prefill_engine.get_num_outputs() > 1:
                talker_hidden = torch.from_numpy(
                    self.talker_prefill_engine.get_output(
                        self.talker_prefill_engine.get_output_name(1)
                    ).numpy()
                )
            past_seq_val += cl

        # Sample first codec token
        first_codec = greedy_or_sample(
            talker_logits.float().numpy(),
            self.args.talker_do_sample,
            self.args.talker_temperature,
            top_k=self.args.talker_top_k,
            repetition_penalty=self.args.talker_repetition_penalty,
            generated_ids=[],
            suppress_tokens=TALKER_SUPPRESS_TOKENS,
        )

        # ── Predictor for first token ──
        step_codes, last_mid_hiddens = self._run_predictor_full(
            talker_hidden, first_codec, num_pred_heads
        )
        all_codes = [step_codes]
        last_primary = first_codec

        # ── Talker Decode Loop ──
        talker_past_seq = past_seq_val
        trailing_idx = 0
        trailing_len = (
            int(trailing_text_hidden.shape[1])
            if trailing_text_hidden is not None
            else 0
        )

        for step in range(1, self.args.talker_max_new_tokens):
            prev_codec = all_codes[-1][0]
            if prev_codec == CODEC_EOS_TOKEN_ID:
                break

            # Build decode bypass: sum(primary_emb + mid_hiddens + last_res_emb) + text_h
            primary_emb = (
                self.talker_embedding[last_primary].reshape(1, 1, -1).to(torch.float16)
            )
            all_parts = [primary_emb]
            for h in last_mid_hiddens:
                if h is not None:
                    all_parts.append(h.to(torch.float16).reshape(1, 1, -1))
                else:
                    all_parts.append(primary_emb)
            last_res_token = all_codes[-1][-1]
            last_res_emb = (
                self.codec_embeddings[num_pred_heads - 1][last_res_token]
                .reshape(1, 1, -1)
                .to(torch.float16)
            )
            all_parts.append(last_res_emb)
            codec_sum = torch.cat(all_parts, dim=1).sum(dim=1, keepdim=True)

            if trailing_idx < trailing_len:
                text_h = trailing_text_hidden[:, trailing_idx : trailing_idx + 1, :]
            else:
                text_h = tts_pad
            trailing_idx += 1

            decode_bypass = (codec_sum + text_h).to(torch.float16)
            d_source = np.zeros((1, 1, THINKER_HIDDEN_SIZE), dtype=np.float16)
            d_role = np.ones((1, 1, 1), dtype=np.float16)
            d_bypass = np.ascontiguousarray(decode_bypass.detach().numpy())
            d_bpm = np.ones((1, 1, 1), dtype=np.float16)

            names = self.talker_decode_names
            self.talker_decode_engine.set_input(names[0], d_source)
            self.talker_decode_engine.set_input(names[1], d_role)
            self.talker_decode_engine.set_input(names[2], d_bypass)
            self.talker_decode_engine.set_input(names[3], d_bpm)
            self.talker_decode_engine.set_input(
                names[4], np.array([talker_past_seq], dtype=np.int32)
            )
            self.talker_decode_engine.set_input(names[5], np.array([1], dtype=np.int32))
            self.talker_decode_engine.run()
            self.talker_decode_engine.sync()

            d_logits = torch.from_numpy(
                self.talker_decode_engine.get_output(
                    self.talker_decode_engine.get_output_name(0)
                ).numpy()
            )
            d_hidden = None
            if self.talker_decode_engine.get_num_outputs() > 1:
                d_hidden = torch.from_numpy(
                    self.talker_decode_engine.get_output(
                        self.talker_decode_engine.get_output_name(1)
                    ).numpy()
                )

            next_codec = greedy_or_sample(
                d_logits.float().numpy(),
                self.args.talker_do_sample,
                self.args.talker_temperature,
                top_k=self.args.talker_top_k,
                repetition_penalty=self.args.talker_repetition_penalty,
                generated_ids=[sc[0] for sc in all_codes],
                suppress_tokens=TALKER_SUPPRESS_TOKENS,
            )
            talker_past_seq += 1

            if next_codec == CODEC_EOS_TOKEN_ID:
                break

            # Run predictor for this step
            step_codes, last_mid_hiddens = self._run_predictor_full(
                d_hidden, next_codec, num_pred_heads
            )
            all_codes.append(step_codes)
            last_primary = next_codec

        # Build codes tensor [1, 16, N]
        num_steps = len(all_codes)
        codes = torch.zeros(1, NUM_CODE_GROUPS, num_steps, dtype=torch.int32)
        for t, sc in enumerate(all_codes):
            for g in range(min(len(sc), NUM_CODE_GROUPS)):
                codes[0, g, t] = sc[g]
        logger.info(f"Talker generated {num_steps} codec steps")
        return codes

    def _run_predictor_full(self, talker_hidden, primary_token_id, num_pred_heads):
        """Run predictor chain: prefill(2 tokens) + decode(14 steps).
        Returns (step_codes_list, mid_hiddens_list).
        """
        # Reset predictor KV cache
        self._reset_pred_cache()

        # Predictor prefill: [talker_hidden, talker_embedding[primary]]
        primary_embed = (
            self.talker_embedding[primary_token_id].reshape(1, 1, -1).to(torch.float16)
        )
        if talker_hidden is not None:
            pred_input = torch.cat(
                [talker_hidden.to(torch.float16).reshape(1, 1, -1), primary_embed],
                dim=1,
            )
        else:
            pred_input = primary_embed.expand(1, 2, -1)

        # head_mask: only group 0 active [1, 2, 15, 1]
        head_mask = np.zeros((1, 2, num_pred_heads, 1), dtype=np.float16)
        head_mask[:, :, 0, 0] = 1.0

        names = self.pred_prefill_names
        self.pred_prefill_engine.set_input(
            names[0], np.ascontiguousarray(pred_input.detach().numpy())
        )
        self.pred_prefill_engine.set_input(names[1], np.ascontiguousarray(head_mask))
        self.pred_prefill_engine.set_input(names[2], np.array([0], dtype=np.int32))
        self.pred_prefill_engine.set_input(names[3], np.array([2], dtype=np.int32))
        self.pred_prefill_engine.run()
        self.pred_prefill_engine.sync()

        pred_logits = torch.from_numpy(
            self.pred_prefill_engine.get_output(
                self.pred_prefill_engine.get_output_name(0)
            ).numpy()
        )
        last_logits = pred_logits.float().reshape(-1, pred_logits.shape[-1])[-1]
        first_res = int(torch.argmax(last_logits[:2048]).item())

        step_codes = [primary_token_id, first_res]
        mid_hiddens = []
        ct = first_res

        # Decode remaining groups (1..14)
        for g in range(1, num_pred_heads):
            codec_embed = (
                self.codec_embeddings[g - 1][ct].reshape(1, 1, -1).to(torch.float16)
            )
            head_mask_d = np.zeros((1, 1, num_pred_heads, 1), dtype=np.float16)
            head_mask_d[:, :, g, 0] = 1.0

            names = self.pred_decode_names
            self.pred_decode_engine.set_input(
                names[0], np.ascontiguousarray(codec_embed.detach().numpy())
            )
            self.pred_decode_engine.set_input(
                names[1], np.ascontiguousarray(head_mask_d)
            )
            self.pred_decode_engine.set_input(
                names[2], np.array([2 + g - 1], dtype=np.int32)
            )
            self.pred_decode_engine.set_input(names[3], np.array([1], dtype=np.int32))
            self.pred_decode_engine.run()
            self.pred_decode_engine.sync()

            pred_logits_d = torch.from_numpy(
                self.pred_decode_engine.get_output(
                    self.pred_decode_engine.get_output_name(0)
                ).numpy()
            )
            pred_hidden_d = None
            if self.pred_decode_engine.get_num_outputs() > 1:
                pred_hidden_d = torch.from_numpy(
                    self.pred_decode_engine.get_output(
                        self.pred_decode_engine.get_output_name(1)
                    ).numpy()
                )

            last_logits = pred_logits_d.float().reshape(-1)
            ct = int(torch.argmax(last_logits[:2048]).item())
            step_codes.append(ct)
            mid_hiddens.append(pred_hidden_d)

        return step_codes, mid_hiddens

    def _reset_pred_cache(self):
        """Zero out predictor KV cache and re-bind prefill->decode."""
        for i in range(self.pred_prefill_engine.get_num_inputs()):
            name = self.pred_prefill_engine.get_input_name(i)
            if "cache" in name:
                info = self.pred_prefill_engine.get_input_info(name)
                shape = tuple(int(d) for d in info.shape)
                self.pred_prefill_engine.set_input(
                    name, np.zeros(shape, dtype=np.float16)
                )
        # Re-bind decode cache to prefill cache (set_input breaks the binding)
        self._bind_cache(self.pred_prefill_engine, self.pred_decode_engine)

    def _reset_thinker_cache(self):
        """Zero out thinker (text LLM) KV cache and re-bind prefill->decode.

        The prefill engine's 96 cache inputs are uninitialized at process start;
        without zeroing them the chunked prefill attends into residual device
        memory, so the generated text is nondeterministic across runs. Counterpart
        to _reset_talker_cache / _reset_pred_cache.
        """
        for i in range(self.prefill_engine.get_num_inputs()):
            name = self.prefill_engine.get_input_name(i)
            if "cache" in name:
                info = self.prefill_engine.get_input_info(name)
                shape = tuple(int(d) for d in info.shape)
                self.prefill_engine.set_input(name, np.zeros(shape, dtype=np.float16))
        # Re-bind decode cache to prefill cache (set_input breaks the binding)
        self._bind_cache(self.prefill_engine, self.decode_engine)

    def _reset_talker_cache(self):
        """Zero out talker KV cache and re-bind prefill->decode.

        The on-chip tcim talker buffers persist across generations and start
        uninitialized on the first run; without this reset the talker carries
        residual state from the previous segment, producing silence or
        nondeterministic output even under deterministic argmax. Counterpart to
        _reset_pred_cache; mirrors minicpmo's per-segment TTS cache reset.
        """
        for i in range(self.talker_prefill_engine.get_num_inputs()):
            name = self.talker_prefill_engine.get_input_name(i)
            if "cache" in name:
                info = self.talker_prefill_engine.get_input_info(name)
                shape = tuple(int(d) for d in info.shape)
                self.talker_prefill_engine.set_input(
                    name, np.zeros(shape, dtype=np.float16)
                )
        # Re-bind decode cache to prefill cache (set_input breaks the binding)
        self._bind_cache(self.talker_prefill_engine, self.talker_decode_engine)

    # ── Code2Wav ──────────────────────────────────────────────────────────────

    def run_code2wav(self, codes: torch.Tensor) -> torch.Tensor:
        """Chunked code2wav decode. codes: [1, 16, N] -> audio waveform."""
        static_len = self.code2wav_static_len
        codes_i32 = codes.to(torch.int32)
        actual_len = codes_i32.shape[2]
        left_context = 25

        def _run_single(chunk):
            cl = chunk.shape[2]
            if cl < static_len:
                chunk = F.pad(chunk, (0, static_len - cl))
            c2w_name = self.code2wav_engine.get_input_name(0)
            self.code2wav_engine.set_input(
                c2w_name, np.ascontiguousarray(chunk.numpy())
            )
            self.code2wav_engine.run()
            self.code2wav_engine.sync()
            out_name = self.code2wav_engine.get_output_name(0)
            out = torch.from_numpy(self.code2wav_engine.get_output(out_name).numpy())
            return out[..., : cl * CODE2WAV_UPSAMPLE]

        if actual_len <= static_len:
            return _run_single(codes_i32)

        safe_chunk = max(1, static_len - left_context)
        all_audio = []
        si = 0
        while si < actual_len:
            ei = min(si + safe_chunk, actual_len)
            ctx = left_context if si - left_context > 0 else si
            chunk_len = ei - si + ctx
            if chunk_len > static_len:
                ctx = max(0, static_len - (ei - si))
            chunk = codes_i32[:, :, si - ctx : ei]
            wav = _run_single(chunk)
            all_audio.append(wav[..., ctx * CODE2WAV_UPSAMPLE :])
            si = ei
        return torch.cat(all_audio, dim=-1)

    # ── Talker synthesis from text ids (shared by single + segmented paths) ────

    def synth_codes_from_text_ids(
        self,
        input_ids,
        inputs_embeds,
        talker_text_ids,
        hidden_states,
        speaker=DEFAULT_SPEAKER,
    ):
        """Build talker context for one assistant text-id list and run the talker.

        Factored out of run() so it can be invoked per sentence segment. Each call
        does its own talker prefill + KV-cache reset, so segments are independent.
        Returns codec codes [1, 16, N].
        """
        gen_embeds = (
            self.text_embedding[torch.tensor(talker_text_ids)]
            .unsqueeze(0)
            .to(torch.float16)
        )
        full_embed = torch.cat([inputs_embeds, gen_embeds], dim=1)
        full_ids = torch.cat([input_ids, torch.tensor([talker_text_ids])], dim=1)

        if hidden_states is not None:
            gap = full_embed.shape[1] - hidden_states.shape[1]
            if gap > 0:
                pad_h = hidden_states[:, -1:, :].expand(1, gap, -1)
                thinker_hidden = torch.cat([hidden_states, pad_h], dim=1)
            else:
                thinker_hidden = hidden_states[:, : full_embed.shape[1], :]
        else:
            thinker_hidden = full_embed

        (
            source,
            role_mask,
            bypass_embeds,
            bypass_mask,
            talker_actual_len,
            trailing_text_hidden,
            tts_pad,
        ) = self.build_talker_prefill_context(
            input_ids, full_ids, full_embed, thinker_hidden, speaker
        )
        return self.run_talker_generate(
            source,
            role_mask,
            bypass_embeds,
            bypass_mask,
            talker_actual_len,
            trailing_text_hidden,
            tts_pad,
        )

    # ── Position IDs ──────────────────────────────────────────────────────────

    def _build_rope_shim(self):
        """Lazily build a lightweight carrier exposing exactly what HF
        get_rope_index needs (config token-ids + spatial_merge_size + the
        vision helper), so we can reuse the native 3D M-RoPE index math WITHOUT
        loading the 30B thinker weights."""
        if getattr(self, "_rope_shim", None) is not None:
            return self._rope_shim
        import types as _types

        pos_per_sec = 13
        try:
            cfg = json.load(open(Path(self.model_dir) / "config.json"))
            tc = cfg.get("thinker_config", cfg)
            for src in (tc, tc.get("text_config", {}), cfg):
                if isinstance(src, dict) and "position_id_per_seconds" in src:
                    pos_per_sec = int(src["position_id_per_seconds"])
                    break
        except Exception:
            pass
        sms = int(getattr(self.processor.image_processor, "merge_size", 2))
        shim = _types.SimpleNamespace()
        shim.config = _types.SimpleNamespace(
            audio_start_token_id=self.audio_start_token_id,
            audio_token_id=AUDIO_TOKEN_ID,
            image_token_id=IMAGE_TOKEN_ID,
            video_token_id=VIDEO_TOKEN_ID,
            vision_start_token_id=self.vision_start_token_id,
            position_id_per_seconds=pos_per_sec,
        )
        shim.spatial_merge_size = sms
        shim.get_llm_pos_ids_for_vision = _types.MethodType(
            Qwen3OmniMoePreTrainedModelForConditionalGeneration.get_llm_pos_ids_for_vision,
            shim,
        )
        self._rope_shim = shim
        return shim

    def compute_position_ids(self, input_ids, attention_mask, inputs):
        """Position ids for the thinker prefill/decode graphs.

        Two modes (args.rope_mode):
          "linear" (default): t=h=w = cumulative position (attention_mask.cumsum-1).
              This matches the deployed prefill/decode .hmm engines, which were
              traced/calibrated with linear positions (the same approximation the
              known-good demo_hmonnx_full.py uses). Correct, coherent output.
          "exact": true 3D M-RoPE via the native HF get_rope_index on a lightweight
              shim (no 30B weights). Matches the HF reference index math exactly, but
              the current engines do NOT honor the compressed t/h/w contract and
              produce garbage with it — kept for analysis/experimentation only.

        Returns (position_ids_3d [3, S] int32, rope_deltas [B, 1] long).
        """
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids)
        input_ids = input_ids.cpu().to(torch.long)
        attention_mask = attention_mask.cpu().to(torch.long)
        mode = self.args.rope_mode

        if mode == "exact":
            img_thw = inputs.get("image_grid_thw")
            if img_thw is not None:
                img_thw = torch.as_tensor(img_thw).cpu().to(torch.long)
            vid_thw = inputs.get("video_grid_thw")
            if vid_thw is not None:
                vid_thw = torch.as_tensor(vid_thw).cpu().to(torch.long)
            second_per_grids = inputs.get("video_second_per_grid")
            if second_per_grids is not None:
                second_per_grids = torch.as_tensor(second_per_grids).cpu().float()
            audio_seqlens = None
            if "feature_attention_mask" in inputs:
                audio_seqlens = inputs["feature_attention_mask"].cpu().sum(-1).long()
            shim = self._build_rope_shim()
            pos, rope_deltas = (
                Qwen3OmniMoePreTrainedModelForConditionalGeneration.get_rope_index(
                    shim,
                    input_ids,
                    img_thw,
                    vid_thw,
                    attention_mask,
                    use_audio_in_video=True,
                    audio_seqlens=audio_seqlens,
                    second_per_grids=second_per_grids,
                )
            )
            delta0 = (1 - attention_mask).sum(dim=-1, keepdim=True)
            rope_deltas = (rope_deltas - delta0).to(torch.long)
            return pos[:, 0, :].to(torch.int32), rope_deltas

        # linear (default)
        position_ids = attention_mask.cumsum(-1) - 1
        position_ids.masked_fill_(attention_mask == 0, 1)
        position_ids_3d = position_ids.unsqueeze(0).expand(3, -1, -1).clone()
        max_position_ids = position_ids.max(-1, keepdim=True)[0]
        rope_deltas = (
            max_position_ids + 1 - attention_mask.sum(dim=-1, keepdim=True)
        ).to(torch.long)
        return position_ids_3d[:, 0, :].to(torch.int32), rope_deltas

    # ── Main Run ──────────────────────────────────────────────────────────────

    def run(
        self,
        image=None,
        audio=None,
        prompt=None,
        speaker=DEFAULT_SPEAKER,
        generate_audio=None,
    ):
        """Single-turn convenience wrapper: assemble a one-user-turn conversation
        from optional image/audio/text and delegate to run_conversation. Any
        subset of modalities is allowed (audio-only for ASR, image+text for VLM,
        etc.) — the audio/visual encoders run conditionally on what's present.
        """
        content = []
        if image is not None:
            image_path = Path(image).resolve()
            if not image_path.exists():
                raise FileNotFoundError(f"Image not found: {image_path}")
            content.append({"type": "image", "image": str(image_path)})
        if audio is not None:
            audio_path = Path(audio).resolve()
            if not audio_path.exists():
                raise FileNotFoundError(f"Audio not found: {audio_path}")
            content.append({"type": "audio", "audio": str(audio_path)})
        if prompt is not None:
            content.append({"type": "text", "text": prompt})
        conversation = [{"role": "user", "content": content}]
        return self.run_conversation(
            conversation, speaker=speaker, generate_audio=generate_audio
        )

    def run_conversation(
        self,
        conversation,
        speaker=DEFAULT_SPEAKER,
        generate_audio=None,
        use_audio_in_video=False,
    ):
        """Core multimodal inference over an arbitrary chat `conversation`.

        `conversation` is a standard Qwen3-Omni message list; it may carry history
        (prior user/assistant turns) so multi-turn speech dialogue stays coherent
        — the thinker prefills the whole history and the talker builds its context
        from the full rendered ids. `generate_audio` overrides whether speech is
        synthesized for this call (None -> follow self.enable_audio_generation).
        Sentence-segmented synthesis is driven by args.talker_segment (an example
        can set self.args.talker_segment in its own block to turn it on).
        """
        args = self.args
        gen_audio = (
            self.enable_audio_generation
            if generate_audio is None
            else (bool(generate_audio) and self.enable_audio_generation)
        )
        seg = args.talker_segment
        logger.info(f"Conversation: {conversation}")
        perf: Dict[str, float] = {}
        t0 = time.perf_counter()

        # ── Preprocess ──
        text = self.processor.apply_chat_template(
            conversation, add_generation_prompt=True, tokenize=False
        )
        audios, images, videos = process_mm_info(
            conversation, use_audio_in_video=use_audio_in_video
        )
        inputs = self.processor(
            text=text,
            audio=audios,
            images=images,
            videos=videos,
            return_tensors="pt",
            padding=True,
            seconds_per_chunk=2.0,
            position_id_per_seconds=13,
            use_audio_in_video=use_audio_in_video,
        )

        input_ids = inputs["input_ids"].cpu()
        attention_mask = inputs.get("attention_mask", torch.ones_like(input_ids)).cpu()
        actual_seq_len = int(input_ids.shape[1])
        logger.info(f"Input sequence length: {actual_seq_len}")
        perf["input_tokens"] = actual_seq_len

        # ── Build multimodal embeddings ──
        inputs_embeds = self.text_embedding[input_ids[0]].unsqueeze(0).to(torch.float16)
        deepstack_tensors = [
            torch.zeros(1, actual_seq_len, THINKER_HIDDEN_SIZE, dtype=torch.float16)
            for _ in range(3)
        ]

        if "input_features" in inputs and "feature_attention_mask" in inputs:
            logger.info("Running audio encoder...")
            _t = time.perf_counter()
            audio_embeds = self.run_audio_encoder(
                inputs["input_features"].cpu(), inputs["feature_attention_mask"].cpu()
            )
            audio_mask = input_ids[0] == AUDIO_TOKEN_ID
            num_audio = int(audio_mask.sum().item())
            audio_embeds = audio_embeds[:num_audio]
            inputs_embeds[0, audio_mask] = audio_embeds.to(torch.float16)
            perf["audio_encoder_s"] = time.perf_counter() - _t
            perf["audio_tokens"] = num_audio
            logger.info(f"  Audio: {num_audio} tokens injected")

        pv_key = "hm_pixel_values" if "hm_pixel_values" in inputs else "pixel_values"
        if pv_key in inputs:
            logger.info("Running visual encoder...")
            _t = time.perf_counter()
            vision_embeds, ds_list = self.run_visual_encoder(inputs[pv_key].cpu())
            image_mask = input_ids[0] == IMAGE_TOKEN_ID
            inputs_embeds[0, image_mask] = vision_embeds.to(torch.float16)
            for i, ds in enumerate(ds_list):
                dense = torch.zeros(
                    1, actual_seq_len, THINKER_HIDDEN_SIZE, dtype=torch.float16
                )
                dense[0, image_mask] = ds.to(torch.float16)
                deepstack_tensors[i] = dense
            perf["visual_encoder_s"] = time.perf_counter() - _t
            perf["image_tokens"] = int(vision_embeds.shape[0])
            logger.info(f"  Visual: {vision_embeds.shape[0]} tokens injected")

        # ── Text LLM Generation ──
        logger.info("Running text LLM...")
        position_ids_3d, rope_deltas = self.compute_position_ids(
            input_ids, attention_mask, inputs
        )

        _t = time.perf_counter()
        first_token, valid_length, all_hidden = self.run_text_prefill(
            inputs_embeds, deepstack_tensors, position_ids_3d, actual_seq_len
        )
        perf["thinker_prefill_s"] = time.perf_counter() - _t
        perf["thinker_ttft_s"] = (
            perf["thinker_prefill_s"]
            + perf.get("audio_encoder_s", 0.0)
            + perf.get("visual_encoder_s", 0.0)
        )
        logger.info(
            f"  Prefill done, first_token={first_token}, valid_length={valid_length}"
        )

        _t = time.perf_counter()
        generated_ids, hidden_states = self.run_text_decode(
            first_token, valid_length, all_hidden, position_ids_3d, rope_deltas
        )
        perf["thinker_decode_s"] = time.perf_counter() - _t
        perf["thinker_decode_tokens"] = max(0, len(generated_ids) - 1)

        output_text = self.tokenizer.batch_decode(
            [generated_ids],
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )
        logger.info(f"Generated text: {output_text}")

        # ── Audio Generation ──
        wav_data = None
        if gen_audio:
            logger.info("Running audio generation...")
            # Sanitize the decoded text BEFORE it reaches the talker. Markdown
            # markup (*, **, #, bullets) and \n have no pronunciation; left in,
            # the codec emits garbled frames around them (confirmed via whisper
            # re-transcription). We strip them, re-tokenize the clean text, and
            # drive the talker from the cleaned ids. The user-facing output_text
            # above is left untouched.
            raw_text = output_text[0] if output_text else ""
            clean_text = sanitize_text_for_talker(raw_text)
            if clean_text != raw_text:
                logger.info(f"Sanitized talker text: {clean_text!r}")
                clean_ids = self.tokenizer(clean_text, add_special_tokens=False)[
                    "input_ids"
                ]
                # Re-append the assistant turn terminator the thinker emitted so
                # the talker still sees an end-of-turn marker.
                eos_tail = [t for t in generated_ids[-1:] if t in EOS_TOKEN_IDS]
                talker_text_ids = clean_ids + eos_tail
            else:
                talker_text_ids = generated_ids

            # Drop the trailing turn terminator(s); we re-add one per segment.
            eos_tail = [t for t in talker_text_ids[-1:] if t in EOS_TOKEN_IDS]
            body_ids = (
                talker_text_ids[: -len(eos_tail)] if eos_tail else talker_text_ids
            )

            _t = time.perf_counter()
            if seg:
                # Sentence-segmented synthesis. The talker codec stream degenerates
                # into unintelligible audio past a few hundred decode steps; voicing
                # each sentence as its own short utterance (cache resets per call)
                # keeps every call inside the proven-reliable regime. Whisper
                # re-transcription confirms long answers stay intelligible end-to-end
                # this way, vs. garbled back-halves when run as one long sequence.
                clean_for_seg = (
                    clean_text
                    if clean_text
                    else (output_text[0] if output_text else "")
                )
                segments = split_text_into_segments(
                    clean_for_seg, max_chars=args.talker_segment_max_chars
                )
                logger.info(f"Talker segmentation: {len(segments)} segment(s)")
                gap = torch.zeros(
                    int(SAMPLE_RATE * args.talker_segment_gap_ms / 1000.0)
                )
                seg_wavs = []
                total_codes = 0
                eos_one = eos_tail or [IM_END_TOKEN_ID]
                for si, seg in enumerate(segments):
                    seg_ids = (
                        self.tokenizer(seg, add_special_tokens=False)["input_ids"]
                        + eos_one
                    )
                    seg_codes = self.synth_codes_from_text_ids(
                        input_ids,
                        inputs_embeds,
                        seg_ids,
                        hidden_states,
                        speaker=speaker,
                    )
                    total_codes += int(seg_codes.shape[2])
                    seg_wav = self.run_code2wav(seg_codes).float().cpu().flatten()
                    seg_wavs.append(seg_wav)
                    if si < len(segments) - 1:
                        seg_wavs.append(gap)
                    logger.info(
                        f"  segment {si+1}/{len(segments)}: "
                        f"{int(seg_codes.shape[2])} codes, {seg!r}"
                    )
                audio_wav = torch.cat(seg_wavs) if seg_wavs else torch.zeros(1)
                perf["talker_s"] = time.perf_counter() - _t
                perf["talker_codes"] = total_codes
                perf["talker_segments"] = len(segments)
                perf["code2wav_s"] = 0.0  # folded into talker loop above
            else:
                codes = self.synth_codes_from_text_ids(
                    input_ids,
                    inputs_embeds,
                    body_ids + (eos_tail or [IM_END_TOKEN_ID]),
                    hidden_states,
                    speaker=speaker,
                )
                perf["talker_s"] = time.perf_counter() - _t
                perf["talker_codes"] = int(codes.shape[2])

                dump = os.environ.get("QWEN_DUMP_TOKENS")
                if dump:
                    np.savez(
                        dump,
                        text_ids=np.asarray(generated_ids, dtype=np.int64),
                        codes=codes.cpu().numpy().astype(np.int64),
                    )
                    logger.info(f"  Dumped token streams -> {dump}")

                logger.info("Running code2wav...")
                _t = time.perf_counter()
                audio_wav = self.run_code2wav(codes)
                perf["code2wav_s"] = time.perf_counter() - _t
            logger.info(f"Audio waveform shape: {audio_wav.shape}")
            wav_data = audio_wav.float().cpu().numpy().flatten()
            perf["audio_out_s"] = wav_data.shape[0] / float(SAMPLE_RATE)
            perf["speech_total_s"] = perf["talker_s"] + perf["code2wav_s"]

        perf["e2e_total_s"] = time.perf_counter() - t0
        logger.info("\n" + _format_perf_report(perf))
        return {"output_text": output_text, "wav_data": wav_data, "perf": perf}


# ═══════════════════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════════════════
def get_args():
    # fmt: off
    p = argparse.ArgumentParser(description="Qwen3-Omni xh2 HMM demo (tcim_lite)")
    p.add_argument("--config", dest="config_path", type=str, default=DEFAULT_CONFIG_PATH,
                   help="path to config.yaml")
    p.add_argument("--model_name", dest="model_name", type=str, default=None,
                   help="model name")
    p.add_argument("--model_size", dest="model_size", type=str, default=None,
                   help="model size")
    p.add_argument("--model_dir", dest="model_dir", type=str, default=None,
                   help="hf model dir (processor / tokenizer)")
    p.add_argument("--embedding_path", dest="embedding_path", type=str,
                   default=str(DEFAULT_HMQUANT_DIR),
                   help="houmo embedding weight path (hmquant dir)")
    p.add_argument("--visual_path", dest="visual_path", type=str, default=None,
                   help="houmo visual model path")
    p.add_argument("--audio_path", dest="audio_path", type=str, default=None,
                   help="houmo audio model path")
    p.add_argument("--prefill_path", dest="prefill_path", type=str, default=None,
                   help="houmo thinker prefill model path")
    p.add_argument("--decode_path", dest="decode_path", type=str, default=None,
                   help="houmo thinker decode model path")
    p.add_argument("--talker_prefill_path", dest="talker_prefill_path", type=str,
                   default=None, help="houmo talker prefill model path")
    p.add_argument("--talker_decode_path", dest="talker_decode_path", type=str,
                   default=None, help="houmo talker decode model path")
    p.add_argument("--talker_prediction_prefill_path",
                   dest="talker_prediction_prefill_path", type=str, default=None,
                   help="houmo talker prediction prefill model path")
    p.add_argument("--talker_prediction_decode_path",
                   dest="talker_prediction_decode_path", type=str, default=None,
                   help="houmo talker prediction decode model path")
    p.add_argument("--text_projection_path", dest="text_projection_path", type=str,
                   default=None, help="houmo text projection model path")
    p.add_argument("--code2wav_path", dest="code2wav_path", type=str, default=None,
                   help="houmo code2wav model path")
    p.add_argument("--ndevice", type=int, default=None, choices=[1, 2],
                   help="model device number")
    p.add_argument("--run_device_num", type=int, default=1, choices=[1, 2],
                   help="run device number")
    p.add_argument("--device_id", dest="device_id", type=int, default=0,
                   help="Houmo device index")
    p.add_argument("--enable_audio_generation", action=argparse.BooleanOptionalAction,
                   dest="enable_audio_generation", default=True,
                   help="generate speech")
    p.add_argument("--output_dir", dest="output_dir", type=str, default="./",
                   help="output dir for generated audio")
    p.add_argument("--example_idx", dest="example_idx", type=int, default=0,
                   help="example mode index: 0(omni), 1(conv, multi-turn speech "
                        "dialogue), 2(vlm), 3(music, talker_segment on), 4(asr), "
                        "5(translate)")
    # ── Post-processing / generation config ──
    p.add_argument("--max_new_tokens", dest="max_new_tokens", type=int, default=512,
                   help="thinker text decode cap")
    p.add_argument("--talker_max_new_tokens", dest="talker_max_new_tokens", type=int,
                   default=1024, help="talker codec decode cap")
    p.add_argument("--rope_mode", dest="rope_mode", type=str, default="linear",
                   choices=["linear", "exact"],
                   help="linear (default): t=h=w cumulative positions, matches the "
                        "deployed engines (correct output). exact: true HF 3D M-RoPE "
                        "via get_rope_index (analysis only; current engines mis-handle it)")
    p.add_argument("--thinker_do_sample", action=argparse.BooleanOptionalAction,
                   dest="thinker_do_sample", default=False,
                   help="thinker text sampling (default: greedy)")
    p.add_argument("--thinker_temperature", dest="thinker_temperature", type=float,
                   default=0.7)
    p.add_argument("--thinker_top_k", dest="thinker_top_k", type=int, default=20)
    p.add_argument("--thinker_top_p", dest="thinker_top_p", type=float, default=0.9)
    p.add_argument("--thinker_repetition_penalty", dest="thinker_repetition_penalty",
                   type=float, default=1.0)
    p.add_argument("--talker_do_sample", action=argparse.BooleanOptionalAction,
                   dest="talker_do_sample", default=True,
                   help="enable talker sampling (repetition-penalty + suppress + "
                        "temperature + top-k + multinomial); --no-talker_do_sample "
                        "for greedy argmax (default: sampling)")
    p.add_argument("--talker_temperature", dest="talker_temperature", type=float,
                   default=0.9)
    p.add_argument("--talker_top_k", dest="talker_top_k", type=int, default=50)
    p.add_argument("--talker_repetition_penalty", dest="talker_repetition_penalty",
                   type=float, default=1.05)
    p.add_argument("--talker_segment", action=argparse.BooleanOptionalAction,
                   dest="talker_segment", default=False,
                   help="synthesize speech sentence-by-sentence and concatenate "
                        "(codec cache resets per segment, and segments can stream "
                        "out incrementally). Off by default; pass --talker_segment "
                        "to force it on, or an example can set args.talker_segment "
                        "in its own block (e.g. the music example).")
    p.add_argument("--talker_segment_max_chars", dest="talker_segment_max_chars",
                   type=int, default=60,
                   help="pack sentences into segments up to this many chars")
    p.add_argument("--talker_segment_gap_ms", dest="talker_segment_gap_ms",
                   type=int, default=120,
                   help="silence inserted between synthesized segments (ms)")
    args = p.parse_args()
    # fmt: on

    default_model_size, default_model_name, model_configs = get_model_configs(
        args.config_path
    )
    args.model_name = first_not_none(args.model_name, default_model_name)
    args.model_size = first_not_none(args.model_size, default_model_size)
    model_config = model_configs.get(args.model_name, {}).get(args.model_size, {})

    if args.model_dir is None:
        repo_ids = model_config.get("modelscope_repo", [])
        if repo_ids:
            args.model_dir = str(SCRIPT_DIR / repo_ids[0].rsplit("/", maxsplit=1)[-1])
        else:
            args.model_dir = DEFAULT_MODEL_DIR
    args.ndevice = first_not_none(args.ndevice, model_config.get("ndevice", 1))

    pfx = f"{args.model_name}-{args.model_size}"
    hmm_dir = os.path.join("output", HOUMO_TARGET)
    engine_defaults = {
        "visual_path": f"{pfx}_visual.hmm",
        "audio_path": f"{pfx}_audio.hmm",
        "prefill_path": f"{pfx}_prefill.hmm",
        "decode_path": f"{pfx}_decode.hmm",
        "talker_prefill_path": f"{pfx}_talker_prefill.hmm",
        "talker_decode_path": f"{pfx}_talker_decode.hmm",
        "talker_prediction_prefill_path": f"{pfx}_talker_prediction_prefill.hmm",
        "talker_prediction_decode_path": f"{pfx}_talker_prediction_decode.hmm",
        "text_projection_path": f"{pfx}_text_projection.hmm",
        "code2wav_path": f"{pfx}_code2wav.hmm",
    }
    for attr, fname in engine_defaults.items():
        if getattr(args, attr) is None:
            setattr(args, attr, os.path.join(hmm_dir, fname))
    return args


def _save_and_report(result, output_dir, wav_name):
    """Print the generated text and, if audio was produced, write the wav."""
    output_text = result["output_text"]
    print(f"\n[Result] text: {output_text}")
    wav_data = result.get("wav_data")
    if wav_data is not None:
        out_dir = Path(output_dir).resolve()
        out_dir.mkdir(parents=True, exist_ok=True)
        wav_path = out_dir / wav_name
        sf.write(str(wav_path), wav_data, SAMPLE_RATE)
        print(f"[Result] audio: {wav_path}")


def main():
    args = get_args()
    pipeline = Qwen3OmniHmmPipeline(args)

    if args.run_device_num < 2 and args.enable_audio_generation:
        logger.warning(
            "Running Omni on a single device requires 48GB of memory; otherwise, multiple devices are needed."
        )

    example_mode = EXAMPLES_MODE[args.example_idx]

    if example_mode == "omni":
        # image + audio + text -> text + speech
        image = str(DEFAULT_SAMPLE_DIR / "cars.jpg")
        audio = str(DEFAULT_SAMPLE_DIR / "cough.wav")
        prompt = "结合图像和音频内容，用一句话简述你看到和听到了什么？"
        result = pipeline.run(
            image=image, audio=audio, prompt=prompt, speaker=DEFAULT_SPEAKER
        )
        _save_and_report(result, args.output_dir, "omni_output.wav")

    elif example_mode == "conv":
        # Multi-turn speech conversation. The user speaks (comment0.wav), the
        # assistant replies with text + speech; the assistant's text reply is then
        # appended to the conversation as history so the second user turn
        # (comment1.wav) is answered with full context — mirrors minicpmo's CONV
        # example, where history is carried turn-to-turn for coherence.
        sys_msg = {
            "role": "system",
            "content": [
                {
                    "type": "text",
                    "text": "你是一个智能助手，可以接受语音和文本输入并输出语音和文本。"
                    "请用自然、口语化的方式回答用户的问题，保持对话的连贯性。如果是创作类问题，只需创作即可，不需要额外说明。",
                }
            ],
        }
        # Turn 0. Speech generation stays in normal single-sequence mode (no
        # talker_segment) — on this chip the talker codec stays intelligible well
        # past a long answer (verified by re-transcribing a ~1000-codec-step /
        # ~80s reply end-to-end), so chat replies don't need segmentation.
        conversation = [
            sys_msg,
            {
                "role": "user",
                "content": [
                    {"type": "audio", "audio": str(DEFAULT_SAMPLE_DIR / "comment0.wav")}
                ],
            },
        ]
        logger.info("=== Conversation turn 0 ===")
        result0 = pipeline.run_conversation(conversation, speaker=DEFAULT_SPEAKER)
        _save_and_report(result0, args.output_dir, "conv_output_0.wav")

        # Carry the assistant's text reply as history, then add turn 1.
        answer0 = result0["output_text"][0] if result0["output_text"] else ""
        conversation.append(
            {"role": "assistant", "content": [{"type": "text", "text": answer0}]}
        )
        conversation.append(
            {
                "role": "user",
                "content": [
                    {"type": "audio", "audio": str(DEFAULT_SAMPLE_DIR / "comment1.wav")}
                ],
            }
        )
        logger.info("=== Conversation turn 1 (with history) ===")
        result1 = pipeline.run_conversation(conversation, speaker=DEFAULT_SPEAKER)
        _save_and_report(result1, args.output_dir, "conv_output_1.wav")

    elif example_mode == "vlm":
        # image + text -> text + speech. Speech generation stays on so the answer
        # is both described in text and spoken.
        image = str(DEFAULT_SAMPLE_DIR / "2233.jpg")
        prompt = "请描述一下这张图片的内容。"
        result = pipeline.run(image=image, prompt=prompt, speaker=DEFAULT_SPEAKER)
        _save_and_report(result, args.output_dir, "vlm_output.wav")

    elif example_mode == "music":
        # Music appreciation: audio + text -> text + speech. The answer is long
        # (style/rhythm/dynamics/emotion/instruments/background); this example
        # turns on sentence-segmented synthesis by flipping args.talker_segment,
        # which synthesizes the reply segment-by-segment (codec cache resets per
        # segment) and lets it stream out incrementally.
        audio = str(DEFAULT_SAMPLE_DIR / "音乐风格-调性.mp3")
        prompt = "描述这首乐曲的风格、节奏、力度和情感表达。指出所使用的乐器，并推测这首乐曲可能的创作背景。用尽量简洁的语言来描述。"
        result = pipeline.run(audio=audio, prompt=prompt, speaker=DEFAULT_SPEAKER)
        _save_and_report(result, args.output_dir, "music_output.wav")

    elif example_mode == "asr":
        # Speech recognition: speech audio -> text only (no speech output).
        audio = str(DEFAULT_SAMPLE_DIR / "asr_zh.wav")
        prompt = "请将这段中文语音转换为纯文本。"
        result = pipeline.run(audio=audio, prompt=prompt, generate_audio=False)
        _save_and_report(result, args.output_dir, "asr_output.wav")

    elif example_mode == "translate":
        # Speech translation: English speech audio -> Chinese text (no speech out).
        audio = str(DEFAULT_SAMPLE_DIR / "asr_en.wav")
        prompt = "请听这段英文语音，并将其内容翻译成中文。"
        result = pipeline.run(audio=audio, prompt=prompt, generate_audio=False)
        _save_and_report(result, args.output_dir, "translate_output.wav")

    else:
        raise ValueError(f"Unsupported example_idx={args.example_idx}")


if __name__ == "__main__":
    main()
