# Copyright (c) 2025 HOUMO AI
#
# File: demo.py
# Description:
#   SenseVoiceSmall ASR with Optimized Feature-Level Chunking.
#   Splits features dynamically based on model capacity while maintaining
#   better context handling through overlapping windows and smarter padding.
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
import argparse
import json
import re
import sys
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torchaudio
import yaml
import librosa

import tcim_lite as tcim
import math
import time
from loguru import logger

HOUMO_TARGET = os.getenv("HOUMO_TARGET")

# --- Constants ---
LANGUAGE_MAP: Dict[str, int] = {"auto": 0, "zh": 3, "en": 4, "yue": 7, "ja": 11, "ko": 12, "nospeech": 13}
TEXTNORM_MAP: Dict[str, int] = {"withitn": 14, "woitn": 15}

# Overlap frames to maintain context between chunks
DEFAULT_OVERLAP_FRAMES = 10  # frames of overlap
DEFAULT_CONTEXT_FRAMES = 20  # extra context frames to pad on both sides


def get_args() -> argparse.Namespace:
    """Parse commandline."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--audio_files",
        nargs="+",
        default=["../../../data/audio/audio.mp3"],
        help="Audio files to transcribe",
    )
    parser.add_argument(
        "--model_path",
        dest="model_path",
        type=str,
        default=os.path.join("output", HOUMO_TARGET, "sensevoice_small.hmm"),
        help="houmo model path",
    )
    parser.add_argument(
        "--assets-dir",
        dest="assets_dir",
        type=str,
        default="SenseVoiceSmall",
        help="Dir with config.yaml/am.mvn/tokens.json",
    )
    parser.add_argument(
        "--language",
        dest="language",
        type=str,
        default="auto",
        choices=LANGUAGE_MAP.keys(),
    )
    parser.add_argument(
        "--textnorm",
        dest="textnorm",
        type=str,
        default="woitn",
        choices=TEXTNORM_MAP.keys()
    )
    parser.add_argument(
        "--overlap_frames",
        dest="overlap_frames",
        type=int,
        default=DEFAULT_OVERLAP_FRAMES,
        help="Number of overlapping frames between chunks for context continuity",
    )
    parser.add_argument(
        "--context_frames",
        dest="context_frames",
        type=int,
        default=DEFAULT_CONTEXT_FRAMES,
        help="Extra context frames to pad on both sides of each chunk",
    )
    parser.add_argument(
        "--raw_result",
        action="store_true",
        help="Show Raw Result"
    )
    parser.add_argument(
        "--device_monitor",
        action="store_true",
        help="Device Monitor During Demo"
    )

    args = parser.parse_args()
    return args


def _read_yaml(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _load_cmvn(cmvn_file: Path) -> Any:
    lines = cmvn_file.read_text(encoding="utf-8").splitlines()
    means_list = []
    vars_list = []
    for i in range(len(lines)):
        line_item = lines[i].split()
        if not line_item:
            continue
        if line_item[0] == "<AddShift>":
            line_item = lines[i + 1].split()
            if line_item and line_item[0] == "<LearnRateCoef>":
                means_list = list(line_item[3 : len(line_item) - 1])
        elif line_item[0] == "<Rescale>":
            line_item = lines[i + 1].split()
            if line_item and line_item[0] == "<LearnRateCoef>":
                vars_list = list(line_item[3 : len(line_item) - 1])

    means = np.array(means_list, dtype=np.float64)
    vars_ = np.array(vars_list, dtype=np.float64)
    if means.size == 0 or vars_.size == 0:
        raise ValueError(f"failed to parse cmvn file: {cmvn_file}")
    return np.stack([means, vars_], axis=0)


def _apply_cmvn(feat: Any, cmvn: Any) -> Any:
    frame, dim = feat.shape
    means = np.tile(cmvn[0:1, :dim], (frame, 1))
    vars_ = np.tile(cmvn[1:2, :dim], (frame, 1))
    return (feat + means) * vars_


def _apply_lfr(inputs: Any, lfr_m: int, lfr_n: int) -> Any:
    if lfr_m == 1 and lfr_n == 1:
        return inputs.astype(np.float32)
    lfr_inputs = []
    t = inputs.shape[0]
    t_lfr = int(np.ceil(t / lfr_n))
    left_padding = np.tile(inputs[0], ((lfr_m - 1) // 2, 1))
    inputs = np.vstack((left_padding, inputs))
    t = t + (lfr_m - 1) // 2
    for i in range(t_lfr):
        if lfr_m <= t - i * lfr_n:
            lfr_inputs.append((inputs[i * lfr_n : i * lfr_n + lfr_m]).reshape(1, -1))
        else:
            num_padding = lfr_m - (t - i * lfr_n)
            frame = inputs[i * lfr_n :].reshape(-1)
            for _ in range(num_padding):
                frame = np.hstack((frame, inputs[-1]))
            lfr_inputs.append(frame)
    return np.vstack(lfr_inputs).astype(np.float32)


@dataclass(frozen=True)
class FrontendConfig:
    fs: int = 16000
    window: str = "hamming"
    n_mels: int = 80
    frame_length: int = 25
    frame_shift: int = 10
    lfr_m: int = 7
    lfr_n: int = 6
    dither: float = 1.0
    cmvn_file: str = ""


class SenseVoiceFrontend:
    def __init__(self, cfg: FrontendConfig):
        self.cfg = cfg
        self.cmvn = _load_cmvn(Path(cfg.cmvn_file)) if cfg.cmvn_file else None

    @classmethod
    def from_model_dir(cls, model_dir: Union[str, Path]) -> "SenseVoiceFrontend":
        model_dir = Path(model_dir).expanduser().resolve()
        cfg_path = model_dir / "config.yaml"
        if not cfg_path.exists():
            print(f"Warning: {cfg_path} not found. Using default config but CMVN might fail.")
            cfg_obj = FrontendConfig(cmvn_file=str(model_dir / "am.mvn"))
        else:
            cfg = _read_yaml(cfg_path)
            frontend_conf = dict(cfg.get("frontend_conf") or {})
            cfg_obj = FrontendConfig(
                fs=int(frontend_conf.get("fs", 16000)),
                window=str(frontend_conf.get("window", "hamming")),
                n_mels=int(frontend_conf.get("n_mels", 80)),
                frame_length=int(frontend_conf.get("frame_length", 25)),
                frame_shift=int(frontend_conf.get("frame_shift", 10)),
                lfr_m=int(frontend_conf.get("lfr_m", 7)),
                lfr_n=int(frontend_conf.get("lfr_n", 6)),
                dither=float(frontend_conf.get("dither", 1.0)),
                cmvn_file=str(model_dir / "am.mvn"),
            )
        return cls(cfg_obj)

    def fbank(self, waveform: Any) -> Tuple[Any, int]:
        waveform_np = np.array(waveform)
        wav = torch.as_tensor(waveform_np, dtype=torch.float32)
        if wav.ndim == 1:
            wav = wav.unsqueeze(0)
        if wav.shape[0] != 1:
            wav = wav.mean(dim=0, keepdim=True)

        feat = torchaudio.compliance.kaldi.fbank(
            wav * (1 << 15),
            num_mel_bins=self.cfg.n_mels,
            sample_frequency=self.cfg.fs,
            frame_length=float(self.cfg.frame_length),
            frame_shift=float(self.cfg.frame_shift),
            dither=float(self.cfg.dither),
            window_type=self.cfg.window,
            snip_edges=True,
            energy_floor=0.0,
            use_energy=False,
        )
        feat = feat.cpu().numpy().astype("float32")
        return feat, int(feat.shape[0])

    def extract(self, waveform: Any) -> Tuple[Any, int]:
        feat, feat_len = self.fbank(waveform)
        feat = _apply_lfr(feat, self.cfg.lfr_m, self.cfg.lfr_n)
        if self.cmvn is not None:
            feat = _apply_cmvn(feat, self.cmvn).astype("float32")
        return feat, int(feat.shape[0])


def load_tokens(tokens_path: Path) -> Optional[List[str]]:
    if not tokens_path.exists():
        return None
    obj = json.loads(tokens_path.read_text(encoding="utf-8"))
    if not isinstance(obj, list):
        return None
    return [str(x) for x in obj]


def load_audio(audio_path: str, fs: int = 16000) -> np.ndarray:
    """Load audio file and return waveform."""
    sample, orig_sr = librosa.load(audio_path)
    if orig_sr != fs:
        waveform = librosa.resample(sample, orig_sr=orig_sr, target_sr=fs)
        logger.info(f"Resampled audio from {orig_sr}Hz to 16000Hz")
    else:
        waveform, _ = librosa.load(audio_path, sr=fs)
    return waveform


def resolve_tag(v: str, mapping: Dict[str, int]) -> int:
    if v.isdigit():
        return int(v)
    key = v.lower().strip()
    if key not in mapping:
        raise ValueError(f"unsupported value: {v}; supported: {sorted(mapping.keys())}")
    return int(mapping[key])


def make_inputs_for_sample(feat: Any, feat_len: int, language: str, textnorm: str) -> Dict[str, Any]:
    speech = feat[None, :, :].astype("float32")
    speech_lengths = np.array([feat_len], dtype="int32")
    lang = np.array([resolve_tag(language, LANGUAGE_MAP)], dtype="int32")
    norm = np.array([resolve_tag(textnorm, TEXTNORM_MAP)], dtype="int32")
    return {"speech": speech, "speech_lengths": speech_lengths, "language": lang, "textnorm": norm}


def ctc_greedy_decode(logits: Any, out_len: int, blank_id: int = 0) -> List[int]:
    x = torch.as_tensor(logits)
    x = x[:out_len]
    y = x.argmax(dim=-1)
    y = torch.unique_consecutive(y, dim=-1)
    y = y[y != blank_id]
    return [int(v) for v in y.cpu().tolist()]


def decode_token_ids(token_ids: List[int], token_list: Optional[List[str]]) -> str:
    if not token_list:
        return " ".join(str(x) for x in token_ids)
    toks = [token_list[i] if 0 <= i < len(token_list) else "" for i in token_ids]
    s = "".join(toks)
    s = s.replace("▁", " ").strip()
    s = re.sub(r"\s+", " ", s)
    return s


def strip_rich_tags(s: str) -> str:
    return re.sub(r"<\|.*?\|>", "", s)


class SenseVoiceSmallFeature:
    """
    SenseVoice with Optimized Feature-Level Chunking.

    Improvements over basic approach:
    1. Dynamic chunk sizing based on model capacity
    2. Context padding on both sides for better boundary accuracy
    3. Overlapping frames for context continuity
    4. Smarter padding (edge-padding vs zero-padding)
    5. Better handling of boundary frames
    """
    def __init__(
        self,
        model_path: str,
        assets_dir: str,
        language: str,
        textnorm: str,
        overlap_frames: int = DEFAULT_OVERLAP_FRAMES,
        context_frames: int = DEFAULT_CONTEXT_FRAMES,
        **kwargs,
    ):
        self.model_file = model_path
        self.assets_dir = assets_dir
        self.overlap_frames = overlap_frames
        self.context_frames = context_frames

        self.frontend = SenseVoiceFrontend.from_model_dir(assets_dir)
        self.target_sr = self.frontend.cfg.fs

        tokens_path = Path(assets_dir) / "tokens.json"
        print(f"Loading tokens from {tokens_path}")
        self.token_list = load_tokens(tokens_path)
        if self.token_list is None:
            print("Warning: Failed to load tokens. Output will be token IDs.")

        weight_manager = tcim.runtime.WeightManager(0)
        option = tcim.runtime.Option(weight_manager)
        print("Loading model from", self.model_file)
        self.sess = tcim.runtime.load(self.model_file, option=option)

        # Get model max feature length
        self.max_feat_len = self.sess.get_input_info(self.sess.get_input_name(0)).shape[1]
        print(f"Model max feature length: {self.max_feat_len}")

        # Reserve space for context padding (max 10 frames per side)
        self.max_context_per_side = 10

        # Effective chunk size considering overlap
        self.effective_chunk_size = self.max_feat_len - overlap_frames
        if self.effective_chunk_size <= 0:
            logger.warning(f"Overlap frames {overlap_frames} >= max_feat_len {self.max_feat_len}. Disabling overlap.")
            self.effective_chunk_size = self.max_feat_len
            self.overlap_frames = 0

        print(f"Effective chunk size: {self.effective_chunk_size} (max_len={self.max_feat_len}, overlap={self.overlap_frames}, max_context_per_side={self.max_context_per_side})")

        self.language = language
        self.textnorm = textnorm

    def calculate_chunk_boundaries(self, total_feat_len: int) -> List[Tuple[int, int]]:
        """
        Calculate chunk boundaries with optional overlap.

        Returns list of (start, end) tuples where end is exclusive.
        """
        if total_feat_len <= self.max_feat_len:
            return [(0, total_feat_len)]

        boundaries = []
        start = 0

        while start < total_feat_len:
            # Use effective_chunk_size for content length to leave room for context
            content_end = start + self.effective_chunk_size
            end = min(content_end, total_feat_len)
            boundaries.append((start, end))

            # Move start forward with overlap consideration
            next_start = start + self.effective_chunk_size
            if next_start >= total_feat_len:
                break
            start = next_start

        return boundaries

    def run_inference(self, audio_files: List[str], raw_result: bool) -> None:
        """
        Process audio files with optimized feature-level chunking.
        """
        for audio_file in audio_files:
            logger.info(f"{'=' * 20} Process {audio_file} {'=' * 20}")

            t0_total = time.time()

            # Load and extract features for full audio
            waveform = load_audio(audio_file, self.target_sr)
            audio_duration = len(waveform) / self.target_sr

            # Extract features
            feat, feat_len = self.frontend.extract(waveform)
            logger.info(f"Audio duration: {audio_duration:.2f}s, Feature frames: {feat_len}")

            # Calculate chunk boundaries
            boundaries = self.calculate_chunk_boundaries(feat_len)
            logger.info(f"Split into {len(boundaries)} feature chunks "
                       f"(max_len={self.max_feat_len}, overlap={self.overlap_frames})")

            # Process each chunk with sliding window
            confirmed_text = ""  # Confirmed text
            pending_text = ""    # Unconfirmed text (sliding window)
            slide_len = 10       # Sliding window length
            total_infer_time = 0.0

            for i, (start, end) in enumerate(boundaries):
                t0 = time.time()

                content_len = end - start

                # Calculate context: up to 10 per side
                left_context = min(self.max_context_per_side, start)
                right_context = min(self.max_context_per_side, feat_len - end)

                # Ensure content + context = max_feat_len exactly
                total_with_context = content_len + left_context + right_context
                if total_with_context > self.max_feat_len:
                    excess = total_with_context - self.max_feat_len
                    content_len = content_len - excess
                    end = start + content_len

                # Calculate context boundaries
                context_start = start - left_context
                context_end = end + right_context

                logger.info(f"Chunk {i+1}/{len(boundaries)}: content={content_len} frames, "
                           f"context=[{left_context}, {right_context}], "
                           f"total={content_len + left_context + right_context} frames")

                # Extract chunk with context
                chunk_feat = feat[context_start:context_end, :]
                chunk_len = context_end - context_start

                # Apply padding if needed
                if chunk_len < self.max_feat_len:
                    pad_width = self.max_feat_len - chunk_len
                    last_frame = chunk_feat[-1:, :]
                    pad_feats = np.tile(last_frame, (pad_width, 1))
                    chunk_feat = np.vstack([chunk_feat, pad_feats])

                inputs = make_inputs_for_sample(chunk_feat, chunk_len, self.language, self.textnorm)

                # Run inference
                for k, v in inputs.items():
                    self.sess.set_input(k, v)
                self.sess.run()
                self.sess.sync()

                # Get outputs
                logits = self.sess.get_output(self.sess.get_output_name(0)).numpy()
                lens = self.sess.get_output(self.sess.get_output_name(1)).numpy()

                # Calculate how many logits correspond to context frames
                ctc_ratio = lens[0] / chunk_len if chunk_len > 0 else 0.25
                left_context_logits = int(left_context * ctc_ratio)
                right_context_logits = int(right_context * ctc_ratio)

                # Crop context from both sides
                start_idx = left_context_logits
                end_idx = int(lens[0]) - right_context_logits

                # Handle overlap with previous chunk
                if i > 0:
                    overlap_logits = int(content_len * ctc_ratio * self.overlap_frames / self.max_feat_len)
                    if overlap_logits > 0:
                        end_idx -= overlap_logits

                chunk_logits = logits[:, start_idx:end_idx, :]
                chunk_lens = end_idx - start_idx

                # Decode this chunk
                chunk_token_ids = ctc_greedy_decode(chunk_logits[0], chunk_lens)
                chunk_text = decode_token_ids(chunk_token_ids, self.token_list)
                chunk_text = strip_rich_tags(chunk_text)

                # Sliding window: keep last slide_len chars as pending
                if len(chunk_text) > slide_len:
                    confirm_part = chunk_text[:-slide_len]
                    pending_part = chunk_text[-slide_len:]

                    if pending_text:
                        merged = self._merge_with_overlap(pending_text, confirm_part)
                        confirmed_text += merged
                    else:
                        confirmed_text += confirm_part

                    pending_text = pending_part
                else:
                    pending_text = chunk_text

                t1 = (time.time() - t0) * 1000.0
                total_infer_time += t1

            # Append final pending
            if pending_text:
                confirmed_text += pending_text

            clean_text = confirmed_text
            if raw_result:
                logger.info(f"Raw Output: {text}")
            logger.info(f"Clean Text: {clean_text}")

            # Performance stats
            t_total = (time.time() - t0_total) * 1000.0
            audio_duration_ms = audio_duration * 1000
            rtf = total_infer_time / audio_duration_ms

            logger.success(f"Performance:")
            logger.success(f"  audio_duration: {audio_duration_ms:.2f} ms")
            logger.success(f"  total infer time: {total_infer_time:.2f} ms")
            logger.success(f"  end-to-end time: {t_total:.2f} ms")
            logger.success(f"  feature chunks: {len(boundaries)}")
            logger.success(f"  rtf (infer_time/audio_duration): {rtf:.2f}")

    def _merge_with_overlap(self, s1: str, s2: str) -> str:
        """
        Merge two strings with overlap detection.
        Find the longest suffix of s1 that matches prefix of s2.
        """
        if not s1 or not s2:
            return s1 + s2

        # Try to find longest overlap
        max_overlap = min(len(s1), len(s2))
        for i in range(max_overlap, 0, -1):
            if s1[-i:] == s2[:i]:
                return s1 + s2[i:]

        # No overlap found
        return s1 + s2


if __name__ == "__main__":
    args = get_args()
    smi_monitor = None
    if args.device_monitor:
        import hmatc.python.smi as smi
        smi_monitor = smi.DeviceMonitor(0, 100)
        smi_monitor.start()
    if HOUMO_TARGET == "xh2":
        sensevoice = SenseVoiceSmallFeature(
            args.model_path,
            args.assets_dir,
            args.language,
            args.textnorm,
            overlap_frames=args.overlap_frames,
            context_frames=args.context_frames,
        )
    else:
        raise ValueError("Unsupported houmo target!")

    sensevoice.run_inference(args.audio_files, args.raw_result)

    if args.device_monitor:
        smi_monitor.stop()
