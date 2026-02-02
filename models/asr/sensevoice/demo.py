# Copyright (c) 2025 HOUMO AI
#
# File: demo.py
# Description:
#   SenseVoiceSmall ASR Inference Demo - Python script for running sensevoice_small
# automatic speech recognition on HOUMO AI device.
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
import sys, os
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


def get_args() -> argparse.Namespace:
    """Parse commandline."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--audio_files",
        nargs="+",
        default=["SenseVoiceSmall/example/zh.mp3", "SenseVoiceSmall/example/en.mp3",
                 "SenseVoiceSmall/example/yue.mp3", "SenseVoiceSmall/example/ja.mp3",
                 "SenseVoiceSmall/example/ko.mp3"],
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
        "--raw_result",
        action="store_true",
        help="Show Raw Result"
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
        # wav = torch.as_tensor(waveform, dtype=torch.float32)
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

def load_data(wav_content: Union[str, np.ndarray, List[str]], fs: int = None) -> List:
    def load_wav(path: str) -> np.ndarray:
        waveform, _ = librosa.load(path, sr=fs)
        return waveform

    if isinstance(wav_content, np.ndarray):
        return [wav_content]

    if isinstance(wav_content, str):
        return [load_wav(wav_content)]

    if isinstance(wav_content, list):
        return [load_wav(path) for path in wav_content]

    raise TypeError(f"The type of {wav_content} is not in [str, np.ndarray, list]")

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
    s = re.sub(r"\\s+", " ", s)
    return s


def strip_rich_tags(s: str) -> str:
    return re.sub(r"<\|.*?\|>", "", s)


class SenseVoiceSmall:
    def __init__(
        self,
        model_path: str,
        assets_dir: str,
        language: str,
        textnorm: str,
        **kwargs,
    ):
        self.model_file = model_path
        self.config_file = os.path.join(assets_dir, "../SenseVoiceSmall/config.yaml")
        self.cmvn_file = os.path.join(assets_dir, "../SenseVoiceSmall/am.mvn")

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

        self.language = language
        self.textnorm = textnorm

    def run_inference(self, audio_files: List, raw_result: bool) -> Tuple[List[int], int]:
        for audio_file in audio_files:
            logger.info(f"{'=' * 20} Process {audio_file} {'=' * 20}")
            wav = load_data(str(audio_file), self.target_sr)
            audio_duration = len(wav[0]) / self.target_sr * 1000
            feat, feat_len = self.frontend.extract(wav)

            inputs = make_inputs_for_sample(feat, feat_len, self.language, self.textnorm)

            t0 = time.time()
            outs = []
            max_feat_len = self.sess.get_input_info(self.sess.get_input_name(0)).shape[1]
            feats = inputs['speech']
            feats_len = np.max(inputs['speech_lengths'])
            loop_round = math.ceil(feats.shape[1] / max_feat_len)
            for loop_idx in range(loop_round):
                cur_feats_len = 0
                if (loop_idx + 1) * max_feat_len > feats_len:
                    feats_end_idx = feats_len
                    cur_feats_len = np.array([feats_len - loop_idx * max_feat_len]).astype(np.int32)
                    cur_feats = feats[:, loop_idx * max_feat_len : feats_end_idx, :]
                    pad_width = max_feat_len - np.max(cur_feats_len)
                    _pad_feats = np.zeros((cur_feats.shape[0], pad_width, cur_feats.shape[2]), dtype=np.float32)
                    cur_feats = np.concatenate([cur_feats, _pad_feats], axis=1)
                else:
                    feats_end_idx = (loop_idx + 1) * max_feat_len
                    cur_feats_len = np.array([max_feat_len]).astype(np.int32)
                    cur_feats = feats[:, loop_idx * max_feat_len : feats_end_idx, :]

                inputs["speech"] = cur_feats
                inputs["speech_lengths"] = cur_feats_len
                for k, v in inputs.items():
                    self.sess.set_input(k, v)
                self.sess.run()
                self.sess.sync()
                if len(outs) < 2:
                    outs.append(
                            self.sess.get_output(self.sess.get_output_name(0)).numpy()
                        )
                    outs.append(
                            self.sess.get_output(self.sess.get_output_name(1)).numpy()
                        )
                else:
                    outs[0] = np.concatenate([outs[0], self.sess.get_output(self.sess.get_output_name(0)).numpy()], axis=1)
                    outs[1] = outs[1] + self.sess.get_output(self.sess.get_output_name(1)).numpy()

            t1 = (time.time() - t0) * 1000.0

            logits, lens = outs
            out_len = int(lens[0]) if hasattr(lens, "__len__") else int(lens)
            token_ids = ctc_greedy_decode(logits[0], out_len)
            text = decode_token_ids(token_ids, self.token_list)

            clean_text = strip_rich_tags(text)
            if raw_result:
                logger.info(f"Raw Output: {text}")
            logger.info(f"Clean Text: {clean_text}")
            rtf = audio_duration / t1
            logger.success(f"Performance: ")
            logger.success(f"  audio_duration: {audio_duration :.2f} ms")
            logger.success(f"  infer time {t1 :.2f} ms")
            logger.success(f"  rtf(audio_duration / infer_time): {rtf:.2f}")

if __name__ == "__main__":
    args = get_args()

    # init houmo whisper model
    if HOUMO_TARGET == "xh2":
        sensevoice_small = SenseVoiceSmall(
            args.model_path,
            args.assets_dir,
            args.language,
            args.textnorm,
        )
    else:
        raise ValueError("Unsupport houmo target!")

    sensevoice_small.run_inference(args.audio_files, args.raw_result)
