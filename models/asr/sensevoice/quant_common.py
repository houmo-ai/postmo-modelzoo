# Copyright (c) 2025 HOUMO AI
#
# File: quant_common.py
# Description:
#   common functions for quantization
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
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

from quant_frontend import SenseVoiceFrontend


LANGUAGE_MAP: Dict[str, int] = {"auto": 0, "zh": 3, "en": 4, "yue": 7, "ja": 11, "ko": 12, "nospeech": 13}
TEXTNORM_MAP: Dict[str, int] = {"withitn": 14, "woitn": 15}
_ONNX_SESS_CACHE: Dict[str, Any] = {}
_HMONNX_SESS_CACHE: Dict[str, Any] = {}

def resolve_tag(v: str, mapping: Dict[str, int]) -> int:
    if v.isdigit():
        return int(v)
    key = v.lower().strip()
    if key not in mapping:
        raise ValueError(f"unsupported value: {v}; supported: {sorted(mapping.keys())}")
    return int(mapping[key])


@dataclass(frozen=True)
class Sample:
    audio: Any
    text: str
    language: str = "auto"
    textnorm: str = "woitn"
    audio_id: str = ""


def read_manifest_jsonl(path: Path) -> List[Sample]:
    out: List[Sample] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            audio = obj.get("audio") or obj.get("source")
            text = obj.get("text") or obj.get("target") or ""
            if not audio:
                raise ValueError(f"manifest line missing audio/source: {obj}")
            out.append(
                Sample(
                    audio=str(audio),
                    text=str(text),
                    language=str(obj.get("language") or "auto"),
                    textnorm=str(obj.get("textnorm") or obj.get("textnorm_type") or "woitn"),
                    audio_id=str(obj.get("id") or audio),
                )
            )
    return out


def read_wav_scp_text(wav_scp: Path, text: Path) -> List[Sample]:
    wav_map: Dict[str, str] = {}
    with wav_scp.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            k, v = line.split(maxsplit=1)
            wav_map[k] = v

    text_map: Dict[str, str] = {}
    with text.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            k, v = line.split(maxsplit=1)
            text_map[k] = v

    keys = sorted(set(wav_map.keys()) & set(text_map.keys()))
    return [Sample(audio=wav_map[k], text=text_map[k], audio_id=k) for k in keys]


def load_hf_dataset(
    dataset: str,
    config: str,
    split: str,
    limit: int,
    streaming: bool = False,
    audio_field: str = "audio",
    text_field: str = "",
) -> List[Sample]:
    from datasets import load_dataset

    kwargs: Dict[str, Any] = {"path": dataset, "split": split}
    if config:
        kwargs["name"] = config
    kwargs["streaming"] = bool(streaming)
    ds = load_dataset(**kwargs)

    out: List[Sample] = []
    n = 0
    for row in ds:
        if limit > 0 and n >= limit:
            break
        if audio_field not in row:
            continue
        audio = row[audio_field]
        if text_field:
            text = row.get(text_field, "")
        else:
            text = row.get("text") or row.get("sentence") or row.get("transcript") or row.get("transcription") or ""
        audio_id = str(row.get("id") or row.get("file") or row.get("__key__") or f"{dataset}:{split}:{n}")
        out.append(Sample(audio=audio, text=str(text), audio_id=audio_id))
        n += 1
    return out


def load_audio_any(sample: Sample, target_sr: int) -> Any:
    if isinstance(sample.audio, dict) and "array" in sample.audio and "sampling_rate" in sample.audio:
        import numpy as np

        wav = np.asarray(sample.audio["array"], dtype=np.float32)
        sr = int(sample.audio["sampling_rate"])
        if sr != target_sr:
            import librosa

            wav = librosa.resample(wav, orig_sr=sr, target_sr=target_sr)
        return wav

    if isinstance(sample.audio, dict) and "path" in sample.audio:
        audio_path = str(sample.audio["path"])
    else:
        audio_path = str(sample.audio)

    import librosa

    wav, _ = librosa.load(audio_path, sr=target_sr, mono=True)
    return wav


def build_frontend(model_dir: Path) -> SenseVoiceFrontend:
    return SenseVoiceFrontend.from_model_dir(model_dir)


def extract_features(frontend: SenseVoiceFrontend, waveform: Any) -> Tuple[Any, int]:
    return frontend.extract(waveform)


def edit_distance(ref: Sequence[Any], hyp: Sequence[Any]) -> int:
    n = len(ref)
    m = len(hyp)
    dp = list(range(m + 1))
    for i in range(1, n + 1):
        prev = dp[0]
        dp[0] = i
        for j in range(1, m + 1):
            cur = dp[j]
            cost = 0 if ref[i - 1] == hyp[j - 1] else 1
            dp[j] = min(dp[j] + 1, dp[j - 1] + 1, prev + cost)
            prev = cur
    return dp[m]


def cer(ref: str, hyp: str) -> float:
    r = list(ref)
    h = list(hyp)
    denom = max(1, len(r))
    return edit_distance(r, h) / denom


def wer(ref: str, hyp: str) -> float:
    r = [w for w in ref.split() if w]
    h = [w for w in hyp.split() if w]
    denom = max(1, len(r))
    return edit_distance(r, h) / denom


def load_tokens(tokens_path: Path) -> Optional[List[str]]:
    if not tokens_path.exists():
        return None
    obj = json.loads(tokens_path.read_text(encoding="utf-8"))
    if not isinstance(obj, list):
        return None
    return [str(x) for x in obj]


def decode_token_ids(token_ids: List[int], token_list: Optional[List[str]]) -> str:
    if not token_list:
        return " ".join(str(x) for x in token_ids)
    toks = [token_list[i] if 0 <= i < len(token_list) else "" for i in token_ids]
    s = "".join(toks)
    s = s.replace("▁", " ").strip()
    s = re.sub(r"\\s+", " ", s)
    return s


def ctc_greedy_decode(logits: Any, out_len: int, blank_id: int = 0) -> List[int]:
    import torch

    x = torch.as_tensor(logits)
    x = x[:out_len]
    y = x.argmax(dim=-1)
    y = torch.unique_consecutive(y, dim=-1)
    y = y[y != blank_id]
    return [int(v) for v in y.cpu().tolist()]


def strip_rich_tags(s: str) -> str:
    return re.sub(r"<\|.*?\|>", "", s)


def run_onnx(onnx_path: Path, inputs: Dict[str, Any]) -> Tuple[Any, Any]:
    import numpy as np
    import onnxruntime as ort

    cache_key = str(onnx_path)
    sess = _ONNX_SESS_CACHE.get(cache_key)
    if sess is None:
        sess = ort.InferenceSession(str(onnx_path), providers=["CUDAExecutionProvider"])
        _ONNX_SESS_CACHE[cache_key] = sess
    feed: Dict[str, Any] = {}
    input_metadata = sess.get_inputs()
    feats_pad_len = 0
    for idx, input_info in enumerate(input_metadata):
        if input_info.name == "speech":
            feats_pad_len = input_info.shape[1]
            feats = pad_feats(inputs["speech"], feats_pad_len)
            inputs["speech"] = feats

    for k, v in inputs.items():
        if hasattr(v, "numpy"):
            v = v.numpy()
        if isinstance(v, np.ndarray):
            feed[k] = v
        else:
            feed[k] = np.asarray(v)
    outs = sess.run(None, feed)
    return outs[0], outs[1]


def run_hmonnx(hmonnx_path: Path, inputs: Dict[str, Any], device: str, fast: bool = False) -> Tuple[Any, Any]:
    import torch
    from xhquant.api import HMONNXInference

    cache_key = str(hmonnx_path)
    sess = _HMONNX_SESS_CACHE.get(cache_key)
    dev = torch.device(device)

    if sess is None:
        sess = HMONNXInference(str(hmonnx_path))
        sess.to(dev)
        _HMONNX_SESS_CACHE[cache_key] = sess

    if hasattr(sess, "device") and sess.device != dev:
        sess.to(dev)

    if fast:
        sess.to_fast_mode()

    input_info = {info.name: info for info in sess.inputs}

    feed: Dict[str, Any] = {}
    for k, v in inputs.items():
        t = torch.as_tensor(v)

        # Auto-cast dtype if mismatch
        if k in input_info:
            target_info = input_info[k]
            target_dtype = target_info.dtype
            if t.dtype != target_dtype:
                t = t.to(dtype=target_dtype)

            # Auto-pad/truncate if shape mismatch (for static shape models)
            # Only apply to 'speech' input which has time dimension at index 1: [B, T, D]
            target_shape = target_info.shape
            if k == "speech" and len(target_shape) == 3 and t.ndim == 3:
                 cur_t = t.shape[1]
                 tgt_t = target_shape[1]
                 if cur_t != tgt_t:
                     import torch.nn.functional as F
                     if cur_t < tgt_t:
                         t = F.pad(t, (0, 0, 0, tgt_t - cur_t))
                     else:
                         t = t[:, :tgt_t, :]

        feed[k] = t.to(dev)
    outs = sess.run(feed)
    return outs[0], outs[1]


def make_inputs_for_sample(feat: Any, feat_len: int, language: str, textnorm: str) -> Dict[str, Any]:
    import numpy as np

    speech = feat[None, :, :].astype("float32")
    speech_lengths = np.array([feat_len], dtype="int32")
    lang = np.array([resolve_tag(language, LANGUAGE_MAP)], dtype="int32")
    norm = np.array([resolve_tag(textnorm, TEXTNORM_MAP)], dtype="int32")
    return {"speech": speech, "speech_lengths": speech_lengths, "language": lang, "textnorm": norm}

import numpy as np
def pad_feats(feats: List[np.ndarray], max_feat_len: int) -> np.ndarray:
    def pad_feat(feat: np.ndarray, cur_len: int) -> np.ndarray:
        pad_width = ((0, max_feat_len - cur_len), (0, 0))
        return np.pad(feat, pad_width, "constant", constant_values=0)

    feat_res = [pad_feat(feat, feat.shape[0]) for feat in feats]
    feats = np.array(feat_res).astype(np.float32)
    return feats

def close_hf_streaming() -> None:
    try:
        import fsspec.asyn

        fsspec.asyn.close()
    except Exception:
        pass
