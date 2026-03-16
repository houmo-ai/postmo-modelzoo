# Copyright (c) 2025 HOUMO AI
#
# File: build.py
# Description:
#  Fun-CosyVoice3-0.5B-2512 Model Demo script.
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

import os
import sys
import re
import math
import regex
import time
import json
import threading
import argparse
import warnings
import torch
from torch import nn
from torch.nn import functional as F
import torchaudio
import torchaudio.compliance.kaldi as kaldi
from scipy.signal import get_window
import whisper
import inflect
import uuid
import glob
import numpy as np
from transformers import AutoTokenizer
from librosa.filters import mel as librosa_mel_fn
from contextlib import nullcontext
from tqdm import tqdm
from functools import partial
from typing import List, Generator, Optional, Dict, Any
from loguru import logger

import tcim_lite as tcim

logger.remove()
logger.add(sys.stderr, level="INFO")

# Suppress noisy UserWarning from torchaudio / torch STFT, keep other logs
warnings.filterwarnings("ignore", category=UserWarning, module="torchaudio")
warnings.filterwarnings("ignore", category=UserWarning, module="torch.functional")


HOUMO_TARGET = os.getenv("HOUMO_TARGET")
assert HOUMO_TARGET in ["xh2"], f"Unsupported HOUMO_TARGET: {HOUMO_TARGET}"


TOKENS_FILE_PATH = "./special_tokens.json"
SUPPORT_DEVICE_NUM = 4
mel_basis = {}
hann_window = {}

# Debug switch: dump model input and output
ENABLE_DUMP = False


def _ms(seconds: float) -> float:
    return seconds * 1000.0


def _fmt_ms(seconds: float, digits: int = 3) -> str:
    return f"{_ms(seconds):.{digits}f} ms"


def _fmt_s(seconds: float, digits: int = 3) -> str:
    return f"{seconds:.{digits}f} s"


def _fmt_toks_per_s(tokens: float, seconds: float, digits: int = 2) -> str:
    if seconds <= 0:
        return "inf tokens/s"
    return f"{(tokens / seconds):.{digits}f} tokens/s"


def _format_perf_report(perf: Dict[str, Any]) -> str:
    """Format a compact multi-line perf report."""
    lines = []
    llm = perf.get("llm", {}) or {}
    tts = perf.get("tts", {}) or {}
    llm_total_s = float(llm.get("llm_total_s", 0.0) or 0.0)
    e2e_total_s = float(perf.get("e2e_total_s", 0.0) or 0.0)

    if llm_total_s > 0:
        lines.append(f"LLM Total Cost {_fmt_ms(llm_total_s)}")

    prefill_s = float(llm.get("prefill_s", 0.0) or 0.0)
    prefill_tokens = float(llm.get("prefill_tokens", 0.0) or 0.0)
    if prefill_s > 0 and prefill_tokens > 0:
        lines.append(f"LLM Prefill Speed: {_fmt_toks_per_s(prefill_tokens, prefill_s)}")

    ttft_s = llm.get("ttft_s", None)
    if ttft_s is not None and float(ttft_s) >= 0:
        lines.append(f"TTFT (Time to First Token): {_fmt_ms(float(ttft_s))}")

    decode_s = float(llm.get("decode_s", 0.0) or 0.0)
    decode_tokens = int(llm.get("decode_tokens", 0) or 0)
    if decode_s > 0 and decode_tokens > 0:
        lines.append(
            f"TPOT (Time Per Output Token): {_fmt_toks_per_s(decode_tokens, decode_s)}"
        )

    tts_total_s = float(tts.get("tts_total_s", 0.0) or 0.0)
    if tts_total_s > 0:
        lines.append(f"TTS Total Cost: {_fmt_ms(tts_total_s)}")

    rtf = tts.get("rtf", None)
    if rtf is not None and float(rtf) > 0:
        rtf_f = float(rtf)
        lines.append(f"TTS Real-Time Factor(RTF): {rtf_f:.6f}")
        lines.append(f"TTS Generate Speed: {(1.0 / rtf_f):.2f} x real-time")

    if e2e_total_s > 0:
        lines.append(f"E2E Latency (End-to-End Latency): {_fmt_s(e2e_total_s)}")

    return "\n".join(lines)


def get_args() -> argparse.Namespace:
    """Parse commandline."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--tokenizer_dir",
        dest="tokenizer_dir",
        type=str,
        default="./Fun-CosyVoice3-0.5B-2512",
        help="Path to CosyVoice3 tokenizer directory",
    )
    parser.add_argument(
        "--ndevice",
        dest="ndevice",
        type=int,
        default=1,
        help="Number of devices to use",
    )
    parser.add_argument(
        "--campplus_path",
        dest="campplus_path",
        type=str,
        default=os.path.join("output", HOUMO_TARGET, "cosyvoice3_campplus.hmm"),
        help="Path to campplus model file",
    )
    parser.add_argument(
        "--speech_tokenizer_path",
        dest="speech_tokenizer_path",
        type=str,
        default=os.path.join("output", HOUMO_TARGET, "cosyvoice3_speech_tokenizer.hmm"),
        help="Path to speech tokenizer model file",
    )
    parser.add_argument(
        "--sos_emb_path",
        type=str,
        default=os.path.join("output", HOUMO_TARGET, "hmquant", "llm_sos_embedding.pt"),
        help="Path to SOS embedding file",
    )
    parser.add_argument(
        "--task_id_emb_path",
        type=str,
        default=os.path.join(
            "output", HOUMO_TARGET, "hmquant", "llm_task_id_embedding.pt"
        ),
        help="Path to task ID embedding file",
    )
    parser.add_argument(
        "--speech_embedding_path",
        type=str,
        default=os.path.join(
            "output", HOUMO_TARGET, "hmquant", "llm_speech_embedding.pt"
        ),
        help="Path to speech embedding file",
    )
    parser.add_argument(
        "--token_embedding_path",
        type=str,
        default=os.path.join("output", HOUMO_TARGET, "hmquant", "quant_embedding.pt"),
        help="Path to token embedding file",
    )
    parser.add_argument(
        "--prefill_model_path",
        type=str,
        default=os.path.join(
            "output", HOUMO_TARGET, "cosyvoice3_llm_qwen2_prefill.hmm"
        ),
        help="Path to LLM prefill model file",
    )
    parser.add_argument(
        "--decode_model_path",
        type=str,
        default=os.path.join("output", HOUMO_TARGET, "cosyvoice3_llm_qwen2_decode.hmm"),
        help="Path to LLM decode model file",
    )
    parser.add_argument(
        "--llm_decoder_path",
        type=str,
        default=os.path.join("output", HOUMO_TARGET, "cosyvoice3_llm_decoder.hmm"),
        help="Path to LLM decoder model file",
    )
    parser.add_argument(
        "--input_embedding",
        type=str,
        default=os.path.join(
            "output", HOUMO_TARGET, "hmquant", "flow_input_embedding.pt"
        ),
        help="Path to flow input embedding file",
    )
    parser.add_argument(
        "--spk_embed_affine_layer",
        type=str,
        default=os.path.join("output", HOUMO_TARGET, "cosyvoice3_flow_spk.hmm"),
        help="Path to speaker embedding affine layer model file",
    )
    parser.add_argument(
        "--pre_lookahead_layer",
        type=str,
        default=os.path.join("output", HOUMO_TARGET, "cosyvoice3_flow_encoder.hmm"),
        help="Path to flow pre-lookahead encoder model file",
    )
    parser.add_argument(
        "--flow_decoder_path",
        type=str,
        default=os.path.join("output", HOUMO_TARGET, "cosyvoice3_flow_decoder.hmm"),
        help="Path to flow decoder model file",
    )
    parser.add_argument(
        "--hift_part1_path",
        type=str,
        default=os.path.join("output", HOUMO_TARGET, "cosyvoice3_hift_part1.hmm"),
        help="Path to HiFT part1 model file",
    )
    parser.add_argument(
        "--hift_part2_path",
        type=str,
        default=os.path.join("output", HOUMO_TARGET, "cosyvoice3_hift_part2.hmm"),
        help="Path to HiFT part2 model file",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./results",
        help="Directory to save the generated audio files",
    )

    args = parser.parse_args()
    return args


def _dump_array2txt(data, file_path):
    """Dump a tensor-like array to a plain-text file for debugging."""
    data_arr = data.numpy().flatten()
    np.savetxt(
        file_path,
        data_arr,
        fmt="%.4f",
        delimiter="\n",
        encoding="utf-8",
    )


def load_special_tokens(file_path):
    """Load the tokenizer special-token definition from JSON."""
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Special tokens file not found at: {file_path}")

    with open(file_path, "r", encoding="utf-8") as f:
        tokens = json.load(f)
    return tokens


def load_wav(wav, target_sr, min_sr=16000):
    """Load an audio file, convert it to mono, and resample if needed."""
    speech, sample_rate = torchaudio.load(wav, backend="soundfile")
    speech = speech.mean(dim=0, keepdim=True)
    if sample_rate != target_sr:
        assert (
            sample_rate >= min_sr
        ), "wav sample rate {} must be greater than {}".format(sample_rate, target_sr)
        speech = torchaudio.transforms.Resample(
            orig_freq=sample_rate, new_freq=target_sr
        )(speech)
    return speech


def contains_chinese(text):
    """Return whether the input text contains any Chinese characters."""
    chinese_char_pattern = re.compile(r"[\u4e00-\u9fff]+")
    return bool(chinese_char_pattern.search(text))


def replace_corner_mark(text):
    """Replace superscript symbols with spoken Chinese equivalents."""
    text = text.replace("²", "平方")
    text = text.replace("³", "立方")
    return text


def remove_bracket(text):
    """Remove decorative brackets and normalize a few special separators."""
    text = text.replace("（", "").replace("）", "")
    text = text.replace("【", "").replace("】", "")
    text = text.replace("`", "").replace("`", "")
    text = text.replace("——", " ")
    return text


def replace_blank(text: str):
    """Keep spaces only when they separate adjacent ASCII segments."""
    out_str = []
    for i, c in enumerate(text):
        if c == " ":
            if (text[i + 1].isascii() and text[i + 1] != " ") and (
                text[i - 1].isascii() and text[i - 1] != " "
            ):
                out_str.append(c)
        else:
            out_str.append(c)
    return "".join(out_str)


def is_only_punctuation(text):
    """Return whether the string contains only punctuation or symbols."""
    # Regular expression: Match strings that consist only of punctuation marks or are empty.
    punctuation_pattern = r"^[\p{P}\p{S}]*$"
    return bool(regex.fullmatch(punctuation_pattern, text))


def spell_out_number(text: str, inflect_parser):
    """
    Convert Arabic numerals in text into spoken-word expressions.

    Args:
        text: Input text that may contain numeric substrings.
        inflect_parser: Converter object that provides `number_to_words`.

    Returns:
        Text with numeric substrings replaced by word forms.
    """
    new_text = []
    st = None
    for i, c in enumerate(text):
        if not c.isdigit():
            if st is not None:
                num_str = inflect_parser.number_to_words(text[st:i])
                new_text.append(num_str)
                st = None
            new_text.append(c)
        else:
            if st is None:
                st = i
    if st is not None and st < len(text):
        num_str = inflect_parser.number_to_words(text[st:])
        new_text.append(num_str)
    return "".join(new_text)


def split_paragraph(
    text: str,
    tokenize,
    lang="zh",
    token_max_n=80,
    token_min_n=60,
    merge_len=20,
    comma_split=False,
):
    """
    Split text into utterances that satisfy token-length constraints.
    split paragrah logic:
    1. per sentence max len token_max_n, min len token_min_n, merge if last sentence len less than merge_len
    2. cal sentence len according to lang
    3. split sentence according to puncatation

    Args:
        text: Input text to segment.
        tokenize: Tokenization function used for non-Chinese length checks.
        lang: Language tag used to choose punctuation and length rules.
        token_max_n: Maximum token length for one utterance.
        token_min_n: Minimum token length before starting a new utterance.
        merge_len: Threshold for merging a short tail segment.
        comma_split: Whether commas should also trigger a split.

    Returns:
        A list of normalized utterance strings.
    """

    def calc_utt_length(_text: str):
        """Compute utterance length with language-specific rules."""
        if lang == "zh":
            return len(_text)
        else:
            return len(tokenize(_text))

    def should_merge(_text: str):
        """Return whether the tail utterance is short enough to merge."""
        if lang == "zh":
            return len(_text) < merge_len
        else:
            return len(tokenize(_text)) < merge_len

    if lang == "zh":
        pounc = ["。", "？", "！", "；", "：", "、", ".", "?", "!", ";"]
    else:
        pounc = [".", "?", "!", ";", ":"]
    if comma_split:
        pounc.extend(["，", ","])

    if text[-1] not in pounc:
        if lang == "zh":
            text += "。"
        else:
            text += "."

    st = 0
    utts = []
    for i, c in enumerate(text):
        if c in pounc:
            if len(text[st:i]) > 0:
                utts.append(text[st:i] + c)
            if i + 1 < len(text) and text[i + 1] in ['"', "”"]:
                tmp = utts.pop(-1)
                utts.append(tmp + text[i + 1])
                st = i + 2
            else:
                st = i + 1

    final_utts = []
    cur_utt = ""
    for utt in utts:
        if (
            calc_utt_length(cur_utt + utt) > token_max_n
            and calc_utt_length(cur_utt) > token_min_n
        ):
            final_utts.append(cur_utt)
            cur_utt = ""
        cur_utt = cur_utt + utt
    if len(cur_utt) > 0:
        if should_merge(cur_utt) and len(final_utts) != 0:
            final_utts[-1] = final_utts[-1] + cur_utt
        else:
            final_utts.append(cur_utt)

    return final_utts


def make_pad_mask(lengths: torch.Tensor, max_len: int = 0) -> torch.Tensor:
    """Make mask tensor containing indices of padded part.

    See description of make_non_pad_mask.

    Args:
        lengths (torch.Tensor): Batch of lengths (B,).
    Returns:
        torch.Tensor: Mask tensor containing indices of padded part.

    Examples:
        >>> lengths = [5, 3, 2]
        >>> make_pad_mask(lengths)
        masks = [[0, 0, 0, 0 ,0],
                 [0, 0, 0, 1, 1],
                 [0, 0, 1, 1, 1]]
    """
    batch_size = lengths.size(0)
    max_len = max_len if max_len > 0 else lengths.max().item()
    seq_range = torch.arange(0, max_len, dtype=torch.int64, device=lengths.device)
    seq_range_expand = seq_range.unsqueeze(0).expand(batch_size, max_len)
    seq_length_expand = lengths.unsqueeze(-1)
    mask = seq_range_expand >= seq_length_expand
    return mask


def random_sampling(weighted_scores, decoded_tokens, sampling):
    """Sample one token directly from the full probability distribution."""
    top_ids = weighted_scores.softmax(dim=0).multinomial(1, replacement=True).item()
    return top_ids


def nucleus_sampling(weighted_scores, top_p=0.8, top_k=25):
    """Sample one token from the truncated top-p and top-k candidate set."""
    prob, indices = [], []
    cum_prob = 0.0
    sorted_value, sorted_idx = weighted_scores.softmax(dim=0).sort(
        descending=True, stable=True
    )
    for i in range(len(sorted_idx)):
        # sampling both top-p and numbers.
        if cum_prob < top_p and len(prob) < top_k:
            cum_prob += sorted_value[i]
            prob.append(sorted_value[i])
            indices.append(sorted_idx[i])
        else:
            break
    prob = torch.tensor(prob).to(weighted_scores)
    indices = torch.tensor(indices, dtype=torch.long).to(weighted_scores.device)
    top_ids = indices[prob.multinomial(1, replacement=True)].item()
    return top_ids


def ras_sampling(
    weighted_scores,
    decoded_tokens,
    sampling,
    top_p=0.8,
    top_k=25,
    win_size=10,
    tau_r=0.1,
):
    """Apply repetition-aware sampling on top of nucleus sampling."""
    top_ids = nucleus_sampling(weighted_scores, top_p=top_p, top_k=top_k)
    rep_num = (
        (torch.tensor(decoded_tokens[-win_size:]).to(weighted_scores.device) == top_ids)
        .sum()
        .item()
    )
    if rep_num >= win_size * tau_r:
        top_ids = random_sampling(weighted_scores, decoded_tokens, sampling)
    return top_ids


def dynamic_range_compression_torch(x, C=1, clip_val=1e-5):
    """Apply log compression to stabilize magnitude ranges."""
    return torch.log(torch.clamp(x, min=clip_val) * C)


def spectral_normalize_torch(magnitudes):
    """Normalize spectrogram magnitudes with log-domain compression."""
    output = dynamic_range_compression_torch(magnitudes)
    return output


def mel_spectrogram(
    y,
    n_fft=1920,
    num_mels=80,
    sampling_rate=24000,
    hop_size=480,
    win_size=1920,
    fmin=0,
    fmax=None,
    center=False,
):
    """Compute a log-mel spectrogram and cache reusable filter kernels."""
    if torch.min(y) < -1.0:
        logger.info("min value is ", torch.min(y))
    if torch.max(y) > 1.0:
        logger.info("max value is ", torch.max(y))

    global mel_basis, hann_window  # pylint: disable=global-statement
    if f"{str(fmax)}_{str(y.device)}" not in mel_basis:
        mel = librosa_mel_fn(
            sr=sampling_rate, n_fft=n_fft, n_mels=num_mels, fmin=fmin, fmax=fmax
        )
        mel_basis[str(fmax) + "_" + str(y.device)] = (
            torch.from_numpy(mel).float().to(y.device)
        )
        hann_window[str(y.device)] = torch.hann_window(win_size).to(y.device)

    y = torch.nn.functional.pad(
        y.unsqueeze(1),
        (int((n_fft - hop_size) / 2), int((n_fft - hop_size) / 2)),
        mode="reflect",
    )
    y = y.squeeze(1)

    spec = torch.view_as_real(
        torch.stft(
            y,
            n_fft,
            hop_length=hop_size,
            win_length=win_size,
            window=hann_window[str(y.device)],
            center=center,
            pad_mode="reflect",
            normalized=False,
            onesided=True,
            return_complex=True,
        )
    )
    spec = torch.sqrt(spec.pow(2).sum(-1) + (1e-9))
    spec = torch.matmul(mel_basis[str(fmax) + "_" + str(y.device)], spec)
    spec = spectral_normalize_torch(spec)

    return spec


def get_module_input_names(module):
    """Return all input names declared by a TCIM module."""
    ipt_names = []
    input_num = module.get_num_inputs()
    for idx in range(input_num):
        input_name = module.get_input_name(idx)
        ipt_names.append(input_name)
    return ipt_names


def get_module_output_names(module):
    """Return all output names declared by a TCIM module."""
    opt_names = []
    output_num = module.get_num_outputs()
    for idx in range(output_num):
        output_name = module.get_output_name(idx)
        opt_names.append(output_name)
    return opt_names


class CosyVoice3LM:
    def __init__(
        self,
        ndevice: int,
        prefill_model_path: str,
        decode_model_path: str,
        token_embedding_path: str,
        speech_embedding_path: str,
        llm_decoder_path: str,
        sos_emb_path: str,
        task_id_emb_path: str,
        llm_input_size: int,
        llm_output_size: int,
        speech_token_size: int,
    ):
        """Initialize the speech-token LLM and its runtime modules."""
        self.llm_input_size = llm_input_size
        self.llm_output_size = llm_output_size
        self.speech_token_size = speech_token_size

        token_embedding_state_dict = torch.load(
            token_embedding_path, map_location="cpu", weights_only=True
        )
        self.token_embedding = nn.Embedding(
            token_embedding_state_dict["weight"].shape[0],
            token_embedding_state_dict["weight"].shape[1],
        )
        self.token_embedding.load_state_dict(token_embedding_state_dict)
        self.token_embedding = self.token_embedding.to(torch.float16)

        speech_embedding_param = torch.load(speech_embedding_path, map_location="cpu")
        speech_embedding_state_dict = {"weight": speech_embedding_param}
        self.speech_embedding = nn.Embedding(6761, 896)
        self.speech_embedding.load_state_dict(speech_embedding_state_dict)
        self.speech_embedding = self.speech_embedding.to(torch.float16)

        self.ndevice = ndevice
        if self.ndevice > SUPPORT_DEVICE_NUM:
            raise ValueError(
                f"CosyVoice3 only supports {SUPPORT_DEVICE_NUM} device(s), please check your config."
            )
        # Initialize device and weight manager based on device count

        dev_manager_0 = tcim.runtime.DevManager([0], "Xh2HalBackend")
        weight_manager_0 = tcim.runtime.WeightManager(dev_manager_0)
        llm_decoder_option = tcim.runtime.Option(weight_manager_0)

        device_list = list(range(self.ndevice))
        dev_manager = tcim.runtime.DevManager(device_list, "Xh2HalBackend")
        weight_manager = tcim.runtime.WeightManager(dev_manager)

        prefill_option = tcim.runtime.Option(weight_manager)
        decode_option = tcim.runtime.Option(weight_manager)

        self.llm_decoder = tcim.runtime.load(
            llm_decoder_path, option=llm_decoder_option
        )
        self.llm_decoder_input_names = get_module_input_names(self.llm_decoder)
        self.llm_decoder_output_names = get_module_output_names(self.llm_decoder)

        self.prefill = tcim.runtime.load(prefill_model_path, option=prefill_option)
        self.prefill_input_names = get_module_input_names(self.prefill)

        dummy_tensor_names = []
        for input_name in self.prefill_input_names:
            if "model_layers" in input_name:
                dummy_tensor_names.append(input_name)
        decode_option.set_dummy_tensors(dummy_tensor_names)

        self.decode = tcim.runtime.load(decode_model_path, option=decode_option)
        self.decode_input_names = get_module_input_names(self.decode)

        # Initialize cache inputs for decode model from prefill model
        for input_name in dummy_tensor_names:
            cache = self.prefill.get_dev_input(input_name)
            self.decode.set_input(input_name, cache)

        # Get model dimension information from input metadata
        self.prefill_length = self.prefill.get_input_info(
            self.prefill_input_names[0]
        ).shape[1]
        self.embedding_len = self.prefill.get_input_info(
            self.prefill_input_names[0]
        ).shape[2]
        self.context_max_length = self.decode.get_input_info(
            self.decode_input_names[3]
        ).shape[2]
        self.batch = self.decode.get_input_info(self.decode_input_names[0]).shape[0]

        # Set initial decode current length input
        current_length_input_1 = np.array([1]).astype("int32")
        decode_current_length_name = self.decode_input_names[2]
        self.decode.set_input(decode_current_length_name, current_length_input_1)

        self.input_sequence_length = 256
        self.context_length = 0

        # Load the start-of-sequence embedding.
        self.sos_emb = torch.load(sos_emb_path, map_location="cpu")
        # Load the task-id embedding.
        self.task_id_emb = torch.load(task_id_emb_path, map_location="cpu")

        # sampling method
        self.sampling = ras_sampling
        self.pad_token_id = 151645

        self.stop_token_ids = [speech_token_size + i for i in range(200)]

    def sampling_ids(
        self,
        weighted_scores: torch.Tensor,
        decoded_tokens: List,
        sampling: int,
        ignore_eos: bool = True,
    ):
        """Sample the next token while optionally avoiding EOS for a while."""
        num_trials, max_trials = 0, 100
        speech_token_size = 6561
        while True:
            top_ids = self.sampling(weighted_scores, decoded_tokens, sampling)
            if (not ignore_eos) or (
                speech_token_size != top_ids
            ):  # (top_ids < self.speech_token_size):
                break
            num_trials += 1
            if num_trials > max_trials:
                raise RuntimeError(
                    "sampling reaches max_trials {} and still get eos when ignore_eos is True, check your input!".format(
                        max_trials
                    )
                )
        return top_ids

    @torch.inference_mode()
    def inference(
        self,
        text: torch.Tensor,
        text_len: torch.Tensor,
        prompt_text: torch.Tensor,
        prompt_text_len: torch.Tensor,
        prompt_speech_token: torch.Tensor,
        prompt_speech_token_len: torch.Tensor,
        sampling: int = 25,
        max_token_text_ratio: float = 20,
        min_token_text_ratio: float = 2,
        perf: Optional[Dict[str, Any]] = None,
    ) -> Generator[torch.Tensor, None, None]:
        """
        Generate speech tokens from text and prompt conditions.

        Args:
            text: Target text token sequence.
            text_len: Length of the target text sequence.
            prompt_text: Prompt text token sequence.
            prompt_text_len: Length of the prompt text sequence.
            prompt_speech_token: Prompt speech-token sequence.
            prompt_speech_token_len: Length of the prompt speech-token sequence.
            sampling: Sampling configuration value.
            max_token_text_ratio: Upper bound on generated speech-token length.
            min_token_text_ratio: Lower bound before EOS is allowed.
            uuid: Request identifier reserved for tracing.

        Returns:
            A generated speech-token sequence.
        """

        t_start = time.perf_counter()

        text = torch.concat([prompt_text, text], dim=1)
        text_len += prompt_text_len

        text = self.token_embedding(text)

        if prompt_speech_token_len != 0:
            prompt_speech_token_emb = self.speech_embedding(prompt_speech_token)
        else:
            prompt_speech_token_emb = torch.zeros(
                1, 0, self.llm_input_size, dtype=text.dtype
            )
        lm_input = torch.concat(
            [self.sos_emb, text, self.task_id_emb, prompt_speech_token_emb], dim=1
        )

        min_len = int((text_len - prompt_text_len) * min_token_text_ratio)
        max_len = int((text_len - prompt_text_len) * max_token_text_ratio)

        lm_input = lm_input.to(torch.float16)

        self.context_length = 0
        seq_length = lm_input.shape[1]
        out_tokens = []

        if perf is not None:
            perf["prefill_tokens"] = int(seq_length)

        # Validate input length against maximum context length
        if seq_length >= self.context_max_length:
            logger.error(
                f"Input sequence length ({seq_length}) exceeds maximum context length ({self.context_max_length}), please shorten your question!"
            )
            sys.exit(1)

        # Process prefill in chunks if input length exceeds prefill length
        input_name = self.prefill_input_names[0]
        valid_length_name = self.prefill_input_names[1]
        current_length_name = self.prefill_input_names[2]
        prefill_loop_round = math.ceil(seq_length / self.prefill_length)
        t_prefill0 = time.perf_counter()
        for round_idx in range(prefill_loop_round):
            valid_length = round_idx * self.prefill_length + self.context_length
            if round_idx == prefill_loop_round - 1:
                current_length = seq_length - round_idx * self.prefill_length
                inputs_embeds = lm_input[
                    :, round_idx * self.prefill_length : seq_length, :
                ]
            else:
                current_length = self.prefill_length
                inputs_embeds = lm_input[
                    :,
                    round_idx
                    * self.prefill_length : (round_idx + 1)
                    * self.prefill_length,
                    :,
                ]

            effective_length = inputs_embeds.shape[1]
            pad_input_ids = torch.zeros(
                (self.prefill_length - effective_length), dtype=torch.long
            )
            pad_input_ids.fill_(self.pad_token_id)
            pad_input_ids = pad_input_ids.unsqueeze(0)
            pad_embeds = self.token_embedding(pad_input_ids)
            input_data = torch.cat([inputs_embeds, pad_embeds], dim=1).detach()

            # Prepare length parameters for prefill input
            valid_length_data = np.array([valid_length]).astype("int32")
            current_length_data = np.array([current_length]).astype("int32")

            if ENABLE_DUMP:
                _dump_array2txt(input_data, f"./prefill_input_round{round_idx}.txt")

            self.prefill.set_input(input_name, input_data.numpy())
            self.prefill.set_input(valid_length_name, valid_length_data)
            self.prefill.set_input(current_length_name, current_length_data)

            self.prefill.run()
            self.prefill.sync()
        t_prefill1 = time.perf_counter()

        prefill_opt = (
            self.prefill.get_dev_output(self.prefill.get_output_name(0))
            .to_host()
            .numpy()
            .squeeze(0)
        )

        self.llm_decoder.set_input(self.llm_decoder_input_names[0], prefill_opt)
        self.llm_decoder.run()
        self.llm_decoder.sync()
        prefill_logp = (
            self.llm_decoder.get_dev_output(self.llm_decoder_output_names[0])
            .to_host()
            .numpy()
        )
        prefill_logp = torch.from_numpy(prefill_logp)

        top_ids = self.sampling_ids(
            prefill_logp.squeeze(dim=0),
            out_tokens,
            sampling,
            ignore_eos=True if 0 < min_len else False,
        )
        t_first_token = time.perf_counter()
        if top_ids in self.stop_token_ids:
            if perf is not None:
                perf["prefill_s"] = float(t_prefill1 - t_prefill0)
                perf["ttft_s"] = float(t_first_token - t_start)
                perf["decode_s"] = 0.0
                perf["decode_tokens"] = 0
                perf["llm_total_s"] = float(t_first_token - t_start)
            return out_tokens
        out_tokens.append(top_ids)
        self.context_length += seq_length

        input_name = self.decode_input_names[0]
        valid_length_name = self.decode_input_names[1]

        t_decode0 = time.perf_counter()
        decode_token_cnt = 0
        for i in range(1, max_len):
            lm_input = self.speech_embedding.weight[top_ids].reshape(1, 1, -1).detach()

            if ENABLE_DUMP:
                _dump_array2txt(lm_input, f"./ground_truth/decode_input_round{i}.txt")

            self.decode.set_input(input_name, lm_input.numpy())
            self.decode.set_input(
                valid_length_name, np.array([self.context_length]).astype("int32")
            )
            self.decode.run()
            self.decode.sync()
            decode_dev_opt = self.decode.get_dev_output(self.decode.get_output_name(0))
            llm_decoder_dev_ipt = decode_dev_opt.select_batch([0])

            if ENABLE_DUMP:
                decode_opt = decode_dev_opt.to_host().numpy().squeeze(0)
                _dump_array2txt(
                    decode_opt, f"./ground_truth/decode_output_round{i}.txt"
                )

            self.llm_decoder.set_input(
                self.llm_decoder_input_names[0], llm_decoder_dev_ipt
            )
            self.llm_decoder.run()
            self.llm_decoder.sync()
            logp = (
                self.llm_decoder.get_dev_output(self.llm_decoder_output_names[0])
                .to_host()
                .numpy()
            )

            if ENABLE_DUMP:
                _dump_array2txt(logp, f"./ground_truth/decode_logp_round{i}.txt")

            logp = torch.from_numpy(logp)
            top_ids = self.sampling_ids(
                logp.squeeze(dim=0),
                out_tokens,
                sampling,
                ignore_eos=True if i < min_len else False,
            )
            self.context_length += 1
            if top_ids in self.stop_token_ids:
                break
            out_tokens.append(top_ids)
            decode_token_cnt += 1
        t_decode1 = time.perf_counter()

        if perf is not None:
            perf["prefill_s"] = float(t_prefill1 - t_prefill0)
            perf["ttft_s"] = float(t_first_token - t_start)
            perf["decode_s"] = float(t_decode1 - t_decode0)
            perf["decode_tokens"] = int(decode_token_cnt)
            perf["llm_total_s"] = float(t_decode1 - t_start)

        return out_tokens


class CausalMaskedDiffWithDiT:
    def __init__(
        self,
        ndevice: int,
        input_embedding_path: str,
        pre_lookahead_layer_path: str,
        spk_embed_affine_layer_path: str,
        flow_decoder_path: str,
        input_size: int = 512,
        output_size: int = 80,
        output_type: str = "mel",
        vocab_size: int = 4096,
        input_frame_rate: int = 50,
        only_mask_loss: bool = True,
        token_mel_ratio: int = 2,
        pre_lookahead_len: int = 3,
    ):
        """Initialize the flow-based acoustic decoder runtime modules."""
        self.input_size = input_size
        self.output_size = output_size
        self.vocab_size = vocab_size
        self.output_type = output_type
        self.input_frame_rate = input_frame_rate

        input_embedding_state_dict = torch.load(
            input_embedding_path, map_location="cpu", weights_only=True
        )
        input_embedding = nn.Embedding(
            input_embedding_state_dict.shape[0],
            input_embedding_state_dict.shape[1],
        )
        input_embedding_state_dict_ = {"weight": input_embedding_state_dict}
        input_embedding.load_state_dict(input_embedding_state_dict_)
        self.input_embedding = input_embedding.to("cpu")

        self.ndevice = ndevice
        if self.ndevice > SUPPORT_DEVICE_NUM:
            raise ValueError(
                f"CosyVoice3 only supports {SUPPORT_DEVICE_NUM} device(s), please check your config."
            )
        # Initialize device and weight manager based on device count
        # device_list = list(range(self.ndevice))
        dev_manager = tcim.runtime.DevManager([0], "Xh2HalBackend")
        weight_manager = tcim.runtime.WeightManager(dev_manager)

        spk_embed_affine_layer_option = tcim.runtime.Option(weight_manager)
        pre_lookahead_layer_option = tcim.runtime.Option(weight_manager)
        flow_decoder_option = tcim.runtime.Option(weight_manager)

        self.spk_embed_affine_layer = tcim.runtime.load(
            spk_embed_affine_layer_path, option=spk_embed_affine_layer_option
        )
        self.spk_embed_affine_layer_input_names = get_module_input_names(
            self.spk_embed_affine_layer
        )
        self.spk_embed_affine_layer_output_names = get_module_output_names(
            self.spk_embed_affine_layer
        )

        self.pre_lookahead_layer = tcim.runtime.load(
            pre_lookahead_layer_path, option=pre_lookahead_layer_option
        )
        self.pre_lookahead_layer_input_names = get_module_input_names(
            self.pre_lookahead_layer
        )
        self.pre_lookahead_layer_output_names = get_module_output_names(
            self.pre_lookahead_layer
        )

        self.flow_decoder = tcim.runtime.load(
            flow_decoder_path, option=flow_decoder_option
        )
        self.flow_decoder_input_names = get_module_input_names(self.flow_decoder)
        self.flow_decoder_output_names = get_module_output_names(self.flow_decoder)

        self.only_mask_loss = only_mask_loss
        self.token_mel_ratio = token_mel_ratio
        self.pre_lookahead_len = pre_lookahead_len

    @torch.inference_mode()
    def inference(
        self,
        token,
        token_len,
        prompt_token,
        prompt_token_len,
        prompt_feat,
        prompt_feat_len,
        embedding,
    ):
        """
        Convert speech tokens into mel spectrogram features.

        Args:
            token: Target speech-token sequence.
            token_len: Length of the target speech-token sequence.
            prompt_token: Prompt speech-token sequence.
            prompt_token_len: Length of the prompt speech-token sequence.
            prompt_feat: Prompt acoustic feature sequence.
            prompt_feat_len: Length of the prompt acoustic feature sequence.
            embedding: Prompt speech embedding vector.

        Returns:
            A tuple of generated mel features and their valid length.
        """
        # This implementation only supports batch size 1.
        assert token.shape[0] == 1

        token, token_len = (
            torch.concat([prompt_token, token], dim=1),
            prompt_token_len + token_len,
        )
        token = self.input_embedding(token)
        token = F.pad(token, (0, 0, 0, 1024 - token.shape[1]), value=0)
        token = token.to(torch.float16).numpy()

        self.pre_lookahead_layer.set_input(
            self.pre_lookahead_layer_input_names[0], token
        )
        self.pre_lookahead_layer.run()
        self.pre_lookahead_layer.sync()
        h = (
            self.pre_lookahead_layer.get_dev_output(
                self.pre_lookahead_layer_output_names[0]
            )
            .to_host()
            .numpy()
        )
        h = torch.from_numpy(h)
        h = h.repeat_interleave(self.token_mel_ratio, dim=1)

        embedding = F.normalize(embedding, dim=1)
        self.spk_embed_affine_layer.set_input(
            self.spk_embed_affine_layer_input_names[0], embedding.numpy()
        )
        self.spk_embed_affine_layer.run()
        self.spk_embed_affine_layer.sync()
        embedding = (
            self.spk_embed_affine_layer.get_dev_output(
                self.spk_embed_affine_layer_output_names[0]
            )
            .to_host()
            .numpy()
        )
        embedding = torch.from_numpy(embedding)

        # `mel_len1` is the prompt length and `mel_len2` is the newly synthesized part.
        mel_len1, mel_len2 = prompt_feat.shape[1], token_len * 2 - prompt_feat.shape[1]
        # Build conditioning tensors for the flow decoder.
        conds = torch.zeros([1, 2048, 80], device=token.device).to(h.dtype)
        conds[:, :mel_len1] = prompt_feat
        conds = conds.transpose(1, 2)

        mask = (~make_pad_mask(torch.tensor([mel_len1 + mel_len2]), 2048)).to(h)
        mask = mask.unsqueeze(1)

        mu = h.transpose(1, 2).contiguous()
        rand_noise = torch.randn([1, 80, 50 * 300])
        x = rand_noise[:, :, : mu.size(2)].to(mu.dtype) * 1.0
        t_span = torch.linspace(0, 1, 10 + 1, dtype=mu.dtype)
        t_span = 1 - torch.cos(t_span * 0.5 * torch.pi)
        t, _, dt = t_span[0], t_span[-1], t_span[1] - t_span[0]
        t = t.unsqueeze(dim=0)
        x_in = torch.zeros([2, 80, x.size(2)], dtype=x.dtype)
        mask_in = torch.zeros([2, 1, x.size(2)], dtype=x.dtype)
        mu_in = torch.zeros([2, 80, x.size(2)], dtype=x.dtype)
        t_in = torch.zeros([2], dtype=x.dtype)
        cond_in = torch.zeros([2, 80, x.size(2)], dtype=x.dtype)
        spks_in = torch.zeros([2, 80], dtype=x.dtype)

        sol = []
        inference_cfg_rate = 0.7
        for step in range(1, len(t_span)):
            # Classifier-Free Guidance inference introduced in VoiceBox
            x_in[:] = x
            mask_in[:] = mask
            mu_in[0] = mu
            t_in[:] = t.unsqueeze(0)
            spks_in[0] = embedding
            cond_in[0] = conds

            self.flow_decoder.set_input(self.flow_decoder_input_names[0], x_in.numpy())
            self.flow_decoder.set_input(
                self.flow_decoder_input_names[1], mask_in.numpy()
            )
            self.flow_decoder.set_input(self.flow_decoder_input_names[2], mu_in.numpy())
            self.flow_decoder.set_input(self.flow_decoder_input_names[3], t_in.numpy())
            self.flow_decoder.set_input(
                self.flow_decoder_input_names[4], spks_in.numpy()
            )
            self.flow_decoder.set_input(
                self.flow_decoder_input_names[5], cond_in.numpy()
            )
            self.flow_decoder.run()
            self.flow_decoder.sync()
            dphi_dt = (
                self.flow_decoder.get_dev_output(self.flow_decoder_output_names[0])
                .to_host()
                .numpy()
            )
            dphi_dt = torch.from_numpy(dphi_dt)

            dphi_dt, cfg_dphi_dt = torch.split(dphi_dt, [x.size(0), x.size(0)], dim=0)
            dphi_dt = (
                1.0 + inference_cfg_rate
            ) * dphi_dt - inference_cfg_rate * cfg_dphi_dt
            x = x + dt * dphi_dt
            t = t + dt
            sol.append(x)
            if step < len(t_span) - 1:
                dt = t_span[step + 1] - t
        feat = sol[-1].float()
        feat = feat[:, :, mel_len1 : mel_len1 + mel_len2]

        return feat, mel_len2


class CosyVoice3Tokenizer:
    def __init__(self, token_path, skip_special_tokens=True):
        """Build the tokenizer and register project-specific special tokens."""
        # NOTE: non-chat model, all these special tokens keep randomly initialized.
        self.special_tokens = load_special_tokens(TOKENS_FILE_PATH)
        self.tokenizer = AutoTokenizer.from_pretrained(token_path)
        self.tokenizer.add_special_tokens(self.special_tokens)
        self.skip_special_tokens = skip_special_tokens

    def encode(self, text, **kwargs):
        """Encode a text string into token ids."""
        tokens = self.tokenizer([text], return_tensors="pt")
        tokens = tokens["input_ids"][0].cpu().tolist()
        return tokens

    def decode(self, tokens):
        """Decode token ids back into text."""
        tokens = torch.tensor(tokens, dtype=torch.int64)
        text = self.tokenizer.batch_decode(
            [tokens], skip_special_tokens=self.skip_special_tokens
        )[0]
        return text


class CosyVoiceFrontEnd:

    def __init__(
        self,
        ndevice: int,
        tokenizer_dir: str,
        campplus_model: str,
        speech_tokenizer_model: str,
        spk2info: str = "",
        allowed_special: str = "all",
    ):
        """Initialize text and audio frontend modules for CosyVoice3."""
        self.tokenizer = CosyVoice3Tokenizer(
            token_path=os.path.join(tokenizer_dir, "CosyVoice-BlankEN"),
            skip_special_tokens=True,
        )
        self.feat_extractor = mel_spectrogram
        self.device = "cpu"

        self.ndevice = ndevice
        if self.ndevice > SUPPORT_DEVICE_NUM:
            raise ValueError(
                f"CosyVoice3 only supports {SUPPORT_DEVICE_NUM} device(s), please check your config."
            )
        # Initialize device and weight manager based on device count
        # device_list = list(range(self.ndevice))
        dev_manager = tcim.runtime.DevManager([0], "Xh2HalBackend")
        weight_manager = tcim.runtime.WeightManager(dev_manager)

        campplus_option = tcim.runtime.Option(weight_manager)
        speech_tokenizer_option = tcim.runtime.Option(weight_manager)

        self.campplus = tcim.runtime.load(campplus_model, option=campplus_option)
        self.speech_tokenizer = tcim.runtime.load(
            speech_tokenizer_model, option=speech_tokenizer_option
        )

        self.campplus_input_names = get_module_input_names(self.campplus)
        self.campplus_output_names = get_module_output_names(self.campplus)
        self.speech_tokenizer_input_names = get_module_input_names(
            self.speech_tokenizer
        )
        self.speech_tokenizer_output_names = get_module_output_names(
            self.speech_tokenizer
        )

        if spk2info and os.path.exists(spk2info):
            logger.info(f"load spk2info from {spk2info}")
            self.spk2info = torch.load(spk2info, map_location="cpu", weights_only=True)
        else:
            self.spk2info = {}

        self.allowed_special = allowed_special
        self.inflect_parser = inflect.engine()
        try:
            from wetext import Normalizer as ZhNormalizer
            from wetext import Normalizer as EnNormalizer

            self.zh_tn_model = ZhNormalizer(remove_erhua=False)
            self.en_tn_model = EnNormalizer()
            self.text_frontend = "wetext"
        except Exception as e:
            self.text_frontend = ""
            logger.warning(f"No frontend is avaliable, err: {e}")

    def _extract_text_token(self, text):
        """Tokenize text and return both token ids and sequence length."""
        text_token = self.tokenizer.encode(text, allowed_special=self.allowed_special)
        text_token = torch.tensor([text_token], dtype=torch.int32)
        text_token_len = torch.tensor([text_token.shape[1]], dtype=torch.int32)
        return text_token, text_token_len

    def _extract_speech_token(self, prompt_wav):
        """Extract prompt speech tokens with the speech tokenizer runtime."""
        speech = load_wav(prompt_wav, 16000)
        assert (
            speech.shape[1] / 16000 <= 30
        ), "do not support extract speech token for audio longer than 30s"
        feat = whisper.log_mel_spectrogram(
            speech, n_mels=128
        )  # torch.Size([1, 128, 348]).

        feat = feat.half()
        feat_len = feat.shape[2]
        # Pad features to the static model input shape `(1, 128, 3000)`.
        padded_input = torch.zeros((1, 128, 3000), dtype=torch.float16)
        padded_input[:, :, :feat_len] = feat

        mask_shape = (1, 20, 750, 750)
        mask = torch.full(
            mask_shape, torch.finfo(torch.float16).min, dtype=torch.float16
        )
        mask[:, :, :, : feat_len // 4] = 0
        mask1 = torch.zeros((1, 750, 1280), dtype=torch.float16)
        mask1[:, 0 : feat_len // 4, :] = 1.0

        self.speech_tokenizer.set_input(
            self.speech_tokenizer_input_names[0], padded_input.numpy()
        )
        self.speech_tokenizer.set_input(
            self.speech_tokenizer_input_names[1], mask.numpy()
        )
        self.speech_tokenizer.set_input(
            self.speech_tokenizer_input_names[2], mask1.numpy()
        )
        self.speech_tokenizer.run()
        self.speech_tokenizer.sync()
        speech_token = (
            self.speech_tokenizer.get_dev_output(self.speech_tokenizer_output_names[0])
            .to_host()
            .numpy()
        )
        speech_token = torch.from_numpy(speech_token)

        speech_token = speech_token[:, : feat_len // 4]
        speech_token_len = torch.tensor([speech_token.shape[1]], dtype=torch.int32)
        return speech_token, speech_token_len

    def _extract_spk_embedding(self, prompt_wav):
        """
        Extract a speaker embedding from the prompt waveform.

        Args:
            prompt_wav: Path to the prompt waveform file.

        Returns:
            A tensor shaped like `[1, embedding_dim]`.
        """
        # Load prompt audio and resample it to 16 kHz.
        speech = load_wav(prompt_wav, 16000)
        # Extract 80-bin FBANK features for speaker embedding inference.
        feat = kaldi.fbank(speech, num_mel_bins=80, dither=0, sample_frequency=16000)
        feat = feat - feat.mean(dim=0, keepdim=True)  # torch.Size([346, 80])

        T_fixed = 1000
        T = feat.shape[0]
        if T < T_fixed:
            feat = F.pad(feat, (0, 0, 0, T_fixed - T))
        else:
            feat = feat[:T_fixed]
        feat = feat.half()

        campplus_ipt = feat.unsqueeze(0).numpy()
        self.campplus.set_input(self.campplus_input_names[0], campplus_ipt)
        self.campplus.run()
        self.campplus.sync()
        embedding = (
            self.campplus.get_dev_output(self.campplus_output_names[0])
            .to_host()
            .numpy()
        )
        embedding = torch.from_numpy(embedding)

        return embedding

    def _extract_speech_feat(self, prompt_wav):
        """Extract prompt mel features and return their valid length."""
        speech = load_wav(prompt_wav, 24000)
        speech_feat = self.feat_extractor(speech).squeeze(dim=0).transpose(0, 1)
        speech_feat = speech_feat.unsqueeze(dim=0)
        speech_feat_len = torch.tensor([speech_feat.shape[1]], dtype=torch.int32)
        return speech_feat, speech_feat_len

    def text_normalize(self, text, split=True, text_frontend=True):
        """Normalize text and optionally split it into synthesis segments."""
        if isinstance(text, Generator):
            logger.warning("get tts_text generator, will skip text_normalize!")
            return [text]
        # NOTE skip text_frontend when ssml symbol in text
        if "<|" in text and "|>" in text:
            text_frontend = False
        if text_frontend is False or text == "":
            return [text] if split is True else text
        text = text.strip()

        if contains_chinese(text):
            if self.text_frontend == "wetext":
                text = self.zh_tn_model.normalize(text)
            text = text.replace("\n", "")
            text = replace_blank(text)
            text = replace_corner_mark(text)
            text = text.replace(".", "。")
            text = text.replace(" - ", "，")
            text = remove_bracket(text)
            text = re.sub(r"[，,、]+$", "。", text)
            texts = list(
                split_paragraph(
                    text,
                    partial(
                        self.tokenizer.encode, allowed_special=self.allowed_special
                    ),
                    "zh",
                    token_max_n=80,
                    token_min_n=60,
                    merge_len=20,
                    comma_split=False,
                )
            )
        else:
            if self.text_frontend == "wetext":
                text = self.en_tn_model.normalize(text)
            text = spell_out_number(text, self.inflect_parser)
            texts = list(
                split_paragraph(
                    text,
                    partial(
                        self.tokenizer.encode, allowed_special=self.allowed_special
                    ),
                    "en",
                    token_max_n=80,
                    token_min_n=60,
                    merge_len=20,
                    comma_split=False,
                )
            )
        texts = [i for i in texts if not is_only_punctuation(i)]
        return texts if split is True else text

    def frontend_sft(self, tts_text, spk_id):
        """Build model inputs for supervised fine-tuning style synthesis."""
        tts_text_token, tts_text_token_len = self._extract_text_token(tts_text)
        embedding = self.spk2info[spk_id]["embedding"]
        model_input = {
            "text": tts_text_token,
            "text_len": tts_text_token_len,
            "llm_embedding": embedding,
            "flow_embedding": embedding,
        }
        return model_input

    def frontend_zero_shot(
        self, tts_text, prompt_text, prompt_wav, resample_rate, zero_shot_spk_id
    ):
        """
        Build frontend inputs for zero-shot voice cloning synthesis.

        Args:
            tts_text: Target text segment.
            prompt_text: Prompt text used to steer style and content.
            prompt_wav: Path to the prompt waveform.
            resample_rate: Target sample rate used by the synthesis path.
            zero_shot_spk_id: Cached speaker id, or an empty string to extract live.
        """
        tts_text_token, tts_text_token_len = self._extract_text_token(tts_text)

        if zero_shot_spk_id == "":
            prompt_text_token, prompt_text_token_len = self._extract_text_token(
                prompt_text
            )
            speech_feat, speech_feat_len = self._extract_speech_feat(prompt_wav)
            speech_token, speech_token_len = self._extract_speech_token(prompt_wav)
            if resample_rate == 24000:
                # cosyvoice2, force speech_feat % speech_token = 2
                token_len = min(int(speech_feat.shape[1] / 2), speech_token.shape[1])
                speech_feat, speech_feat_len[:] = (
                    speech_feat[:, : 2 * token_len],
                    2 * token_len,
                )
                speech_token, speech_token_len[:] = (
                    speech_token[:, :token_len],
                    token_len,
                )
            embedding = self._extract_spk_embedding(prompt_wav)
            model_input = {
                "prompt_text": prompt_text_token,
                "prompt_text_len": prompt_text_token_len,
                "llm_prompt_speech_token": speech_token,
                "llm_prompt_speech_token_len": speech_token_len,
                "flow_prompt_speech_token": speech_token,
                "flow_prompt_speech_token_len": speech_token_len,
                "prompt_speech_feat": speech_feat,
                "prompt_speech_feat_len": speech_feat_len,
                "llm_embedding": embedding,
                "flow_embedding": embedding,
            }
        else:
            logger.info(f"Reuse the features of {zero_shot_spk_id}.")
            model_input = {**self.spk2info[zero_shot_spk_id]}

        model_input["text"] = tts_text_token
        model_input["text_len"] = tts_text_token_len
        return model_input

    def frontend_cross_lingual(
        self, tts_text, prompt_wav, resample_rate, zero_shot_spk_id
    ):
        """Build frontend inputs for cross-lingual synthesis."""
        model_input = self.frontend_zero_shot(
            tts_text, "", prompt_wav, resample_rate, zero_shot_spk_id
        )
        # in cross lingual mode, we remove prompt in llm
        del model_input["prompt_text"]
        del model_input["prompt_text_len"]
        del model_input["llm_prompt_speech_token"]
        del model_input["llm_prompt_speech_token_len"]
        return model_input

    def frontend_instruct(self, tts_text, spk_id, instruct_text):
        """Build frontend inputs for instruction-guided synthesis."""
        model_input = self.frontend_sft(tts_text, spk_id)
        # in instruct mode, we remove spk_embedding in llm due to information leakage
        del model_input["llm_embedding"]
        instruct_text_token, instruct_text_token_len = self._extract_text_token(
            instruct_text
        )
        model_input["prompt_text"] = instruct_text_token
        model_input["prompt_text_len"] = instruct_text_token_len
        return model_input

    def frontend_instruct2(
        self, tts_text, instruct_text, prompt_wav, resample_rate, zero_shot_spk_id
    ):
        """Build zero-shot instruction-following frontend inputs."""
        model_input = self.frontend_zero_shot(
            tts_text, instruct_text, prompt_wav, resample_rate, zero_shot_spk_id
        )
        del model_input["llm_prompt_speech_token"]
        del model_input["llm_prompt_speech_token_len"]
        return model_input


class CosyVoice3Model:

    def __init__(
        self,
        ndevice: int,
        prefill_model_path: str,
        decode_model_path: str,
        token_embedding_path: str,
        speech_embedding_path: str,
        llm_decoder_path: str,
        sos_emb_path: str,
        task_id_emb_path: str,
        input_embedding_path: str,
        pre_lookahead_layer: str,
        spk_embed_affine_layer_path: str,
        flow_decoder_path: str,
        hift_part1_path: str,
        hift_part2_path: str,
        llm_input_size: int,
        lm_output_size: int,
        token_frame_rate: int,
        token_mel_ratio: int,
        sampling_rate: int,
    ):
        """Initialize the end-to-end acoustic and vocoder pipeline."""
        self.device = "cpu"
        self.sampling_rate = sampling_rate
        self.llm = CosyVoice3LM(
            ndevice=ndevice,
            prefill_model_path=prefill_model_path,
            decode_model_path=decode_model_path,
            token_embedding_path=token_embedding_path,
            speech_embedding_path=speech_embedding_path,
            llm_decoder_path=llm_decoder_path,
            sos_emb_path=sos_emb_path,
            task_id_emb_path=task_id_emb_path,
            llm_input_size=llm_input_size,
            llm_output_size=lm_output_size,
            speech_token_size=6561,
        )

        self.flow = CausalMaskedDiffWithDiT(
            ndevice=ndevice,
            input_embedding_path=input_embedding_path,
            pre_lookahead_layer_path=pre_lookahead_layer,
            spk_embed_affine_layer_path=spk_embed_affine_layer_path,
            flow_decoder_path=flow_decoder_path,
            input_size=80,
            output_size=80,
            output_type="mel",
            vocab_size=6561,
            input_frame_rate=token_frame_rate,
            only_mask_loss=True,
            token_mel_ratio=token_mel_ratio,
            pre_lookahead_len=3,
        )

        self.ndevice = ndevice
        if self.ndevice > SUPPORT_DEVICE_NUM:
            raise ValueError(
                f"CosyVoice3 only supports {SUPPORT_DEVICE_NUM} device(s), please check your config."
            )
        # Initialize device and weight manager based on device count
        # device_list = list(range(self.ndevice))
        dev_manager = tcim.runtime.DevManager([0], "Xh2HalBackend")
        weight_manager = tcim.runtime.WeightManager(dev_manager)

        hift_part1_option = tcim.runtime.Option(weight_manager)
        hift_part2_option = tcim.runtime.Option(weight_manager)

        self.hift_part1 = tcim.runtime.load(hift_part1_path, option=hift_part1_option)
        self.hift_part1_input_names = get_module_input_names(self.hift_part1)
        self.hift_part1_output_names = get_module_output_names(self.hift_part1)

        dummy_input_name = self.hift_part1_input_names[0]
        hift_part2_option.set_dummy_tensors([dummy_input_name])
        self.hift_part2 = tcim.runtime.load(hift_part2_path, option=hift_part2_option)
        dummy_input = self.hift_part1.get_dev_input(dummy_input_name)
        self.hift_part2.set_input(dummy_input_name, dummy_input)
        self.hift_part2_input_names = get_module_input_names(self.hift_part2)
        self.hift_part2_output_names = get_module_output_names(self.hift_part2)

        # NOTE must matching training static_chunk_size
        self.token_hop_len = 25
        # rtf and decoding related
        self.llm_context = nullcontext()
        self.lock = threading.Lock()
        # dict used to store session related variable
        self.tts_speech_token_dict = {}
        self.llm_end_dict = {}
        self.hift_cache_dict = {}
        self.silent_tokens = [1, 2, 28, 29, 55, 248, 494, 2241, 2242, 2322, 2323]
        # perf
        self._perf_by_uuid: Dict[str, Dict[str, Any]] = {}
        self.perf_history: List[Dict[str, Any]] = []

    def llm_job(self, text, prompt_text, llm_prompt_speech_token, llm_embedding, uuid):
        """
        Run the LLM stage and store generated speech tokens for a request.

        Args:
            text: Target text token sequence.
            prompt_text: Prompt text token sequence.
            llm_prompt_speech_token: Prompt speech-token sequence.
            llm_embedding: Speaker embedding used by the LLM.
            uuid: Unique request identifier for shared state.
        """
        cur_silent_token_num, max_silent_token_num = 0, 5
        llm_perf: Dict[str, Any] = {}
        token_generator = self.llm.inference(
            text=text,
            text_len=torch.tensor([text.shape[1]], dtype=torch.int32),
            prompt_text=prompt_text,
            prompt_text_len=torch.tensor([prompt_text.shape[1]], dtype=torch.int32),
            prompt_speech_token=llm_prompt_speech_token,
            prompt_speech_token_len=torch.tensor(
                [llm_prompt_speech_token.shape[1]], dtype=torch.int32
            ),
            perf=llm_perf,
        )

        for i in token_generator:
            if i in self.silent_tokens:
                cur_silent_token_num += 1
                if cur_silent_token_num > max_silent_token_num:
                    logger.debug(f"skip current silent_token {cur_silent_token_num}")
                    continue
            else:
                cur_silent_token_num = 0
            self.tts_speech_token_dict[uuid].append(i)
        self.llm_end_dict[uuid] = True
        with self.lock:
            self._perf_by_uuid[uuid] = {"llm": llm_perf}

    def token2wav(
        self,
        token,
        prompt_token,
        prompt_feat,
        embedding,
        speed=1.0,
    ):
        """
        Convert generated speech tokens into a waveform.

        Args:
            token: Generated speech-token sequence.
            prompt_token: Prompt speech-token sequence for conditioning.
            prompt_feat: Prompt acoustic features for conditioning.
            embedding: Speaker embedding used by the flow decoder.
            speed: Playback speed ratio.

        Returns:
            A waveform tensor.
        """

        tts_mel, mel_len2 = self.flow.inference(
            token=token.to(dtype=torch.int32),
            token_len=torch.tensor([token.shape[1]], dtype=torch.int32),
            prompt_token=prompt_token,
            prompt_token_len=torch.tensor([prompt_token.shape[1]], dtype=torch.int32),
            prompt_feat=prompt_feat,
            prompt_feat_len=torch.tensor([prompt_feat.shape[1]], dtype=torch.int32),
            embedding=embedding,
        )

        tts_mel = tts_mel[:, :, 0:]
        needed = 1024 - tts_mel.size(2)
        if needed > 0:
            tts_mel = F.pad(tts_mel, (0, needed), value=0)
        else:
            tts_mel = tts_mel[:, :, :1024]

        if speed != 1.0:
            # Resample the mel frames to approximate the requested speed.
            tts_mel = F.interpolate(
                tts_mel, size=int(tts_mel.shape[2] / speed), mode="linear"
            )

        speech_feat = tts_mel.to(torch.float16).numpy()
        self.hift_part1.set_input(self.hift_part1_input_names[0], speech_feat)
        self.hift_part1.run()
        self.hift_part1.sync()
        hift_part1_opt = (
            self.hift_part1.get_dev_output(self.hift_part1_output_names[0])
            .to_host()
            .numpy()
        )
        stft_ipt = torch.from_numpy(hift_part1_opt)

        stft_window = torch.from_numpy(
            get_window("hann", 16, fftbins=True).astype(np.float32)
        )
        stft_opt = torch.stft(
            stft_ipt,
            16,
            4,
            16,
            window=stft_window,
            onesided=True,
            center=False,
            return_complex=False,
        ).permute(0, 2, 1, 3)

        self.hift_part2.set_input(
            self.hift_part2_input_names[0], stft_opt.to(torch.float16).to("cpu").numpy()
        )
        # self.hift_part2.set_input(self.hift_part2_input_names[1], speech_feat)
        self.hift_part2.run()
        self.hift_part2.sync()
        hift_part2_opt = (
            self.hift_part2.get_dev_output(self.hift_part2_output_names[0])
            .to_host()
            .numpy()
        )
        tts_speech = torch.from_numpy(hift_part2_opt)
        tts_speech = tts_speech[:, : 480 * mel_len2]

        return tts_speech

    def tts(
        self,
        text=torch.zeros(1, 0, dtype=torch.int32),  # tts tokens
        flow_embedding=torch.zeros(0, 192),  # speech embedding - from prompt wav
        llm_embedding=torch.zeros(0, 192),  # speech embedding - from prompt wav
        prompt_text=torch.zeros(1, 0, dtype=torch.int32),  # prompt text tokens
        llm_prompt_speech_token=torch.zeros(
            1, 0, dtype=torch.int32
        ),  # prompt wav tokens
        flow_prompt_speech_token=torch.zeros(
            1, 0, dtype=torch.int32
        ),  # prompt wav tokens
        prompt_speech_feat=torch.zeros(1, 0, 80),  # speech feature - from prompt wav
        source_speech_token=torch.zeros(
            1, 0, dtype=torch.int32
        ),  # speech feature length - from prompt wav
        speed=1.0,
        **kwargs,
    ):
        """Run the full text-to-speech pipeline for one request."""
        e2e_t0 = time.perf_counter()
        # this_uuid is used to track variables related to this inference thread
        this_uuid = str(uuid.uuid1())
        with self.lock:
            self.tts_speech_token_dict[this_uuid], self.llm_end_dict[this_uuid] = (
                [],
                False,
            )
            self.hift_cache_dict[this_uuid] = None
            self._perf_by_uuid[this_uuid] = {}

        if source_speech_token.shape[1] == 0:
            llm_t0 = time.perf_counter()
            p = threading.Thread(
                target=self.llm_job,
                args=(
                    text,
                    prompt_text,
                    llm_prompt_speech_token,
                    llm_embedding,
                    this_uuid,
                ),
            )
        else:
            assert False, "source_speech_token.shape[1] != 0"
        p.start()

        # deal with all tokens
        p.join()
        llm_t1 = time.perf_counter()
        this_tts_speech_token = torch.tensor(
            self.tts_speech_token_dict[this_uuid]
        ).unsqueeze(dim=0)
        tts_t0 = time.perf_counter()
        this_tts_speech = self.token2wav(
            token=this_tts_speech_token,
            prompt_token=flow_prompt_speech_token,
            prompt_feat=prompt_speech_feat,
            embedding=flow_embedding,
            speed=speed,
        )
        tts_t1 = time.perf_counter()
        result_speech = this_tts_speech.to(torch.float32).cpu()
        tts_wall_s = float(time.perf_counter() - e2e_t0)

        with self.lock:
            perf = self._perf_by_uuid.get(this_uuid, {}) or {}
            llm_perf = perf.get("llm", {}) or {}
            # wall time for llm thread (includes prefill+decode+host overhead)
            llm_perf["llm_stage_wall_s"] = float(llm_t1 - llm_t0)
            perf["llm"] = llm_perf

            tts_compute_s = float(tts_t1 - tts_t0)
            audio_s = (
                float(this_tts_speech.shape[1]) / float(self.sampling_rate)
                if this_tts_speech.ndim == 2
                else 0.0
            )
            perf["tts"] = {
                "tts_total_s": tts_compute_s,
                "tts_wall_s": tts_wall_s,
                "audio_s": audio_s,
                "rtf": (tts_wall_s / audio_s) if audio_s > 0 else None,
            }
            perf["e2e_total_s"] = tts_wall_s
            self.perf_history.append(perf)

            logger.info(f"tts, uuid: {this_uuid}, release resources.")
            self.tts_speech_token_dict.pop(this_uuid)
            self.llm_end_dict.pop(this_uuid)
            self.hift_cache_dict.pop(this_uuid)
            self._perf_by_uuid.pop(this_uuid, None)

        return result_speech

    def flush_perf_history(self) -> List[Dict[str, Any]]:
        """Return and clear accumulated perf stats."""
        with self.lock:
            out = list(self.perf_history)
            self.perf_history.clear()
            return out


class CosyVoice3:
    def __init__(
        self,
        args,
        llm_input_size=896,
        llm_output_size=896,
        sample_rate=24000,
        token_frame_rate=25,
        token_mel_ratio=2,
        allowed_special="all",
    ):
        """Construct the public CosyVoice3 inference wrapper."""
        # load args
        ndevice = args.ndevice
        tokenizer_dir = args.tokenizer_dir
        output_dir = args.output_dir
        prefill_model_path = args.prefill_model_path
        decode_model_path = args.decode_model_path
        token_embedding_path = args.token_embedding_path
        speech_embedding_path = args.speech_embedding_path
        llm_decoder_path = args.llm_decoder_path
        sos_emb_path = args.sos_emb_path
        task_id_emb_path = args.task_id_emb_path
        campplus_model = args.campplus_path
        speech_tokenizer_model = args.speech_tokenizer_path
        input_embedding_path = args.input_embedding
        pre_lookahead_layer = args.pre_lookahead_layer
        spk_embed_affine_layer_path = args.spk_embed_affine_layer
        flow_decoder_path = args.flow_decoder_path
        hift_part1_path = args.hift_part1_path
        hift_part2_path = args.hift_part2_path

        os.makedirs(output_dir, exist_ok=True)

        self.output_dir = output_dir
        self.sample_rate = sample_rate
        self._perf_reports: List[Dict[str, Any]] = []

        self.frontend = CosyVoiceFrontEnd(
            ndevice=ndevice,
            tokenizer_dir=tokenizer_dir,
            campplus_model=campplus_model,
            speech_tokenizer_model=speech_tokenizer_model,
            allowed_special=allowed_special,
        )

        self.model = CosyVoice3Model(
            ndevice=ndevice,
            prefill_model_path=prefill_model_path,
            decode_model_path=decode_model_path,
            token_embedding_path=token_embedding_path,
            speech_embedding_path=speech_embedding_path,
            llm_decoder_path=llm_decoder_path,
            sos_emb_path=sos_emb_path,
            task_id_emb_path=task_id_emb_path,
            input_embedding_path=input_embedding_path,
            pre_lookahead_layer=pre_lookahead_layer,
            spk_embed_affine_layer_path=spk_embed_affine_layer_path,
            flow_decoder_path=flow_decoder_path,
            hift_part1_path=hift_part1_path,
            hift_part2_path=hift_part2_path,
            llm_input_size=llm_input_size,
            lm_output_size=llm_output_size,
            token_frame_rate=token_frame_rate,
            token_mel_ratio=token_mel_ratio,
            sampling_rate=sample_rate,
        )

    def collect_perf(self):
        """Collect perf stats from model into wrapper buffer."""
        self._perf_reports.extend(self.model.flush_perf_history())

    def print_perf_summary(self):
        """Print all collected perf reports (after all inference is done)."""
        if not self._perf_reports:
            logger.info("No perf reports collected.")
            return
        logger.info(f"Collected perf reports: {len(self._perf_reports)}")
        for idx, perf in enumerate(self._perf_reports):
            report = _format_perf_report(perf)
            if report:
                logger.info(f"\n[Perf #{idx}]\n{report}")

    def add_zero_shot_spk(self, prompt_text, prompt_wav, zero_shot_spk_id):
        """Cache prompt-derived speaker features under a reusable speaker id."""
        assert zero_shot_spk_id != "", "do not use empty zero_shot_spk_id"
        model_input = self.frontend.frontend_zero_shot(
            "", prompt_text, prompt_wav, self.sample_rate, ""
        )
        del model_input["text"]
        del model_input["text_len"]
        self.frontend.spk2info[zero_shot_spk_id] = model_input
        return True

    def save_spkinfo(self):
        """Persist cached speaker information to the output directory."""
        torch.save(self.frontend.spk2info, "{}/spk2info.pt".format(self.output_dir))

    def inference_zero_shot(
        self,
        tts_text,
        prompt_text,
        prompt_wav,
        zero_shot_spk_id="",
        speed=1.0,
        text_frontend=True,
    ):
        """Yield synthesized audio chunks for zero-shot voice cloning."""
        logger.info(
            f"Start inference zero shot, zero_shot_spk_id: {zero_shot_spk_id}, speed: {speed}, text_frontend: {text_frontend}"
        )
        if (
            self.__class__.__name__ == "CosyVoice3"
            and "<|endofprompt|>" not in prompt_text + tts_text
            and not zero_shot_spk_id
        ):
            logger.warning(
                "<|endofprompt|> not found in CosyVoice3 inference, check your input text"
            )
        prompt_text = self.frontend.text_normalize(
            prompt_text, split=False, text_frontend=text_frontend
        )
        for i in tqdm(
            self.frontend.text_normalize(
                tts_text, split=True, text_frontend=text_frontend
            )
        ):
            if (not isinstance(i, Generator)) and len(i) < 0.5 * len(prompt_text):
                logger.warning(
                    "synthesis text {} too short than prompt text {}, this may lead to bad performance".format(
                        i, prompt_text
                    )
                )
            model_input = self.frontend.frontend_zero_shot(
                i, prompt_text, prompt_wav, self.sample_rate, zero_shot_spk_id
            )
            logger.info("synthesis text {}".format(i))
            tts_speech = self.model.tts(**model_input, speed=speed)
            logger.success("generate tts speech successfully.")
            # Collect perf stats (no printing here; printing happens after all inference).
            self.collect_perf()
            yield tts_speech

    def inference_cross_lingual(
        self,
        tts_text,
        prompt_wav,
        zero_shot_spk_id="",
        speed=1.0,
        text_frontend=True,
    ):
        """Yield synthesized audio chunks for cross-lingual generation."""
        for i in tqdm(
            self.frontend.text_normalize(
                tts_text, split=True, text_frontend=text_frontend
            )
        ):
            model_input = self.frontend.frontend_cross_lingual(
                i, prompt_wav, self.sample_rate, zero_shot_spk_id
            )
            logger.info("synthesis text {}".format(i))
            tts_speech = self.model.tts(**model_input, speed=speed)
            logger.success("Generate tts speech successfully.")
            self.collect_perf()
            yield tts_speech

    def inference_instruct2(
        self,
        tts_text,
        instruct_text,
        prompt_wav,
        zero_shot_spk_id="",
        speed=1.0,
        text_frontend=True,
    ):
        """Yield synthesized audio chunks for instruction-following generation."""
        for i in tqdm(
            self.frontend.text_normalize(
                tts_text, split=True, text_frontend=text_frontend
            )
        ):
            model_input = self.frontend.frontend_instruct2(
                i, instruct_text, prompt_wav, self.sample_rate, zero_shot_spk_id
            )
            logger.info("synthesis text {}".format(i))
            tts_speech = self.model.tts(**model_input, speed=speed)
            logger.success("generate tts speech successfully.")
            self.collect_perf()
            yield tts_speech


if __name__ == "__main__":
    args = get_args()

    # find prompt wav in current directory
    prompt_wav_matches = glob.glob(os.path.join(".", "zero_shot_prompt.wav"))
    if not prompt_wav_matches:
        raise FileNotFoundError("zero_shot_prompt.wav not found in current directory")
    prompt_wav = prompt_wav_matches[0]

    cosyvoice = CosyVoice3(args)

    for i, this_tts_speech in enumerate(
        cosyvoice.inference_zero_shot(
            "下面为您朗诵一段绕口令，希望您[j][ǐ]予好评，朗诵开始: 八百标兵奔北坡，北坡炮兵并排跑，炮兵怕把标兵碰，标兵怕碰炮兵炮。",
            "You are a helpful assistant.<|endofprompt|>希望你以后能够做的比我还好呦。",
            prompt_wav,
        )
    ):
        torchaudio.save(
            f"{cosyvoice.output_dir}/cosyvoice3_zero_shot_{i}.wav",
            this_tts_speech,
            cosyvoice.sample_rate,
        )

    # fine grained control
    for i, this_tts_speech in enumerate(
        cosyvoice.inference_cross_lingual(
            "You are a helpful assistant.<|endofprompt|>[breath]因为他们那一辈人[breath]在乡里面住的要习惯一点，[breath]邻居都很活络，[breath]嗯，都很熟悉。[breath]",
            prompt_wav,
        )
    ):
        torchaudio.save(
            f"{cosyvoice.output_dir}/cosyvoice3_fine_grained_control_{i}.wav",
            this_tts_speech,
            cosyvoice.sample_rate,
        )

    # convert to cantonese
    for i, this_tts_speech in enumerate(
        cosyvoice.inference_instruct2(
            "好少咯，一般系放嗰啲国庆啊，中秋嗰啲可能会咯。",
            "You are a helpful assistant. 请用广东话表达。<|endofprompt|>",
            prompt_wav,
        )
    ):
        torchaudio.save(
            f"{cosyvoice.output_dir}/cosyvoice3_instruct_cantonese_{i}.wav",
            this_tts_speech,
            cosyvoice.sample_rate,
        )

    # Print perf reports once, after all inference is finished.
    cosyvoice.print_perf_summary()
