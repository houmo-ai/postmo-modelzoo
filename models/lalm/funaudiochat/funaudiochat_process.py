# Copyright (c) 2026 HOUMO AI
#
# File: funaudiochat_process.py
# Description:
#   CPU preprocessing and postprocessing for Fun-Audio-Chat.
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

"""CPU-side preprocessing, feature preparation, and postprocessing helpers."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import librosa
import numpy as np
import torch
import torch.nn.functional as functional
import torchaudio
import torchaudio.compliance.kaldi as kaldi
import yaml
from scipy.signal import get_window
from transformers import AutoTokenizer, Qwen2Tokenizer, WhisperFeatureExtractor
from transformers.feature_extraction_utils import BatchFeature

from houmo_engine import ModelProcess
from houmo_engine.core.types import StageInputs, StageOutputs
from houmo_engine.perf import PerfTracker

from funaudiochat_types import (
    FunAudioChatRequest,
    FunAudioChatState,
    LanguagePrefill,
    PreparedAudioRequest,
    VadPreparedRequest,
)

DEFAULT_STATIC_AUDIO_SAMPLES = 126799
DEFAULT_SYSTEM_PROMPT = "You are asked to generate text tokens."
SPOKEN_PROMPT = (
    "You are asked to generate both text and speech tokens at the same time. "
    "你的名字是小云。你是一位来自杭州的温柔友善的女孩，声音甜美，举止亲切。"
    "你的回复语气自然友好，力求沟通简洁明了。你的回复简短，通常只有一到三句话，"
    "避免使用正式的称谓和重复的短语。你能用恰当的声音回复，遵循用户的指示，"
    "并能共情他们的情绪。你能用恰当的方言回复，会说四川话和粤语。"
)
MASK_VALUE = np.finfo(np.float16).min
GROUP_SIZE = 5
AUDIO_BOS_ID = 6561
AUDIO_EOS_ID = 6562
TEXT_AUDIO_BOS_ID = 151670
TEXT_AUDIO_EOS_ID = 151671
SAMPLE_RATE = 24000
SAMPLES_PER_TOKEN = 960
TOKEN_MEL_RATIO = 2
FLOW_STEPS = 10
FLOW_CFG_RATE = 0.7


def require_file(path: str | Path, label: str) -> Path:
    """Resolve a required file path and fail with a descriptive error if absent."""
    value = Path(path).expanduser().resolve()
    if not value.is_file():
        raise FileNotFoundError(f"missing {label}: {value}")
    return value


def load_embedding(path: Path, dtype=torch.float16) -> np.ndarray:
    """Load an embedding checkpoint and return it as a contiguous NumPy array."""
    value = torch.load(require_file(path, "embedding"), map_location="cpu", weights_only=False)
    if isinstance(value, dict):
        value = value["weight"]
    elif hasattr(value, "weight"):
        value = value.weight
    return np.ascontiguousarray(value.detach().to(dtype).cpu().numpy())


def language_attention_mask(valid_key_length: int, query_length: int, key_length: int) -> np.ndarray:
    """Build the padded attention mask used by the language model stages."""
    mask = np.zeros((query_length, key_length), dtype=np.float16)
    if valid_key_length < key_length:
        mask[:, valid_key_length:] = MASK_VALUE
        if query_length == key_length:
            mask[valid_key_length:, :] = MASK_VALUE
    return mask[None, None, :, :]


def crq_attention_mask(valid_key_length: int, query_length: int, key_length: int) -> np.ndarray:
    """Build the two-dimensional attention mask required by the CRQ decoder."""
    return language_attention_mask(valid_key_length, query_length, key_length)[0, 0]


class FunAudioChatProcessor:
    """Convert audio requests into model inputs and generated tokens into results."""

    def __init__(self, model_dir):
        self.model_dir = str(model_dir)
        self.feature_extractor = WhisperFeatureExtractor.from_pretrained(self.model_dir)
        self.speech_tokenizer = AutoTokenizer.from_pretrained(self.model_dir, subfolder="speech_tokenizer")
        self.tokenizer = Qwen2Tokenizer.from_pretrained(self.model_dir)
        self.audio_token = self.tokenizer.audio_token
        self.audio_bos_token = self.tokenizer.audio_bos_token
        self.audio_eos_token = self.tokenizer.audio_eos_token
        self.audio_pad_token = self.tokenizer.audio_pad_token
        self.audio_token_id = self.tokenizer.convert_tokens_to_ids(self.audio_token)
        self.audio_group_size = int(self.speech_tokenizer.init_kwargs.get("audio_group_size", 5))
        self.audio_sampling_rate = int(self.feature_extractor.sampling_rate or 16000)

    def apply_chat_template(self, *args, **kwargs):
        return self.tokenizer.apply_chat_template(*args, **kwargs)

    def _prepare_waveforms(self, audio: list[np.ndarray]) -> list[np.ndarray]:
        waveforms = []
        for value in audio:
            if isinstance(value, (str, bytes)):
                value, _ = librosa.load(value, sr=self.audio_sampling_rate, mono=True)
            value = np.asarray(value, dtype=np.float32)
            if value.ndim != 1:
                raise ValueError(f"audio must be mono 1-D data, got shape {value.shape}")
            waveforms.append(value)
        return waveforms

    def _expand_audio_tokens(self, texts: list[str], speech_lengths: list[int]) -> list[str]:
        expanded = []
        for sample in texts:
            replacements = []
            while self.audio_token in sample:
                speech_length = int(speech_lengths.pop(0))
                token_count = (speech_length + self.audio_group_size - 1) // self.audio_group_size
                start = sample.find(self.audio_token)
                end = start + len(self.audio_token)
                has_bos = sample[start - len(self.audio_bos_token) : start] == self.audio_bos_token
                has_eos = sample[end : end + len(self.audio_eos_token)] == self.audio_eos_token
                replacement = self.audio_token * token_count
                if not has_bos and not has_eos:
                    replacement = self.audio_bos_token + replacement + self.audio_eos_token
                replacements.append(replacement)
                sample = sample.replace(self.audio_token, f"<|funaudio_placeholder_{len(replacements)}|>", 1)
            for index, replacement in enumerate(replacements, start=1):
                sample = sample.replace(f"<|funaudio_placeholder_{index}|>", replacement, 1)
            expanded.append(sample)
        return expanded

    def __call__(self, text: str, audio: list[np.ndarray], **kwargs: Any) -> BatchFeature:
        if not isinstance(audio, list) or not audio:
            raise ValueError("audio must be a non-empty list of waveforms")
        texts = [text] if isinstance(text, str) else text
        waveforms = self._prepare_waveforms(audio)
        if sum(sample.count(self.audio_token) for sample in texts) != len(waveforms):
            raise ValueError("the number of audio placeholders must match audio inputs")

        speech = [self.audio_pad_token * int(len(waveform) / self.audio_sampling_rate * 25) for waveform in waveforms]
        speech_inputs = self.speech_tokenizer(
            speech,
            padding=True,
            pad_to_multiple_of=self.audio_group_size,
            return_attention_mask=True,
            return_token_type_ids=False,
            return_tensors="pt",
        )
        speech_lengths = speech_inputs["attention_mask"].sum(-1).tolist()
        expanded = self._expand_audio_tokens(texts, speech_lengths)
        features = self.feature_extractor(
            waveforms,
            sampling_rate=self.audio_sampling_rate,
            padding="max_length",
            return_attention_mask=True,
            return_tensors="pt",
        )
        text_inputs = self.tokenizer(expanded, return_tensors="pt", padding=True, add_special_tokens=False)
        return BatchFeature(
            data={
                **text_inputs,
                "speech_ids": speech_inputs["input_ids"],
                "speech_attention_mask": speech_inputs["attention_mask"],
                "input_features": features["input_features"],
                "feature_attention_mask": features["attention_mask"],
                "feature_exist_mask": torch.ones(len(waveforms), dtype=torch.bool),
            },
            tensor_type="pt",
        )


class FunAudioChatProcess(ModelProcess):
    """Prepare stage-specific inputs for the Fun-Audio-Chat model graphs."""

    def __init__(
        self,
        tokenizer_dir,
        embedding_path,
        audio_embedding_path,
        pre_matching_path,
        flow_input_embedding_path,
        speaker_info_path,
        config_path,
        cmvn_path,
        *,
        prefill_length: int,
        hidden_size: int,
        audio_encoder_shapes: dict[str, tuple[int, ...]],
        flow_token_capacity: int,
        flow_mel_capacity: int,
        hift_mel_capacity: int,
        load_s2s: bool,
        load_vad: bool,
        static_audio_samples: int,
        perf: PerfTracker,
    ):
        self.perf = perf
        self.processor = FunAudioChatProcessor(tokenizer_dir)
        self.tokenizer = self.processor.tokenizer
        self.text_embedding = load_embedding(embedding_path)
        self.audio_embedding = load_embedding(audio_embedding_path)
        self.pre_matching_weight = self.pre_matching_bias = None
        self.flow_input_embedding = None
        if load_s2s:
            pre_matching = torch.load(require_file(pre_matching_path, "audio decoder pre-matching weights"), map_location="cpu", weights_only=True)
            self.pre_matching_weight = pre_matching["weight"].to("cpu")
            self.pre_matching_bias = pre_matching["bias"].to("cpu")
            self.flow_input_embedding = torch.from_numpy(load_embedding(flow_input_embedding_path, torch.float32))
        self.prefill_length = prefill_length
        self.hidden_size = hidden_size
        self.audio_encoder_shapes = audio_encoder_shapes
        self.static_audio_samples = static_audio_samples
        self.audio_token_id = int(self.processor.audio_token_id)
        self.max_chunk_tokens = min(
            flow_token_capacity,
            flow_mel_capacity // TOKEN_MEL_RATIO,
            hift_mel_capacity // TOKEN_MEL_RATIO,
        )
        self.flow_token_capacity = flow_token_capacity
        self.flow_mel_capacity = flow_mel_capacity
        self.hift_mel_capacity = hift_mel_capacity
        self.speaker_embedding = None
        if load_s2s:
            profile = torch.load(require_file(speaker_info_path, "default speaker profile"), map_location="cpu", weights_only=False)
            if "中文女" not in profile or "embedding" not in profile["中文女"]:
                raise KeyError(f"missing speaker embedding '中文女' in {speaker_info_path}")
            self.speaker_embedding = profile["中文女"]["embedding"].detach().to(torch.float16)
            if self.speaker_embedding.shape != (1, 192):
                raise ValueError(f"default speaker embedding must be (1, 192), got {self.speaker_embedding.shape}")
        generation_config = json.loads(require_file(Path(tokenizer_dir) / "generation_config.json", "generation config").read_text(encoding="utf-8"))
        eos = generation_config.get("eos_token_id", self.tokenizer.eos_token_id)
        self.stop_token_ids = set(eos if isinstance(eos, list) else [eos])
        self.vad_config = None
        self.frontend_config = None
        self.vad_model_config = None
        self.vad_sample_rate = 0
        self.vad_feature_dim = 0
        self.cmvn = None
        if load_vad:
            self.vad_config = yaml.safe_load(require_file(config_path, "VAD config").read_text(encoding="utf-8"))
            self.frontend_config = self.vad_config["frontend_conf"]
            self.vad_model_config = self.vad_config["model_conf"]
            self.vad_encoder_config = self.vad_config["encoder_conf"]
            self.vad_sample_rate = int(self.frontend_config["fs"])
            self.vad_feature_dim = int(self.vad_encoder_config["input_dim"])
            self.cmvn = self._load_cmvn(require_file(cmvn_path, "VAD CMVN"))

    def load_audio(self, audio, *, reject_long: bool = False) -> np.ndarray:
        with self.perf.scope("lalm.audio.preprocess"):
            if isinstance(audio, np.ndarray):
                waveform = np.asarray(audio, dtype=np.float32).reshape(-1)
            else:
                waveform, _ = librosa.load(require_file(audio, "input audio"), sr=16000, mono=True)
                waveform = np.asarray(waveform, dtype=np.float32)
            if waveform.size == 0:
                raise ValueError("input waveform is empty")
            if waveform.size > self.static_audio_samples:
                if reject_long:
                    raise ValueError("input waveform exceeds static audio capacity")
                waveform = waveform[: self.static_audio_samples]
            return waveform

    def preprocess(self, request: FunAudioChatRequest) -> PreparedAudioRequest:
        waveform = self.load_audio(request.audio, reject_long=request.stage != "s2t")
        audio_template = "<|audio_bos|><|AUDIO|><|audio_eos|>"
        conversation = [
            {"role": "system", "content": request.system_prompt},
            {"role": "user", "content": audio_template},
        ]
        with self.perf.scope("lalm.processor"):
            text = self.processor.apply_chat_template(conversation, add_generation_prompt=True, tokenize=False)
            inputs = self.processor(text=text, audio=[waveform], return_tensors="pt", return_token_type_ids=False)
        return PreparedAudioRequest(waveform, inputs, waveform.size / 16000)

    def prepare_audio_encode(self, request: PreparedAudioRequest) -> StageInputs:
        inputs = request.processor_inputs
        speech_shape = self.audio_encoder_shapes["speech_ids"]
        speech_maxlen = speech_shape[1]
        speech_raw = inputs["speech_ids"].to(torch.int32).cpu().numpy()
        if speech_raw.shape[1] > speech_maxlen:
            raise ValueError(f"processed speech length {speech_raw.shape[1]} exceeds static audio encoder length {speech_maxlen}")
        speech_ids = np.full(speech_shape, 6563, dtype=np.int32)
        speech_ids[:, : speech_raw.shape[1]] = speech_raw
        audio_inputs_embeds = self.audio_embedding[speech_ids.astype(np.int64)]
        feature_mask = inputs["feature_attention_mask"].bool()
        feature_length = int(feature_mask.sum().item())
        flat = inputs["input_features"].permute(0, 2, 1)[feature_mask].permute(1, 0).to(torch.float16).cpu().numpy()
        padded_shape = self.audio_encoder_shapes["padded_input_features"]
        num_chunks, feature_size, max_chunk_len = padded_shape
        if flat.shape[0] != feature_size:
            raise ValueError(f"feature size {flat.shape[0]} does not match HMM {feature_size}")
        required = (feature_length + max_chunk_len - 1) // max_chunk_len
        if required > num_chunks:
            raise ValueError(f"audio produces {feature_length} Mel frames and {required} chunks, but HMM supports {num_chunks}")
        lengths = []
        remaining = feature_length
        for _ in range(num_chunks):
            length = min(remaining, max_chunk_len)
            lengths.append(length)
            remaining -= length
        padded = np.zeros(padded_shape, dtype=np.float16)
        chunk_mask = np.zeros(self.audio_encoder_shapes["chunk_padded_mask"], dtype=np.float16)
        cursor = 0
        for index, length in enumerate(lengths):
            if length:
                padded[index, :, :length] = flat[:, cursor : cursor + length]
                chunk_mask[index, :, :length] = 1
                cursor += length
        cnn_lengths = [(length - 1) // 2 + 1 if length else 0 for length in lengths]
        cnn_mask = np.zeros(self.audio_encoder_shapes["aftercnn_valid_mask"], dtype=np.float16)
        for index, length in enumerate(cnn_lengths):
            cnn_mask[index, :length, 0] = 1
        audio_mask = np.full(self.audio_encoder_shapes["audio_attention_mask"], MASK_VALUE, dtype=np.float16)
        max_cnn = cnn_mask.shape[1]
        for index, length in enumerate(cnn_lengths):
            start = index * max_cnn
            audio_mask[:, :, start : start + length, start : start + length] = 0
        continuous = np.zeros(self.audio_encoder_shapes["continuous_audio_valid_mask"], dtype=np.float16)
        fixed_pool = max_cnn if max_cnn < 2 else max_cnn // 2
        offset = 0
        for length in (value if value < 2 else value // 2 for value in cnn_lengths):
            continuous[:, offset : offset + length, :] = 1
            offset += fixed_pool
        return StageInputs(
            tensors=(speech_ids, audio_inputs_embeds, padded, chunk_mask, cnn_mask, audio_mask, continuous),
            metadata={"feature_length": feature_length},
        )

    def prepare_language_prefill(self, request: PreparedAudioRequest, outputs: StageOutputs) -> LanguagePrefill:
        inputs = request.processor_inputs
        ids = inputs["input_ids"].cpu().numpy().astype(np.int64, copy=False)
        valid_ids = ids[inputs["attention_mask"].cpu().numpy().astype(bool)].reshape(1, -1)
        if valid_ids.shape[1] > self.prefill_length:
            raise ValueError(f"prompt length {valid_ids.shape[1]} exceeds HMM prefill length {self.prefill_length}")
        original = self.text_embedding[valid_ids]
        embeds = original.copy()
        audio_features = outputs.tensors[0]
        positions = np.nonzero(valid_ids[0] == self.audio_token_id)[0]
        if positions.size > audio_features.shape[1]:
            raise ValueError("prompt audio placeholders exceed encoder feature slots")
        embeds[:, positions, :] = audio_features[:, : positions.size, :].astype(np.float16, copy=False)
        length = int(valid_ids.shape[1])
        padded = np.zeros((1, self.prefill_length, self.hidden_size), dtype=np.float16)
        padded[:, :length, :] = embeds
        return LanguagePrefill(padded, language_attention_mask(length, self.prefill_length, self.prefill_length), length, original)

    def language_prefill_inputs(self, request: LanguagePrefill) -> StageInputs:
        return StageInputs(
            tensors=(request.embeds, request.attention_mask, np.array([0], np.int32), np.array([request.prompt_length], np.int32)),
            metadata={"prompt_length": request.prompt_length},
        )

    def language_decode_inputs(self, token: int, context_length: int, audio_features=None) -> StageInputs:
        embedding = self.text_embedding[token : token + 1].reshape(1, 1, self.hidden_size)
        llm_input = embedding if audio_features is None else (embedding + audio_features) / 2
        return StageInputs(
            tensors=(llm_input, language_attention_mask(context_length + 1, 1, self.prefill_length), np.array([context_length], np.int32), np.array([1], np.int32)),
            metadata={"text_embedding": embedding},
        )

    def audio_tower_inputs(self, speech_group: list[int]) -> StageInputs:
        return StageInputs(tensors=(np.asarray(speech_group, dtype=np.int32).reshape(1, GROUP_SIZE),))

    def crq_prefill_inputs(self, expanded: np.ndarray, previous_audio_embedding: np.ndarray, context_length: int) -> tuple[StageInputs, int]:
        valid_length = expanded.shape[1] - (GROUP_SIZE - 1)
        if valid_length <= 0 or valid_length > context_length:
            raise ValueError(f"invalid CRQ prefill length {valid_length}")
        values = expanded + previous_audio_embedding.reshape(1, 1, -1)
        padded = np.zeros((1, context_length, self.hidden_size), dtype=np.float16)
        padded[:, :valid_length, :] = values[:, :valid_length, :]
        return StageInputs(tensors=(padded, np.array([0], np.int32), np.array([valid_length], np.int32), crq_attention_mask(valid_length, context_length, context_length))), valid_length

    def crq_decode_inputs(self, hidden: np.ndarray, previous_audio_embedding: np.ndarray, past_length: int, context_length: int) -> StageInputs:
        return StageInputs(tensors=(hidden + previous_audio_embedding.reshape(1, 1, -1), np.array([past_length], np.int32), np.array([1], np.int32), crq_attention_mask(past_length + 1, 1, context_length)))

    def pre_matching(self, values) -> np.ndarray:
        if self.pre_matching_weight is None:
            raise RuntimeError("S2S pre-matching weights are not loaded")
        with self.perf.scope("lalm.pre_matching"):
            tensor = torch.from_numpy(np.ascontiguousarray(values, dtype=np.float16))
            with torch.no_grad():
                projected = functional.linear(tensor, self.pre_matching_weight, self.pre_matching_bias)
            return projected.cpu().numpy().astype(np.float16, copy=False)

    def postprocess(self, state: FunAudioChatState, *, final: bool = False) -> str:
        if not final:
            return ""
        with self.perf.scope("lalm.text.postprocess"):
            text = self.tokenizer.decode(state.generated_ids, skip_special_tokens=True)
        delta = text[len(state.emitted_text) :]
        state.emitted_text = text
        return delta

    def prepare_flow(self, tokens: list[int], generator: torch.Generator):
        if self.flow_input_embedding is None or self.speaker_embedding is None:
            raise RuntimeError("Token2Wav resources are not loaded")
        count = len(tokens)
        if count > self.max_chunk_tokens:
            raise ValueError(f"chunk has {count} tokens, HMM chain supports at most {self.max_chunk_tokens}")
        embedded = self.flow_input_embedding[torch.tensor(tokens, dtype=torch.long)].unsqueeze(0)
        encoder = torch.zeros((1, self.flow_token_capacity, embedded.shape[-1]), dtype=torch.float16)
        encoder[:, :count] = embedded.to(torch.float16)
        normalized = functional.normalize(self.speaker_embedding.to(torch.float32), dim=1).to(torch.float16)
        mel_length = count * TOKEN_MEL_RATIO
        mask = torch.zeros((2, 1, self.flow_mel_capacity), dtype=torch.float16)
        mask[:, :, :mel_length] = 1
        x = torch.randn((1, 80, self.flow_mel_capacity), generator=generator).to(torch.float16)
        return encoder.numpy(), normalized.numpy(), mask, x, mel_length

    def prepare_flow_decoder(self, x, mask, mu, speaker, t):
        x_in = x.repeat(2, 1, 1)
        mu_in = torch.zeros_like(x_in)
        mu_in[0] = mu[0]
        cond = torch.zeros((2, 80, self.flow_mel_capacity), dtype=torch.float16)
        spks = torch.zeros((2, 80), dtype=torch.float16)
        spks[0] = speaker.to(torch.float16)
        return (x_in, mask, mu_in, torch.full((2,), t.item(), dtype=torch.float16), spks, cond)

    def stft(self, source: np.ndarray) -> np.ndarray:
        window = torch.from_numpy(get_window("hann", 16, fftbins=True).astype(np.float32))
        stft = torch.stft(torch.from_numpy(source), n_fft=16, hop_length=4, win_length=16, window=window, center=False, onesided=True, return_complex=True)
        return torch.view_as_real(stft).permute(0, 2, 1, 3).numpy()

    @staticmethod
    def fade(waveform: torch.Tensor, fade_ms: float) -> torch.Tensor:
        samples = int(fade_ms / 1000 * SAMPLE_RATE)
        if samples <= 0 or waveform.shape[1] <= 2 * samples:
            return waveform
        result = waveform.clone()
        result[:, :samples] *= torch.linspace(0, 1, samples)
        result[:, -samples:] *= torch.linspace(1, 0, samples)
        return result

    @staticmethod
    def _load_cmvn(path: Path) -> tuple[np.ndarray, np.ndarray]:
        text = path.read_text(encoding="utf-8")
        values = []
        for tag in ("AddShift", "Rescale"):
            match = re.search(rf"<{tag}>.*?<LearnRateCoef>\s+0\s+\[([^\]]+)\]", text, flags=re.DOTALL)
            if match is None:
                raise ValueError(f"cannot find {tag} vector in {path}")
            values.append(np.fromstring(match.group(1), sep=" ", dtype=np.float32))
        if values[0].size != 400 or values[1].size != 400:
            raise ValueError("unexpected CMVN size")
        return values[0], values[1]

    @staticmethod
    def _apply_lfr(inputs: torch.Tensor, lfr_m: int, lfr_n: int) -> torch.Tensor:
        steps = inputs.shape[0]
        output_steps = int(np.ceil(steps / lfr_n))
        inputs = torch.vstack((inputs[0].repeat((lfr_m - 1) // 2, 1), inputs))
        total = inputs.shape[0]
        dim = inputs.shape[-1]
        last_idx = (total - lfr_m) // lfr_n + 1
        padding = lfr_m - (total - last_idx * lfr_n)
        if padding > 0:
            padding = (2 * lfr_m - 2 * total + (output_steps - 1 + last_idx) * lfr_n) / 2 * (output_steps - last_idx)
            inputs = torch.vstack([inputs] + [inputs[-1:]] * int(padding))
        return inputs.as_strided((output_steps, lfr_m * dim), (lfr_n * dim, 1)).clone().to(torch.float32)

    def prepare_vad(self, audio) -> VadPreparedRequest:
        if self.vad_config is None or self.cmvn is None:
            raise RuntimeError("VAD resources are not loaded")
        if isinstance(audio, np.ndarray):
            waveform = np.asarray(audio, dtype=np.float32).reshape(-1)
            sample_rate = self.vad_sample_rate
        else:
            tensor, sample_rate = torchaudio.load(str(require_file(audio, "input audio")))
            waveform = tensor.mean(dim=0).numpy().astype(np.float32)
        if sample_rate != self.vad_sample_rate:
            waveform = torchaudio.functional.resample(torch.from_numpy(waveform).unsqueeze(0), sample_rate, self.vad_sample_rate)[0].numpy()
            sample_rate = self.vad_sample_rate
        scaled = (torch.from_numpy(waveform) * (1 << 15)).unsqueeze(0)
        fbank = kaldi.fbank(
            scaled,
            num_mel_bins=int(self.frontend_config["n_mels"]),
            frame_length=int(self.frontend_config["frame_length"]),
            frame_shift=int(self.frontend_config["frame_shift"]),
            dither=float(self.frontend_config.get("dither", 0.0)),
            energy_floor=0.0,
            window_type=str(self.frontend_config.get("window", "hamming")),
            sample_frequency=sample_rate,
            snip_edges=True,
        )
        lfr = self._apply_lfr(fbank, int(self.frontend_config.get("lfr_m", 5)), int(self.frontend_config.get("lfr_n", 1)))
        add, scale = self.cmvn
        features = ((lfr + torch.from_numpy(add)) * torch.from_numpy(scale)).numpy().astype(np.float32)
        return VadPreparedRequest(waveform, sample_rate, features)

    def vad_postprocess(self, scores: np.ndarray, waveform: np.ndarray) -> tuple[list[list[int]], dict[str, Any]]:
        if self.vad_model_config is None:
            raise RuntimeError("VAD resources are not loaded")
        config = self.vad_model_config
        frame_shift = int(config.get("frame_in_ms", 10))
        frame_length = int(config.get("frame_length_ms", 25))
        if len(scores) == 0:
            return [], {"speech_frames": 0, "total_frames": 0, "raw_segments": []}
        silence = scores[:, config.get("sil_pdf_ids", [0])].sum(axis=1)
        speech = 1.0 - silence
        sample_length = int(self.vad_sample_rate * frame_length / 1000)
        sample_shift = int(self.vad_sample_rate * frame_shift / 1000)
        energy = np.asarray([10 * np.log10(np.sum(waveform[i : i + sample_length] ** 2) + 1e-6) for i in range(0, max(0, len(waveform) - sample_length + 1), sample_shift)], dtype=np.float32)
        if len(energy) < len(scores):
            energy = np.pad(energy, (0, len(scores) - len(energy)), constant_values=-100.0)
        flags = (speech >= silence + float(config.get("speech_noise_thres", 0.6))) & (energy[: len(scores)] >= float(config.get("decibel_thres", -100.0)))
        window = max(1, int(config.get("window_size_ms", 200) / frame_shift))
        start_frames = max(1, int(config.get("sil_to_speech_time_thres", 150) / frame_shift))
        end_frames = max(1, int(config.get("speech_to_sil_time_thres", 150) / frame_shift))
        start_padding = window + int(config.get("lookback_time_start_point", 200) / frame_shift)
        max_silence = max(1, int((config.get("max_end_silence_time", 800) - config.get("speech_to_sil_time_thres", 150)) / frame_shift))
        end_lookahead = int(config.get("lookahead_time_end_point", 100) / frame_shift)
        raw = self._find_vad_segments(
            flags, frame_shift, start_frames, end_frames, max_silence, end_lookahead, start_padding
        )
        merged = self._merge_vad_segments(raw)
        stats = {
            "speech_frames": int(flags.sum()),
            "total_frames": int(len(flags)),
            "speech_ratio": float(flags.mean()),
            "posterior_min": float(scores.min()),
            "posterior_max": float(scores.max()),
            "posterior_sum_max_error": float(np.max(np.abs(scores.sum(axis=1) - 1.0))),
            "raw_segments": raw,
            "sample_rate": self.vad_sample_rate,
            "feature_frames": int(len(scores)),
        }
        return merged, stats

    @staticmethod
    def _find_vad_segments(
        flags: np.ndarray,
        frame_shift: int,
        start_frames: int,
        end_frames: int,
        max_silence: int,
        end_lookahead: int,
        start_padding: int,
    ) -> list[list[int]]:
        raw = []
        state = "silence"
        start = None
        speech_run = silence_run = 0
        for index, is_speech in enumerate(flags):
            if is_speech:
                speech_run += 1
                silence_run = 0
            else:
                silence_run += 1
                speech_run = 0
            if state == "silence" and speech_run >= start_frames:
                start = max(0, index - start_padding + 1)
                state = "speech"
                silence_run = 0
            elif state == "speech" and silence_run >= max(end_frames, max_silence):
                end = max(start + 1, index - max_silence + 1 - end_lookahead)
                raw.append([start * frame_shift, end * frame_shift])
                start = None
                state = "silence"
                speech_run = 0
        if state == "speech" and start is not None:
            raw.append([start * frame_shift, len(flags) * frame_shift])
        return raw

    @staticmethod
    def _merge_vad_segments(raw: list[list[int]]) -> list[list[int]]:
        merged = []
        for segment in raw:
            if merged and segment[0] - merged[-1][1] <= 200:
                merged[-1][1] = segment[1]
            else:
                merged.append(segment)
        return merged


__all__ = [
    "AUDIO_BOS_ID",
    "AUDIO_EOS_ID",
    "DEFAULT_STATIC_AUDIO_SAMPLES",
    "DEFAULT_SYSTEM_PROMPT",
    "FLOW_CFG_RATE",
    "FLOW_STEPS",
    "FunAudioChatProcess",
    "GROUP_SIZE",
    "SAMPLE_RATE",
    "SAMPLES_PER_TOKEN",
    "SPOKEN_PROMPT",
    "TEXT_AUDIO_BOS_ID",
    "TEXT_AUDIO_EOS_ID",
    "TOKEN_MEL_RATIO",
]
