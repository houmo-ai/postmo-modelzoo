# Copyright (c) 2025 HOUMO AI
#
# File: qwen3_tts_runtime.py
# Description:
#   Shared runtime utilities and HMM wrappers for Qwen3-TTS demos.
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
import os
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
from torch import Tensor

import tcim_lite as tcim
from hmatc.python.get_hm_devices import get_hm_devices
from loguru import logger

HOUMO_TARGET = os.getenv("HOUMO_TARGET")
assert HOUMO_TARGET in ["xh2"], f"Unsupported HOUMO_TARGET: {HOUMO_TARGET}"

script_dir = os.path.dirname(os.path.abspath(__file__))


def get_hmm_path(model_name: str, model_size: str, sub_model_name: str) -> str:
    """Build default HMM path from model_name/model_size and sub-model name."""
    model_tag = f"{model_name}-{model_size}"
    return os.path.join("output", HOUMO_TARGET, f"{model_tag}_{sub_model_name}.hmm")


def get_default_hf_model_dir(
    model_config: dict,
    model_folder: str = script_dir,
    default_model_size: str = "0.6b-customvoice",
) -> str:
    """Infer the local HF model directory from config.yaml."""
    repo_ids = model_config.get("modelscope_repo", [])
    if repo_ids:
        model_dir = repo_ids[0].rsplit("/", maxsplit=1)[-1]
    else:
        model_name = model_config.get("model_name", "qwen3-tts")
        model_size = model_config.get("model_size", default_model_size)
        model_dir = f"{model_name}-{model_size}"
    return os.path.join(model_folder, model_dir)


def pad_or_trim_mels(
    mels: torch.Tensor, target_frames: int, target_dim: int
) -> torch.Tensor:
    """Pad/trim speaker mels to static speaker encoder input shape."""
    if mels.shape[-1] != target_dim:
        raise ValueError(
            f"speaker mels dim mismatch: expected {target_dim}, got {mels.shape[-1]}"
        )
    out = torch.zeros((mels.shape[0], target_frames, target_dim), dtype=torch.float32)
    valid_frames = min(int(mels.shape[1]), target_frames)
    if valid_frames > 0:
        out[:, :valid_frames, :] = mels[:, :valid_frames, :].to(torch.float32)
    return out


def to_numpy(value):
    """Convert torch.Tensor or other array-like values to numpy arrays."""
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().numpy()
    if isinstance(value, np.ndarray):
        return value
    return np.asarray(value)


def to_torch_dtype(dtype):
    """Convert numpy dtype or torch dtype to torch dtype."""
    if isinstance(dtype, torch.dtype):
        return dtype
    return torch.from_numpy(np.empty((), dtype=np.dtype(dtype))).dtype


def elapsed_ms(start_time: float) -> float:
    """Return elapsed wall time in milliseconds."""
    return (time.time() - start_time) * 1000


def elapsed_s(start_time: float) -> float:
    """Return elapsed wall time in seconds."""
    return time.time() - start_time


def get_module_input_names(module):
    """Get all input names from a tcim module."""
    return [module.get_input_name(i) for i in range(module.get_num_inputs())]


def get_module_output_names(module):
    """Get all output names from a tcim module."""
    return [module.get_output_name(i) for i in range(module.get_num_outputs())]


def build_assistant_text(text: str) -> str:
    """Build assistant-formatted text."""
    return f"<|im_start|>assistant\n{text}<|im_end|>\n<|im_start|>assistant\n"


def tokenize_texts(processor, texts):
    """Tokenize a list of texts to token ids."""
    input_ids = []
    for text in texts:
        input_data = processor(text=text, return_tensors="pt", padding=True)
        input_id = input_data["input_ids"]
        input_id = input_id.unsqueeze(0) if input_id.dim() == 1 else input_id
        input_ids.append(input_id)
    return input_ids


class Qwen3TTSTextProjectionInference(nn.Module):
    """Qwen3 TTS Text Projection inference."""

    def __init__(self, hmm_file: str, ndevice: int = 1):
        super().__init__()
        self.ndevice = ndevice

        dev_manager = tcim.runtime.DevManager(get_hm_devices(ndevice), "Xh2HalBackend")
        weight_manager = tcim.runtime.WeightManager(dev_manager)
        option = tcim.runtime.Option(weight_manager)

        self.model = tcim.runtime.load(hmm_file, option=option)
        self.input_names = get_module_input_names(self.model)
        self.output_names = get_module_output_names(self.model)

        input_info = self.model.get_input_info(self.input_names[0])
        self.net_input_seq_len = input_info.shape[1]

        logger.info("TextProjection loaded")

    def forward(self, hidden_state: Tensor) -> Tensor:
        """Run Text Projection by static sequence chunks."""
        hidden_state_chunks = hidden_state.split(self.net_input_seq_len, dim=1)
        outputs = []

        for chunk in hidden_state_chunks:
            self.model.set_input(self.input_names[0], to_numpy(chunk))
            self.model.run()
            self.model.sync()
            output = self.model.get_dev_output(self.output_names[0]).to_host().numpy()
            outputs.append(torch.from_numpy(output))

        return torch.cat(outputs, dim=1)


class Qwen3TTSCodePredictorInference:
    """Qwen3 TTS Code Predictor inference (prefill + decode)."""

    def __init__(
        self,
        prefill_hmm: str,
        decode_hmm: str,
        token_embedding: str,
        ndevice: int = 1,
    ) -> None:
        self.ndevice = ndevice
        dev_manager = tcim.runtime.DevManager(get_hm_devices(ndevice), "Xh2HalBackend")
        weight_manager = tcim.runtime.WeightManager(dev_manager)

        prefill_option = tcim.runtime.Option(weight_manager)
        self.prefill_model = tcim.runtime.load(prefill_hmm, option=prefill_option)
        self.prefill_input_names = get_module_input_names(self.prefill_model)
        self.prefill_output_names = get_module_output_names(self.prefill_model)

        dummy_tensor_names = [
            name for name in self.prefill_input_names if "model_layers" in name
        ]

        decode_option = tcim.runtime.Option(weight_manager)
        decode_option.set_dummy_tensors(dummy_tensor_names)
        self.decode_model = tcim.runtime.load(decode_hmm, option=decode_option)
        self.decode_input_names = get_module_input_names(self.decode_model)
        self.decode_output_names = get_module_output_names(self.decode_model)

        for input_name in dummy_tensor_names:
            cache = self.prefill_model.get_dev_input(input_name)
            self.decode_model.set_input(input_name, cache)

        self.prefill_length = self.prefill_model.get_input_info(
            self.prefill_input_names[0]
        ).shape[1]
        self.embedding_dim = self.prefill_model.get_input_info(
            self.prefill_input_names[0]
        ).shape[2]

        current_length_input = np.array([1]).astype("int32")
        self.decode_model.set_input(self.decode_input_names[2], current_length_input)
        self._zero_cache_inputs: Dict[int, np.ndarray] = {}

        embedding_state_dict = torch.load(token_embedding, map_location="cpu")
        num_embeddings, embedding_dim = embedding_state_dict["0.weight"].shape
        token_embedding_list = []
        for _ in range(len(embedding_state_dict)):
            token_embedding_list.append(
                nn.Embedding(num_embeddings=num_embeddings, embedding_dim=embedding_dim)
            )
        self.token_embedding = nn.ModuleList(token_embedding_list)
        self.token_embedding.load_state_dict(embedding_state_dict)
        self.token_embedding.to(torch.float16)

        logger.info(f"CodePredictor loaded: prefill_length={self.prefill_length}")

    def reset_cache_inputs(self, start_index: int = 3):
        """Clear KV cache inputs."""
        for index in range(start_index, len(self.prefill_input_names)):
            if index not in self._zero_cache_inputs:
                input_info = self.prefill_model.get_input_info(
                    self.prefill_input_names[index]
                )
                self._zero_cache_inputs[index] = np.zeros(
                    input_info.shape, dtype=input_info.dtype
                )
            self.prefill_model.set_input(
                self.prefill_input_names[index],
                self._zero_cache_inputs[index],
            )

    def prefill(
        self,
        inputs_embeds: np.ndarray,
        valid_length: np.ndarray,
        current_length: np.ndarray,
        generation_steps: np.ndarray,
        fetch_output: bool = True,
    ) -> np.ndarray:
        """Run one CodePredictor prefill chunk."""
        self.prefill_model.set_input(self.prefill_input_names[0], inputs_embeds)
        self.prefill_model.set_input(self.prefill_input_names[1], valid_length)
        self.prefill_model.set_input(self.prefill_input_names[2], current_length)
        self.prefill_model.set_input(self.prefill_input_names[-1], generation_steps)

        self.prefill_model.run()
        self.prefill_model.sync()

        if fetch_output:
            return (
                self.prefill_model.get_dev_output(self.prefill_output_names[0])
                .to_host()
                .numpy()
            )
        return None

    def decode(
        self,
        inputs_embeds: np.ndarray,
        valid_length: np.ndarray,
        generation_steps: np.ndarray,
    ) -> np.ndarray:
        """Run one CodePredictor decode step."""
        self.decode_model.set_input(self.decode_input_names[0], inputs_embeds)
        self.decode_model.set_input(self.decode_input_names[1], valid_length)
        self.decode_model.set_input(self.decode_input_names[-1], generation_steps)

        self.decode_model.run()
        self.decode_model.sync()

        return (
            self.decode_model.get_dev_output(self.decode_output_names[0])
            .to_host()
            .numpy()
        )


class Qwen3TTSTalkerInference:
    """Qwen3 TTS Talker inference (prefill + decode)."""

    def __init__(
        self,
        prefill_hmm: str,
        decode_hmm: str,
        token_embedding: str,
        text_embedding: str,
        ndevice: int = 1,
    ) -> None:
        self.ndevice = ndevice
        dev_manager = tcim.runtime.DevManager(get_hm_devices(ndevice), "Xh2HalBackend")
        weight_manager = tcim.runtime.WeightManager(dev_manager)

        prefill_option = tcim.runtime.Option(weight_manager)
        self.prefill_model = tcim.runtime.load(prefill_hmm, option=prefill_option)
        self.prefill_input_names = get_module_input_names(self.prefill_model)
        self.prefill_output_names = get_module_output_names(self.prefill_model)

        dummy_tensor_names = [
            name for name in self.prefill_input_names if "model_layers" in name
        ]

        decode_option = tcim.runtime.Option(weight_manager)
        decode_option.set_dummy_tensors(dummy_tensor_names)
        self.decode_model = tcim.runtime.load(decode_hmm, option=decode_option)
        self.decode_input_names = get_module_input_names(self.decode_model)
        self.decode_output_names = get_module_output_names(self.decode_model)

        for input_name in dummy_tensor_names:
            cache = self.prefill_model.get_dev_input(input_name)
            self.decode_model.set_input(input_name, cache)

        self.prefill_length = self.prefill_model.get_input_info(
            self.prefill_input_names[0]
        ).shape[1]
        self.embedding_dim = self.prefill_model.get_input_info(
            self.prefill_input_names[0]
        ).shape[2]
        self.context_max_length = self.decode_model.get_input_info(
            self.decode_input_names[3]
        ).shape[2]

        current_length_input = np.array([1]).astype("int32")
        self.decode_model.set_input(self.decode_input_names[2], current_length_input)
        self._zero_cache_inputs: Dict[int, np.ndarray] = {}

        self.token_embedding = torch.nn.Embedding.from_pretrained(
            torch.load(token_embedding, map_location="cpu")["weight"], freeze=True
        )
        self.text_embedding = torch.nn.Embedding.from_pretrained(
            torch.load(text_embedding, map_location="cpu")["weight"], freeze=True
        )
        self.token_embedding.to(torch.float16)
        self.text_embedding.to(torch.float16)

        logger.info(f"Talker loaded, context_length={self.context_max_length}")

    def get_input_embeddings(self):
        return self.token_embedding

    def get_text_embeddings(self):
        return self.text_embedding

    def reset_cache_inputs(self, start_index: int = 3):
        """Clear KV cache inputs."""
        for index in range(start_index, len(self.prefill_input_names)):
            if index not in self._zero_cache_inputs:
                input_info = self.prefill_model.get_input_info(
                    self.prefill_input_names[index]
                )
                self._zero_cache_inputs[index] = np.zeros(
                    input_info.shape, dtype=input_info.dtype
                )
            self.prefill_model.set_input(
                self.prefill_input_names[index],
                self._zero_cache_inputs[index],
            )

    def prefill(
        self,
        inputs_embeds: np.ndarray,
        valid_length: np.ndarray,
        current_length: np.ndarray,
        fetch_output: bool = True,
    ) -> tuple:
        """Run one Talker prefill chunk."""
        self.prefill_model.set_input(self.prefill_input_names[0], inputs_embeds)
        self.prefill_model.set_input(self.prefill_input_names[1], valid_length)
        self.prefill_model.set_input(self.prefill_input_names[2], current_length)

        self.prefill_model.run()
        self.prefill_model.sync()

        if fetch_output:
            logits = (
                self.prefill_model.get_dev_output(self.prefill_output_names[0])
                .to_host()
                .numpy()
            )
            past_hidden = (
                self.prefill_model.get_dev_output(self.prefill_output_names[1])
                .to_host()
                .numpy()
            )
            return logits, past_hidden
        return None, None

    def decode(
        self,
        inputs_embeds: np.ndarray,
        valid_length: np.ndarray,
    ) -> tuple:
        """Run one Talker decode step."""
        self.decode_model.set_input(self.decode_input_names[0], inputs_embeds)
        self.decode_model.set_input(self.decode_input_names[1], valid_length)

        self.decode_model.run()
        self.decode_model.sync()

        logits = (
            self.decode_model.get_dev_output(self.decode_output_names[0])
            .to_host()
            .numpy()
        )
        past_hidden = (
            self.decode_model.get_dev_output(self.decode_output_names[1])
            .to_host()
            .numpy()
        )
        return logits, past_hidden


class Qwen3TTSSpeechTokenizerInference(nn.Module):
    """Qwen3 TTS Speech Tokenizer inference."""

    def __init__(
        self,
        hmm_file: str,
        decode_padding_shapes: Union[str, Dict[str, List[int]]],
        ndevice: int = 1,
    ) -> None:
        super().__init__()

        if isinstance(decode_padding_shapes, str):
            with open(decode_padding_shapes, "r") as f:
                self.decode_padding_shapes = json.load(f)
        else:
            self.decode_padding_shapes = decode_padding_shapes

        self.ndevice = ndevice

        dev_manager = tcim.runtime.DevManager(get_hm_devices(ndevice), "Xh2HalBackend")
        weight_manager = tcim.runtime.WeightManager(dev_manager)
        option = tcim.runtime.Option(weight_manager)

        self.model = tcim.runtime.load(hmm_file, option=option)
        self.input_names = get_module_input_names(self.model)
        self.output_names = get_module_output_names(self.model)

        input_info = self.model.get_input_info(self.input_names[0])
        self.net_input_seq_len = input_info.shape[2]

        logger.info(f"SpeechTokenizer loaded: input_seq_len={self.net_input_seq_len}")

    def forward(self, codes: Tensor) -> Tensor:
        """Run speech tokenizer inference, input codes shape [b, c, seq]."""
        b, c, seq = codes.shape
        output_shape = self.decode_padding_shapes[f"{b}_{c}_{seq}"]

        input_codes = torch.nn.functional.pad(codes, (0, self.net_input_seq_len - seq))
        input_codes = input_codes.to(torch.int32)

        self.model.set_input(self.input_names[0], to_numpy(input_codes))
        self.model.run()
        self.model.sync()

        output = self.model.get_dev_output(self.output_names[0]).to_host().numpy()
        output = output[:, :, : output_shape[2]]
        return torch.from_numpy(output)

    def decode(self, encoded) -> tuple:
        """Decode generated audio codes into wav arrays."""
        if isinstance(encoded, dict):
            audio_codes_list = [encoded["audio_codes"]]
        elif isinstance(encoded, list):
            audio_codes_list = [item["audio_codes"] for item in encoded]
        else:
            raise TypeError(
                "`encoded` must be a dict or a list of dicts with `audio_codes`."
            )

        wavs = []
        for codes in audio_codes_list:
            if not isinstance(codes, torch.Tensor):
                codes = torch.from_numpy(np.asarray(codes))
            codes = codes.to(torch.long)

            if codes.dim() == 2:
                codes = codes.transpose(0, 1).unsqueeze(0)
            elif codes.dim() == 3:
                pass
            else:
                raise ValueError(f"Unsupported audio_codes shape: {tuple(codes.shape)}")

            wav = self.chunked_decode(codes)
            wavs.append(wav.squeeze().to(torch.float32).detach().cpu().numpy())

        return wavs, 24000

    def chunked_decode(
        self, codes: Tensor, chunk_size: int = 300, left_context_size: int = 25
    ) -> Tensor:
        """Decode long audio by overlapping chunks."""
        wavs = []
        start_index = 0
        while start_index < codes.shape[-1]:
            context_size = (
                left_context_size
                if start_index - left_context_size > 0
                else start_index
            )
            max_chunk_tokens = self.net_input_seq_len - context_size
            if max_chunk_tokens <= 0:
                raise ValueError(
                    f"left_context_size={left_context_size} is too large for "
                    f"net_input_seq_len={self.net_input_seq_len}"
                )
            end_index = min(
                start_index + min(chunk_size, max_chunk_tokens),
                codes.shape[-1],
            )
            codes_chunk = codes[..., start_index - context_size : end_index]
            wav_chunk = self.forward(codes_chunk)
            upsample_rate = 1920
            wavs.append(wav_chunk[..., context_size * upsample_rate :])
            start_index = end_index
        return torch.cat(wavs, dim=-1)


@dataclass
class StatefulDecoderState:
    """Stateful decoder runtime state."""

    pre_conv_history: Any
    latent_buffer: Any
    conv_history: Any
    kv_cache: List[Any] = field(default_factory=list)
    kv_valid_len: int = 0
    skip_samples: int = 0
    latent_audio: Optional[np.ndarray] = None


@dataclass
class StreamingPlaybackGapTracker:
    """Track whether streaming audio chunks can be played continuously."""

    prev_emit_time: Optional[float] = None
    prev_audio_ms: float = 0.0
    gap_chunks: int = 0
    max_gap_ms: float = 0.0
    total_gap_ms: float = 0.0

    def update(
        self,
        num_samples: int,
        sample_rate: int,
        emit_time: Optional[float] = None,
    ) -> Optional[float]:
        """Record one emitted audio chunk and return its playback gap in ms.

        The first chunk has no previous chunk to compare with, so the return
        value is None. Later chunks return 0 when playback can continue
        without an underrun, or a positive gap in milliseconds otherwise.
        """
        now = time.perf_counter() if emit_time is None else emit_time
        audio_ms = num_samples / sample_rate * 1000
        if self.prev_emit_time is None:
            self.prev_emit_time = now
            self.prev_audio_ms = audio_ms
            return None

        since_prev_ms = (now - self.prev_emit_time) * 1000
        gap_ms = max(0.0, since_prev_ms - self.prev_audio_ms)
        if gap_ms > 0:
            self.gap_chunks += 1
            self.total_gap_ms += gap_ms
            self.max_gap_ms = max(self.max_gap_ms, gap_ms)
        self.prev_emit_time = now
        self.prev_audio_ms = audio_ms
        return gap_ms


class Qwen3TTSStatefulDecoderInference:
    """Qwen3 TTS Stateful Decoder inference for streaming audio decode."""

    NUM_LAYERS = 8
    NUM_HEADS = 16
    HEAD_DIM = 64
    KV_CACHE_WINDOW = 72
    SAMPLES_PER_FRAME = 1920
    INITIAL_OUTPUT_SKIP_FRAMES = 4

    def __init__(self, hmm_file: str, chunk_size: int = 12, ndevice: int = 1):
        self.chunk_size = chunk_size
        self.ndevice = ndevice

        dev_manager = tcim.runtime.DevManager(get_hm_devices(ndevice), "Xh2HalBackend")
        weight_manager = tcim.runtime.WeightManager(dev_manager)
        option = tcim.runtime.Option(weight_manager)

        self.model = tcim.runtime.load(hmm_file, option=option)
        self.input_names = get_module_input_names(self.model)
        self.output_names = get_module_output_names(self.model)
        input_info = self.model.get_input_info(self.input_names[0])
        self.net_chunk_size = int(input_info.shape[1])
        if int(chunk_size) != self.net_chunk_size:
            raise ValueError(
                f"StatefulDecoder HMM expects chunk_size={self.net_chunk_size}, "
                f"got {chunk_size}. Rebuild the model or use --chunk_size "
                f"{self.net_chunk_size}."
            )
        self._is_last_buf = np.empty((1,), dtype=np.float16)
        self._kv_valid_len_buf = np.empty((1,), dtype=np.int32)
        self._valid_frames_buf = np.empty((1,), dtype=np.int32)

        logger.info(f"StatefulDecoder loaded: chunk_size={self.net_chunk_size}")

    def create_state(self) -> StatefulDecoderState:
        """Create initial zero decoder state and upload it to device inputs."""
        pre_conv_history = np.zeros((1, 512, 2), dtype=np.float16)
        latent_buffer = np.zeros((1, 1024, 4), dtype=np.float16)
        conv_history = np.zeros((1, 1024, 4), dtype=np.float16)
        kv_cache = []
        for _ in range(self.NUM_LAYERS):
            kv_cache.append(
                np.zeros(
                    (1, self.NUM_HEADS, self.KV_CACHE_WINDOW, self.HEAD_DIM),
                    dtype=np.float16,
                )
            )
            kv_cache.append(
                np.zeros(
                    (1, self.NUM_HEADS, self.KV_CACHE_WINDOW, self.HEAD_DIM),
                    dtype=np.float16,
                )
            )

        state_input_index = 1
        self.model.set_input(self.input_names[state_input_index], pre_conv_history)
        pre_conv_history_dev = self.model.get_dev_input(
            self.input_names[state_input_index]
        )
        state_input_index += 1

        self.model.set_input(self.input_names[state_input_index], latent_buffer)
        latent_buffer_dev = self.model.get_dev_input(
            self.input_names[state_input_index]
        )
        state_input_index += 1

        self.model.set_input(self.input_names[state_input_index], conv_history)
        conv_history_dev = self.model.get_dev_input(self.input_names[state_input_index])
        state_input_index = 7

        kv_cache_dev = []
        for cache in kv_cache:
            input_name = self.input_names[state_input_index]
            self.model.set_input(input_name, cache)
            kv_cache_dev.append(self.model.get_dev_input(input_name))
            state_input_index += 1

        return StatefulDecoderState(
            pre_conv_history=pre_conv_history_dev,
            latent_buffer=latent_buffer_dev,
            conv_history=conv_history_dev,
            kv_cache=kv_cache_dev,
            kv_valid_len=0,
            skip_samples=0,
            latent_audio=None,
        )

    def _set_state_input(self, input_name: str, value: Any) -> None:
        if isinstance(value, np.ndarray):
            self.model.set_input(input_name, value)
        else:
            self.model.set_dev_input(input_name, value)

    def decode(
        self,
        audio_codes: np.ndarray,
        state: StatefulDecoderState,
        is_final: bool = False,
    ) -> Tuple[np.ndarray, StatefulDecoderState]:
        """Decode one codec chunk and return audio samples plus next state."""
        audio_codes = np.asarray(audio_codes, dtype=np.int32)
        if audio_codes.ndim == 1:
            audio_codes = audio_codes.reshape(-1, 16)

        n_frames = audio_codes.shape[0]
        skip_counter = state.skip_samples

        if n_frames == 0:
            if is_final and state.latent_audio is not None:
                audio = state.latent_audio.astype(np.float32)
                state.latent_audio = None
                return audio, state
            return np.array([], dtype=np.float32), state

        if n_frames > self.chunk_size:
            raise ValueError(
                f"expected at most {self.chunk_size} codec frames, got {n_frames}"
            )
        if not is_final and n_frames < self.chunk_size:
            raise ValueError(
                f"non-final chunks must contain exactly {self.chunk_size} codec frames, got {n_frames}"
            )

        valid_frames = n_frames
        if n_frames < self.chunk_size:
            pad = np.zeros((self.chunk_size - n_frames, 16), dtype=np.int32)
            audio_codes = np.concatenate([audio_codes, pad], axis=0)

        audio_codes_input = audio_codes[np.newaxis, ...]
        self._is_last_buf[0] = 1.0 if is_final else 0.0
        self._kv_valid_len_buf[0] = state.kv_valid_len
        self._valid_frames_buf[0] = valid_frames

        idx = 0
        self.model.set_input(self.input_names[idx], audio_codes_input)
        idx += 1
        self._set_state_input(self.input_names[idx], state.pre_conv_history)
        idx += 1
        self._set_state_input(self.input_names[idx], state.latent_buffer)
        idx += 1
        self._set_state_input(self.input_names[idx], state.conv_history)
        idx += 1
        self.model.set_input(self.input_names[idx], self._is_last_buf)
        idx += 1
        self.model.set_input(self.input_names[idx], self._kv_valid_len_buf)
        idx += 1
        self.model.set_input(self.input_names[idx], self._valid_frames_buf)
        idx += 1
        for i in range(self.NUM_LAYERS * 2):
            self._set_state_input(self.input_names[idx], state.kv_cache[i])
            idx += 1

        self.model.run()
        self.model.sync()

        final_wav = self.model.get_dev_output(self.output_names[0]).to_host().numpy()
        valid_samples = int(
            self.model.get_dev_output(self.output_names[1])
            .to_host()
            .numpy()
            .reshape(-1)[0]
        )

        new_state = StatefulDecoderState(
            pre_conv_history=self.model.get_dev_output(self.output_names[2]),
            latent_buffer=self.model.get_dev_output(self.output_names[3]),
            conv_history=self.model.get_dev_output(self.output_names[4]),
            kv_cache=[],
            kv_valid_len=min(self.KV_CACHE_WINDOW, state.kv_valid_len + valid_frames),
            skip_samples=0,
            latent_audio=None,
        )
        key_start = 5
        for i in range(self.NUM_LAYERS):
            new_state.kv_cache.append(
                self.model.get_dev_output(self.output_names[key_start + i])
            )
        for i in range(self.NUM_LAYERS):
            new_state.kv_cache.append(
                self.model.get_dev_output(
                    self.output_names[key_start + self.NUM_LAYERS + i]
                )
            )

        initial_skip = 0
        if state.kv_valid_len == 0:
            initial_skip = self.INITIAL_OUTPUT_SKIP_FRAMES * self.SAMPLES_PER_FRAME
        audio_start = initial_skip
        audio_end = audio_start + valid_samples

        if is_final:
            audio = (
                final_wav[0, audio_start:audio_end]
                if valid_samples > 0
                else np.array([], dtype=np.float32)
            )
            new_state.latent_audio = None
        elif valid_samples > 0:
            audio = final_wav[0, audio_start:audio_end]
            new_state.latent_audio = final_wav[0, audio_end:]
        else:
            audio = np.array([], dtype=np.float32)
            new_state.latent_audio = final_wav[0, audio_start:]

        if skip_counter > 0 and len(audio) > 0:
            if len(audio) <= skip_counter:
                skip_counter -= len(audio)
                audio = np.array([], dtype=np.float32)
            else:
                audio = audio[skip_counter:]
                skip_counter = 0
        new_state.skip_samples = (
            4 * self.SAMPLES_PER_FRAME if is_final else skip_counter
        )

        return audio.astype(np.float32), new_state


class CodeChunkBuffer:
    """Frame buffer that emits fixed-size codec chunks."""

    def __init__(self, chunk_size: int = 12):
        self.chunk_size = chunk_size
        self._frames: List[torch.Tensor] = []

    def push(self, codec_ids: torch.Tensor):
        self._frames.append(codec_ids.view(16))

    def flush(self) -> Optional[torch.Tensor]:
        if len(self._frames) >= self.chunk_size:
            chunk = torch.stack(self._frames[: self.chunk_size])
            self._frames = self._frames[self.chunk_size :]
            return chunk
        return None

    def finalize(self) -> Optional[torch.Tensor]:
        if self._frames:
            chunk = torch.stack(self._frames)
            self._frames = []
            return chunk
        return None
