# Copyright (c) 2026 HOUMO AI
#
# File: qwen3_tts.py
# Description:
#   Qwen3-TTS runtime Module implementation.
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

"""Qwen3-TTS device graphs, KV/state caches, and stage execution.

The Module owns every HMM graph and device-side buffer used by Qwen3-TTS:

- ``text_projection``: projects text hidden states into the Talker input space.
- ``talker`` prefill/decode: autoregressive codec-group-0 generation.
- ``code_predictor`` prefill/decode: remaining codec groups per frame.
- ``speech_tokenizer``: oneshot codec-frame to waveform decode.
- ``stateful_decoder``: streaming chunked codec to waveform decode.

CPU-side embedding lookups and tokenization live in :class:`Qwen3TtsProcess`.
Stage ordering, sampling, and stop decisions live in :class:`Qwen3TtsEngine`.
"""

import json
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np
import tcim_lite as tcim
import torch

from ..core import HoumoModule
from ..core.types import Stage, StageInputs, StageOutputs
from ..perf import PerfTracker


def _to_numpy(value):
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().numpy()
    if isinstance(value, np.ndarray):
        return value
    return np.asarray(value)


def _input_names(model) -> List[str]:
    return [model.get_input_name(i) for i in range(model.get_num_inputs())]


def _output_names(model) -> List[str]:
    return [model.get_output_name(i) for i in range(model.get_num_outputs())]


@dataclass
class _StatefulDecoderState:
    """Streaming decoder device state carried between chunks."""

    pre_conv_history: Any
    latent_buffer: Any
    conv_history: Any
    kv_cache: List[Any] = field(default_factory=list)
    kv_valid_len: int = 0
    skip_samples: int = 0
    latent_audio: Optional[np.ndarray] = None


class Qwen3TtsModule(HoumoModule):
    """Qwen3-TTS HMM graphs, cache bindings, and stage execution."""

    # Stateful decoder static configuration (matches the compiled graph).
    _SD_NUM_LAYERS = 8
    _SD_NUM_HEADS = 16
    _SD_HEAD_DIM = 64
    _SD_KV_CACHE_WINDOW = 72
    _SD_SAMPLES_PER_FRAME = 1920
    _SD_INITIAL_OUTPUT_SKIP_FRAMES = 4

    def __init__(
        self,
        text_projection_path,
        talker_prefill_path,
        talker_decode_path,
        code_predictor_prefill_path,
        code_predictor_decode_path,
        *,
        mode: str = "oneshot",
        speech_tokenizer_path=None,
        stateful_decoder_path=None,
        decode_padding_shapes_path=None,
        chunk_size: int = 12,
        ndevice: int = 1,
        perf: PerfTracker,
    ):
        self.perf = perf
        self.mode = mode
        self.chunk_size = int(chunk_size)
        self._stage_metadata: Dict[Stage, dict] = {}
        self._decoder_state: Optional[_StatefulDecoderState] = None
        self.load(
            text_projection_path,
            talker_prefill_path,
            talker_decode_path,
            code_predictor_prefill_path,
            code_predictor_decode_path,
            speech_tokenizer_path=speech_tokenizer_path,
            stateful_decoder_path=stateful_decoder_path,
            decode_padding_shapes_path=decode_padding_shapes_path,
            ndevice=ndevice,
        )

    # ------------------------------------------------------------------
    # Loading
    # ------------------------------------------------------------------
    def _weight_manager(self, ndevice: int):
        from hmatc.python.get_hm_devices import get_hm_devices

        dev_manager = tcim.runtime.DevManager(get_hm_devices(ndevice), "Xh2HalBackend")
        return tcim.runtime.WeightManager(dev_manager)

    def load(
        self,
        text_projection_path,
        talker_prefill_path,
        talker_decode_path,
        code_predictor_prefill_path,
        code_predictor_decode_path,
        *,
        speech_tokenizer_path=None,
        stateful_decoder_path=None,
        decode_padding_shapes_path=None,
        ndevice: int = 1,
    ) -> None:
        with self.perf.scope("tts.init"):
            self._load_text_projection(text_projection_path, ndevice)
            self._load_talker(talker_prefill_path, talker_decode_path, ndevice)
            self._load_code_predictor(code_predictor_prefill_path, code_predictor_decode_path, ndevice)
            if self.mode == "streaming":
                if stateful_decoder_path is None:
                    raise ValueError("streaming mode requires a stateful decoder graph")
                self._load_stateful_decoder(stateful_decoder_path, ndevice)
            else:
                if speech_tokenizer_path is None:
                    raise ValueError("oneshot mode requires a speech tokenizer graph")
                self._load_speech_tokenizer(speech_tokenizer_path, decode_padding_shapes_path, ndevice)

    def _load_text_projection(self, path, ndevice: int) -> None:
        option = tcim.runtime.Option(self._weight_manager(ndevice))
        with self.perf.scope("tts.init.text_projection_load"):
            self.text_projection = tcim.runtime.load(str(path), option=option)
        self._tp_inputs = _input_names(self.text_projection)
        self._tp_outputs = _output_names(self.text_projection)
        self._tp_seq_len = int(self.text_projection.get_input_info(self._tp_inputs[0]).shape[1])

    def _load_talker(self, prefill_path, decode_path, ndevice: int) -> None:
        weight_manager = self._weight_manager(ndevice)
        with self.perf.scope("tts.init.talker_prefill_load"):
            self.talker_prefill = tcim.runtime.load(str(prefill_path), option=tcim.runtime.Option(weight_manager))
        self._talker_prefill_inputs = _input_names(self.talker_prefill)
        self._talker_prefill_outputs = _output_names(self.talker_prefill)

        dummy = [n for n in self._talker_prefill_inputs if "model_layers" in n]
        decode_option = tcim.runtime.Option(weight_manager)
        decode_option.set_dummy_tensors(dummy)
        with self.perf.scope("tts.init.talker_decode_load"):
            self.talker_decode = tcim.runtime.load(str(decode_path), option=decode_option)
        self._talker_decode_inputs = _input_names(self.talker_decode)
        self._talker_decode_outputs = _output_names(self.talker_decode)

        for name in dummy:
            self.talker_decode.set_input(name, self.talker_prefill.get_dev_input(name))

        self.talker_prefill_length = int(self.talker_prefill.get_input_info(self._talker_prefill_inputs[0]).shape[1])
        self.embedding_size = int(self.talker_prefill.get_input_info(self._talker_prefill_inputs[0]).shape[2])
        self.talker_context_max_length = int(self.talker_decode.get_input_info(self._talker_decode_inputs[3]).shape[2])
        # Talker decode current_length is always a single token.
        self.talker_decode.set_input(self._talker_decode_inputs[2], np.array([1], dtype=np.int32))
        self._talker_prefill_zero_cache: Dict[int, np.ndarray] = {}

    def _load_code_predictor(self, prefill_path, decode_path, ndevice: int) -> None:
        weight_manager = self._weight_manager(ndevice)
        with self.perf.scope("tts.init.code_predictor_prefill_load"):
            self.cp_prefill = tcim.runtime.load(str(prefill_path), option=tcim.runtime.Option(weight_manager))
        self._cp_prefill_inputs = _input_names(self.cp_prefill)
        self._cp_prefill_outputs = _output_names(self.cp_prefill)

        dummy = [n for n in self._cp_prefill_inputs if "model_layers" in n]
        decode_option = tcim.runtime.Option(weight_manager)
        decode_option.set_dummy_tensors(dummy)
        with self.perf.scope("tts.init.code_predictor_decode_load"):
            self.cp_decode = tcim.runtime.load(str(decode_path), option=decode_option)
        self._cp_decode_inputs = _input_names(self.cp_decode)
        self._cp_decode_outputs = _output_names(self.cp_decode)

        for name in dummy:
            self.cp_decode.set_input(name, self.cp_prefill.get_dev_input(name))

        self.cp_prefill_length = int(self.cp_prefill.get_input_info(self._cp_prefill_inputs[0]).shape[1])
        self.cp_decode.set_input(self._cp_decode_inputs[2], np.array([1], dtype=np.int32))
        self._cp_prefill_zero_cache: Dict[int, np.ndarray] = {}

    def _load_speech_tokenizer(self, path, padding_shapes_path, ndevice: int) -> None:
        if padding_shapes_path is None:
            raise ValueError("speech tokenizer requires decode_padding_shapes")
        with open(str(padding_shapes_path), "r", encoding="utf-8") as stream:
            self._st_padding_shapes = json.load(stream)
        option = tcim.runtime.Option(self._weight_manager(ndevice))
        with self.perf.scope("tts.init.speech_tokenizer_load"):
            self.speech_tokenizer = tcim.runtime.load(str(path), option=option)
        self._st_inputs = _input_names(self.speech_tokenizer)
        self._st_outputs = _output_names(self.speech_tokenizer)
        self._st_seq_len = int(self.speech_tokenizer.get_input_info(self._st_inputs[0]).shape[2])

    def _load_stateful_decoder(self, path, ndevice: int) -> None:
        option = tcim.runtime.Option(self._weight_manager(ndevice))
        with self.perf.scope("tts.init.stateful_decoder_load"):
            self.stateful_decoder = tcim.runtime.load(str(path), option=option)
        self._sd_inputs = _input_names(self.stateful_decoder)
        self._sd_outputs = _output_names(self.stateful_decoder)
        net_chunk = int(self.stateful_decoder.get_input_info(self._sd_inputs[0]).shape[1])
        if self.chunk_size != net_chunk:
            raise ValueError(f"StatefulDecoder HMM expects chunk_size={net_chunk}, " f"got {self.chunk_size}.")
        self._sd_is_last_buf = np.empty((1,), dtype=np.float16)
        self._sd_kv_valid_len_buf = np.empty((1,), dtype=np.int32)
        self._sd_valid_frames_buf = np.empty((1,), dtype=np.int32)

    # ------------------------------------------------------------------
    # Cache management (timing decided by the Engine)
    # ------------------------------------------------------------------
    def reset_talker_cache(self, start_index: int = 3) -> None:
        for index in range(start_index, len(self._talker_prefill_inputs)):
            if index not in self._talker_prefill_zero_cache:
                info = self.talker_prefill.get_input_info(self._talker_prefill_inputs[index])
                self._talker_prefill_zero_cache[index] = np.zeros(info.shape, dtype=info.dtype)
            self.talker_prefill.set_input(
                self._talker_prefill_inputs[index],
                self._talker_prefill_zero_cache[index],
            )

    def reset_code_predictor_cache(self, start_index: int = 3) -> None:
        for index in range(start_index, len(self._cp_prefill_inputs)):
            if index not in self._cp_prefill_zero_cache:
                info = self.cp_prefill.get_input_info(self._cp_prefill_inputs[index])
                self._cp_prefill_zero_cache[index] = np.zeros(info.shape, dtype=info.dtype)
            self.cp_prefill.set_input(
                self._cp_prefill_inputs[index],
                self._cp_prefill_zero_cache[index],
            )

    # ------------------------------------------------------------------
    # Named text projection graph (used during embedding preparation)
    # ------------------------------------------------------------------
    def run_text_projection(self, hidden_state) -> np.ndarray:
        """Project text hidden states by fixed sequence chunks."""
        hidden = _to_numpy(hidden_state)
        seq_len = hidden.shape[1]
        outputs = []
        with self.perf.scope("tts.text_projection.infer"):
            for start in range(0, seq_len, self._tp_seq_len):
                chunk = hidden[:, start : start + self._tp_seq_len, :]
                self.text_projection.set_input(self._tp_inputs[0], chunk)
                self.text_projection.run()
                self.text_projection.sync()
                out = self.text_projection.get_dev_output(self._tp_outputs[0]).to_host().numpy()
                outputs.append(out)
        return np.concatenate(outputs, axis=1)

    # ------------------------------------------------------------------
    # Standard four-step stage interface
    # ------------------------------------------------------------------
    def set_input(self, stage: Stage, inputs: StageInputs) -> None:
        self._stage_metadata[stage] = dict(inputs.metadata)
        if stage == Stage.PREFILL:
            model, names, path = (
                self.talker_prefill,
                self._talker_prefill_inputs,
                "tts.talker.prefill",
            )
        elif stage == Stage.DECODE:
            model, names, path = (
                self.talker_decode,
                self._talker_decode_inputs,
                "tts.talker.decode",
            )
        elif stage == Stage.CODE_PREDICTOR_PREFILL:
            model, names, path = (
                self.cp_prefill,
                self._cp_prefill_inputs,
                "tts.code_predictor.prefill",
            )
        elif stage == Stage.CODE_PREDICTOR_DECODE:
            model, names, path = (
                self.cp_decode,
                self._cp_decode_inputs,
                "tts.code_predictor.decode",
            )
        else:
            raise ValueError(f"unsupported stage: {stage}")

        with self.perf.scope(f"{path}.set_input"):
            if stage == Stage.PREFILL:
                embeds, valid_length, current_length = inputs.tensors
                model.set_input(names[0], _to_numpy(embeds))
                model.set_input(names[1], _to_numpy(valid_length))
                model.set_input(names[2], _to_numpy(current_length))
            elif stage == Stage.DECODE:
                embeds, valid_length = inputs.tensors
                model.set_input(names[0], _to_numpy(embeds))
                model.set_input(names[1], _to_numpy(valid_length))
            elif stage == Stage.CODE_PREDICTOR_PREFILL:
                embeds, valid_length, current_length, generation_steps = inputs.tensors
                model.set_input(names[0], _to_numpy(embeds))
                model.set_input(names[1], _to_numpy(valid_length))
                model.set_input(names[2], _to_numpy(current_length))
                model.set_input(names[-1], _to_numpy(generation_steps))
            else:  # CODE_PREDICTOR_DECODE
                embeds, valid_length, generation_steps = inputs.tensors
                model.set_input(names[0], _to_numpy(embeds))
                model.set_input(names[1], _to_numpy(valid_length))
                model.set_input(names[-1], _to_numpy(generation_steps))

    def run(self, stage: Stage) -> None:
        model, path = self._stage_model(stage)
        with self.perf.scope(f"{path}.infer"):
            model.run()
            model.sync()

    def get_output(self, stage: Stage) -> StageOutputs:
        model, path = self._stage_model(stage)
        metadata = self._stage_metadata.pop(stage, {})
        with self.perf.scope(f"{path}.get_output"):
            if stage in (Stage.PREFILL, Stage.DECODE):
                outputs = self._talker_prefill_outputs if stage == Stage.PREFILL else self._talker_decode_outputs
                logits = model.get_dev_output(outputs[0]).to_host().numpy()
                past_hidden = model.get_dev_output(outputs[1]).to_host().numpy()
                return StageOutputs(tensors=(logits, past_hidden), metadata=metadata)
            outputs = self._cp_prefill_outputs if stage == Stage.CODE_PREDICTOR_PREFILL else self._cp_decode_outputs
            logits = model.get_dev_output(outputs[0]).to_host().numpy()
            return StageOutputs(tensors=(logits,), metadata=metadata)

    def _stage_model(self, stage: Stage):
        if stage == Stage.PREFILL:
            return self.talker_prefill, "tts.talker.prefill"
        if stage == Stage.DECODE:
            return self.talker_decode, "tts.talker.decode"
        if stage == Stage.CODE_PREDICTOR_PREFILL:
            return self.cp_prefill, "tts.code_predictor.prefill"
        if stage == Stage.CODE_PREDICTOR_DECODE:
            return self.cp_decode, "tts.code_predictor.decode"
        raise ValueError(f"unsupported stage: {stage}")

    # ------------------------------------------------------------------
    # Oneshot audio decode (speech tokenizer)
    # ------------------------------------------------------------------
    def run_speech_tokenizer(self, audio_codes_list):
        """Decode a list of codec-frame tensors into waveforms."""
        wavs = []
        with self.perf.scope("tts.speech_tokenizer.infer"):
            for codes in audio_codes_list:
                if not isinstance(codes, torch.Tensor):
                    codes = torch.from_numpy(np.asarray(codes))
                codes = codes.to(torch.long)
                if codes.dim() == 2:
                    codes = codes.transpose(0, 1).unsqueeze(0)
                elif codes.dim() != 3:
                    raise ValueError(f"unsupported audio_codes shape: {tuple(codes.shape)}")
                wav = self._speech_tokenizer_chunked_decode(codes)
                wavs.append(wav.squeeze().to(torch.float32).detach().cpu().numpy())
        return wavs, 24000

    def _speech_tokenizer_forward(self, codes: torch.Tensor) -> torch.Tensor:
        b, c, seq = codes.shape
        output_shape = self._st_padding_shapes[f"{b}_{c}_{seq}"]
        input_codes = torch.nn.functional.pad(codes, (0, self._st_seq_len - seq))
        input_codes = input_codes.to(torch.int32)
        self.speech_tokenizer.set_input(self._st_inputs[0], _to_numpy(input_codes))
        self.speech_tokenizer.run()
        self.speech_tokenizer.sync()
        output = self.speech_tokenizer.get_dev_output(self._st_outputs[0]).to_host().numpy()
        output = output[:, :, : output_shape[2]]
        return torch.from_numpy(output)

    def _speech_tokenizer_chunked_decode(
        self, codes: torch.Tensor, chunk_size: int = 300, left_context_size: int = 25
    ) -> torch.Tensor:
        wavs = []
        start_index = 0
        while start_index < codes.shape[-1]:
            context_size = left_context_size if start_index - left_context_size > 0 else start_index
            max_chunk_tokens = self._st_seq_len - context_size
            if max_chunk_tokens <= 0:
                raise ValueError(
                    f"left_context_size={left_context_size} is too large for " f"net_input_seq_len={self._st_seq_len}"
                )
            end_index = min(start_index + min(chunk_size, max_chunk_tokens), codes.shape[-1])
            codes_chunk = codes[..., start_index - context_size : end_index]
            wav_chunk = self._speech_tokenizer_forward(codes_chunk)
            upsample_rate = 1920
            wavs.append(wav_chunk[..., context_size * upsample_rate :])
            start_index = end_index
        return torch.cat(wavs, dim=-1)

    # ------------------------------------------------------------------
    # Streaming audio decode (stateful decoder)
    # ------------------------------------------------------------------
    def create_decoder_state(self) -> None:
        """Create initial zero decoder state and upload it to device inputs."""
        pre_conv_history = np.zeros((1, 512, 2), dtype=np.float16)
        latent_buffer = np.zeros((1, 1024, 4), dtype=np.float16)
        conv_history = np.zeros((1, 1024, 4), dtype=np.float16)
        kv_cache = []
        for _ in range(self._SD_NUM_LAYERS):
            for _ in range(2):
                kv_cache.append(
                    np.zeros(
                        (
                            1,
                            self._SD_NUM_HEADS,
                            self._SD_KV_CACHE_WINDOW,
                            self._SD_HEAD_DIM,
                        ),
                        dtype=np.float16,
                    )
                )

        idx = 1
        self.stateful_decoder.set_input(self._sd_inputs[idx], pre_conv_history)
        pre_conv_history_dev = self.stateful_decoder.get_dev_input(self._sd_inputs[idx])
        idx += 1
        self.stateful_decoder.set_input(self._sd_inputs[idx], latent_buffer)
        latent_buffer_dev = self.stateful_decoder.get_dev_input(self._sd_inputs[idx])
        idx += 1
        self.stateful_decoder.set_input(self._sd_inputs[idx], conv_history)
        conv_history_dev = self.stateful_decoder.get_dev_input(self._sd_inputs[idx])
        idx = 7

        kv_cache_dev = []
        for cache in kv_cache:
            name = self._sd_inputs[idx]
            self.stateful_decoder.set_input(name, cache)
            kv_cache_dev.append(self.stateful_decoder.get_dev_input(name))
            idx += 1

        self._decoder_state = _StatefulDecoderState(
            pre_conv_history=pre_conv_history_dev,
            latent_buffer=latent_buffer_dev,
            conv_history=conv_history_dev,
            kv_cache=kv_cache_dev,
            kv_valid_len=0,
            skip_samples=0,
            latent_audio=None,
        )

    def _sd_set_state_input(self, name: str, value: Any) -> None:
        if isinstance(value, np.ndarray):
            self.stateful_decoder.set_input(name, value)
        else:
            self.stateful_decoder.set_dev_input(name, value)

    def run_stateful_decoder(self, audio_codes, is_final: bool = False) -> np.ndarray:
        """Decode one codec chunk into audio samples, threading device state."""
        with self.perf.scope("tts.stateful_decoder.infer"):
            return self._run_stateful_decoder(audio_codes, is_final)

    def _prepare_decoder_codes(self, audio_codes, state, is_final: bool) -> tuple[np.ndarray, int]:
        audio_codes = np.asarray(audio_codes, dtype=np.int32)
        if audio_codes.ndim == 1:
            audio_codes = audio_codes.reshape(-1, 16)

        n_frames = audio_codes.shape[0]
        if n_frames == 0:
            if is_final and state.latent_audio is not None:
                audio = state.latent_audio.astype(np.float32)
                state.latent_audio = None
                return audio, 0
            return np.array([], dtype=np.float32), 0

        if n_frames > self.chunk_size:
            raise ValueError(f"expected at most {self.chunk_size} codec frames, got {n_frames}")
        if not is_final and n_frames < self.chunk_size:
            raise ValueError(
                f"non-final chunks must contain exactly {self.chunk_size} codec frames, " f"got {n_frames}"
            )

        if n_frames < self.chunk_size:
            pad = np.zeros((self.chunk_size - n_frames, 16), dtype=np.int32)
            audio_codes = np.concatenate([audio_codes, pad], axis=0)
        return audio_codes, n_frames

    def _set_decoder_inputs(self, audio_codes: np.ndarray, state, is_final: bool, valid_frames: int) -> None:
        audio_codes_input = audio_codes[np.newaxis, ...]
        self._sd_is_last_buf[0] = 1.0 if is_final else 0.0
        self._sd_kv_valid_len_buf[0] = state.kv_valid_len
        self._sd_valid_frames_buf[0] = valid_frames

        idx = 0
        self.stateful_decoder.set_input(self._sd_inputs[idx], audio_codes_input)
        idx += 1
        for value in (state.pre_conv_history, state.latent_buffer, state.conv_history):
            self._sd_set_state_input(self._sd_inputs[idx], value)
            idx += 1
        for value in (
            self._sd_is_last_buf,
            self._sd_kv_valid_len_buf,
            self._sd_valid_frames_buf,
        ):
            self.stateful_decoder.set_input(self._sd_inputs[idx], value)
            idx += 1
        for cache in state.kv_cache:
            self._sd_set_state_input(self._sd_inputs[idx], cache)
            idx += 1

    def _read_decoder_state(self, state, valid_frames: int) -> tuple[np.ndarray, int, _StatefulDecoderState]:
        final_wav = self.stateful_decoder.get_dev_output(self._sd_outputs[0]).to_host().numpy()
        valid_samples = int(self.stateful_decoder.get_dev_output(self._sd_outputs[1]).to_host().numpy().reshape(-1)[0])
        new_state = _StatefulDecoderState(
            pre_conv_history=self.stateful_decoder.get_dev_output(self._sd_outputs[2]),
            latent_buffer=self.stateful_decoder.get_dev_output(self._sd_outputs[3]),
            conv_history=self.stateful_decoder.get_dev_output(self._sd_outputs[4]),
            kv_cache=[],
            kv_valid_len=min(self._SD_KV_CACHE_WINDOW, state.kv_valid_len + valid_frames),
            skip_samples=0,
            latent_audio=None,
        )
        key_start = 5
        for i in range(self._SD_NUM_LAYERS * 2):
            new_state.kv_cache.append(self.stateful_decoder.get_dev_output(self._sd_outputs[key_start + i]))
        return final_wav, valid_samples, new_state

    def _select_decoder_audio(
        self,
        final_wav: np.ndarray,
        valid_samples: int,
        state,
        new_state,
        is_final: bool,
    ) -> np.ndarray:
        initial_skip = (
            self._SD_INITIAL_OUTPUT_SKIP_FRAMES * self._SD_SAMPLES_PER_FRAME if state.kv_valid_len == 0 else 0
        )
        audio_start = initial_skip
        audio_end = audio_start + valid_samples
        if is_final:
            audio = final_wav[0, audio_start:audio_end] if valid_samples > 0 else np.array([], dtype=np.float32)
        elif valid_samples > 0:
            audio = final_wav[0, audio_start:audio_end]
            new_state.latent_audio = final_wav[0, audio_end:]
        else:
            audio = np.array([], dtype=np.float32)
            new_state.latent_audio = final_wav[0, audio_start:]
        return audio

    def _apply_skip_samples(self, audio: np.ndarray, skip_counter: int) -> tuple[np.ndarray, int]:
        if skip_counter <= 0 or len(audio) == 0:
            return audio, skip_counter
        if len(audio) <= skip_counter:
            return np.array([], dtype=np.float32), skip_counter - len(audio)
        return audio[skip_counter:], 0

    def _run_stateful_decoder(self, audio_codes, is_final: bool) -> np.ndarray:
        state = self._decoder_state
        if state is None:
            raise RuntimeError("decoder state was not created")

        audio_codes, valid_frames = self._prepare_decoder_codes(audio_codes, state, is_final)
        if valid_frames == 0:
            return audio_codes

        self._set_decoder_inputs(audio_codes, state, is_final, valid_frames)
        self.stateful_decoder.run()
        self.stateful_decoder.sync()
        final_wav, valid_samples, new_state = self._read_decoder_state(state, valid_frames)
        audio = self._select_decoder_audio(final_wav, valid_samples, state, new_state, is_final)
        audio, skip_counter = self._apply_skip_samples(audio, state.skip_samples)
        new_state.skip_samples = 4 * self._SD_SAMPLES_PER_FRAME if is_final else skip_counter

        self._decoder_state = new_state
        return audio.astype(np.float32)


__all__ = ["Qwen3TtsModule"]
