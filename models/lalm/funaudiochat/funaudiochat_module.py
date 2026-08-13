# Copyright (c) 2026 HOUMO AI
#
# File: funaudiochat_module.py
# Description:
#   HMM graphs and device cache ownership for Fun-Audio-Chat.
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

"""Own Fun-Audio-Chat HMM graphs, runtime execution, and device caches."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import tcim_lite as tcim
import torch

from houmo_engine import HoumoModule
from houmo_engine.core.types import Stage, StageInputs, StageOutputs
from houmo_engine.perf import PerfTracker

from funaudiochat_types import FunAudioChatPaths


def _names(model, kind: str) -> list[str]:
    """Return all runtime input or output names for an HMM model."""
    if kind == "input":
        return [model.get_input_name(index) for index in range(model.get_num_inputs())]
    return [model.get_output_name(index) for index in range(model.get_num_outputs())]


class FunAudioChatModule(HoumoModule):
    """Load and execute the model graphs used by the Fun-Audio-Chat engine."""

    def __init__(
        self,
        paths: FunAudioChatPaths,
        *,
        device: int,
        ndevice: int,
        load_s2s: bool,
        load_token2wav: bool,
        load_vad: bool,
        perf: PerfTracker,
    ):
        self.perf = perf
        self._stage_metadata = {}
        self._stage_models = {}
        self.load(paths, device=device, ndevice=ndevice, load_s2s=load_s2s, load_token2wav=load_token2wav, load_vad=load_vad)

    @staticmethod
    def _require(path: Path, label: str) -> Path:
        value = Path(path).expanduser().resolve()
        if not value.is_file():
            raise FileNotFoundError(f"missing {label}: {value}")
        return value

    def _load(self, path, label: str, option=None):
        with self.perf.scope(f"lalm.init.load_{label}"):
            return tcim.runtime.load(str(self._require(path, label)), option=option or tcim.runtime.Option(self.weight_manager))

    def load(self, paths: FunAudioChatPaths, *, device: int, ndevice: int, load_s2s: bool, load_token2wav: bool, load_vad: bool) -> None:
        if ndevice <= 0:
            raise ValueError("ndevice must be greater than zero")
        devices = list(range(device, device + ndevice))
        manager = tcim.runtime.DevManager(devices, "Xh2HalBackend")
        self.weight_manager = tcim.runtime.WeightManager(manager)
        self.audio_encoder = self._load(paths.audio_encoder_path, "audio_encoder")
        self.prefill = self._load(paths.prefill_path, "prefill")
        self.llm_cache_names = [name for name in _names(self.prefill, "input") if "model_layers" in name]
        decode_option = tcim.runtime.Option(self.weight_manager)
        decode_option.set_dummy_tensors(self.llm_cache_names)
        self.decode = self._load(paths.decode_path, "decode", decode_option)
        self._bind_llm_caches()
        prefill_shape = tuple(self.prefill.get_input_info("input_1").shape)
        self.prefill_length = int(prefill_shape[1])
        self.hidden_size = int(prefill_shape[2])
        self.audio_encoder_shapes = {name: tuple(int(value) for value in self.audio_encoder.get_input_info(name).shape) for name in _names(self.audio_encoder, "input")}
        self._stage_models.update({Stage.ENCODE: (self.audio_encoder, "lalm.audio_encoder"), Stage.PREFILL: (self.prefill, "lalm.prefill"), Stage.DECODE: (self.decode, "lalm.decode")})

        self.audio_tower = self.crq_prefill = self.crq_decode = None
        if load_s2s:
            self.audio_tower = self._load(paths.audio_tower_path, "audio_tower")
            self.crq_prefill = self._load(paths.audio_decoder_prefill_path, "audio_decoder_prefill")
            self.crq_cache_names = [name for name in _names(self.crq_prefill, "input") if "cache" in name]
            option = tcim.runtime.Option(self.weight_manager)
            option.set_dummy_tensors(self.crq_cache_names)
            self.crq_decode = self._load(paths.audio_decoder_decode_path, "audio_decoder_decode", option)
            self._bind_crq_caches()
            self.crq_context_length = int(self.crq_decode.get_input_info("attention_mask").shape[-1])

        self.flow_encoder = self.flow_spk = self.flow_decoder = None
        self.hift_part1 = self.hift_part2 = None
        self.flow_token_capacity = self.flow_mel_capacity = self.hift_mel_capacity = 0
        if load_token2wav:
            self.flow_encoder = self._load(paths.flow_encoder_path, "flow_encoder")
            self.flow_spk = self._load(paths.flow_spk_path, "flow_spk")
            self.flow_decoder = self._load(paths.flow_decoder_path, "flow_decoder")
            self.hift_part1 = self._load(paths.hift_part1_path, "hift_part1")
            hift_name = self.hift_part1.get_input_name(0)
            option = tcim.runtime.Option(self.weight_manager)
            option.set_dummy_tensors([hift_name])
            self.hift_part2 = self._load(paths.hift_part2_path, "hift_part2", option)
            self.hift_part2.set_input(hift_name, self.hift_part1.get_dev_input(hift_name))
            self.flow_token_capacity = int(self.flow_encoder.get_input_info(self.flow_encoder.get_input_name(0)).shape[1])
            self.flow_mel_capacity = int(self.flow_decoder.get_input_info(self.flow_decoder.get_input_name(0)).shape[2])
            self.hift_mel_capacity = int(self.hift_part1.get_input_info(hift_name).shape[2])

        self.vad = None
        if load_vad:
            self.vad = self._load(paths.vad_path, "vad")

    def _bind_llm_caches(self) -> None:
        decode_inputs = set(_names(self.decode, "input"))
        for name in self.llm_cache_names:
            if name not in decode_inputs:
                raise RuntimeError(f"decode HMM is missing cache input {name}")
            cache = self.prefill.get_dev_input(name)
            self.decode.set_dev_input(name, cache)
            cache.set_zero()

    def _bind_crq_caches(self) -> None:
        decode_inputs = set(_names(self.crq_decode, "input"))
        for name in self.crq_cache_names:
            if name not in decode_inputs:
                raise RuntimeError(f"audio decoder decode is missing cache {name}")
            cache = self.crq_prefill.get_dev_input(name)
            self.crq_decode.set_dev_input(name, cache)
            cache.set_zero()

    def clear_language_session(self) -> None:
        self._bind_llm_caches()

    def clear_crq_session(self) -> None:
        if self.crq_prefill is not None:
            self._bind_crq_caches()

    @staticmethod
    def _set(model, name: str, value) -> None:
        if isinstance(value, torch.Tensor):
            value = value.detach().cpu().numpy()
        value = np.asarray(value)
        info = model.get_input_info(name)
        shape = tuple(int(item) for item in info.shape)
        if value.shape != shape:
            if value.size != int(np.prod(shape)):
                raise ValueError(f"input {name!r} expects {shape}, got {value.shape}")
            value = value.reshape(shape)
        model.set_input(name, np.ascontiguousarray(value, dtype=np.dtype(info.dtype)))

    def set_input(self, stage: Stage, inputs: StageInputs) -> None:
        if stage not in self._stage_models:
            raise ValueError(f"unsupported stage: {stage}")
        model, path = self._stage_models[stage]
        self._stage_metadata[stage] = dict(inputs.metadata)
        names = _names(model, "input")
        with self.perf.scope(f"{path}.set_input"):
            if stage == Stage.ENCODE:
                expected = ["speech_ids", "audio_inputs_embeds", "padded_input_features", "chunk_padded_mask", "aftercnn_valid_mask", "audio_attention_mask", "continuous_audio_valid_mask"]
                if set(names) != set(expected):
                    raise RuntimeError(f"unexpected audio encoder inputs: {names}")
                for name, value in zip(expected, inputs.tensors, strict=True):
                    self._set(model, name, value)
            else:
                for name, value in zip(("input_1", "attention_mask", "valid_length", "current_length"), inputs.tensors, strict=True):
                    self._set(model, name, value)

    def run(self, stage: Stage) -> None:
        model, path = self._stage_models[stage]
        with self.perf.scope(f"{path}.infer"):
            model.run()
            model.sync()

    def get_output(self, stage: Stage) -> StageOutputs:
        model, path = self._stage_models[stage]
        with self.perf.scope(f"{path}.get_output"):
            if stage == Stage.ENCODE:
                tensors = (model.get_output("audio_features").numpy(),)
            elif "last_hidden_state" in _names(model, "output"):
                tensors = (model.get_output("logits").numpy(), model.get_output("last_hidden_state").numpy())
            else:
                tensors = (model.get_output("logits").numpy(),)
        return StageOutputs(tensors=tensors, metadata=self._stage_metadata.pop(stage, {}))

    def run_audio_tower(self, inputs: StageInputs) -> StageOutputs:
        with self.perf.scope("lalm.audio_tower"):
            with self.perf.scope("lalm.audio_tower.set_input"):
                self._set(self.audio_tower, "speech_ids", inputs.tensors[0])
            with self.perf.scope("lalm.audio_tower.infer"):
                self.audio_tower.run()
                self.audio_tower.sync()
            with self.perf.scope("lalm.audio_tower.get_output"):
                output = self.audio_tower.get_output("audio_features").numpy()
        return StageOutputs(tensors=(output,))

    def _run_crq(self, model, inputs: StageInputs, path: str) -> StageOutputs:
        names = ("crq_inputs_embeds", "past_seq_length", "current_input_length", "attention_mask")
        with self.perf.scope(path):
            with self.perf.scope(f"{path}.set_input"):
                for name, value in zip(names, inputs.tensors, strict=True):
                    self._set(model, name, value)
            with self.perf.scope(f"{path}.infer"):
                model.run()
                model.sync()
            with self.perf.scope(f"{path}.get_output"):
                output = model.get_output("speech_logits").numpy()
        return StageOutputs(tensors=(output,))

    def run_crq_prefill(self, inputs: StageInputs) -> StageOutputs:
        return self._run_crq(self.crq_prefill, inputs, "lalm.crq_prefill")

    def run_crq_decode(self, inputs: StageInputs) -> StageOutputs:
        return self._run_crq(self.crq_decode, inputs, "lalm.crq_decode")

    def run_flow_encoder(self, value) -> StageOutputs:
        return self._run_single(self.flow_encoder, value, "lalm.flow_encoder")

    def run_flow_spk(self, value) -> StageOutputs:
        return self._run_single(self.flow_spk, value, "lalm.flow_spk")

    def _run_single(self, model, value, path: str) -> StageOutputs:
        with self.perf.scope(path):
            with self.perf.scope(f"{path}.set_input"):
                self._set(model, model.get_input_name(0), value)
            with self.perf.scope(f"{path}.infer"):
                model.run()
                model.sync()
            with self.perf.scope(f"{path}.get_output"):
                output = model.get_dev_output(model.get_output_name(0)).to_host().numpy()
        return StageOutputs(tensors=(output,))

    def run_flow_decoder(self, values: tuple) -> StageOutputs:
        names = _names(self.flow_decoder, "input")
        if names != ["x", "mask", "mu", "t", "spks", "cond"]:
            raise RuntimeError(f"unexpected Flow decoder inputs: {names}")
        with self.perf.scope("lalm.flow_decoder"):
            with self.perf.scope("lalm.flow_decoder.set_input"):
                for name, value in zip(names, values, strict=True):
                    self._set(self.flow_decoder, name, value)
            with self.perf.scope("lalm.flow_decoder.infer"):
                self.flow_decoder.run()
                self.flow_decoder.sync()
            with self.perf.scope("lalm.flow_decoder.get_output"):
                output = self.flow_decoder.get_dev_output(self.flow_decoder.get_output_name(0)).to_host().numpy()
        return StageOutputs(tensors=(output,))

    def run_hift_part1(self, mel) -> StageOutputs:
        return self._run_single(self.hift_part1, mel, "lalm.hift_part1")

    def run_hift_part2(self, stft) -> StageOutputs:
        name = next(name for name in _names(self.hift_part2, "input") if name == "stft" or "stft" in name.lower())
        with self.perf.scope("lalm.hift_part2"):
            with self.perf.scope("lalm.hift_part2.set_input"):
                self._set(self.hift_part2, name, stft)
            with self.perf.scope("lalm.hift_part2.infer"):
                self.hift_part2.run()
                self.hift_part2.sync()
            with self.perf.scope("lalm.hift_part2.get_output"):
                output = self.hift_part2.get_dev_output(self.hift_part2.get_output_name(0)).to_host().numpy()
        return StageOutputs(tensors=(output,))

    def run_vad(self, features: np.ndarray) -> StageOutputs:
        caches = {f"in_cache{index}": np.zeros((1, 128, 19, 1), dtype=np.float16) for index in range(4)}
        scores = []
        with self.perf.scope("lalm.vad"):
            for offset in range(0, len(features), 256):
                valid = min(256, len(features) - offset)
                speech = np.zeros((1, 256, features.shape[1]), dtype=np.float16)
                speech[0, :valid] = features[offset : offset + valid]
                self._set(self.vad, "speech", speech)
                for name, value in caches.items():
                    self._set(self.vad, name, value)
                with self.perf.scope("lalm.vad.infer"):
                    self.vad.run()
                    self.vad.sync()
                scores.append(self.vad.get_output("logits").numpy()[0, :valid].astype(np.float32))
                for index in range(4):
                    caches[f"in_cache{index}"] = self.vad.get_output(f"out_cache{index}").numpy().copy()
        output = np.concatenate(scores, axis=0) if scores else np.empty((0, 248), dtype=np.float32)
        return StageOutputs(tensors=(output,))

__all__ = ["FunAudioChatModule"]
