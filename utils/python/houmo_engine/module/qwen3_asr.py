# Copyright (c) 2026 HOUMO AI
#
# File: qwen3_asr.py
# Description:
#   Qwen3-ASR runtime Module implementation.
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

import numpy as np
import tcim_lite as tcim
import torch

from ..core import HoumoModule
from ..core.types import Stage, StageInputs, StageOutputs
from ..perf import PerfTracker


class Qwen3AsrModule(HoumoModule):
    """Qwen3-ASR HMM graphs, cache bindings, and stage execution."""

    def __init__(
        self,
        encode_path,
        prefill_path,
        decode_path,
        *,
        ndevice: int = 1,
        perf: PerfTracker,
    ):
        self.perf = perf
        self._stage_metadata = {}
        self.load(
            encode_path,
            prefill_path,
            decode_path,
            ndevice=ndevice,
        )

    def load(
        self,
        encode_path,
        prefill_path,
        decode_path,
        *,
        ndevice: int = 1,
    ) -> None:
        if ndevice == 1:
            weight_manager = tcim.runtime.WeightManager(0)
        elif ndevice == 2:
            dev_manager = tcim.runtime.DevManager([0, 1], "Xh2HalBackend")
            weight_manager = tcim.runtime.WeightManager(dev_manager)
        else:
            raise ValueError("unsupported device number")

        with self.perf.scope("asr.init.encode_load"):
            self.encode = tcim.runtime.load(str(encode_path), option=tcim.runtime.Option(weight_manager))
        with self.perf.scope("asr.init.prefill_load"):
            self.prefill = tcim.runtime.load(str(prefill_path), option=tcim.runtime.Option(weight_manager))
        decode_option = tcim.runtime.Option(weight_manager)
        decode_option.set_dummy_tensors(
            [
                self.prefill.get_input_name(index)
                for index in range(self.prefill.get_num_inputs())
                if "model_layers" in self.prefill.get_input_name(index)
            ]
        )
        with self.perf.scope("asr.init.decode_load"):
            self.decode = tcim.runtime.load(str(decode_path), option=decode_option)

        prefill_shape = self.prefill.get_input_info(self.prefill.get_input_name(0)).shape
        self.prefill_length = int(prefill_shape[1])
        self.embedding_size = int(prefill_shape[2])
        self.context_max_length = int(self.prefill.get_input_info(self.prefill.get_input_name(3)).shape[2])
        self.encode_feature_length = int(self.encode.get_input_info(self.encode.get_input_name(0)).shape[2])
        self._bind_caches()
        # self.clear_session()

    def _bind_caches(self) -> None:
        for index in range(3, self.prefill.get_num_inputs()):
            name = self.prefill.get_input_name(index)
            if "model_layers" in name:
                self.decode.set_dev_input(
                    self.decode.get_input_name(index),
                    self.prefill.get_dev_input(name),
                )

    @staticmethod
    def _input_shape(model, name: str) -> tuple[int, ...]:
        return tuple(int(value) for value in model.get_input_info(name).shape)

    def _set_input(self, model, name: str, value) -> None:
        value = value.detach().cpu().numpy() if isinstance(value, torch.Tensor) else np.asarray(value)
        shape = self._input_shape(model, name)
        if value.shape != shape:
            if value.size != int(np.prod(shape)):
                raise RuntimeError(f"input {name!r} expects {shape}, got {value.shape}")
            value = value.reshape(shape)
        model.set_input(name, value)

    def clear_session(self) -> None:
        for index in range(3, self.prefill.get_num_inputs()):
            name = self.prefill.get_input_name(index)
            if "model_layers" not in name:
                continue
            info = self.prefill.get_dev_input(name).info
            self._set_input(self.prefill, name, np.zeros(info.shape, dtype=np.float16))

    def _stage_model(self, stage: Stage):
        if stage == Stage.ENCODE:
            return self.encode, "asr.encode"
        elif stage == Stage.PREFILL:
            return self.prefill, "asr.prefill"
        elif stage == Stage.DECODE:
            return self.decode, "asr.decode"
        raise ValueError(f"unsupported stage: {stage}")

    def set_input(self, stage: Stage, inputs: StageInputs) -> None:
        model, path = self._stage_model(stage)
        self._stage_metadata[stage] = dict(inputs.metadata)
        with self.perf.scope(f"{path}.set_input"):
            for index, value in enumerate(inputs.tensors):
                self._set_input(model, model.get_input_name(index), value)

    def run(self, stage: Stage) -> None:
        model, path = self._stage_model(stage)
        with self.perf.scope(f"{path}.infer"):
            model.run()
            model.sync()

    def get_output(self, stage: Stage) -> StageOutputs:
        model, path = self._stage_model(stage)
        with self.perf.scope(f"{path}.get_output"):
            output = model.get_output(model.get_output_name(0)).numpy()
        return StageOutputs(
            tensors=(output,),
            metadata=self._stage_metadata.pop(stage, {}),
        )


__all__ = ["Qwen3AsrModule"]
