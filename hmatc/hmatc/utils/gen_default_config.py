# Copyright 2025 HOUMO AI
#
# File: gen_default_config.py
# Description:
#     Generate default config file for hmatc.
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
from . import logger
from .utils import get_onnx_inputs_info, save_dict_to_yaml


def generate_default_config(onnx_file, config_file):
    if not os.path.exists(onnx_file):
        logger.error("ONNX file does not exist")
        exit(-1)
    inputs_info, _ = get_onnx_inputs_info(onnx_file)
    filename = os.path.basename(onnx_file)
    basename, ext = os.path.splitext(filename)
    default_config = dict(
        model=dict(
            name=basename,
            save_dir="output",
            model_path=onnx_file,
            inputs={
                key: {"shape": inputs_info[key]["shape"], "data_format": None}
                for key in inputs_info
            },
        ),
        quant=dict(
            calib_data=None,
            calib_num=1,
        ),
        build=dict(
            ncore=1,
            opt_level=2,
        ),
    )
    logger.info(f"default config:\n {default_config}")
    save_dict_to_yaml(default_config, config_file)
