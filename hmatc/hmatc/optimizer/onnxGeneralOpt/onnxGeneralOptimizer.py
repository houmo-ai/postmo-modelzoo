#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
* Copyright 2025 HOUMO AI
*
* File: onnxGeneralOptimizer.py
* Description:
*   General optimization entry.
*
* Licensed under the Apache License, Version 2.0 (the "License");
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at
*
*     https://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.
*
* SPDX-License-Identifier: Apache-2.0
*
"""
from ..onnxBaseOpt.onnxOptimizerManager import OnnxOptimizerManager
from .onnxNpuPlatformOptimizerManager import NpuOptimizerManager
from ..onnxBaseOpt.onnxConfigController import OnnxCfg

from ...utils import logger

@OnnxOptimizerManager("general_opt")
class OnnxGeneralOptimizer(object):

    @classmethod
    def opt(cls, onnx_model):
        platform = OnnxCfg.get_val("platform", None)
        if platform is None:
            logger.warning(f"Not support platform {platform}!")
            return onnx_model
        domain = f"houmo.{platform}"
        optimizer = NpuOptimizerManager.get_npu_optimizer(domain)
        onnx_model = optimizer.opt(onnx_model)
        return onnx_model