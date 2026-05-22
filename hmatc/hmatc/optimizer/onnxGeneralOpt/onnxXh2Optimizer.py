#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
* Copyright 2025 HOUMO AI
*
* File: onnxXh2Optimizer.py
* Description:
*   Registration of optimized functions for xh2.
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

import numpy as np  # type: ignore
import onnxruntime as rt  # type: ignore

from .onnxNpuPlatformOptimizerManager import NpuOptimizerManager
from ..onnxBaseOpt.onnxBaseOptimizer import OnnxBaseOptimizer
from ..onnxBaseOpt.onnxConfigController import OnnxCfg
from ..onnxBaseOpt.onnxBaseFunctions import *
from .onnxGeneralManager.onnxGeneralDeleteFunctions import *
from .onnxGeneralManager.onnxGeneralFusionFunctions import *
from .onnxGeneralManager.onnxGeneralReplaceFunctions import *

from .onnxXh2OptManager.onnxXh2ReplaceFunctions import *
from .onnxXh2OptManager.onnxXh2FusionFunctions import *


TensorShape = List[int]
TensorShapes = Dict[Optional[str], TensorShape]


@NpuOptimizerManager("houmo.xh2")
class OnnxXh2Optimizer(OnnxBaseOptimizer):
    '''
    Explanation: This optimizer takes charge of general onnx optimization functions.
    Author: Nan Xu
    Maintainer: Nan Xu
    '''

    @classmethod
    def opt(cls, onnx_model, *args):
        '''
        :onnx_model input onnx model
        :return: onnx model
        '''
        # loop opt, single loop opt is not enough sometimes
        logger.info("\033[1;32m============== General Optimizer ==============\033[0m")
        onnx_model_ori = copy.deepcopy(onnx_model)
        onnx_model = cls.opt_loop(onnx_model)
        loop_cnt = 1
        while onnx_model_ori != onnx_model:
            onnx_model_ori = copy.deepcopy(onnx_model)
            onnx_model = cls.opt_loop(onnx_model)
            loop_cnt += 1
            if loop_cnt > 10:
                raise AssertionError("General opt loop_cnt exceed 10!!!")
        return onnx_model

    @classmethod
    def opt_loop(cls, onnx_model):
        onnx_model = fusion_SliceSlice(onnx_model)
        onnx_model = replace_GatherUnsqueeze_of_Slice(onnx_model)
        onnx_model = delete_useless_pool(onnx_model)
        onnx_model = replace_MaxPool1D_of_MaxPool2D(onnx_model)
        onnx_model = replace_SqueezeTranspose_of_TransposeReshape(onnx_model)
        onnx_model = replace_TransposeUnsqueeze_of_ReshapeTranspose(onnx_model)
        onnx_model = fusion_TransposeReshapePoolReshapeTranspose(onnx_model)
        onnx_model = fusion_TransposePoolTranspose(onnx_model)
        onnx_model = fusion_ReshapeReshape(onnx_model)
        onnx_model = fusion_TransposeTranspose(onnx_model)
        return onnx_model
