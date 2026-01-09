#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
* Copyright 2025 HOUMO AI
*
* File: onnxOptimizerManager.py
* Description:
*   Graph optimization Manager.
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

from typing import List, Dict, Optional
TensorShape = List[int]
TensorShapes = Dict[Optional[str], TensorShape]


class OnnxOptimizerManager(object):
    OPTIMIZER_DICT={}
    def __init__(self,art_onnx_optimizer_method):
        self.current_art_onnx_optimizer_method=art_onnx_optimizer_method
        
    def __call__(self,art_onnx_optimizer_ptr):
        OnnxOptimizerManager.OPTIMIZER_DICT[self.current_art_onnx_optimizer_method]=art_onnx_optimizer_ptr
        return art_onnx_optimizer_ptr

    @staticmethod
    def get(art_onnx_optimizer_method):
        return OnnxOptimizerManager.OPTIMIZER_DICT[art_onnx_optimizer_method]


