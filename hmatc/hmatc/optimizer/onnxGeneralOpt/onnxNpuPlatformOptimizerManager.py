#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
* Copyright 2025 HOUMO AI
*
* File: onnxNpuPlatformOptimizerManager.py
* Description:
*   Chip Platform Manager.
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
class NpuOptimizerManager:
    NpuOptimizerDict = {}

    def __init__(self, npu_platform):
        self.npu_platform = npu_platform

    def __call__(self, npu_optimizer):
        NpuOptimizerManager.NpuOptimizerDict[self.npu_platform] = npu_optimizer

    @classmethod
    def get_npu_optimizer(cls, npu_platform):
        return cls.NpuOptimizerDict[npu_platform]