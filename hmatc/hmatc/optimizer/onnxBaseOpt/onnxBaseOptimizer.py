#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
* Copyright 2025 HOUMO AI
*
* File: onnxBaseOptimizer.py
* Description:
*   Graph optimization basic optimizer.
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
from functools import wraps
import onnx
import copy
from onnxsim import onnx_simplifier
from typing import List, Dict, Optional

from .onnxConfigController import OnnxCfg
from .onnxOptimizerManager import OnnxOptimizerManager
from ..onnxBaseOpt.onnxBaseFunctions import *
from ..onnxBaseOpt.onnxVersionControlFunctions import *

TensorShape = List[int]
TensorShapes = Dict[Optional[str], TensorShape]

@OnnxOptimizerManager("base_opt")
class OnnxBaseOptimizer(object):

    @classmethod
    def opt(cls, onnx_model):
        '''
        Explanation:
        Execute with ort compare result

        Args:
        :onnx_model input onnx model

        Returns:
        :return: onnx model
        '''
        onnx_model = clean_useless_input(onnx_model)
        onnx_model = delete_useless_input_in_initializer(onnx_model)
        onnx_model = test_infer_onnx(onnx_model)
        onnx_model = cls.upgrade_onnx_version(onnx_model)
        onnx_model = estimate_gops(onnx_model)
        onnx_model = cls.execute_onnx_sim(onnx_model)
        onnx_model = infer_shapes(onnx_model)
        onnxsim_save_path = '{}/{}-sim.onnx'.format(OnnxCfg.get_val("out_path", "./"), OnnxCfg.get_val("model_name", "convert"))
        onnx.save(onnx_model, onnxsim_save_path)
        logger.info(f"save {onnxsim_save_path}...")
        return onnx_model
    
    @classmethod
    def upgrade_onnx_version(cls, onnx_model):
        '''
        Explanation:
        upgrade onnx version, the baseline opset is 18 op

        Args:
        :onnx_model input onnx model

        Returns:
        :return: onnx model
        '''
        ir_version = 8
        opset_version = 18
        ori_model = copy.deepcopy(onnx_model)

        if onnx_model.ir_version < ir_version:
            logger.info(f"MODIFY ONNX IR VERSION:{onnx_model.ir_version} -> {ir_version}")
            onnx_model.ir_version = ir_version
        if onnx_model.opset_import[0].version < opset_version:
            origin_version = onnx_model.opset_import[0].version
            logger.info(f"MODIFY ONNX OPSET VERSION:{origin_version} -> {opset_version}")
            onnx_model = onnx_version_upgrade(ori_model, onnx_model, opset_version)
        return onnx_model
    
    @classmethod
    def execute_onnx_sim(cls, onnx_model):
        logger.info("Execute onnx-simplifier...")
        test_inputs = {}
        for info in onnx_model.graph.input:
            input_name = info.name
            input_shape_proto = info.type.tensor_type.shape
            input_shape = [d.dim_value for d in input_shape_proto.dim]
            if input_shape[0] <= 0:
                input_shape[0] = 1
            test_inputs[input_name] = input_shape
        onnx_model_ori = copy.deepcopy(onnx_model)
        try:
            onnx_model_opt, check = onnx_simplifier.simplify(onnx_model_ori, check_n=3, test_input_shapes=test_inputs)
        except Exception as e:
            logger.info(f"Fail in onnx-simplifier for: {e}")
            logger.info("Return origin model.")
            return onnx_model_ori
        # assert check, "Simplified ONNX model could not be validated."
        if not check:
            logger.info("Simplified model check failed, return origin model.")
            return onnx_model_ori
        logger.info("Simplified model check success.")
        return onnx_model_opt
    
    @staticmethod
    def onnx_opt_traverse_wrapper(func):
        '''
        Explanation:
        The onnx finding traverse is excluded for each function

        Args:
        :func decorator func pointer

        Returns:
        :return: onnx model
        '''

        @wraps(func)
        def standard_opt(*arg, **kwargs):
            restart = True
            activated = False
            work_mode = OnnxDebugger.work_mode
            onnx_model_ori = arg[0] if isinstance(arg[0], onnx.ModelProto) else arg[1]  # for class method
            if work_mode in ["debug", "release"]:
                onnx_model = copy.deepcopy(onnx_model_ori)
            else:
                onnx_model = onnx_model_ori

            while restart:
                restart = False
                for node_index, node in enumerate(onnx_model.graph.node):
                    arg_new = (onnx_model, node, node_index, *arg[1:]) if isinstance(arg[0], onnx.ModelProto) \
                        else (arg[0], onnx_model, node, node_index, *arg[2:])  # for class method
                    kwargs_new = {}
                    if work_mode in ["release", "product"]:
                        try:
                            onnx_model, restart = func(*arg_new, **kwargs_new)
                        except Exception as e:
                            logger.error(f"Failing to execute:{func.__name__}")
                            logger.error(e)
                            if work_mode == "product":
                                raise ValueError(f"Failing to execute:{func.__name__}")
                            onnx_model = copy.deepcopy(onnx_model_ori)
                    else:
                        onnx_model, restart = func(*arg_new, **kwargs_new)
                    if restart:
                        activated = True
                        if work_mode == "debug":
                            onnx_model = infer_shapes(onnx_model)
                            onnx_model = OnnxRuntimeEngine().ort_check_precision(onnx_model_ori, onnx_model)
                            if onnx_model_ori == onnx_model:
                                activated = False
                            else:
                                onnx_model_ori = copy.deepcopy(onnx_model)
                        if activated:
                            break

            if activated:
                if work_mode == "debug":
                    delete_useless_input_in_initializer(onnx_model)
                elif work_mode == "release":
                    delete_useless_input_in_initializer(onnx_model)
                    onnx_model = infer_shapes(onnx_model)
                    onnx_model = OnnxRuntimeEngine().ort_check_precision(onnx_model_ori, onnx_model)
                elif work_mode == "product":
                    onnx_model = infer_shapes(onnx_model)
                return onnx_model, True
            else:
                return onnx_model, False

        return standard_opt

    @staticmethod
    def onnx_opt_once_wrapper(func):
        '''
        Explanation:
        The onnx opt once for each function

        Args:
        :func decorator func pointer

        Returns:
        :return: onnx model
        '''

        @wraps(func)
        def standard_opt(*arg, **kwargs):
            restart = False
            activated = False
            work_mode = OnnxDebugger.work_mode
            onnx_model_ori = arg[0] if isinstance(arg[0], onnx.ModelProto) else arg[1]  # for class method
            onnx_model = copy.deepcopy(onnx_model_ori)
            onnx_model_debug = copy.deepcopy(onnx_model_ori)

            for ori_node in onnx_model_ori.graph.node:
                if ori_node.output[0] in [node.output[0] for node in onnx_model.graph.node]:
                    node = get_node_by_output(onnx_model, ori_node.output[0])
                else:
                    continue
                arg_new = (onnx_model, node, *arg[1:]) if isinstance(arg[0], onnx.ModelProto) \
                    else (arg[0], onnx_model, node, *arg[2:])  # for class method
                kwargs_new = {}
                if work_mode in ["release", "product"]:
                    try:
                        onnx_model, restart = func(*arg_new, **kwargs_new)
                    except Exception as e:
                        logger.error(f"Failing to execute:{func.__name__}")
                        logger.error(e)
                        if work_mode == "product":
                            raise ValueError(f"Failing to execute:{func.__name__}")
                        onnx_model = copy.deepcopy(onnx_model_ori)
                else:
                    onnx_model, restart = func(*arg_new, **kwargs_new)
                if restart:
                    activated = True
                    if work_mode == "debug":
                        onnx_model = infer_shapes(onnx_model)
                        onnx_model = OnnxRuntimeEngine().ort_check_precision(onnx_model_debug, onnx_model)
                        if onnx_model_debug == onnx_model:
                            activated = False
                        else:
                            onnx_model_debug = copy.deepcopy(onnx_model)

            if activated:
                if work_mode == "release":
                    onnx_model = OnnxRuntimeEngine().ort_check_precision(onnx_model_ori, onnx_model)
                return onnx_model, True
            else:
                return onnx_model, False

        return standard_opt
    




