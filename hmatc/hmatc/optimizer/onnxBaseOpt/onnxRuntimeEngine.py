#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
* Copyright 2025 HOUMO AI
*
* File: onnxRuntimeEngine.py
* Description:
*   Graph-optimized inference engine packaging.
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
import copy
import os
import random

import numpy as np  # type: ignore
import onnx  # type: ignore
import onnxruntime as rt  # type: ignore

from .onnxConfigController import OnnxCfg
from ..onnxUtils.generalClassUtils import Singleton
from ..onnxUtils.onnxBasicUtils import *

TensorShape = List[int]
TensorShapes = Dict[Optional[str], TensorShape]


class OnnxRuntimeEngine(metaclass=Singleton):
    def __init__(self):
        self.ort_compare_max_threshold = OnnxCfg.get_val("ort_compare_max_threshold", 0.01)
        self.ort_compare_mse_threshold = OnnxCfg.get_val("ort_compare_mse_threshold", 0.01)
        if OnnxCfg.check_exist("custom_lib"):
            self.so_custom = rt.SessionOptions()
            custom_lib_path = OnnxCfg.get_val("custom_lib")
            files = os.listdir(custom_lib_path)
            for f in files:
                if f.endswith('.so'):
                    self.so_custom.register_custom_ops_library(custom_lib_path + '/' + f)

    def reset_custom_lib(self):
        if OnnxCfg.check_exist("custom_lib"):
            self.so_custom = rt.SessionOptions()
            # self.so_custom.register_custom_ops_library(OnnxCfg.get_val("custom_lib"))
            custom_lib_path = OnnxCfg.get_val("custom_lib")
            files = os.listdir(custom_lib_path)
            for f in files:
                if f.endswith('.so'):
                    self.so_custom.register_custom_ops_library(custom_lib_path + '/' + f)

    def ort_infer_shape(self, onnx_model: onnx.ModelProto):
        ori_model_output_names = [output.name for output in onnx_model.graph.output]
        onnx_model_all_output = copy.deepcopy(onnx_model)
        del onnx_model_all_output.graph.value_info[:]
        del onnx_model_all_output.graph.output[:]
        shaped_onnx_model = copy.deepcopy(onnx_model_all_output)
        for node in onnx_model_all_output.graph.node:
            onnx_model_all_output.graph.output.extend([onnx.ValueInfoProto(name=output) for output in node.output])

        ort_outs = self.ort_run(onnx_model_all_output)
        for name, array in ort_outs.items():
            value_info = onnx.helper.make_tensor_value_info(name, onnx.mapping.NP_TYPE_TO_TENSOR_TYPE[array.dtype],
                                                            array.shape)
            if name not in ori_model_output_names:
                shaped_onnx_model.graph.value_info.append(value_info)
            else:
                shaped_onnx_model.graph.output.append(value_info)
        return shaped_onnx_model

    def ort_run(self, onnx_model: onnx.ModelProto, input_scale: float = 1.0):
        '''
        msg: forward the model to get all constant tensors' value
        '''
        # patch for model with TopK node
        new_input_array = {}
        indices_nodes_list = [node for node in onnx_model.graph.node if node.op_type in ["TopK", "ArgMax"]]
        if input_scale != 1.0 and indices_nodes_list:
            onnx_model = copy.deepcopy(onnx_model)
            new_input_names, new_output_names = [], []
            for indices_node in indices_nodes_list:
                if indices_node.op_type == "TopK":
                    new_input_names.append(indices_node.output[1])
                    new_output_names.append(indices_node.input[0])
                elif indices_node.op_type == "ArgMax":
                    new_input_names.append(indices_node.output[0])
                    new_output_names.append(indices_node.input[0])
                if new_output_names:
                    input_shape = get_shape_by_name(onnx_model, new_output_names[-1])
                    np.random.seed(1)
                    new_input_array[new_input_names[-1]] = np.random.randint(0, input_shape[1], input_shape)
            model_output_names = [output.name for output in onnx_model.graph.output]
            for output_name in new_output_names:
                if output_name not in model_output_names:
                    onnx_model.graph.output.extend([onnx.ValueInfoProto(name=output_name)])
            for input_name in new_input_names:
                new_name = f"{input_name}_opt_input"
                onnx_model.graph.input.append(onnx.helper.make_tensor_value_info(name=new_name,
                                                                                 elem_type=TensorProto.INT64,
                                                                                 shape=get_shape_by_name(
                                                                                     onnx_model, input_name)))
                replace_input_of_all_nodes(onnx_model, input_name, new_name)

        providers = OnnxCfg.get_val('provider', ['CPUExecutionProvider'])

        if OnnxCfg.check_exist("custom_lib"):
            # Model loading successfully indicates that the custom op node could be resolved successfully
            ort_session = rt.InferenceSession(onnx_model.SerializeToString(), self.so_custom, providers=providers)
        else:
            ort_session = rt.InferenceSession(onnx_model.SerializeToString(), providers=providers)
        ort_inputs = {}
        for input_id, input_ in enumerate(ort_session.get_inputs()):
            net_input = onnx_model.graph.input[input_id]
            tensor_type = net_input.type.tensor_type
            input_shape = []
            for d in tensor_type.shape.dim:
                if d.HasField("dim_value"):
                    input_shape.append(d.dim_value)
                elif d.HasField("dim_param"):
                    dim = 1 if isinstance(d.dim_param, str) else int(d.dim_param)
                    input_shape.append(dim)
                    d.dim_value = dim
                else:
                    assert 0, f"unknown {input_.name} dimension"
            np.random.seed(1)
            if tensor_type.elem_type == TensorProto.BOOL:
                array = np.random.randint(0, 2, input_shape)
            elif input_.name in new_input_array.keys():
                array = new_input_array[input_.name][:np.prod(input_shape)]
            else:
                if "ori_input_type" in OnnxCfg.cfg.keys() and input_.name in OnnxCfg.cfg["ori_input_type"].keys():
                    ori_elem_type = OnnxCfg.cfg["ori_input_type"][input_.name]
                    if ori_elem_type == TensorProto.BOOL:
                        array = np.random.randint(0, 2, input_shape)
                    elif ori_elem_type in [TensorProto.INT32, TensorProto.INT64]:
                        array = np.random.randint(0, input_shape[1], input_shape)
                else:
                    array = np.random.random(input_shape) * input_scale
            array = array.astype(onnx.mapping.TENSOR_TYPE_TO_NP_TYPE[tensor_type.elem_type])
            ort_inputs[input_.name] = array
        outputs = [x.name for x in ort_session.get_outputs()]
        ort_outs = ort_session.run(outputs, ort_inputs)
        return OrderedDict(zip(outputs, ort_outs))

    def ort_compare_result(self, model_old: onnx.ModelProto, model_new: onnx.ModelProto, input_scale: float = 1.0):
        out_old = self.ort_run(model_old, input_scale)
        out_new = self.ort_run(model_new, input_scale)
        for k in out_old.keys():
            _o_tensor = np.reshape(out_old[k], -1)
            _n_tensor = np.reshape(out_new[k], -1)
            mse = np.mean((_o_tensor - _n_tensor) * (_o_tensor - _n_tensor))
            abs_max = np.max(abs(_o_tensor - _n_tensor))
            mean_ref = np.mean(abs(_o_tensor)) / 2 ** 10
            max_ref = np.max(abs(_o_tensor)) / 2 ** 16
            # if abs_max>self.ort_compare_max_threshold and mse>self.ort_compare_mse_threshold or np.isnan(abs_max):
            cmp = np.isnan(abs_max)
            cmp |= abs_max > mean_ref and abs_max > max_ref
            cmp &= not (mean_ref == 0 and max_ref == 0 and abs_max < 1e-3)
            if cmp and np.max(abs(_o_tensor)) <= 1.0:
                cmp &= not ((abs_max < 1e-5) or (abs_max / 10 < max_ref) or (abs_max / 10 < mean_ref))
            if cmp:
                logger.warning(
                    f"ONNXRUNTIME COMPARE FAILURE, {k} abs_max:{abs_max} mean_ref:{mean_ref} max_ref:{max_ref}")
                logger.debug(_o_tensor)
                logger.debug(_n_tensor)
                if input_scale == 1.0:
                    logger.warning(f"ONNXRUNTIME COMPARE RESCALE INPUT")  # patch for superglue
                    # return self.ort_compare_result(model_old, model_new, 0.5)
                    status = False
                    for i in range(3):
                        input_scale = round(random.uniform(0.45, 0.55), 2)
                        status |= self.ort_compare_result(model_old, model_new, input_scale)
                    return status
                logger.error(
                    f"ONNXRUNTIME COMPARE FAILURE, {k} abs_max:{abs_max} mean_ref:{mean_ref} max_ref:{max_ref}")
                return False
            logger.info(f"ONNXRUNTIME COMPARE SUCCESS, {k} abs_max:{abs_max} mean_ref:{mean_ref} max_ref:{max_ref}")
        return True

    def ort_check_precision(self, model_old: onnx.ModelProto, model_new: onnx.ModelProto):
        save_path = f'{OnnxCfg.get_val("out_path")}/{OnnxCfg.get_val("model_name")}-opt-debug.onnx'
        try:
            cmp_result = self.ort_compare_result(model_old, model_new)
            if cmp_result:
                return model_new
            else:
                if OnnxCfg.get_val("work_mode") == "product":
                    onnx.save(model_new, save_path)
                    raise ValueError(f'ONNXRUNTIME COMPARE FAILURE')
                return model_old
        except Exception as e:
            logger.warning("[onnxOpt] Failing to execute: ONNXRUNTIME COMPARE")
            logger.warning(e)
            if OnnxCfg.get_val("work_mode") == "product":
                onnx.save(model_new, save_path)
                raise ValueError("[onnxOpt] Failing to execute: ONNXRUNTIME COMPARE")
            return model_old
