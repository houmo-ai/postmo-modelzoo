#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
* Copyright 2025 HOUMO AI
*
* File: onnxVersionControlFunctions.py
* Description:
*   Version controller for ONNX models.
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
import os
import copy
import numpy as np  # type: ignore
import onnx  # type: ignore
import onnx.helper  # type: ignore
import onnx.shape_inference  # type: ignore
import onnx.numpy_helper  # type: ignore
import onnxruntime as rt  # type: ignore
from onnx import ModelProto, version_converter
from .onnxRuntimeEngine import OnnxRuntimeEngine

from .onnxDebugger import OnnxDebugger
from .onnxBaseFunctions import infer_shapes
from ..onnxUtils.onnxBasicUtils import *
from .onnxConfigController import OnnxCfg


def get_node_id(node):
    id_string = "id_" + node.name
    for n in node.output:
        id_string += n
    return id_string

def remove_initializer_from_input(model: ModelProto) -> ModelProto:
    if model.ir_version < 4:
        logger.info(
            'Model with ir_version below 4 requires to include initializer in '
            'graph input'
        )
        return model

    inputs = model.graph.input
    name_to_input = {}
    for input in inputs:
        name_to_input[input.name] = input

    for initializer in model.graph.initializer:
        if initializer.name in name_to_input:
            inputs.remove(name_to_input[initializer.name])

    return model


def onnx_version_upgrade(ori_model, onnx_model, target_opset):
    onnx_model = pre_convert_fix(onnx_model)
    convert_model = version_converter.convert_version(onnx_model, target_opset)
    convert_model = convert_fix(ori_model, convert_model, target_opset)
    onnx.checker.check_model(convert_model)
    net_path = OnnxCfg.get_val("model_path")
    out_path = os.path.join(OnnxCfg.get_val("out_path"), f"{Path(net_path).stem}_opset{target_opset}.onnx")
    onnx.save(convert_model, out_path)
    if ori_model.graph.output:
        OnnxRuntimeEngine().ort_check_precision(ori_model, convert_model)
    return convert_model


def convert_fix(ori_model, onnx_model, cur_opset):
    """
    fix some convert error op
    """
    def split_fix(onnx_model, node, cur_opset):
        # opset >= 18: add num_outputs attribute
        if not check_node_serial_group(onnx_model, node, ["Split"]):
            return False
        split_node = get_node_serial_group(onnx_model, node, ["Split"])[0]
        new_node_attr = attribute_to_dict(split_node.attribute)
        if cur_opset >= 18 and "num_outputs" not in new_node_attr.keys() and len(split_node.input) < 2:
            logger.debug(f"Split node({split_node.name}): add num_outputs attr for opset{cur_opset}")
            num_outputs = len(split_node.output)
            new_node_attr["num_outputs"] = num_outputs
            del split_node.attribute[:]
            split_node.attribute.extend(onnx.helper.make_attribute(key, value)
                                        for key, value in sorted(new_node_attr.items()) if value is not None)
            return True
        return False

    def dropout_fix(onnx_model, node, cur_opset):
        if check_node_serial_group(onnx_model, node, ["Dropout"]) and cur_opset >= 12:
            if len(node.output) > 1:
                mask_output = node.output[1]
                model_outputs_names = [output.name for output in onnx_model.graph.output]
                if mask_output in model_outputs_names:
                    return False
                next_nodes = get_node_by_input(onnx_model, [mask_output])
                if not next_nodes:
                    node.output.remove(mask_output)
                    value_info = get_value_info_by_name(onnx_model, mask_output)
                    onnx_model.graph.value_info.remove(value_info)
                return True
        return False

    def resize_fix(ori_model, onnx_model, node):
        if node.op_type == "Resize":
            ori_nodes = get_node_by_input(ori_model, [node.input[0]])
            for ori_node in ori_nodes:
                if ori_node.op_type == "Upsample":
                    if node.name == "":
                        node.name = ori_node.name
                        replace_input_of_all_nodes(onnx_model, node.output[0], ori_node.output[0])
                        node.output[0] = ori_node.output[0]
                    resize_attr = attribute_to_dict(node.attribute)
                    if resize_attr.get("mode", "nearest") == "linear" and \
                            "coordinate_transformation_mode" not in resize_attr:
                        new_attr = onnx.helper.make_attribute("coordinate_transformation_mode", "asymmetric")
                        node.attribute.append(new_attr)
                    return True
        return False

    activate = False
    for node_index, node in enumerate(onnx_model.graph.node):
        activate |= split_fix(onnx_model, node, cur_opset)
        activate |= dropout_fix(onnx_model, node, cur_opset)
        activate |= resize_fix(ori_model, onnx_model, node)
    onnx_model = delete_useless_input_in_initializer(onnx_model)
    onnx_model = remove_initializer_from_input(onnx_model)
    if activate:
        return infer_shapes(onnx_model)
    else:
        return onnx_model

def pre_convert_fix(onnx_model):
    """
    fix some convert error op
    """

    def resize_fix(node, cur_opset):
        if node.op_type == "Resize" and cur_opset == 10:
            resize_attr = attribute_to_dict(node.attribute)
            if "coordinate_transformation_mode" not in resize_attr.keys():
                new_attr = onnx.helper.make_attribute("coordinate_transformation_mode", "asymmetric")
                node.attribute.append(new_attr)
                return True
        return False

    cur_opset_version = onnx_model.opset_import[0].version
    activate = False
    for node_index, node in enumerate(onnx_model.graph.node):
        activate |= resize_fix(node, cur_opset_version)
    return onnx_model
