#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
* Copyright 2025 HOUMO AI
*
* File: onnxBasicUtils.py
* Description:
*   general basic utils.
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

from collections import OrderedDict
import onnx  # type: ignore
from onnx import TensorProto
import onnx.helper as helper  # type: ignore
import numpy as np  # type: ignore
from typing import List, Dict, Union, Optional, Tuple, Sequence
from ...utils import logger
import json
import re

from .constUtils import *
from .commonUtils import *


def bytes_to_str(s):
    if isinstance(s, bytes):
        return s.decode()
    return s


def attribute_to_dict(attribute: Sequence[onnx.AttributeProto]):
    attr_dict = {}
    for att in attribute:
        value = bytes_to_str(helper.get_attribute_value(att))
        if isinstance(value, list):
            value = [bytes_to_str(item) for item in value]
        attr_dict[att.name] = value
    return attr_dict


def add_model_output_by_outputs(onnx_model: onnx.ModelProto, outputs: List[str]):
    for output in outputs:
        onnx_model.graph.output.extend([onnx.ValueInfoProto(name=output)])
    return onnx_model


def delete_model_output_by_outputs(onnx_model: onnx.ModelProto, outputs: List[str]):
    output_remove = []
    for output in onnx_model.graph.output:
        if output.name in outputs:
            output_remove.append(output)
    for i in output_remove:
        onnx_model.graph.output.remove(i)
    return onnx_model


def delete_useless_input_in_initializer(onnx_model: onnx.ModelProto):
    ini_to_keep_list = []
    init_need_remove = []
    input_need_remove = []
    for node in onnx_model.graph.node:
        ini_to_keep_list.extend(node.input)
    for init in onnx_model.graph.initializer:
        if init.name not in ini_to_keep_list and init.name not in init_need_remove:
            init_need_remove.append(init)
    for input_info in onnx_model.graph.input:
        if input_info.name not in ini_to_keep_list:
            input_need_remove.append(input_info)
    for i in init_need_remove:
        onnx_model.graph.initializer.remove(i)
    for i in input_need_remove:
        onnx_model.graph.input.remove(i)
    return onnx_model


def delete_initializer_by_name(onnx_model: onnx.ModelProto, name: str):
    init_need_remove = []
    for init in onnx_model.graph.initializer:
        if init.name == name:
            init_need_remove.append(init)
    for i in init_need_remove:
        onnx_model.graph.initializer.remove(i)
    return onnx_model


def delete_initializer_by_names(onnx_model: onnx.ModelProto, names: list):
    init_need_remove = []
    for init in onnx_model.graph.initializer:
        if init.name in names:
            init_need_remove.append(init)
    for i in init_need_remove:
        onnx_model.graph.initializer.remove(i)
    return onnx_model


def get_node_by_input(onnx_model: onnx.ModelProto, input_list: Sequence[str]):
    nodes = []
    for i in input_list:
        for node in onnx_model.graph.node:
            if i in node.input:
                nodes.append(node)
    return nodes


def check_node_serial_group(onnx_model: onnx.ModelProto, node: onnx.NodeProto, op_patch_list: List[str]):
    for list_index in range(len(op_patch_list)):
        if list_index > 0:
            nodes = get_node_by_input(onnx_model, node.output)
            if len(nodes) != 1:
                return False
            node = nodes[0]
        if node.op_type != op_patch_list[list_index]:
            return False
    return True


def check_node_reverse_serial_group(onnx_model: onnx.ModelProto, node: onnx.NodeProto, op_patch_list: List[str]):
    # op_patch_list in reverse order, one dynamic input
    for list_index in range(len(op_patch_list)):
        if list_index > 0:
            node = get_node_by_output(onnx_model, node.input[0])
            if node is None or len(get_node_by_input(onnx_model, node.output)) != 1:
                return False
        if node.op_type != op_patch_list[list_index]:
            return False
    return True


def get_node_by_output(onnx_model: onnx.ModelProto, name: str):
    for node in onnx_model.graph.node:
        if name in node.output:
            return node
    return None


def replace_node_input(node: onnx.NodeProto, old_input_name: str, new_input_name: str):
    if old_input_name not in node.input:
        return

    if "art" not in node.domain:
        for i, name in enumerate(node.input):
            if name == old_input_name:
                node.input[i] = new_input_name
    else:
        node_attr = attribute_to_dict(node.attribute)
        crop_param = {}
        if "crop_param" in node_attr.keys() and not isinstance(node_attr["crop_param"], list):
            crop_param = json.loads(node_attr["crop_param"])
        input_crop_key = [f"{input_}_{node.input[:i].count(input_)}" for i, input_ in enumerate(node.input)]
        old_crop_key_to_new_crop_key = {}
        for i, (name, crop_key) in enumerate(zip(node.input, input_crop_key)):
            if name == old_input_name:
                node.input[i] = new_input_name
            if crop_key in crop_param:
                new_crop_key = f"{node.input[i]}_{node.input[:i].count(node.input[i])}"
                if crop_key != new_crop_key:
                    old_crop_key_to_new_crop_key[crop_key] = new_crop_key
        if old_crop_key_to_new_crop_key:
            for old_key, new_key in old_crop_key_to_new_crop_key.items():
                crop_param[new_key] = crop_param.pop(old_key)
            crop_param_str = json.dumps(crop_param)
            node_attr["crop_param"] = crop_param_str
            node.ClearField("attribute")
            node.attribute.extend(
                helper.make_attribute(key, value) for key, value in sorted(node_attr.items()) if value is not None)


def replace_input_of_all_nodes(onnx_model: onnx.ModelProto, old_input_name: str, new_input_name: str,
                               excluded_node_names: List[str]=[]):
    for node in onnx_model.graph.node:
        if node.name in excluded_node_names:
            continue
        replace_node_input(node, old_input_name, new_input_name)


def replace_node_output(node: onnx.NodeProto, old_output_name: str, new_output_name: str):
    for i, name in enumerate(node.output):
        if name == old_output_name:
            node.output[i] = new_output_name


def delete_nodes(onnx_model: onnx.ModelProto, nodes_to_remove: List[onnx.NodeProto]):
    for node in nodes_to_remove:
        onnx_model.graph.node.remove(node)
    return onnx_model


def delete_useless_nodes(onnx_model: onnx.ModelProto):
    restart = True
    while restart:
        node_to_move = []
        for node in onnx_model.graph.node:
            if node.op_type == "Constant":
                if len(get_node_by_input(onnx_model, node.output)) == 0:
                    node_to_move.append(node)
            else:
                model_output_names = [output.name for output in onnx_model.graph.output]
                next_nodes = get_node_by_input(onnx_model, node.output)
                node_output_list = [output for output in node.output if output in model_output_names]
                if len(next_nodes) == 0 and len(node_output_list) == 0:
                    node_to_move.append(node)
        for i in node_to_move:
            onnx_model.graph.node.remove(i)
        restart = True if node_to_move else False
    delete_useless_input_in_initializer(onnx_model)
    return onnx_model


def graph_topological_sort(graph: onnx.GraphProto):
    deps_set = set()  # dependency set of all node
    sorted_node_set = set()  # sorted node set
    sorted_nodes = []  # initialize sorted_nodes
    need_sort = False

    initializer_names = [init.name for init in graph.initializer]
    graph_input_names = [input_.name for input_ in graph.input]
    input_names = initializer_names + graph_input_names

    for input_name in input_names:
        deps_set.add(input_name)

    sorted_node_set_len = -1
    last_node_name = None
    while len(sorted_node_set) != len(graph.node):
        if len(sorted_node_set) == sorted_node_set_len:
            break
        sorted_node_set_len = len(sorted_node_set)
        for node_idx, node in enumerate(graph.node):
            if node_idx in sorted_node_set:
                continue
            input_count = sum(1 for _ in node.input if _)
            if input_count == 0:
                sorted_nodes.append(node)
                sorted_node_set.add(node_idx)
                for output in node.output:
                    if output:
                        deps_set.add(output)
                continue
            failed = False
            for input_name in node.input:
                if input_name and input_name not in deps_set:
                    failed = need_sort = True
                    last_node_name = node.name
            if not failed:
                sorted_nodes.append(node)
                sorted_node_set.add(node_idx)
                for output in node.output:
                    if output:
                        deps_set.add(output)
            else:
                continue

    if len(sorted_node_set) != len(graph.node):
        raise RuntimeError(f"Graph is not a DAG: len(sorted_node_set)={len(sorted_node_set)}, "
                           f"len(graph.node)={len(graph.node)}, failed at node {last_node_name}")
    if need_sort:
        graph.ClearField("node")
        graph.node.extend(sorted_nodes)
    return need_sort


def get_sub_graph(onnx_model: onnx.ModelProto, node: onnx.NodeProto, op_patch_template):
    def node_to_dict(node):
        info_dict = {}
        info_dict["op_type"] = node.op_type
        info_dict["input"] = node.input
        info_dict["output"] = node.output
        return info_dict

    def get_template_by_input(template_model, input_list):
        nodes = []
        for i in input_list:
            for node in template_model:
                if i in node["input"]:
                    nodes.append(node)
        return nodes

    def get_template_by_output(template_model, output_list):
        nodes = []
        for i in output_list:
            for node in template_model:
                if i in node["output"]:
                    nodes.append(node)
        return nodes

    return_nodes = []
    searched_template_node_names_list = []

    if node.op_type != op_patch_template[0]["op_type"]:
        return []
    search_node_list = get_node_by_input(onnx_model, node.output)
    search_template_list = get_template_by_input(op_patch_template, op_patch_template[0]["output"])
    searched_template_node_names_list.append(op_patch_template[0]["name"])
    for search_template_node in search_template_list:
        if search_template_node["name"] not in searched_template_node_names_list:
            searched_template_node_names_list.append(search_template_node["name"])
    op_patch_template[0]["node"] = node
    return_nodes.append(op_patch_template[0])
    while len(search_template_list) > 0:
        current_node = search_node_list.pop(0)
        current_template = search_template_list.pop(0)
        if current_node.op_type != current_template["op_type"]:
            return []
        # search by input
        if len(current_template["output"]) > 0:
            search_template_nodes_by_input = get_template_by_input(op_patch_template, current_template["output"])
            search_nodes_by_input = get_node_by_input(onnx_model, current_node.output)
            for template_node_id, template_node in enumerate(search_template_nodes_by_input):
                if template_node["name"] not in searched_template_node_names_list:
                    searched_template_node_names_list.append(template_node["name"])
                    # add node to search_node_list
                    search_node_list.append(search_nodes_by_input[template_node_id])
                    # add node to search_template_list
                    search_template_list.append(template_node)
        # search by output
        if len(current_template["input"]) > 0:
            search_template_nodes_by_output = get_template_by_output(op_patch_template, current_template["input"])
            search_nodes_by_output = []
            for name in current_node.input:
                node_temp = get_node_by_output(onnx_model, name)
                if node_temp is not None:
                    search_nodes_by_output.append(node_temp)
            for template_node_id, template_node in enumerate(search_template_nodes_by_output):
                template_node = search_template_nodes_by_output[template_node_id]
                if template_node["name"] not in searched_template_node_names_list:
                    searched_template_node_names_list.append(template_node["name"])
                    # add node to search_node_list
                    search_node_list.append(search_nodes_by_output[template_node_id])
                    # add node to search_template_list
                    search_template_list.append(template_node)
        # if current_node not in return_nodes:
        current_template["node"] = current_node
    return op_patch_template


def check_sub_graph(onnx_model: onnx.ModelProto, node: onnx.NodeProto, op_patch_template):
    def node_to_dict(node):
        info_dict = {}
        info_dict["op_type"] = node.op_type
        info_dict["input"] = node.input
        info_dict["output"] = node.output
        return info_dict

    def get_template_by_input(template_model, input_list):
        nodes = []
        for i in input_list:
            for node in template_model:
                if i in node["input"]:
                    nodes.append(node)
        return nodes

    def get_template_by_output(template_model, output_list):
        nodes = []
        for i in output_list:
            for node in template_model:
                if i in node["output"]:
                    nodes.append(node)
        return nodes

    search_node_list = []
    search_template_list = []
    searched_template_node_names_list = []

    if node.op_type == op_patch_template[0]["op_type"]:
        search_node_list = get_node_by_input(onnx_model, node.output)
        search_template_list = get_template_by_input(op_patch_template, op_patch_template[0]["output"])

        searched_template_node_names_list.append(op_patch_template[0]["name"])
        for search_template_node in search_template_list:
            if search_template_node["name"] not in searched_template_node_names_list:
                searched_template_node_names_list.append(search_template_node["name"])

        while len(search_template_list) > 0:
            if (len(search_node_list) == 0):
                return False
            current_node = search_node_list[0]
            search_node_list = search_node_list[1:]
            current_template = search_template_list[0]
            search_template_list = search_template_list[1:]
            if (current_node.op_type != current_template["op_type"]):
                return False
            # search by input
            if len(current_template["output"]) > 0:
                search_template_nodes_by_input = get_template_by_input(op_patch_template, current_template["output"])
                search_nodes_by_input = get_node_by_input(onnx_model, current_node.output)
                if len(search_template_nodes_by_input) != len(search_nodes_by_input):
                    return False
                for template_node_id in range(len(search_template_nodes_by_input)):
                    template_node = search_template_nodes_by_input[template_node_id]
                    if template_node["name"] not in searched_template_node_names_list:
                        searched_template_node_names_list.append(template_node["name"])
                        # add node to search_node_list
                        search_node_list.append(search_nodes_by_input[template_node_id])
                        # add node to search_template_list
                        search_template_list.append(template_node)
            # search by output
            if len(current_template["input"]) > 0:
                search_template_nodes_by_output = get_template_by_output(op_patch_template, current_template["input"])
                search_nodes_by_output = []
                for name in current_node.input:
                    node_temp = get_node_by_output(onnx_model, name)
                    if node_temp is not None:
                        search_nodes_by_output.append(node_temp)
                if len(search_template_nodes_by_output) != len(search_nodes_by_output):
                    return False
                for template_node_id in range(len(search_template_nodes_by_output)):
                    template_node = search_template_nodes_by_output[template_node_id]
                    if template_node["name"] not in searched_template_node_names_list:
                        searched_template_node_names_list.append(template_node["name"])
                        # add node to search_node_list
                        search_node_list.append(search_nodes_by_output[template_node_id])
                        # add node to search_template_list
                        search_template_list.append(template_node)

        return True
    return False


def get_tensor_from_initializer(onnx_model: onnx.ModelProto, name: str):
    for init in onnx_model.graph.initializer:
        if init.name == name:
            return onnx.numpy_helper.to_array(init)
    return np.array([])


def get_tensor_from_initializer_with_default(onnx_model: onnx.ModelProto, name: str, default=None):
    for init in onnx_model.graph.initializer:
        if init.name == name:
            return onnx.numpy_helper.to_array(init)
    if default is None:
        return np.array([])
    else:
        return np.array(default)


def get_arr_from_initializer(initializer, name: str, default=None):
    """
    set up: initializer is not Constant node output
    """
    for init in initializer:
        if init.name == name:
            return onnx.numpy_helper.to_array(init)

    if default is None:
        return np.array([])
    else:
        return np.array(default)


def get_tensor_type_by_name(onnx_model: onnx.ModelProto, name: str):
    for init in onnx_model.graph.initializer:
        if init.name == name:
            return init.data_type
    for value_info in onnx_model.graph.value_info:
        if value_info.name == name:
            return value_info.type.tensor_type.elem_type
    for value_info in onnx_model.graph.input:
        if value_info.name == name:
            return value_info.type.tensor_type.elem_type
    for value_info in onnx_model.graph.output:
        if value_info.name == name:
            return value_info.type.tensor_type.elem_type
    return 1


def check_constant(onnx_model: onnx.ModelProto, name: str):
    for init in onnx_model.graph.initializer:
        if init.name == name:
            return True
    return False


def get_node_id(onnx_model: onnx.ModelProto, n_node: onnx.NodeProto):
    for idx, node in enumerate(onnx_model.graph.node):
        if node.name == n_node.name:
            return idx


def get_initializer(onnx_model: onnx.ModelProto, name: str):
    for init in onnx_model.graph.initializer:
        if init.name == name:
            return init
    return None


def get_node_serial_group(onnx_model: onnx.ModelProto, node: onnx.NodeProto, op_patch_list: List[str]):
    node_serial_list = []
    for list_index in range(len(op_patch_list)):
        if list_index > 0:
            nodes = get_node_by_input(onnx_model, node.output)
            node = nodes[0]
            assert len(nodes) == 1
        if node.op_type == op_patch_list[list_index]:
            node_serial_list.append(node)
    return node_serial_list


def get_node_reverse_serial_group(onnx_model: onnx.ModelProto, node: onnx.NodeProto, op_patch_list: List[str]):
    # op_patch_list in reverse order, one dynamic input
    node_serial_list = []
    for list_index in range(len(op_patch_list)):
        if list_index > 0:
            node = get_node_by_output(onnx_model, node.input[0])
        if node.op_type == op_patch_list[list_index]:
            node_serial_list.append(node)
    return node_serial_list


def create_conv_node(name: str, inputs: Sequence[str], outputs: Sequence[str], kernel_shape=[1, 1], group=1,
                     dilations=[1, 1], strides=[1, 1], pads=[0, 0, 0, 0]):
    attribute_dict = {'dilations': dilations, 'group': group,
                      'kernel_shape': kernel_shape, 'pads': pads,
                      'strides': strides}
    conv_node = onnx.helper.make_node(op_type="Conv", name=name, inputs=inputs, outputs=outputs, **attribute_dict)
    return conv_node


def create_maxpool_node(name: str, inputs: Sequence[str], outputs: Sequence[str], kernel_shape=[1, 1], ceil_mode=0,
                        dilations=[1, 1], strides=[1, 1], pads=[0, 0, 0, 0]):
    attribute_dict = {'dilations': dilations, 'ceil_mode': ceil_mode,
                      'kernel_shape': kernel_shape, 'pads': pads,
                      'strides': strides}
    maxpool_node = onnx.helper.make_node(op_type="MaxPool", name=name, inputs=inputs, outputs=outputs, **attribute_dict)
    return maxpool_node


def create_convTranspose_node(name: str, inputs: Sequence[str], outputs: Sequence[str], auto_pad="NOTSET",
                              dilations=[1, 1], kernel_shape=[1, 1], group=1, strides=[1, 1], pads=[0, 0, 0, 0],
                              output_padding=[0, 0]):
    attribute_dict = {'dilations': dilations, 'group': group,
                      'kernel_shape': kernel_shape, 'pads': pads,
                      'strides': strides, "auto_pad": auto_pad,
                      "output_padding": output_padding}
    deconv_node = onnx.helper.make_node(op_type="ConvTranspose", name=name, inputs=inputs, outputs=outputs,
                                        **attribute_dict)
    return deconv_node


def get_attribute_value(attribute_dict: dict, key, default_value=None):
    return attribute_dict.get(key, default_value)


def get_bn_params_in_constant(onnx_model: onnx.ModelProto, node: onnx.NodeProto):
    if not node.op_type == "BatchNormalization":
        logger.error("" + node.name + " is not BatchNormalization")
        return "Not BatchNormalization"
    bn_scale = get_tensor_from_initializer(onnx_model, node.input[1])
    bn_B = get_tensor_from_initializer(onnx_model, node.input[2])
    bn_mean = get_tensor_from_initializer(onnx_model, node.input[3])
    bn_var = get_tensor_from_initializer(onnx_model, node.input[4])
    return bn_scale, bn_B, bn_mean, bn_var


def get_conv_params_in_constant(onnx_model: onnx.ModelProto, node: onnx.NodeProto):
    if not node.op_type == "Conv" and not node.op_type == "ConvTranspose":
        logger.error("" + node.name + " Not Conv or ConvTranspose")
        return "Not Conv"

    conv_W = get_tensor_from_initializer(onnx_model, node.input[1])
    if len(node.input) == 2:
        conv_B = None
    else:
        conv_B = get_tensor_from_initializer(onnx_model, node.input[2])
    return conv_W, conv_B


def check(onnx_model: onnx.ModelProto):
    onnx.checker.check_model(onnx_model)


def save_model(onnx_model: onnx.ModelProto, out_path):
    # onnx.checker.check_model(onnx_model)
    onnx.save(onnx_model, out_path)


def get_value_info_by_name(onnx_model: onnx.ModelProto, name: str):
    for input_ in onnx_model.graph.input:
        if input_.name == name:
            return input_
    for output in onnx_model.graph.output:
        if output.name == name:
            return output
    for value_info in onnx_model.graph.value_info:
        if value_info.name == name:
            return value_info
    return None


def get_shape_by_name(onnx_model: onnx.ModelProto, name: str):
    # search
    value_info = get_value_info_by_name(onnx_model, name)
    if value_info is not None:
        shape = [d.dim_value if d.dim_value > 0 else 1 for d in value_info.type.tensor_type.shape.dim]
        return shape
    tensor = get_tensor_from_initializer(onnx_model, name)
    shape = list(tensor.shape)
    return shape


def get_slice_param(onnx_model: onnx.ModelProto, node: onnx.NodeProto):
    start = get_tensor_from_initializer(onnx_model, node.input[1])
    end = get_tensor_from_initializer(onnx_model, node.input[2])
    axis = get_tensor_from_initializer(onnx_model, node.input[3])
    step = get_tensor_from_initializer(onnx_model, node.input[4])
    return start, end, axis, step


def refine_reshape_tensor(input_shape, reshape_shape, allow_zero=0):
    if allow_zero != 0:
        return reshape_shape
    input_shape = list(input_shape)
    reshape_shape = list(reshape_shape)
    for axis, shape in enumerate(reshape_shape):
        if shape == 0:
            reshape_shape[axis] = input_shape[axis]
    output_shape = []
    for axis, shape in enumerate(reshape_shape):
        if shape >= 0:
            output_shape.append(shape)
        elif shape == -1:
            total_size = int(np.prod(input_shape))
            remain_size = int(np.prod(reshape_shape))
            output_shape.append(-total_size // remain_size)
        else:
            raise ValueError(f"Invalid dimension value: {shape}")
    return output_shape


def opset_import_extend(onnx_model, domain, version=1):
    for opset_import in onnx_model.opset_import:
        if opset_import.domain == domain:
            opset_import.version = version
            return
    onnx_model.opset_import.append(onnx.helper.make_opsetid(domain, version))


def get_all_match_nodes_lists(onnx_model: onnx.ModelProto, op_patch_lists: List[List[str]]):
    all_match_nodes_list = []
    for node in onnx_model.graph.node:
        for op_patch_list in op_patch_lists:
            nodes_list = []
            if isinstance(op_patch_list[0], str) and check_node_serial_group(onnx_model, node, op_patch_list):
                nodes_list = get_node_serial_group(onnx_model, node, op_patch_list)
            elif isinstance(op_patch_list[0], dict) and check_sub_graph(onnx_model, node, op_patch_list):
                sub_graph = get_sub_graph(onnx_model, node, op_patch_list)
                nodes_list = [_node["node"] for _node in sub_graph]
            if nodes_list and nodes_list not in all_match_nodes_list:
                all_match_nodes_list.append(nodes_list)
    return all_match_nodes_list


def setName(onnx_model, name):
    str_pattern = re.compile(r".*[0-9]$")
    node_names = [n.name for n in onnx_model.graph.node]
    count = 0
    while name in node_names:
        if "_" in name and re.match(str_pattern, name[name.rindex("_") + 1:]):
            suffix_id = name.rindex("_") + 1
            base_name = name[:suffix_id]
            for node_name in node_names:
                node_names.remove(node_name)
                if node_name.find(base_name) == 0:
                    matchStr = node_name[node_name.index(base_name) + len(base_name):]
                    matchObj = all(char.isdigit() for char in matchStr)
                    count = max(count, int(matchStr) + 1) if matchObj else count
                    # matchObj = re.match(str_pattern, node_name[node_name.index(base_name) + len(base_name):])
                    # count = max(count, int(matchObj.group()) + 1) if matchObj else count
                    break
            name = f"{base_name}{count}"
        else:
            name = f"{name}_{count}"
    return name


def set_node_attribute(target_node, attr_name, attr_value):
    r'''
    :param target_node: target node
    :param attr_name: attribute name
    :param attr_value: new attribute value
    :return: flag
    '''
    flag = False
    for attr in target_node.attribute:
        if (attr.name == attr_name):
            if attr.type == 1:
                attr.f = attr_value
            elif attr.type == 2:
                attr.i = attr_value
            elif attr.type == 3:
                attr.s = attr_value
            elif attr.type == 4:
                attr.t = attr_value
            elif attr.type == 5:
                attr.g = attr_value
            # NOTE: For repeated composite types, we should use something like
            # del attr.xxx[:]
            # attr.xxx.extend([n1, n2, n3])
            elif attr.type == 6:
                attr.floats[:] = attr_value
            elif attr.type == 7:
                attr.ints[:] = attr_value
            elif attr.type == 8:
                attr.strings[:] = attr_value
            else:
                print("unsupported attribute data type with attribute name")
                return False
            flag = True
    if not flag:
        # attribute not in original node
        print("Warning: you are appending a new attribute to the node!")
        target_node.attribute.append(helper.make_attribute(attr_name, attr_value))
        flag = True
    return flag


def set_tensor_array(model: onnx.ModelProto, weight: str, data_numpy: np.ndarray):

    weight_tensor = None
    for tensor in model.graph.initializer:
        if tensor.name == weight:
            weight_tensor = tensor
    if weight_tensor:
        raw_shape = tuple([i for i in weight_tensor.dims])
        new_shape = np.shape(data_numpy)
        if weight_tensor.data_type == 8:
            raise ValueError("Can NOT handle string data type right now...")
        if new_shape != raw_shape:
            weight_tensor.dims[:] = list(new_shape)
            for model_input in model.graph.input:
                if model_input.name == weight_tensor.name:
                    tensor_shape_proto = model_input.type.tensor_type.shape
                    tensor_shape_proto.ClearField("dim")
                    tensor_shape_proto.dim.extend([])
                    for d in new_shape:
                        dim = tensor_shape_proto.dim.add()
                        dim.dim_value = d
        weight_tensor.ClearField("float_data")
        weight_tensor.ClearField("int32_data")
        weight_tensor.ClearField("int64_data")
        weight_tensor.raw_data = data_numpy.tobytes()


def get_node_dynamic_input_list(model: onnx.ModelProto, node: onnx.NodeProto):
    dynamic_input_list = []
    for i in node.input:
        if not check_constant(model, i):
            dynamic_input_list.append(i)
    return dynamic_input_list


def set_value_info_shape(onnx_model: onnx.ModelProto, name: str, shape: list):
    value = get_value_info_by_name(onnx_model, name)
    if value:
        shape_proto = value.type.tensor_type.shape
        shape_proto.ClearField('dim')
        shape_proto.dim.extend([])
        for d in shape:
            dim = shape_proto.dim.add()
            dim.dim_value = d
    else:
        assert False, f"Cannot find {name} value info"


def check_is_graph_input(onnx_model: onnx.ModelProto, name: str):
    for input in onnx_model.graph.input:
        if input.name == name:
            return True
    return False


def check_is_graph_output(onnx_model: onnx.ModelProto, name: str):
    for output in onnx_model.graph.output:
        if output.name == name:
            return True
    return False


def create_reshape_node(onnx_model, reshape_input, reshape_shape, insert_id, elem_type=TensorProto.FLOAT):
    new_name = setName(onnx_model, f"{reshape_input}_reshape")
    new_value_info_name = f"{new_name}_out"
    new_reshape_node = onnx.helper.make_node(op_type="Reshape", name=new_name,
                                             inputs=[reshape_input, f"{new_name}_shape"],
                                             outputs=[new_value_info_name])
    reshape_init = onnx.numpy_helper.from_array(np.array(reshape_shape, dtype=np.int64), new_reshape_node.input[1])
    onnx_model.graph.initializer.append(reshape_init)
    new_value_info = onnx.helper.make_tensor_value_info(new_value_info_name, elem_type, reshape_shape)
    onnx_model.graph.value_info.append(new_value_info)
    onnx_model.graph.node.insert(insert_id, new_reshape_node)
    return new_reshape_node

def create_transpose_node(onnx_model, input, perm, insert_id, elem_type=TensorProto.FLOAT):
    input_shape = get_shape_by_name(onnx_model, input)
    transpose_shape = [input_shape[p] for p in perm]
    if check_is_graph_output(onnx_model, input):
        new_name = setName(onnx_model, f"{input}_transpose_input")
        new_value_info_name = f"{new_name}_out"
        new_shape = input_shape
        new_transpose_node = onnx.helper.make_node(op_type="Transpose", name=new_name,
                                                   inputs=[new_value_info_name], outputs=[input], **{"perm": perm})
        second_output_value_info = get_value_info_by_name(onnx_model, input)
        tensor_type_proto = onnx.helper.make_tensor_type_proto(elem_type, transpose_shape)
        second_output_value_info.type.CopyFrom(tensor_type_proto)
    else:
        new_name = setName(onnx_model, f"{input}_transpose")
        new_value_info_name = f"{new_name}_out"
        new_shape = [input_shape[p] for p in perm]
        new_transpose_node = onnx.helper.make_node(op_type="Transpose", name=new_name,
                                                 inputs=[input], outputs=[new_value_info_name], **{"perm": perm})
    new_value_info = onnx.helper.make_tensor_value_info(new_value_info_name, elem_type, new_shape)
    onnx_model.graph.value_info.append(new_value_info)
    onnx_model.graph.node.insert(insert_id, new_transpose_node)
    return new_transpose_node

def create_slice_node(onnx_model, node_input, slice_inputs, insert_id, output_shape, elem_type=TensorProto.FLOAT):
    slice_name = setName(onnx_model, f"{node_input}_slice")
    slice_start = np.array(slice_inputs[0], dtype=np.int64)
    slice_end = np.array(slice_inputs[1], dtype=np.int64)
    slice_axis = np.array(slice_inputs[2], dtype=np.int64)
    slice_step = np.array(slice_inputs[3], dtype=np.int64)
    slice_input = [node_input, slice_name + "_starts", slice_name + "_ends",
                   slice_name + "_axes", slice_name + "_steps"]
    slice_node = onnx.helper.make_node(op_type="Slice", name=slice_name, inputs=slice_input,
                                       outputs=[slice_name + "_out"])
    current_init = onnx.numpy_helper.from_array(slice_start, slice_input[1])
    onnx_model.graph.initializer.append(current_init)
    current_init = onnx.numpy_helper.from_array(slice_end, slice_input[2])
    onnx_model.graph.initializer.append(current_init)
    current_init = onnx.numpy_helper.from_array(slice_axis, slice_input[3])
    onnx_model.graph.initializer.append(current_init)
    current_init = onnx.numpy_helper.from_array(slice_step, slice_input[4])
    onnx_model.graph.initializer.append(current_init)
    new_value_info = onnx.helper.make_tensor_value_info(slice_node.output[0], elem_type, output_shape)
    onnx_model.graph.value_info.append(new_value_info)
    onnx_model.graph.node.insert(insert_id, slice_node)
    return slice_node