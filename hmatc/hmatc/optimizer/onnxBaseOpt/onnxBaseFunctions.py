"""

Author: Nan Xu
Maintainer: Nan Xu
Date: 2025/08/04
Company: Houmo

"""
import numpy as np  # type: ignore
import onnx  # type: ignore
import onnxruntime as rt  # type: ignore
import copy

from .onnxDebugger import OnnxDebugger
from ..onnxUtils.onnxBasicUtils import *
from .onnxConfigController import OnnxCfg
from .onnxRuntimeEngine import OnnxRuntimeEngine

def test_infer_onnx(onnx_model: onnx.ModelProto):
    try:
        return OnnxRuntimeEngine().ort_infer_shape(onnx_model)

    except Exception as e:
        logger.error("invalid onnx model, please check!")
        logger.error(e)
        raise ValueError("invalid onnx model, please check!")

def infer_shapes(onnx_model: onnx.ModelProto):
    def onnx_infer_shape(model):
        model = copy.deepcopy(model)  # protect the original model
        del model.graph.value_info[:]
        ori_model_output_list = [out.name for out in model.graph.output]
        del model.graph.output[:]
        model.graph.output.extend([onnx.ValueInfoProto(name=output_name) for output_name in ori_model_output_list])
        ori_input = copy.deepcopy(model.graph.input)
        for input_ in model.graph.input:
            for d in input_.type.tensor_type.shape.dim:
                if d.dim_value < 1:
                    d.dim_value = 1
        sort = graph_topological_sort(model.graph)
        if sort:
            logger.debug("Reorder the nodes in the graph because they are not topologically sorted.")
        model = onnx.shape_inference.infer_shapes(model)
        del model.graph.input[:]
        model.graph.input.MergeFrom(ori_input)
        return model

    def check_infer_model(model):
        all_value_info = list(model.graph.input) + list(model.graph.value_info) + list(model.graph.output)
        tensor_type_map = {value_info.name: value_info.type.tensor_type for value_info in all_value_info}
        for node in model.graph.node:
            for output in node.output:
                if output not in tensor_type_map:
                    return False
                tensor_type = tensor_type_map[output]
                if not tensor_type.HasField("shape") or not tensor_type.HasField("elem_type"):
                    return False
                # patch for Einsum without shape
                if node.op_type == "Einsum":
                    for dim in tensor_type.shape.dim:
                        if not dim.HasField("dim_value"):
                            return False
        return True

    art_model = any("art" in opset_import.domain for opset_import in onnx_model.opset_import)
    if not art_model and not OnnxCfg.check_exist("custom_lib"):
        onnx_model = onnx_infer_shape(onnx_model)
        if check_infer_model(onnx_model):
            return onnx_model
    onnx_model = OnnxRuntimeEngine().ort_infer_shape(onnx_model)
    return onnx_model

@OnnxDebugger.onnx_opt_func_debug_wrapper
def clean_useless_input(onnx_model: onnx.ModelProto):
    input_need_remove = []
    init_name_list = [init.name for init in onnx_model.graph.initializer]
    for input_ in onnx_model.graph.input:
        if input_.name in init_name_list:
            input_need_remove.append(input_)
    for i in input_need_remove:
        onnx_model.graph.input.remove(i)
    return onnx_model, len(input_need_remove) > 0

@OnnxDebugger.onnx_opt_func_debug_wrapper
def onnx_name_checker(onnx_model: onnx.ModelProto):
    def check_empty_name(model):
        name_empty_nodes_list = []
        for node_id, node in enumerate(model.graph.node):
            if len(node.name) == 0:
                name_empty_nodes_list.append(node)

        if name_empty_nodes_list:
            for i, node in enumerate(name_empty_nodes_list):
                node.name = setName(model, f"{node.op_type}_0")
        return len(name_empty_nodes_list) > 0

    def check_name_illegal_character(model):
        illegal_pattern = r"[\/\\\:\*\?\"\<\>\|]"  # '/ \ : * ? " < > |'
        name_mapper = {}
        # search
        for node in model.graph.node:
            for input_name in node.input:
                if input_name not in name_mapper.keys() and re.search(illegal_pattern, input_name):
                    name_mapper[input_name] = re.sub(illegal_pattern, "_", input_name)
        for output in model.graph.output:
            output_name = output.name
            if re.search(illegal_pattern, output_name):
                name_mapper[output_name] = re.sub(illegal_pattern, "_", output_name)
        # update
        for node in model.graph.node:
            for input_id, input_ in enumerate(node.input):
                if input_ in name_mapper.keys():
                    node.input[input_id] = name_mapper[input_]
            for output_id, output in enumerate(node.output):
                if output in name_mapper.keys():
                    node.output[output_id] = name_mapper[output]
        for init in model.graph.initializer:
            if init.name in name_mapper.keys():
                init.name = name_mapper[init.name]
        for value_info in model.graph.value_info:
            if value_info.name in name_mapper.keys():
                value_info.name = name_mapper[value_info.name]
        for input_ in model.graph.input:
            if input_.name in name_mapper.keys():
                logger.info(f"Rename input: {input_.name} -> {name_mapper[input_.name]}")
                input_.name = name_mapper[input_.name]
        for output in model.graph.output:
            if output.name in name_mapper.keys():
                logger.info(f"Rename output: {output.name} -> {name_mapper[output.name]}")
                output.name = name_mapper[output.name]
        return bool(name_mapper)

    def check_name_length(model):
        for node in model.graph.node:
            if len(node.name) > 512:
                raise ValueError(f"Node name too long: {node.name}")
            for input_ in node.input:
                if len(input_) > 512:
                    raise ValueError(f"Tensor name too long: {input_}")
            for output in node.output:
                if len(output) > 512:
                    raise ValueError(f"Tensor name too long: {output}")

    status = check_empty_name(onnx_model)
    #status |= check_name_illegal_character(onnx_model)
    check_name_length(onnx_model)

    return onnx_model, status

@OnnxDebugger.onnx_opt_func_debug_wrapper
def estimate_gops(onnx_model: onnx.ModelProto):
    def conv_macs(node, input_shape, output_shape, attrs):
        kernel_shape = attrs.get('kernel_shape', [1, 1])
        kernel_ops = np.prod(kernel_shape)  # Kw x Kh
        bias_ops = len(node.input) == 3
        group = 1
        if 'group' in attrs:
            group = attrs['group']
        in_channels = input_shape[1]
        return np.prod(output_shape) * (in_channels // group * kernel_ops + bias_ops)

    def gemm_macs(node, input_shape, output_shape, attrs):
        return np.prod(input_shape) * np.prod(output_shape)

    def bn_macs(node, input_shape, output_shape, attrs):
        batch_macs = np.prod(output_shape)
        if len(node.input) == 5:
            batch_macs *= 2
        return batch_macs

    def linear_activation_macs(node, input_shape, output_shape, attrs):
        return np.prod(input_shape)

    def sigmoid_macs(node, input_shape, output_shape, attrs):
        return np.prod(input_shape) * 4

    def elt_macs(node, input_shape, output_shape, attrs):
        return np.prod(input_shape)

    macs = 0
    nodes = onnx_model.graph.node
    for node_id, node in enumerate(nodes):
        if node.op_type == "Constant":
            continue
        input_shape = get_shape_by_name(onnx_model, node.input[0])
        output_shape = get_shape_by_name(onnx_model, node.output[0])
        attrs = attribute_to_dict(node.attribute)
        if node.op_type == "Conv":
            macs += conv_macs(node, input_shape, output_shape, attrs) / 1000000000
        elif node.op_type == "Gemm":
            macs += gemm_macs(node, input_shape, output_shape, attrs) / 1000000000
        elif node.op_type == "BatchNormalization":
            macs += bn_macs(node, input_shape, output_shape, attrs) / 1000000000
        elif node.op_type in ["Relu", "LeakyRelu"]:
            macs += linear_activation_macs(node, input_shape, output_shape, attrs) / 1000000000
        elif node.op_type in ["Sigmoid", "Tanh", "Softmax"]:
            macs += sigmoid_macs(node, input_shape, output_shape, attrs) / 1000000000
        elif node.op_type in ["Add", "Mul", "Div", "Sub"]:
            macs += elt_macs(node, input_shape, output_shape, attrs) / 1000000000
    logger.info("======== Calculate model gops ========")
    logger.info("Total Gops:" + str(macs * 2))
    logger.info("======================================")
    return onnx_model, True

