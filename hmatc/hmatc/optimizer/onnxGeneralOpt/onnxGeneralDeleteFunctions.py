import copy
from ...utils import logger
import math

import numpy as np  # type: ignore
import onnx  # type: ignore

#from ..onnxBaseOpt.onnxRuntimeEngine import OnnxRuntimeEngine
from ..onnxBaseOpt.onnxDebugger import OnnxDebugger
#from ..onnxBaseOpt.onnxBaseFunctions import infer_shapes
from ..onnxBaseOpt.onnxBaseOptimizer import OnnxBaseOptimizer
from ..onnxUtils.onnxBasicUtils import *
#from ..onnxBaseOpt.onnxConfigController import OnnxCfg

# delete shape useless func region
def delete_shape_useless_node(onnx_model, node, op_type):
    if check_node_serial_group(onnx_model, node, [op_type]):
        del_node = node
        constant_names = [init.name for init in onnx_model.graph.initializer]
        if del_node.input[0] in constant_names:
            return onnx_model, False
        if not all(input_name in constant_names for input_name in del_node.input[1:]):
            return onnx_model, False
        input_value_info = get_value_info_by_name(onnx_model, del_node.input[0])
        output_value_info = get_value_info_by_name(onnx_model, del_node.output[0])
        if input_value_info is None or output_value_info is None:
            return onnx_model, False
        input_shape = [d.dim_value if d.dim_value > 0 else 1 for d in input_value_info.type.tensor_type.shape.dim]
        output_shape = [d.dim_value if d.dim_value > 0 else 1 for d in output_value_info.type.tensor_type.shape.dim]
        if input_shape != output_shape:
            return onnx_model, False
        if input_value_info.type.tensor_type.elem_type != output_value_info.type.tensor_type.elem_type:
            return onnx_model, False
        model_inputs_names = [input_.name for input_ in onnx_model.graph.input]
        model_outputs_names = [output.name for output in onnx_model.graph.output]
        if del_node.input[0] in [*model_inputs_names, *model_outputs_names] and \
                del_node.output[0] in model_outputs_names:
            return onnx_model, False
        logger.debug(f"Delete:{op_type}")
        logger.debug(f"Nodes:{del_node.name}")
        logger.debug(f"Input:{del_node.input}")
        onnx_model.graph.node.remove(del_node)
        if del_node.output[0] not in model_outputs_names:
            replace_input_of_all_nodes(onnx_model, del_node.output[0], del_node.input[0])
        else:
            prev_node = get_node_by_output(onnx_model, del_node.input[0])
            replace_node_output(prev_node, del_node.input[0], del_node.output[0])
            replace_input_of_all_nodes(onnx_model, del_node.input[0], del_node.output[0])
        delete_useless_input_in_initializer(onnx_model)
        return onnx_model, True
    return onnx_model, False

@OnnxDebugger.onnx_opt_func_debug_wrapper
@OnnxBaseOptimizer.onnx_opt_once_wrapper
def delete_useless_pool(onnx_model, node):
    '''
    Explanation: This function deletes some invalid 1D and 2D Pool operators.
    example: dilations, kernel_shape and strides belongs to [1] or [1, 1] and pads belongs to [0, 0] or [0, 0, 0, 0] and storage_order equal 0.
    Author: Nan Xu
    '''
    if node.op_type not in ["MaxPool", "AveragePool"]:
        return onnx_model, False
    input_shape = get_shape_by_name(onnx_model, node.input[0])
    if len(input_shape) not in [3, 4]:
        return onnx_model, False
    pool_attribute = attribute_to_dict(node.attribute)
    dilations = pool_attribute.get("dilations", [1] if len(input_shape) == 3 else [1, 1])
    kernel_shape = pool_attribute.get("kernel_shape", [1] if len(input_shape) == 3 else [1, 1])
    pads = pool_attribute.get("pads", [0, 0] if len(input_shape) == 3 else [0, 0, 0, 0])
    strides = pool_attribute.get("strides", [1] if len(input_shape) == 3 else [1, 1])
    storage_order = pool_attribute.get("storage_order", 0)
    if set(dilations) != {1} or set(kernel_shape) != {1} or set(pads) != {0} or set(strides) != {1} or storage_order != 0:
        return onnx_model, False
    logger.debug("Delete:Pool")
    logger.debug(f"Nodes:{node.name}")
    logger.debug(f"Input:{node.input}")
    model_outputs = [output.name for output in onnx_model.graph.output]
    if node.output[0] not in model_outputs:
        next_nodes = get_node_by_input(onnx_model, node.output)
        for next_node in next_nodes:
            for idx, input in enumerate(next_node.input):
                if input == node.output[0]:
                    next_node.input[idx] = node.input[0]
        pool_out_value_info = get_value_info_by_name(onnx_model, node.output[0])
        onnx_model.graph.value_info.remove(pool_out_value_info)
    else:
        samelevel_nodes = get_node_by_input(onnx_model, [node.input[0]])
        for samelevel_node in samelevel_nodes:
            if samelevel_node.name == node.name:
                continue
            for idx, input in enumerate(samelevel_node.input):
                if input == node.input[0]:
                    samelevel_node.input[idx] = node.output[0]
        pre_node = get_node_by_output(onnx_model, node.input[0])
        for idx, output in enumerate(pre_node.output):
            if output == node.input[0]:
                pre_node.output[idx] = node.output[0]
        pool_input_value_info = get_value_info_by_name(onnx_model, node.input[0])
        onnx_model.graph.value_info.remove(pool_input_value_info)
    onnx_model.graph.node.remove(node)
    return onnx_model, True