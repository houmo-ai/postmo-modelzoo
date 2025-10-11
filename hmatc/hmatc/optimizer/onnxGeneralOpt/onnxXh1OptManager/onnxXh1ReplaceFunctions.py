#import copy
from ....utils import logger
#import math

import numpy as np  # type: ignore
import onnx  # type: ignore

#from ..onnxBaseOpt.onnxRuntimeEngine import OnnxRuntimeEngine
#from ..onnxGeneralManager.onnxGeneralDeleteFunctions import delete_shape_useless_node
from ...onnxBaseOpt.onnxDebugger import OnnxDebugger
#from ..onnxBaseOpt.onnxBaseFunctions import infer_shapes
from ...onnxBaseOpt.onnxBaseOptimizer import OnnxBaseOptimizer
from ...onnxUtils.onnxBasicUtils import *

@OnnxDebugger.onnx_opt_func_debug_wrapper
@OnnxBaseOptimizer.onnx_opt_traverse_wrapper
def replace_focus_layer_of_Conv(onnx_model, node, node_index):
    '''
    Explanation: 4xSlice is converted to Conv2D.
    example: 
        (data 1x3x384x640) -> 4xSlice(axes=[2, 3], steps=[2,2]) -> Concat(axis=1) -> (1x12x192x320)
                           -> Slice0(axes=[2, 3], steps=[2,2], starts=[0,0], ends=[H0,W0])
                          |                                                                |
        (data Nx3xHxW)  ->|-> Slice1(axes=[2, 3], steps=[2,2], starts=[0,1], ends=[H0,W0]) |
                          |                                                                |-> Concat(axis=1) -> (Nx12x(H/2)x(W/2))
                          |-> Slice1(axes=[2, 3], steps=[2,2], starts=[1,0], ends=[H0,W0]) |
                          |                                                                |
                           -> Slice1(axes=[2, 3], steps=[2,2], starts=[1,1], ends=[H0,W0])
    Author: Nan Xu
    '''
    sub_graph_match_template = [
        {"name": "node_1", "op_type": "Slice", "input": [], "output": ["tensor_1"]},
        {"name": "node_2", "op_type": "Slice", "input": [], "output": ["tensor_2"]},
        {"name": "node_3", "op_type": "Slice", "input": [], "output": ["tensor_3"]},
        {"name": "node_4", "op_type": "Slice", "input": [], "output": ["tensor_4"]},
        {"name": "node_5", "op_type": "Concat", "input": ["tensor_1", "tensor_2", "tensor_3", "tensor_4"],
         "output": []},
    ]
    # for YOLO focus block --> conv
    if check_sub_graph(onnx_model, node, sub_graph_match_template):
        nodes_graph = get_sub_graph(onnx_model, node, sub_graph_match_template)
        nodes_list = [_node["node"] for _node in nodes_graph]
        input_shape = get_shape_by_name(onnx_model, nodes_list[0].input[0])
        if len(input_shape) != 4:
            return onnx_model, False
        slice_start_end_step = {}
        for i, cur_node in enumerate(nodes_list):
            if cur_node.op_type != "Slice":
                continue
            starts_ori = list(get_tensor_from_initializer(onnx_model, cur_node.input[1]))
            ends_ori = list(get_tensor_from_initializer(onnx_model, cur_node.input[2]))
            axes = list(get_tensor_from_initializer_with_default(onnx_model, cur_node.input[3], list(range(len(starts_ori))))) \
                    if len(cur_node.input) > 3 else list(range(len(starts_ori)))
            axes = sorted([(a + 4) % 4 for a in axes])
            steps_ori = list(get_tensor_from_initializer_with_default(onnx_model, cur_node.input[4], [1 for i in range(len(starts_ori))])) \
                    if len(cur_node.input) > 4 else [1 for i in range(len(starts_ori))]
            starts = [0 for i in range(len(input_shape))]
            ends = [i for i in input_shape]
            steps = [1 for i in range(len(input_shape))]
            for a, axis in enumerate(axes):
                starts[axis] = max(starts_ori[a], 0)
                ends[axis] = min(ends_ori[a], input_shape[axis])
                steps[axis] = steps_ori[a]
            slice_start_end_step[i] = starts + ends + steps
            if axes != [4 - len(axes) + a for a in range(len(axes))]:
                return onnx_model, False
        if slice_start_end_step != {0: [0, 0, 0, 0] + input_shape + [1, 1, 2, 2], 
                                    1: [0, 0, 1, 0] + input_shape + [1, 1, 2, 2],
                                    2: [0, 0, 0, 1] + input_shape + [1, 1, 2, 2], 
                                    3: [0, 0, 1, 1] + input_shape + [1, 1, 2, 2]}:
            return onnx_model, False
        concat_node = nodes_list[4]
        axis = attribute_to_dict(concat_node.attribute).get('axis')
        axis = (len(input_shape) + axis) % len(input_shape)
        if axis != 1:
            return onnx_model, False

        logger.debug("YOLO focus block->Conv")
        logger.debug(f"Nodes:{[cur_node.name for cur_node in nodes_list]}")
        logger.debug(f"Input:{nodes_list[0].input}")

        conv_attr = {'dilations': [1, 1],
                     'group': 1,
                     'kernel_shape': [2, 2],
                     'pads': [0, 0, 0, 0],
                     'strides': [2, 2]}
        output_shape = get_shape_by_name(onnx_model, concat_node.output[0])
        conv_weights_array = np.zeros((output_shape[1], input_shape[1], *conv_attr["kernel_shape"]), dtype=np.float32)
        for in_c in range(input_shape[1]):
            conv_weights_array[in_c, in_c, 0, 0] = 1.0
            conv_weights_array[in_c + 3, in_c, 1, 0] = 1.0
            conv_weights_array[in_c + 2 * 3, in_c, 0, 1] = 1.0
            conv_weights_array[in_c + 3 * 3, in_c, 1, 1] = 1.0
        conv_weights = onnx.numpy_helper.from_array(conv_weights_array, concat_node.name+"_focus_conv_weight")
        onnx_model.graph.initializer.insert(0, conv_weights)
        conv_node = onnx.helper.make_node(name=concat_node.name+"_focus_conv",
                                          op_type="Conv",
                                          inputs=[nodes_list[0].input[0], conv_weights.name],
                                          outputs=concat_node.output,
                                          **conv_attr)
        for slice_node in nodes_list[:4]:
            out_value_info = get_value_info_by_name(onnx_model, slice_node.output[0])
            onnx_model.graph.value_info.remove(out_value_info)
        onnx_model = delete_nodes(onnx_model, nodes_list)
        onnx_model.graph.node.insert(node_index, conv_node)
        delete_useless_input_in_initializer(onnx_model)
        return onnx_model, True

    return onnx_model, False