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
def fusion_focus_layer_of_Conv(onnx_model, node, node_index):
    '''
    Explanation: 4xSlice is converted to Conv2D and fusion with after Conv2D.
    example: 
        (data 1x3x384x640) -> 4xSlice(axes=[2, 3], steps=[2,2]) -> Concat(axis=1) -> Conv2D(32x12x3x3) -> (1x32x192x320)
                           -> Slice0(axes=[2, 3], steps=[2,2], starts=[0,0], ends=[H0,W0])
                          |                                                                |
        (data Nx3xHxW)  ->|-> Slice1(axes=[2, 3], steps=[2,2], starts=[0,1], ends=[H0,W0]) |
                          |                                                                |-> Concat(axis=1) -> Conv(Cx12xkernel) -> (NxCx(H/2)x(W/2))
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
         "output": ["tensor_5"]},
        {"name": "node_6", "op_type": "Conv", "input": ["tensor_5"], "output": []}
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

        # update next conv
        conv_origin_node = nodes_list[-1]
        conv_origin_wt_init = get_initializer(onnx_model, conv_origin_node.input[1])
        conv_origin_wt_array = onnx.numpy_helper.to_array(conv_origin_wt_init)
        conv_attr = attribute_to_dict(conv_origin_node.attribute)
        kernel_shape = conv_attr.get("kernel_shape", conv_origin_wt_array.shape[2:])
        strides = conv_attr.get("strides", [1, 1])
        pads = conv_attr.get("pads", [0, 0, 0, 0])

        conv_attr["kernel_shape"] = [kernel_shape[0] * 2 + 1, kernel_shape[1] * 2 + 1]
        #conv_attr["kernel_shape"] = [kernel_shape[0] * 2, kernel_shape[1] * 2]
        conv_attr["strides"] = [_s * 2 for _s in strides]
        conv_attr["pads"] = [_p * 2 for _p in pads]
        conv_attr["pads"][2] += 1
        conv_attr["pads"][3] += 1
        weight_tensor_new = np.zeros((conv_origin_wt_array.shape[0], input_shape[1], *conv_attr["kernel_shape"]),
                                     dtype=np.float32)
        for _h in range(conv_origin_wt_array.shape[2]):
            for _w in range(conv_origin_wt_array.shape[3]):
                for _c in range(input_shape[1]):
                    weight_tensor_new[:, _c, _h * 2, _w * 2] = conv_origin_wt_array[:, _c, _h, _w]
                    weight_tensor_new[:, _c, _h * 2 + 1, _w * 2] = conv_origin_wt_array[:, _c + input_shape[1], _h,
                                                                   _w]
                    weight_tensor_new[:, _c, _h * 2, _w * 2 + 1] = conv_origin_wt_array[:, _c + input_shape[1] * 2,
                                                                   _h, _w]
                    weight_tensor_new[:, _c, _h * 2 + 1, _w * 2 + 1] = conv_origin_wt_array[:,
                                                                       _c + input_shape[1] * 3, _h, _w]
        conv_origin_wt_init.CopyFrom(onnx.numpy_helper.from_array(weight_tensor_new,
                                                                  f"{conv_origin_wt_init.name}_new"))
        conv_origin_node.input[:2] = [nodes_list[0].input[0], conv_origin_wt_init.name]
        del conv_origin_node.attribute[:]
        conv_origin_node.attribute.extend(onnx.helper.make_attribute(key, value)
                                          for key, value in sorted(conv_attr.items()) if value is not None)
        onnx_model = delete_nodes(onnx_model, nodes_list[:-1])
        delete_useless_input_in_initializer(onnx_model)
        return onnx_model, True

    return onnx_model, False