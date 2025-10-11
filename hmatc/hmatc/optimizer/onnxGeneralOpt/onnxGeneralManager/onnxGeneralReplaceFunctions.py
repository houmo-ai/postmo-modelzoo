import copy
from ....utils import logger
import math

import numpy as np  # type: ignore
import onnx  # type: ignore

#from ..onnxBaseOpt.onnxRuntimeEngine import OnnxRuntimeEngine
from ...onnxBaseOpt.onnxDebugger import OnnxDebugger
#from ..onnxBaseOpt.onnxBaseFunctions import infer_shapes
from ...onnxBaseOpt.onnxBaseOptimizer import OnnxBaseOptimizer
from ...onnxUtils.onnxBasicUtils import *
#from ..onnxBaseOpt.onnxConfigController import OnnxCfg


@OnnxDebugger.onnx_opt_func_debug_wrapper
@OnnxBaseOptimizer.onnx_opt_once_wrapper
def replace_Div_of_Mul(onnx_model, node):
    '''
    Explanation: This function converts the Div operator with only one dynamic input into the Mul operator.
    example: y = x / [10, 12, 6] ==> y = x * [0.1, 0.8333333, 0.166666667]
    Author: Nan Xu
    '''
    if check_node_serial_group(onnx_model, node, ["Div"]):
        div_node = get_node_serial_group(onnx_model, node, ["Div"])[0]
        if check_constant(onnx_model, div_node.input[0]):
            return onnx_model, False
        if get_tensor_type_by_name(onnx_model, div_node.input[0]) != TensorProto.FLOAT:
            return onnx_model, False
        if check_constant(onnx_model, div_node.input[1]):
            logger.debug("Div->Mul")
            logger.debug(f"Nodes:{div_node.name}")
            div_init = get_initializer(onnx_model, div_node.input[1])
            div_tensor = onnx.numpy_helper.to_array(div_init)
            onnx_model.graph.initializer.remove(div_init)
            mul_tensor = np.array(1 / div_tensor, dtype=np.float32)
            mul_init = onnx.numpy_helper.from_array(mul_tensor, div_init.name)
            onnx_model.graph.initializer.append(mul_init)
            div_node.op_type = "Mul"
            onnx_model = delete_useless_input_in_initializer(onnx_model)
            return onnx_model, True
    return onnx_model, False

@OnnxDebugger.onnx_opt_func_debug_wrapper
@OnnxBaseOptimizer.onnx_opt_traverse_wrapper
def replace_GatherUnsqueeze_of_Slice(onnx_model, node, node_index):
    '''
    Explanation: Gather+Unsqueeze is converted to Slice or eliminated directly.
    example: 
        (data 1x64x22x92) -> Gather(indices=0, axis=0) -> Unsqueeze(axes=0) -> (1x64x22x92)
        (data NxCxHxW) -> Gather(indices=n, axis=a) -> Unsqueeze(axes=a) -> (1xCxHxW or Nx1xHxW or NxCx1xW or NxCxHx1)
    Author: Nan Xu
    '''
    if check_node_serial_group(onnx_model, node, ["Gather", "Unsqueeze"]):
        gather_node, unsqueeze_node = get_node_serial_group(onnx_model, node, ["Gather", "Unsqueeze"])
        gather_indices_init = get_initializer(onnx_model, gather_node.input[1])
        if gather_indices_init is None:
            return onnx_model, False
        gather_indices = onnx.numpy_helper.to_array(gather_indices_init)
        if gather_indices.size > 1:
            return onnx_model, False
        gather_indice = gather_indices.flatten()[0]
        gather_input_shape = get_shape_by_name(onnx_model, gather_node.input[0])
        gather_output_shape = get_shape_by_name(onnx_model, gather_node.output[0])
        gather_axis = attribute_to_dict(gather_node.attribute)["axis"]
        gather_axis = gather_axis + len(gather_input_shape) if gather_axis < 0 else gather_axis
        unsqueeze_axes_init = get_initializer(onnx_model, unsqueeze_node.input[1])
        unsqueeze_axes = 0 if unsqueeze_axes_init is None else onnx.numpy_helper.to_array(unsqueeze_axes_init).flatten()[0]
        unsqueeze_axes = unsqueeze_axes + len(gather_output_shape) if unsqueeze_axes < 0 else unsqueeze_axes
        if unsqueeze_axes != gather_axis and unsqueeze_axes < len(gather_output_shape) and gather_axis < len(gather_input_shape):
            return onnx_model, False
        logger.debug("Replace:Gather+Unsqueeze->Slice")
        logger.debug(f"Nodes:{node.name}")
        logger.debug(f"Input:{node.input}")
        unsqueeze_output_shape = get_shape_by_name(onnx_model, unsqueeze_node.output[0])
        next_nodes = get_node_by_input(onnx_model, unsqueeze_node.output)
        onnx_model.graph.value_info.remove(get_value_info_by_name(onnx_model, gather_node.output[0]))
        onnx_model.graph.value_info.remove(get_value_info_by_name(onnx_model, unsqueeze_node.output[0]))
        if unsqueeze_output_shape == gather_input_shape:
            swap_input = [gather_node.input[0], unsqueeze_node.output[0]]
        else:
            slice_axes = np.array([gather_axis])
            slice_starts = np.array([gather_indice])
            slice_ends = np.array([gather_indice+1])
            slice_steps = np.array([1])
            slice_node = onnx.helper.make_node(name=gather_node.name,
                                               op_type="Slice",
                                               inputs=[gather_node.input[0],
                                                       gather_node.name+"_slice_starts",
                                                       gather_node.name+"_slice_ends",
                                                       gather_node.name+"_slice_axes",
                                                       gather_node.name+"_slice_steps"],
                                                outputs=unsqueeze_node.output)
            starts_init = onnx.numpy_helper.from_array(slice_starts, slice_node.input[1])
            ends_init = onnx.numpy_helper.from_array(slice_ends, slice_node.input[2])
            axes_init = onnx.numpy_helper.from_array(slice_axes, slice_node.input[3])
            steps_init = onnx.numpy_helper.from_array(slice_steps, slice_node.input[4])
            swap_input = [slice_node.output[0], unsqueeze_node.output[0]]
            onnx_model.graph.initializer.extend([starts_init, ends_init, axes_init, steps_init])
            onnx_model.graph.node.insert(node_index, slice_node)
        onnx_model = delete_nodes(onnx_model, [gather_node, unsqueeze_node])
        if swap_input[0] != swap_input[1]:
            for next_node in next_nodes:
                for index, input in enumerate(next_node.input):
                    next_node.input[index] = swap_input[0] if input == swap_input[1] else input
        onnx_model = delete_useless_input_in_initializer(onnx_model)
        return onnx_model, True
    return onnx_model, False

@OnnxDebugger.onnx_opt_func_debug_wrapper
@OnnxBaseOptimizer.onnx_opt_traverse_wrapper
def replace_SqueezeTranspose_of_TransposeReshape(onnx_model, node, node_index):
    '''
    Explanation: Squeeze+Transpose1D is converted to Transpose2D+Reshape or eliminated directly.
    example: 
        (data 1x128x18x44) -> Squeeze(axes=[0]) -> Transpose(1, 2, 0) -> (18x44x128)
        (data 1xCxHxW or Bx1xHxW or BxCx1xW or BxCxHx1) -> Squeeze(axes=x, data.shape[x] = 1) -> Transpose(perm) -> (CxHxW or BxHxW or BxCxW or BxCxH).perm
    Author: Nan Xu
    '''
    if check_node_serial_group(onnx_model, node, ["Squeeze",  "Transpose"]):
        squeeze_node, transpose_node = get_node_serial_group(onnx_model, node, ["Squeeze",  "Transpose"])
        input_shape = get_shape_by_name(onnx_model, squeeze_node.input[0])
        axes = get_tensor_from_initializer(onnx_model, squeeze_node.input[1]).tolist() \
                if len(squeeze_node.input) > 1 else [i for i, x in enumerate(input_shape) if x == 1]
        axes = sorted([axis % len(input_shape) for axis in axes])
        perm = attribute_to_dict(transpose_node.attribute).get("perm", list(range(len(input_shape) - len(axes))))
        keep_axes = [i for i in range(len(input_shape)) if i not in axes]
        new_perm = axes + [keep_axes[p] for p in perm]
        logger.debug("Replace:Squeeze+Transpose->Transpose+Reshape")
        logger.debug(f"Nodes:{node.name}")
        logger.debug(f"Input:{node.input}")
        new_transpose_node = onnx.helper.make_node(name=transpose_node.name,
                                                   op_type="Transpose",
                                                   inputs=[squeeze_node.input[0]],
                                                   outputs=[squeeze_node.output[0]],
                                                   **{'perm': new_perm})
        output_shape = get_shape_by_name(onnx_model, transpose_node.output[0])
        reshape_node = onnx.helper.make_node(name=squeeze_node.name+"_reshape",
                                             op_type="Reshape",
                                             inputs=[new_transpose_node.output[0],
                                                     transpose_node.output[0]+"_shape"],
                                             outputs=transpose_node.output)
        shape_init = onnx.numpy_helper.from_array(np.array(output_shape), reshape_node.input[1])
        squeeze_value_info = get_value_info_by_name(onnx_model, squeeze_node.output[0])
        onnx_model.graph.value_info.remove(squeeze_value_info)
        new_transpose_out_shape = [input_shape[x] for x in new_perm]
        data_type = get_tensor_type_by_name(onnx_model, squeeze_node.output[0])
        new_transpose_value_info = onnx.helper.make_tensor_value_info(name=squeeze_node.output[0],
                                                                      elem_type=data_type,
                                                                      shape=new_transpose_out_shape)
        onnx_model = delete_nodes(onnx_model, [squeeze_node, transpose_node])
        onnx_model.graph.node.insert(node_index, reshape_node)
        onnx_model.graph.node.insert(node_index, new_transpose_node)
        onnx_model.graph.initializer.append(shape_init)
        onnx_model.graph.value_info.append(new_transpose_value_info)
        onnx_model = delete_useless_input_in_initializer(onnx_model)
        return onnx_model, True
    return onnx_model, False

@OnnxDebugger.onnx_opt_func_debug_wrapper
@OnnxBaseOptimizer.onnx_opt_traverse_wrapper
def replace_TransposeUnsqueeze_of_ReshapeTranspose(onnx_model, node, node_index):
    '''
    Explanation: Transpose1D+Unsqueeze is converted to Reshape+Transpose2D or eliminated directly.
    example: 
        (data 18x44x64) -> Transpose(2, 0, 1) -> Unsqueeze(axes=[0]) -> (1x64x18x44)
        (data CxHxW or BxHxW or BxCxW or BxCxH) -> Transpose(perm) -> Unsqueeze(axes=[x]) -> BxC1xH1xW1
    Author: Nan Xu
    '''

    if check_node_serial_group(onnx_model, node, ["Transpose", "Unsqueeze"]):
        transpose_node, unsqueeze_node = get_node_serial_group(onnx_model, node, ["Transpose", "Unsqueeze"])
        input_shape = get_shape_by_name(onnx_model, transpose_node.input[0])
        perm = attribute_to_dict(transpose_node.attribute).get("perm", list(range(len(input_shape))))
        #transpose_out_shape = [input_shape[p] for p in perm]
        #output_shape = get_shape_by_name(onnx_model, unsqueeze_node.output[0])
        axes = get_tensor_from_initializer(onnx_model, unsqueeze_node.input[1]).tolist() \
                if len(unsqueeze_node.input) > 1 else [0]
        reshape_out_shape = copy.deepcopy(input_shape)
        perm_new = copy.deepcopy(perm)
        for axis in axes:
            reshape_out_shape.insert(axis, 1)
            for j, p in enumerate(perm_new):
                if p >= axis:
                    perm_new[j] += 1
            perm_new.insert(axis, axis)
        logger.debug("Replace:Transpose+Unsqueeze->Reshape+Transpose")
        logger.debug(f"Nodes:{node.name}")
        logger.debug(f"Input:{node.input}")
        reshape_node = onnx.helper.make_node(name=unsqueeze_node.name+"_reshape",
                                             op_type="Reshape",
                                             inputs=[transpose_node.input[0],
                                                     unsqueeze_node.name+"_new_shape"],
                                             outputs=[transpose_node.output[0]])
        shape_init = onnx.numpy_helper.from_array(np.array(reshape_out_shape), reshape_node.input[1])
        new_transpose_node = onnx.helper.make_node(name=transpose_node.name,
                                                   op_type="Transpose",
                                                   inputs=[reshape_node.output[0]],
                                                   outputs=[unsqueeze_node.output[0]],
                                                   perm=perm_new)
        transpose_value_info = get_value_info_by_name(onnx_model, transpose_node.output[0])
        onnx_model.graph.value_info.remove(transpose_value_info)
        data_type = get_tensor_type_by_name(onnx_model, transpose_node.input[0])
        reshape_value_info = onnx.helper.make_tensor_value_info(name=reshape_node.output[0],
                                                                elem_type=data_type,
                                                                shape=reshape_out_shape)
        onnx_model.graph.initializer.append(shape_init)
        onnx_model = delete_nodes(onnx_model, [transpose_node, unsqueeze_node])
        onnx_model.graph.node.insert(node_index, new_transpose_node)
        onnx_model.graph.node.insert(node_index, reshape_node)
        onnx_model.graph.value_info.append(reshape_value_info)
        onnx_model = delete_useless_input_in_initializer(onnx_model)
        return onnx_model, True
    return onnx_model, False

@OnnxDebugger.onnx_opt_func_debug_wrapper
@OnnxBaseOptimizer.onnx_opt_once_wrapper
def replace_MaxPool1D_of_MaxPool2D(onnx_model, node):
    '''
    Explanation: MaxPool1D is converted to Reshape+MaxPool2D+Reshape or eliminated directly.
    example: 
        (data 20x90x64) -> MaxPool1D -> (20x90x32)
        (data CxHxW0) -> MaxPool1D -> (CxHxW1)
    Author: Nan Xu
    '''
    if node.op_type != "MaxPool":
        return onnx_model, False
    input_shape = get_shape_by_name(onnx_model, node.input[0])
    if len(input_shape) != 3:
        return onnx_model, False
    logger.debug("Replace:MaxPool1D->Reshape+MaxPool2D+Reshape")
    logger.debug(f"Nodes:{node.name}")
    logger.debug(f"Input:{node.input}")
    top_shape = copy.deepcopy(input_shape)
    top_shape.insert(2, 1)
    output_shape = get_shape_by_name(onnx_model, node.output[0])
    top_reshape_node = onnx.helper.make_node(name=node.name+"_top_reshape",
                                        op_type="Reshape",
                                        inputs=[node.input[0],
                                                node.input[0]+"_new_shape"],
                                        outputs=[node.input[0]+"_new_shape_out"])
    top_shape_init = onnx.numpy_helper.from_array(np.array(top_shape), top_reshape_node.input[1])
    maxpool_attribute = attribute_to_dict(node.attribute)
    dilations = maxpool_attribute.get("dilations", [1])
    kernel_shape = maxpool_attribute.get("kernel_shape")
    pads = maxpool_attribute.get("pads", [0, 0])
    strides = maxpool_attribute.get("strides", [1])
    maxpool2d = create_maxpool_node(name=node.name,
                                    inputs=[top_reshape_node.output[0]],
                                    outputs=[node.output[0]+"_4dim"],
                                    kernel_shape=[1, kernel_shape[-1]],
                                    dilations=[1, dilations[-1]],
                                    strides=[1, strides[-1]],
                                    pads=[0, 0, pads[-2], pads[-1]])
    bot_reshape_node = onnx.helper.make_node(name=node.name+"_bot_reshape",
                                            op_type="Reshape",
                                            inputs=[maxpool2d.output[0],
                                                     node.output[0]+"_new_shape"],
                                            outputs=node.output)
    bot_shape_init = onnx.numpy_helper.from_array(np.array(output_shape), bot_reshape_node.input[1])
    maxpool2d_output_shape = copy.deepcopy(output_shape)
    maxpool2d_output_shape.insert(2, 1)
    data_type = get_tensor_type_by_name(onnx_model, node.input[0])
    top_reshape_value_info = onnx.helper.make_tensor_value_info(name=top_reshape_node.output[0],
                                                                elem_type=data_type,
                                                                shape=top_shape)
    maxpool2d_value_info = onnx.helper.make_tensor_value_info(name=maxpool2d.output[0],
                                                              elem_type=data_type,
                                                              shape=maxpool2d_output_shape)
    onnx_model.graph.value_info.extend([top_reshape_value_info, maxpool2d_value_info])
    onnx_model.graph.initializer.extend([top_shape_init, bot_shape_init])
    node_index = get_node_id(onnx_model, node)
    onnx_model.graph.node.remove(node)
    onnx_model.graph.node.insert(node_index, bot_reshape_node)
    onnx_model.graph.node.insert(node_index, maxpool2d)
    onnx_model.graph.node.insert(node_index, top_reshape_node)
    onnx_model = delete_useless_input_in_initializer(onnx_model)
    return onnx_model, True
