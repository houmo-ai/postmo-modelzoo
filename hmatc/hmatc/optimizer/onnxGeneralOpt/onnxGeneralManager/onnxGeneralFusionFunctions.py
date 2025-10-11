import copy
from ....utils import logger
#import math

import numpy as np  # type: ignore
import onnx  # type: ignore

#from ..onnxBaseOpt.onnxRuntimeEngine import OnnxRuntimeEngine
from .onnxGeneralDeleteFunctions import delete_shape_useless_node
from ...onnxBaseOpt.onnxDebugger import OnnxDebugger
#from ..onnxBaseOpt.onnxBaseFunctions import infer_shapes
from ...onnxBaseOpt.onnxBaseOptimizer import OnnxBaseOptimizer
from ...onnxUtils.onnxBasicUtils import *
#from ..onnxBaseOpt.onnxConfigController import OnnxCfg


@OnnxDebugger.onnx_opt_func_debug_wrapper
@OnnxBaseOptimizer.onnx_opt_once_wrapper
def fusion_ReshapeReshape(onnx_model, node):
    '''
    Explanation:This function completes the fusion of two reshape.
    example: Reshape + Reshape -> Reshape
    Author: Nan.xu
    '''
    if node.op_type == "Reshape":
        reshape_node = node
        model_output_names = [output.name for output in onnx_model.graph.output]
        if reshape_node.output[0] in model_output_names:
            return onnx_model, False
        next_node_list = get_node_by_input(onnx_model, reshape_node.output)
        next_reshape_list = [next_node for next_node in next_node_list if next_node.op_type == "Reshape"
                             and reshape_node.output[0] == next_node.input[0]]
        if not next_reshape_list:
            return onnx_model, False
        input_shape = get_shape_by_name(onnx_model, reshape_node.input[0])
        if len(next_reshape_list) != len(next_node_list) and len(input_shape) > 4:
            return onnx_model, False
        logger.debug("Fusion:Reshape+Reshape->Reshape")
        logger.debug(f"Nodes:{reshape_node.name}")
        logger.debug(f"Input:{reshape_node.input}")
        shape_to_name = {}
        for next_reshape_node in next_reshape_list:
            next_reshape_node.input[0] = reshape_node.input[0]
            onnx_model, status = delete_shape_useless_node(onnx_model, next_reshape_node, "Reshape")
            if not status:
                output_shape = get_shape_by_name(onnx_model, next_reshape_node.output[0])
                next_reshape_shape_init = get_initializer(onnx_model, next_reshape_node.input[1])
                if output_shape != onnx.numpy_helper.to_array(next_reshape_shape_init).tolist():
                    next_reshape_shape_arr = np.array(output_shape, dtype=np.int64)
                    new_next_reshape_shape_init = onnx.numpy_helper.from_array(next_reshape_shape_arr,
                                                                               next_reshape_shape_init.name)
                    next_reshape_shape_init.CopyFrom(new_next_reshape_shape_init)
                if next_reshape_node.output[0] not in model_output_names:
                    out_shape = tuple(output_shape)
                    if out_shape in shape_to_name.keys():
                        replace_input_of_all_nodes(onnx_model, next_reshape_node.output[0], shape_to_name[out_shape])
                        onnx_model.graph.node.remove(next_reshape_node)
                    else:
                        shape_to_name[out_shape] = next_reshape_node.output[0]

        if len(next_node_list) == len(next_reshape_list):
            onnx_model.graph.node.remove(reshape_node)
            onnx_model.graph.initializer.remove(get_initializer(onnx_model, reshape_node.input[1]))
        return onnx_model, True
    return onnx_model, False

@OnnxDebugger.onnx_opt_func_debug_wrapper
@OnnxBaseOptimizer.onnx_opt_once_wrapper
def fusion_TransposeTranspose(onnx_model, node):
    '''
    Explanation:This function completes the fusion of two tranpose.
    example: Transpose + Transpose -> Transpose
    Author: Nan.xu
    '''
    if node.op_type == "Transpose":
        transpose_node = node
        model_outputs_names = [output.name for output in onnx_model.graph.output]
        if transpose_node.output[0] in model_outputs_names:
            return onnx_model, False
        next_nodes_list = get_node_by_input(onnx_model, transpose_node.output)
        next_transpose_list = [node for node in next_nodes_list if node.op_type == "Transpose"]
        if not next_transpose_list:
            return onnx_model, False
        logger.debug("Fusion:Transpose+Transpose->Transpose")
        logger.debug(f"Nodes:{transpose_node.name}")
        logger.debug(f"Input:{transpose_node.input}")
        input_shape = get_shape_by_name(onnx_model, transpose_node.input[0])
        first_transpose_attr = attribute_to_dict(transpose_node.attribute)
        first_transpose_perm = first_transpose_attr.get("perm", list(reversed(range(len(input_shape)))))
        for next_transpose_node in next_transpose_list:
            perm = attribute_to_dict(next_transpose_node.attribute).get("perm", list(reversed(range(len(input_shape)))))
            fusion_perm = [first_transpose_perm[perm_id] for perm_id in perm]
            next_transpose_node.input[0] = transpose_node.input[0]
            del next_transpose_node.attribute[:]
            next_transpose_node.attribute.append(onnx.helper.make_attribute("perm", fusion_perm))

            if fusion_perm == list(range(len(input_shape))):
                delete_shape_useless_node(onnx_model, next_transpose_node, "Transpose")
        if len(next_nodes_list) == len(next_transpose_list):
            onnx_model.graph.node.remove(transpose_node)
        return onnx_model, True
    return onnx_model, False

@OnnxDebugger.onnx_opt_func_debug_wrapper
@OnnxBaseOptimizer.onnx_opt_traverse_wrapper
def fusion_TransposePoolTranspose(onnx_model, node, node_index):
    '''
    Explanation: Transpose+MaxPool2D+Transpose is fused with the previous or subsequent conv or converted to conv.
    example: 
        (data 1x128x18x44) -> Transpose(perm=[0,2,3,1]) -> 
            Pool(kernel_shape=[1, 1],dilations=[1, 1],pads=[0,0,0,0],strides=[1, 2]) -> Transpose(perm=[0,3,1,2]) -> (1x64x18x44)
        (data NxC0xHxW) -> Transpose(perm=[1 -> 2 or 3]) -> Pool(only strides is valid(C0 -> C1)) -> Transpose(perm=[2 or 3 -> 1]) -> (NxC1xHxW)
    Author: Nan Xu
    '''
    def convert_conv(onnx_model, nodes_list, parameters):
        conv_attr = {"dilations": [1, 1],
                     "group": 1,
                     "kernel_shape": [1, 1],
                     "pads": parameters['pool_pads'],
                     "strides": [1, 1]}
        conv_input_channel = parameters['input_shape'][1]
        conv_output_channel = conv_input_channel / parameters['pool_stride']
        weights = np.zeros((conv_output_channel, conv_input_channel, 1, 1), dtype=np.float32)
        for h in conv_output_channel:
            weights[h, h * parameters['pool_stride'], 0, 0] = 1.0
        pool_input_value_info = get_value_info_by_name(onnx_model, nodes_list[1].input[0])
        pool_output_value_info = get_value_info_by_name(onnx_model, nodes_list[1].output[0])
        onnx_model.graph.value_info.remove(pool_input_value_info)
        onnx_model.graph.value_info.remove(pool_output_value_info)
        if parameters['new_perm'] != list(range(len(parameters['input_shape']))):
            del nodes_list[2].attribute[:]
            nodes_list[2].attribute.append(onnx.helper.make_attribute('perm', parameters['new_perm']))
            conv_output = nodes_list[2].input[0]
            conv_output_shape = parameters['input_shape']
            conv_output_shape[1] = conv_output_shape[1] / parameters['pool_stride']
            conv_value_info = onnx.helper.make_tensor_value_info(name=conv_output,
                                                                 elem_type=1,
                                                                 shape=conv_output_shape)
            onnx_model.graph.value_info.append(conv_value_info)
            onnx_model = delete_nodes(onnx_model, nodes_list[:2])
        else:
            conv_output = nodes_list[2].output[0]
            onnx_model = delete_nodes(onnx_model, nodes_list)
        conv_node = onnx.helper.make_node(op_type="Conv", 
                                          name=nodes_list[1].name+"_conv", 
                                          inputs=[nodes_list[0].input[0], nodes_list[1].name+"_weight"], 
                                          outputs=nodes_list[2].output, 
                                          **conv_attr)
        onnx_model.graph.node.insert(parameters['node_index'], conv_node)
        onnx_model = delete_useless_input_in_initializer(onnx_model)
        return onnx_model, True

    def convert_convpool(onnx_model, nodes_list, conv_node, parameters):
        conv_attr = attribute_to_dict(conv_node.attribute)
        conv_group = conv_attr.get('group', 1)
        if conv_group != 1 or set(parameters['pool_pads']) != {0}:
            return onnx_model, False
        conv_weights = get_tensor_from_initializer(onnx_model, conv_node.input[1])
        if conv_weights.size == 0:
            return onnx_model, False 
        new_conv_weights = conv_weights[::parameters['pool_stride'], :, :, :]
        new_weights_init = onnx.numpy_helper.from_array(new_conv_weights, conv_node.input[1]+"_fusion_pool")
        onnx_model.graph.initializer.append(new_weights_init)
        conv_node.input[1] = new_weights_init.name
        if len(conv_node.input) > 1:
            conv_bais = get_tensor_from_initializer(onnx_model, conv_node.input[2])
            if conv_bais.size != 0:
                new_conv_bais = conv_bais[::parameters['pool_stride']]
                new_bais_init = onnx.numpy_helper.from_array(new_conv_bais, conv_node.input[2]+"_fusion_pool")
                onnx_model.graph.initializer.append(new_bais_init)
                conv_node.input[2] = new_bais_init.name
        value_info_top = get_value_info_by_name(onnx_model, nodes_list[0].input[0])
        value_info_mid = get_value_info_by_name(onnx_model, nodes_list[1].input[0])
        value_info_bot = get_value_info_by_name(onnx_model, nodes_list[2].input[0])
        onnx_model.graph.value_info.remove(value_info_bot)
        onnx_model.graph.value_info.remove(value_info_top)
        onnx_model.graph.value_info.remove(value_info_mid)
        if parameters['new_perm'] != list(range(len(parameters['input_shape']))):
            del nodes_list[2].attribute[:]
            nodes_list[2].attribute.append(onnx.helper.make_attribute("perm", parameters['new_perm']))
            conv_node.output[0] = nodes_list[2].input[0]
            new_conv_out_shape = parameters['input_shape']
            new_conv_out_shape[1] = new_conv_out_shape[1] / parameters['pool_stride']
            new_conv_value_info = onnx.helper.make_tensor_value_info(name=conv_node.output[0],
                                                                     elem_type=1,
                                                                     shape=new_conv_out_shape)
            onnx_model.graph.value_info.append(new_conv_value_info)
            onnx_model = delete_nodes(onnx_model, nodes_list[:2])
        else:
            conv_node.output[0] = nodes_list[2].output[0]
            onnx_model = delete_nodes(onnx_model, nodes_list)
        onnx_model = delete_useless_input_in_initializer(onnx_model)
        return onnx_model, True
    
    def convert_poolconv(onnx_model, nodes_list, conv_node, parameters):
        conv_attr = attribute_to_dict(conv_node.attribute)
        conv_group = conv_attr.get('group', 1)
        if conv_group != 1:
            return onnx_model, False
        conv_pads = conv_attr.get('pads', [0, 0, 0, 0])
        conv_weights = get_tensor_from_initializer(onnx_model, conv_node.input[1])
        if conv_weights.size == 0:
            return onnx_model, False
        new_conv_weights = np.zeros(
            (conv_weights.shape[0], conv_weights.shape[1] * parameters['pool_stride'], *conv_weights.shape[2:]),
            dtype=conv_weights.dtype)
        new_conv_weights[:, ::parameters['pool_stride'], ...] = conv_weights
        new_weights_init = onnx.numpy_helper.from_array(new_conv_weights, conv_node.input[1]+"_fusion_pool")
        onnx_model.graph.initializer.append(new_weights_init)
        # new_pads = [conv_pads[i] + v for i, v in enumerate(parameters['pool_pads'])]
        # if conv_attr.get('auto_pad', 'VALID') == 'VALID':
        #     conv_attr['pads'] = new_pads
        #     conv_attr.pop('auto_pad', None)
        # del conv_node.attribute[:]
        # conv_node.attribute.extend(onnx.helper.make_attribute(key, value) for key, value in sorted(conv_attr.items()))
        value_info_top = get_value_info_by_name(onnx_model, nodes_list[0].output[0])
        value_info_mid = get_value_info_by_name(onnx_model, nodes_list[1].output[0])
        value_info_bot = get_value_info_by_name(onnx_model, nodes_list[2].output[0])
        onnx_model.graph.value_info.remove(value_info_top)
        onnx_model.graph.value_info.remove(value_info_mid)
        onnx_model.graph.value_info.remove(value_info_bot)
        conv_node.input[1] = new_weights_init.name
        if parameters['new_perm'] != list(range(len(parameters['input_shape']))):
            del nodes_list[0].attribute[:]
            nodes_list[0].attribute.append(onnx.helper.make_attribute('perm', parameters['new_perm']))
            new_transpose_output_shape = [parameters['input_shape'][p] for p in parameters['new_perm']]
            new_transpose_output_value_info = onnx.helper.make_tensor_value_info(name=nodes_list[0].output[0],
                                                                                 elem_type=1,
                                                                                 shape=new_transpose_output_shape)
            onnx_model.graph.value_info.append(new_transpose_output_value_info)
            conv_node.input[0] = nodes_list[1].input[0]
            onnx_model = delete_nodes(onnx_model, nodes_list[1:])
        else:
            conv_node.input[0] = nodes_list[0].input[0]
            onnx_model = delete_nodes(onnx_model, nodes_list)
        onnx_model = delete_useless_input_in_initializer(onnx_model)
        return onnx_model, True

    if check_node_serial_group(onnx_model, node, ["Transpose", "MaxPool", "Transpose"]):
        nodes_type_list = ["Transpose", "MaxPool", "Transpose"]
    elif check_node_serial_group(onnx_model, node, ["Transpose", "AveragePool", "Transpose"]):
        nodes_type_list = ["Transpose", "AveragePool", "Transpose"]
    else:
        nodes_type_list = None
    if nodes_type_list is not None:
        nodes_list = get_node_serial_group(onnx_model, node, nodes_type_list)
        top_transpose_node, pool_node, bot_transpose_node = nodes_list
        input_shape = get_shape_by_name(onnx_model, top_transpose_node.input[0])
        if len(input_shape) != 4:
            return onnx_model, False
        pool_attr = attribute_to_dict(pool_node.attribute)
        dilations = pool_attr.get("dilations", [1, 1])
        kernel_shape = pool_attr.get("kernel_shape")
        pads = pool_attr.get("pads", [0, 0, 0, 0])
        strides = pool_attr.get("strides", [1, 1])
        indices = [i for i, s in enumerate(strides) if s != 1]
        if set(dilations) != {1} or set(kernel_shape) != {1} or set(pads) != {0} or len(indices) != 1:
            return onnx_model, False
        reduce_axis = indices[0] + 2
        top_perm = attribute_to_dict(top_transpose_node.attribute).get("perm", list(range(len(input_shape))))
        bot_perm = attribute_to_dict(bot_transpose_node.attribute).get("perm", list(range(len(input_shape))))
        new_perm = [top_perm[p] for p in bot_perm]
        if top_perm[reduce_axis] != 1 and bot_perm.index(reduce_axis) != 1:
            return onnx_model, False
        pre_node = get_node_by_output(onnx_model, top_transpose_node.input[0])
        samelevel_nodes = get_node_by_input(onnx_model, pre_node.output)
        next_nodes = get_node_by_input(onnx_model, bot_transpose_node.output)
        parameters = {'input_shape': input_shape,
                      'pool_pads': pads,
                      'node_index': node_index,
                      'reduce_axis': reduce_axis,
                      'pool_stride': strides[indices[0]],
                      'top_perm': top_perm,
                      'new_perm': new_perm}
        logger.debug("Fusion:Transpose+Pool+Transpose->Conv")
        logger.debug(f"Nodes:{pool_node.name}")
        logger.debug(f"Input:{pool_node.input}")
        if pre_node.op_type == "Conv" and len(samelevel_nodes) == 1 and top_perm[reduce_axis] == 1:
            onnx_model, status = convert_convpool(onnx_model, nodes_list, pre_node, parameters)
            if status:
                return onnx_model, status
        if next_nodes[0].op_type == "Conv" and len(next_nodes) == 1 and bot_perm.index(reduce_axis) == 1:
            onnx_model, status = convert_poolconv(onnx_model, nodes_list, next_nodes[0], parameters) ##if convert poolconv false, convert conv
            if status:
                return onnx_model, status
        return convert_conv(onnx_model, nodes_list, parameters)
    return onnx_model, False

@OnnxDebugger.onnx_opt_func_debug_wrapper
def fusion_TransposeReshapePoolReshapeTranspose(onnx_model):
    '''
    Explanation: Transpose+Reshape+MaxPool2D+Reshape+Transpose is fused with the previous or subsequent conv or converted to conv.
    example: 
        (data 1x128x18x44) -> Transpose(perm=[0,2,3,1]) -> Reshape(18,44,1,128) -> 
            Pool(kernel_shape=[1, 1],dilations=[1, 1],pads=[0,0,0,0],strides=[1, 2]) -> Reshape(1,18,44,64) -> Transpose(perm=[0,3,1,2]) -> (1x64x18x44)
        (data NxC0xHxW) -> Transpose(perm=[1 -> 2 or 3]) -> Reshape(keep C0 unchanged) -> Pool(only strides is valid(C0 -> C1)) 
            -> Reshape(keep C1 unchanged) -> Transpose(perm=[2 or 3 -> 1]) -> (NxC1xHxW)
    Author: Nan Xu
    '''
    @OnnxBaseOptimizer.onnx_opt_once_wrapper
    def fusion_ReshapePoolReshape(onnx_model, node):
        if check_node_serial_group(onnx_model, node, ["Reshape", "MaxPool", "Reshape"]):
            nodes_type_list = ["Reshape", "MaxPool", "Reshape"]
        elif check_node_serial_group(onnx_model, node, ["Reshape", "AveragePool", "Reshape"]):
            nodes_type_list = ["Reshape", "AveragePool", "Reshape"]
        else:
            nodes_type_list = None
        if nodes_type_list is not None:
            nodes_list = get_node_serial_group(onnx_model, node, nodes_type_list)
            pool_input_shape = get_shape_by_name(onnx_model, nodes_list[1].input[0])
            output_shape = get_shape_by_name(onnx_model, nodes_list[2].output[0])
            if len(pool_input_shape) != 4 or len(output_shape) != 4:
                return onnx_model, False
            #top_perm = attribute_to_dict(nodes_list[0].attribute).get("perm", list(range(len(input_shape))))
            pool_attr = attribute_to_dict(nodes_list[1].attribute)
            dilations = pool_attr.get("dilations", [1, 1])
            kernel_shape = pool_attr.get("kernel_shape")
            pads = pool_attr.get("pads", [0, 0, 0, 0])
            strides = pool_attr.get("strides", [1, 1])
            indices = [i for i, s in enumerate(strides) if s != 1]
            if set(dilations) != {1} or set(kernel_shape) != {1} or set(pads) != {0} or len(indices) != 1:
                return onnx_model, False
            reduce_axis = indices[0] + 2
            input_shape = get_shape_by_name(onnx_model, nodes_list[0].input[0])
            rm_input_shape = copy.deepcopy(input_shape)
            rm_input_shape.pop(reduce_axis)
            rm_output_shape = copy.deepcopy(output_shape) 
            rm_output_shape.pop(reduce_axis)
            if rm_input_shape != rm_output_shape:
                return onnx_model, False
            logger.debug("Fusion:Reshape+Pool+Reshape->Pool")
            logger.debug(f"Nodes:{nodes_list[1].name}")
            logger.debug(f"Input:{nodes_list[1].input}")
            pool_node = nodes_list[1]
            top_reshape_value_info = get_value_info_by_name(onnx_model, nodes_list[0].output[0])
            onnx_model.graph.value_info.remove(top_reshape_value_info)
            pool_value_info = get_value_info_by_name(onnx_model, pool_node.output[0])
            onnx_model.graph.value_info.remove(pool_value_info)
            pool_node.input[0] = nodes_list[0].input[0]
            pool_node.output[0] = nodes_list[2].output[0]
            onnx_model = delete_nodes(onnx_model, [nodes_list[0], nodes_list[2]])
            onnx_model = delete_useless_input_in_initializer(onnx_model)
            return onnx_model, True
        return onnx_model, False
    onnx_model = fusion_ReshapeReshape(onnx_model)
    onnx_model, restart = fusion_ReshapePoolReshape(onnx_model)
    if restart:
        onnx_model = fusion_TransposePoolTranspose(onnx_model)
    return onnx_model, restart