import argparse
import copy
import os

import hdpl
import numpy as np
import onnx
import tvm
import tvm.contrib.graph_executor as runtime
import tvm.relay as relay
import tvm.relay.frontend.hmonnx as hm_onnx
import tvm.tcim as tcim
from tvm import te
from tvm.contrib import graph_executor
from tvm.contrib import hdpl_graph_executor
from tvm.relay.frontend.hmonnx import ResizerAttr


def get_args() -> argparse.Namespace:
    """Parse commandline."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--model-path',
        dest='model_path',
        type=str,
        default='../../../data/models/quant_yolov3.onnx',
        help='path to the model root path',
    )
    parser.add_argument(
        '--output',
        type=str,
        default='yolov3',
        help='output houmo model path',
    )
    parser.add_argument(
        '--batch',
        type=int,
        default=1,
        help='Set batch size for implicit batch houmo model',
    )
    args = parser.parse_args()
    return args


def modify_onnx_model(onnxfile):
    onnx_model = onnx.load(onnxfile)
    graph = onnx_model.graph
    nodes = graph.node
    graph.node.remove(nodes[206])
    for i in range(205, 196, -1):
        graph.node.remove(nodes[i])
    for i in range(192, 183, -1):
        graph.node.remove(nodes[i])
    for i in range(179, 170, -1):
        graph.node.remove(nodes[i])

    graph.output[0].name = '341'
    graph.output[0].type.tensor_type.elem_type = 1
    out_0 = graph.output[0]
    out_1 = copy.deepcopy(out_0)
    out_2 = copy.deepcopy(out_0)
    out_1.name = '325'
    out_2.name = '309'
    graph.output.append(out_1)
    graph.output.append(out_2)
    # print(graph.output)
    return onnx_model


if __name__ == '__main__2':
    batch = 4
    filename = './libyolov3'
    onnxfile = get_onnx_module()
    onnx_model = modify_onnx_model(onnxfile)

    input_name = 'data'
    input_shape = (batch, 3, 416, 416)
    n, c, preprocess_h, preprocess_w = input_shape
    print('input name:', input_name)
    print('input shape:', input_shape)

    resizer_attr = ResizerAttr(enfold=1)
    type_dict = {input_name: 'uint8'}
    shape_dict = {input_name: input_shape}
    convert_config = {'layout': 'NHWC'}
    mod = relay.frontend.from_hmonnx(
        onnx_model, shape_dict, type_dict, resizer_attr=resizer_attr, convert_config=convert_config,
    )
    with relay.build_config(opt_level=3):
        graph, lib, params = relay.build(mod, 'hdpl --host=llvm')

    # store model as one fusedop
    #rt_opt = "-load -reiszer"
    tcim.store_as_fusedop(filename, graph, params, shape_dict, lib)


def compile(args=None):
    """Compile quanted model to tcim model"""
    if args is None:
        args = get_args()
    # Compile model
    filename = args.output
    batch = args.batch
    onnxfile = args.model_path
    onnx_model = modify_onnx_model(onnxfile)
#    onnx_model = onnx.load(onnxfile)
    input_name = onnx_model.graph.input[0].name
    dims = onnx_model.graph.input[0].type.tensor_type.shape.dim
    input_shape = (
        batch, dims[1].dim_value,
        dims[2].dim_value, dims[3].dim_value,
    )
    # TODO: remove the next clause after quantool release 1.2
    input_shape = (batch, 3, 416, 416)
    print('input name:', input_name)
    print('input shape:', input_shape)

    convert_config = {'layout': 'NHWC'}
    type_dict = {input_name: 'uint8'}
    shape_dict = {input_name: input_shape}
    resizer_attr = ResizerAttr(enfold=1)
    mod = relay.frontend.from_hmonnx(
        onnx_model, shape_dict, type_dict, resizer_attr=resizer_attr, convert_config=convert_config,
    )
    with relay.build_config(opt_level=3):
        graph, lib, params = relay.build(mod, 'hdpl --host=llvm')

    # store model as one fusedop
    tcim.store_as_fusedop(filename, graph, params, shape_dict, lib)

    print(filename, ' saved as one fusedop model.')


if __name__ == '__main__':
    compile()
