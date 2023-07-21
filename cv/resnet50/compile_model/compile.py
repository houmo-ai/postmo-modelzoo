import argparse
import os

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
        default='./quant_resnet50.onnx',
        help='path to the model root path',
    )
    parser.add_argument(
        '--output',
        type=str,
        default='resnet50',
        help='output houmo model path',
    )
    parser.add_argument(
        '--batch',
        type=int,
        default=28,
        help='Set batch size for implicit batch houmo model',
    )
    args = parser.parse_args()
    return args


def reset_so():
    os.system('cp /opt/resnet50*.so .')


def compile(args=None):
    if args is None:
        args = get_args()
    # Compile model
    filename = args.output
    batch = args.batch
    onnxfile = args.model_path
    onnx_model = onnx.load(onnxfile)
    input_name = onnx_model.graph.input[0].name
    dims = onnx_model.graph.input[0].type.tensor_type.shape.dim
    input_shape = (
        batch, dims[1].dim_value,
        dims[2].dim_value, dims[3].dim_value,
    )
    print('input name:', input_name)
    print('input shape:', input_shape)

    convert_config = {'layout': 'NHWC'}
    type_dict = {input_name: 'uint8'}
    shape_dict = {input_name: input_shape}
    mod = relay.frontend.from_hmonnx(
        onnx_model, shape_dict, type_dict, resizer_attr=None, convert_config=convert_config,
    )
    with relay.build_config(opt_level=3):
        graph, lib, params = relay.build(mod, 'hdpl --host=llvm')

    # store model as one fusedop
    rt_opt = '-resizer'
    tcim.store_as_fusedop(filename, graph, params, shape_dict, lib, rt_opt)
    if batch == 28:
        reset_so()

    print(filename, ' saved as one fusedop model.')


if __name__ == '__main__':
    compile()
