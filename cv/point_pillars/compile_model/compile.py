import argparse
import os

import numpy as np
import onnx
import tvm
import tvm.relay as relay
import tvm.relay.frontend.hmonnx as hm_onnx
import tvm.tcim as tcim
from tvm import te
from tvm.relay.backend import Executor
from tvm.relay.backend import Runtime


def get_args() -> argparse.Namespace:
    """Parse commandline."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--pfe-model-path',
        dest='pfe_model_path',
        type=str,
        help='path to the pfe model path',
    )
    parser.add_argument(
        '--pfe-output',
        dest='pfe_output',
        type=str,
        help='output houmo pfe model path',
    )
    parser.add_argument(
        '--rpn-model-path',
        dest='rpn_model_path',
        type=str,
        help='path to the rpn model path',
    )
    parser.add_argument(
        '--rpn-output',
        dest='rpn_output',
        type=str,
        help='output houmo rpn model path',
    )
    args = parser.parse_args()
    return args


def compile_pfe(args=None):
    # Compile model
    filename = args.pfe_output
    onnxfile = args.pfe_model_path
    onnx_model = onnx.load(onnxfile)
    input_name = onnx_model.graph.input[0].name
    dims = onnx_model.graph.input[0].type.tensor_type.shape.dim
    channel = 12032
    input_shape = (
        dims[0].dim_value, dims[1].dim_value,
        channel, dims[3].dim_value,
    )
    print('input name:', input_name)
    print('input shape:', input_shape)

    type_dict = {input_name: 'int8'}
    shape_dict = {input_name: input_shape}
    layout_dict = {input_name: 'NHWC'}
    convert_config = {'transpose_axes': [0, 2, 3, 1]}
    mod = relay.frontend.from_hmonnx(
        onnx_model, shape_dict, type_dict, layout=layout_dict, resizer_attr=None, convert_config=convert_config,
    )
    executor = Executor('aot')
    compile_config = {
        'tcim.fuse_strategy': 1,
        'tcim.codegen_pic': True,
        'tcim.sync_strategy': 0,
    }
    target = tvm.target.Target('hdpl', host='c')
    with tvm.transform.PassContext(opt_level=3, config=compile_config):
        graph, lib, params = relay.build(
            mod, target, executor=executor, mod_name='pfe',
        )
    tcim.store_so(filename, lib)

    print(filename, ' saved as a aot model.')


def compile_rpn(args=None):
    # Compile model
    filename = args.rpn_output
    onnxfile = args.rpn_model_path
    onnx_model = onnx.load(onnxfile)
    input_name = onnx_model.graph.input[0].name
    dims = onnx_model.graph.input[0].type.tensor_type.shape.dim
    input_shape = (
        dims[0].dim_value, dims[1].dim_value,
        dims[2].dim_value, dims[3].dim_value,
    )
    print('input name:', input_name)
    print('input shape:', input_shape)

    type_dict = {input_name: 'int8'}
    shape_dict = {input_name: input_shape}
    mod = relay.frontend.from_hmonnx(
        onnx_model, shape_dict, type_dict, resizer_attr=None,
    )
    executor = Executor('aot')
    compile_config = {
        'tcim.fuse_strategy': 1,
        'tcim.gen_intrinsic': False,
        'tcim.sync_strategy': 0,
        'tcim.codegen_pic': True,
        'tcim.for_benchmark': False,
        'tcim.core_num': 1,
    }
    target = tvm.target.Target('hdpl', host='c')
    with tvm.transform.PassContext(opt_level=3, config=compile_config):
        graph, lib, params = relay.build(
            mod, target, executor=executor, mod_name='rpn',
        )
    tcim.store_so(filename, lib)

    print(filename, ' saved as a aot model.')


if __name__ == '__main__':
    args = get_args()
    if args.pfe_output is not None:
        compile_pfe(args)
    if args.rpn_output is not None:
        compile_rpn(args)
