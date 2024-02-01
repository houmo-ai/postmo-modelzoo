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
from tvm.relay.backend import Executor
from tvm.relay.backend import Runtime
from tvm.relay.frontend.hmonnx import ResizerAttr


def get_args() -> argparse.Namespace:
    """Parse commandline."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--model-path',
        dest='model_path',
        type=str,
        default='./quant_vit.onnx',
        help='path to the model root path',
    )
    parser.add_argument(
        '--output',
        type=str,
        default='vit',
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
        #batch, dims[1].dim_value,
        dims[0].dim_value, dims[1].dim_value,
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
    executor = Executor('aot')
    if batch == 1:
        compile_config = {
            'tcim.fuse_strategy': 1,
            'tcim.gen_intrinsic': 0,
            'tcim.sync_strategy': 0,
            'tcim.for_benchmark': True,
            'tcim.spec_batch_num': batch,
            "tcim.special_model_name": "vit_small"
        }
    else:
        compile_config = {
            'tcim.fuse_strategy': 1,
            'tcim.gen_intrinsic': 0,
            'tcim.sync_strategy': 0,
            'tcim.for_benchmark': True,
            'tcim.spec_batch_num': batch,
            "tcim.special_model_name": "vit_small"
        }
    target = tvm.target.Target('hdpl', host='c')
    with tvm.transform.PassContext(opt_level=3, config=compile_config):
        graph, lib, params = relay.build(
            mod, target, executor=executor, mod_name='vit',
        )

    tcim.store_so(filename, lib, 'vit', hdplcc_options=['-O2'])
    print(filename, ' saved as one fusedop model.')


if __name__ == '__main__':
    compile()
