import argparse
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
        default='./hmquant_bicubic_pp_x3_with_act.onnx',
        help='path to the model root path',
    )
    parser.add_argument(
        '--output',
        type=str,
        default='bicubic_pp_x3',
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
    """Compile quanted model to tcim model"""
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

    convert_config = {'layout': 'NCHW'}
    # convert_config = {"transpose_axes": [0, 3, 1, 2]}
    type_dict = {input_name: 'uint8'}
    shape_dict = {input_name: input_shape}
    mod = relay.frontend.from_hmonnx(
        onnx_model, shape_dict, type_dict, resizer_attr=None,
    )

    from tvm.relay.backend import Executor
    executor = Executor('aot')
    target = tvm.target.Target('hdpl', host='c')
    compile_config = {}
    #compile_config = {"tcim.fuse_strategy": 1, "tcim.codegen_pic": True, "tcim.core_mask": 0b11111, "tcim.for_benchmark": True}

    with tvm.transform.PassContext(opt_level=3, config=compile_config):
        graph, lib, params = relay.build(
            mod, target, executor=executor, mod_name=filename,
        )
    tcim.store_so(filename, lib, hdplcc_options=['-O2'])

    print(filename, ' saved as aot model.')


if __name__ == '__main__':
    model_name = 'bicubic_pp_x3'
    local_path = model_name
    quant_path = 'hmquant_bicubic_pp_x3.zip'
    if not os.path.exists(quant_path):
        os.system('wget http://10.10.1.53:8082/artifactory/toolchain/support/models/bicubic_pp_x3/hmquant_bicubic_pp_x3.zip')
        os.system('unzip hmquant_bicubic_pp_x3.zip')
    compile()
