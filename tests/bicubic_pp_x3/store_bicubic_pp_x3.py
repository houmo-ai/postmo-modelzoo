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
        default='./hmquant_bicubic_pp_with_act.onnx',
        help='path to the model root path',
    )
    parser.add_argument(
        '--output',
        type=str,
        default='bicubic_pp',
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

    convert_config = {'layout': 'NHWC'}
    type_dict = {input_name: 'uint8'}
    shape_dict = {input_name: input_shape}
    mod = relay.frontend.from_hmonnx(
        onnx_model, shape_dict, type_dict, resizer_attr=None, convert_config=convert_config
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
    model_name = 'bicubic_pp'
    local_path = model_name
    quant_path = 'bicubic_pp.zip'
    if not os.path.exists(quant_path):
        os.system('curl -upublic:Password@123 \
            http://10.10.1.53:8082/artifactory/hdpl_test_data/quant_models/bicubic_pp.zip -o bicubic_pp.zip')
        os.system('unzip -o bicubic_pp.zip')
    compile()
