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
        default='./hmquant_lane_512x1536_with_act.onnx',
        help='path to the model root path',
    )
    parser.add_argument(
        '--output',
        type=str,
        default='lane_512x1536',
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
    # TODO: remove the next clause after quantool release 1.2
    print('input name:', input_name)
    print('input shape:', input_shape)

    # convert_config = {'layout': 'NCHW'}
    type_dict = {input_name: 'uint8'}
    shape_dict = {input_name: input_shape}
    mod = relay.frontend.from_hmonnx(
        onnx_model, shape_dict, type_dict, resizer_attr=None,
    )

    from tvm.relay.backend import Executor
    executor = Executor('aot')
    target = tvm.target.Target('hdpl', host='c')
    #compile_config = {}
    compile_config = {"tcim.for_benchmark": True, "tcim.fuse_strategy": 1, "tcim.codegen_pic": True}

    with tvm.transform.PassContext(opt_level=3, config=compile_config):
        graph, lib, params = relay.build(
            mod, target, executor=executor, mod_name=filename,
        )
    tcim.store_so(filename, lib, hdplcc_options=['-O2'])

    print(filename, ' saved as aot model.')


if __name__ == '__main__':
    model_name = 'lane_512x1536'
    local_path = model_name
    quant_path = 'hmquant_lane_512x1536_with_act.onnx'
    # if not os.path.exists(quant_path):
        # os.system('wget http://10.10.1.53:8082/artifactory/model_zoo2/houmo/yolov5/yolov5s_640x640_without_ptprocess.onnx')
    compile()
