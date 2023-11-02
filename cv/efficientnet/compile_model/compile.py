import argparse
import os
import shutil
import time

import numpy as np
import onnx
import tvm
import tvm.relay as relay
import tvm.tcim as tcim
from tvm.relay import param_dict
from tvm.relay.backend import Executor
from tvm.relay.backend import Runtime


def get_args() -> argparse.Namespace:
    """Parse commandline."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--model-path',
        dest='model_path',
        type=str,
        default='hmquant_efficient_with_act.onnx',
        help='path to the model root path',
    )
    parser.add_argument(
        '--output',
        type=str,
        default='efficientnet',
        help='output houmo model path',
    )
    parser.add_argument(
        '--batch',
        type=int,
        default=1,
        help='Set batch size for implicit batch houmo model',
    )
    parser.add_argument(
        '--mode',
        type=str,
        default='eval',
        help='Set batch size for implicit batch houmo model',
    )
    args = parser.parse_args()
    return args


def compile(args=None):
    # Compile model
    if args is None:
        args = get_args()
    filename = args.output
    batch = args.batch
    onnxfile = args.model_path
    onnx_model = onnx.load(onnxfile)
    input_name = onnx_model.graph.input[0].name
    print('input name:', input_name)
    type_dict = {input_name: 'uint8'}
    shape_dict = {input_name: (1, 3, 224, 224)}
    layout_dict = {input_name, 'NHWC'}
    mod = relay.frontend.from_hmonnx(
        onnx_model, shape_dict, type_dict,
        layout=layout_dict, resizer_attr=None, convert_config=None,
    )
    executor = Executor('aot')
    target = tvm.target.Target('hdpl', host='c')
    if args.mode == 'eval':
        compile_config = {
            'tcim.gen_intrinsic': 1,
            'tcim.sync_strategy': 1,
            'tcim.for_benchmark': True,
        }
    else:
        compile_config = {
            'tcim.fuse_strategy': 0,
            'tcim.spec_batch_num': batch,
            'tcim.input_data_fmt': 'YUV422SP',
            'tcim.gen_intrinsic': 1,
            'tcim.sync_strategy': 1,
            'tcim.for_benchmark': True,
        }

    with tvm.transform.PassContext(opt_level=3, config=compile_config):
        graph, lib, params = relay.build(
            mod, target, executor=executor, mod_name=filename,
        )
    tcim.store_so(filename, lib, hdplcc_options=['-O2'])
    print('store model done.')


if __name__ == '__main__':
    compile()
