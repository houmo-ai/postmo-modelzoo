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


def get_onnx_module():
    return './quant_resnet50.onnx'


def compile():
    # Compile model
    filename = 'resnet50'
    onnxfile = get_onnx_module()
    onnx_model = onnx.load(onnxfile)
    input_name = onnx_model.graph.input[0].name
    input_format = 'YUV422SP'
    dims = onnx_model.graph.input[0].type.tensor_type.shape.dim
    input_shape = (1, dims[1].dim_value, dims[2].dim_value, dims[3].dim_value)
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
#    tcim.store_model(filename, graph, params, lib)
    rt_opt = '-resizer'
    tcim.store_as_fusedop(filename, graph, params, shape_dict, lib, rt_opt)

    print(filename, ' saved as one fusedop model.')


if __name__ == '__main__':
    compile()
