import os
import onnx
import json
import tvm.relay.frontend.hmonnx as hm_onnx
from tvm.relay.frontend.hmonnx import ResizerAttr
import numpy as np
import tvm
from tvm import te
import tvm.relay as relay
from PIL import Image
from tvm.contrib import graph_executor
import tvm.contrib.graph_executor as runtime
def get_onnx_module():
    if not os.path.exists('box_head_without_softmax.zip'):
        os.system('curl -upublic:Password@123 http://10.10.1.53:8082/artifactory/hdpl_test_data/quant_models/apollo/obstacle/box_head_without_softmax.zip -o ./box_head_without_softmax.zip')
    if not os.path.exists('box_head_without_softmax'):
        os.system('unzip box_head_without_softmax.zip')
    return "./box_head_without_softmax/hmquant_box_head_with_act_without_softmax.onnx"

if __name__ == '__main__':
    #Compile model
    onnxfile = get_onnx_module()
    onnx_model = onnx.load(onnxfile)
    input_name = onnx_model.graph.input[0].name
    input_shape = (1, 512, 8, 8)
    print("input name:",input_name)

    graph = onnx_model.graph
    nodes = graph.node
    resizer_attr = None
    type_dict={input_name:"int8"}
    shape_dict = {input_name: input_shape}
    mod = relay.frontend.from_hmonnx(onnx_model, shape_dict, type_dict,\
        resizer_attr=None)
    with relay.build_config(opt_level=3):
        graph, lib, params = relay.build(mod, "hdpl --host=llvm")
    print("build model done.")

    from tvm.relay import param_dict
    filename = "libboxhead_without_softmax"
    lib.export_library(filename + ".so", tvm.contrib.cc.create_shared)
    with open(filename + ".json", "w") as fp:
      fp.write(graph)
    params_ba = param_dict.save_param_dict(params)
    with open(filename + ".params", "wb") as fp:
      fp.write(params_ba)
    print('tvm runtime saved')
