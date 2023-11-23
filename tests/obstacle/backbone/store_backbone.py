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
from tvm.relay.backend import Executor
import tvm.tcim as tcim


def get_onnx_module():
    print("use local onnx file--szj")
    return "./backbone/hmquant_backbone_rpn_with_act.onnx"

if __name__ == '__main__':
    #Compile model
    onnxfile = get_onnx_module()
    onnx_model = onnx.load(onnxfile)
    input_name = onnx_model.graph.input[0].name
    input_shape = (1, 3, 720, 1280)
    input_format="YUV422SP"
    print("input name:",input_name)
    print("input shape:",input_shape)

    resizer_attr = None
    type_dict={input_name:"uint8"}
    shape_dict = {input_name: input_shape}
    mod = relay.frontend.from_hmonnx(onnx_model, shape_dict, type_dict, resizer_attr=resizer_attr)
    executor = Executor('aot')
    compile_config={
        'tcim.fuse_strategy': 1,
        'tcim.codegen_pic': True,
#        'tcim.gen_intrinsic': 2,
        'tcim.schedule_strategy': 2,
        'tcim.sync_strategy': 1,
    }
    target = tvm.target.Target('hdpl', host='c')
    with tvm.transform.PassContext(opt_level=3, config=compile_config):
        graph, lib, params = relay.build(mod, target, executor=executor)
    tcim.store_so('backbone', lib)

