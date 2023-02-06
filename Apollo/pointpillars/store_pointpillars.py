import os
import onnx
import tvm.relay.frontend.hmonnx as hm_onnx
from tvm.relay.frontend.hmonnx import ResizerAttr
import numpy as np
import tvm
from tvm import te
import tvm.relay as relay
from tvm.contrib import graph_executor
import tvm.contrib.graph_executor as runtime
def get_onnx_module():
    if not os.path.exists('traffic_light_recog.zip'):
        os.system('curl -upublic:Password@123 \
            http://10.10.1.53:8082/artifactory/hdpl_test_data/quant_models/pointpillars_rpn_116x116_20230202.zip \
            -o pointpillars_rpn.zip')
    if not os.path.exists('pointpillars_rpn'):
        os.system('unzip pointpillars_rpn.zip')
    return "./pointpillars_rpn/hmquant_pointpillars2_with_act.onnx"

if __name__ == '__main__':
    #Compile model
    onnxfile = get_onnx_module()
    onnx_model = onnx.load(onnxfile)
    input_name = onnx_model.graph.input[0].name
    input_shape = (1, 64, 116, 116)
    input_format="YUV420SP"
    print("input name:",input_name)
    print("input shape:",input_shape)

    type_dict={input_name:"int8"}
    shape_dict = {input_name: input_shape}
    mod = relay.frontend.from_hmonnx(onnx_model, shape_dict, type_dict, resizer_attr=None)
    with relay.build_config(opt_level=3):
        graph, lib, params = relay.build(mod, "hdpl --host=llvm")

    print("build model done.")

    from tvm.relay import param_dict
    filename = "libpointpillars2"
    lib.export_library(filename + ".so", tvm.contrib.cc.create_shared)
    with open(filename + ".json", "w") as fp:
      fp.write(graph)
    params_ba = param_dict.save_param_dict(params)
    with open(filename + ".params", "wb") as fp:
      fp.write(params_ba)
    print('tvm runtime saved')
