import os
import onnx
import numpy as np
import tvm
import tvm.relay as relay
import tvm.contrib.graph_executor as runtime
def get_onnx_module():
    if not os.path.exists('traffic_light_recog.zip'):
        os.system('curl -upublic:Password@123 \
            http://10.10.1.53:8082/artifactory/hdpl_test_data/quant_models/apollo/traffic_light/traffic_light_recog.zip \
            -o traffic_light_recog.zip')
    if not os.path.exists('traffic_light_recog'):
        os.system('unzip traffic_light_recog.zip')
    return "./traffic_light_recog/hmquant_traffic_light_recog_with_act.onnx"

if __name__ == '__main__':
    #Compile model
    onnxfile = get_onnx_module()
    onnx_model = onnx.load(onnxfile)
    input_name = onnx_model.graph.input[0].name
    input_shape = (1, 3, 96, 96)
    print("input name:",input_name)

    graph = onnx_model.graph
    print(graph.output)

    resizer_attr = None

    type_dict={input_name:"uint8"}
    shape_dict = {input_name: input_shape}
    mod = relay.frontend.from_hmonnx(onnx_model, shape_dict, type_dict, resizer_attr=None)
    with relay.build_config(opt_level=3):
        graph, lib, params = relay.build(mod, "hdpl --host=llvm")
    print("build model done.")

    from tvm.relay import param_dict
    filename = "liblight_recog"
    lib.export_library(filename + ".so", tvm.contrib.cc.create_shared)
    with open(filename + ".json", "w") as fp:
      fp.write(graph)
    params_ba = param_dict.save_param_dict(params)
    with open(filename + ".params", "wb") as fp:
      fp.write(params_ba)
    print('tvm runtime saved')
