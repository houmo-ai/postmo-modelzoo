import os
import onnx
import numpy as np
import tvm
import tvm.relay as relay
import tvm.contrib.graph_executor as runtime
def get_onnx_module():
    if not os.path.exists('cruise_cutin_vehicle.zip'):
        os.system('curl -upublic:Password@123 \
            http://10.10.1.53:8082/artifactory/hdpl_test_data/quant_models/cruise_cutin_vehicle.zip \
            -o cruise_cutin_vehicle.zip')
    if not os.path.exists('cruise_cutin_vehicle'):
        os.system('unzip cruise_cutin_vehicle.zip')
    return "./cruise_cutin_vehicle/hmquant_cruise_cutin_vehicle_model.3D_shape_namecorrect_quant.onnx_with_act.onnx"

if __name__ == '__main__':
    #Compile model
    onnxfile = get_onnx_module()
    onnx_model = onnx.load(onnxfile)
    input_name = onnx_model.graph.input[0].name
    input_shape = (1, 148)
    print("input name:",input_name)

    graph = onnx_model.graph
    print(graph.output)

    resizer_attr = None

    type_dict={input_name:"int8"}
    shape_dict = {input_name: input_shape}
    mod = relay.frontend.from_hmonnx(onnx_model, shape_dict, type_dict, resizer_attr=None)
    with relay.build_config(opt_level=3):
        graph, lib, params = relay.build(mod, "hdpl --host=llvm")
    print("build model done.")

    from tvm.relay import param_dict
    filename = "libcruise_cutin_vehicle"
    lib.export_library(filename + ".so", tvm.contrib.cc.create_shared)
    with open(filename + ".json", "w") as fp:
      fp.write(graph)
    params_ba = param_dict.save_param_dict(params)
    with open(filename + ".params", "wb") as fp:
      fp.write(params_ba)
    print('tvm runtime saved')