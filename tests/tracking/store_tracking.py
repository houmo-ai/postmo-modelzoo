import os

import numpy as np
import onnx
import tvm
import tvm.relay as relay
from tvm import tcim
from tvm.contrib import hdpl_graph_executor

local_path = 'apollo/tracking'

def get_onnx_module_and_golden():
    if not os.path.exists(local_path + '/hmquant_tracking_with_act.onnx'):
        os.system('curl -upublic:Password@123 \
            http://10.10.1.53:8082/artifactory/hdpl_test_data/quant_models/apollo/tracking/tracking_v2.zip  -o tracking.zip')
        if not os.path.exists(local_path):
            os.makedirs(local_path)
        os.system('unzip -o tracking.zip -d %s'%(local_path))

def gen_model():
    #Compile model
    get_onnx_module_and_golden()
    onnx_model = onnx.load(local_path + '/hmquant_tracking_with_act.onnx')

    input_shape_nwhc = onnx_model.graph.input[1].type.tensor_type.shape.dim
    input_shape = (input_shape_nwhc[0].dim_value, input_shape_nwhc[3].dim_value, input_shape_nwhc[1].dim_value, input_shape_nwhc[2].dim_value)

    input_name = onnx_model.graph.input[1].name
    print('input name:',input_name, 'with shape(ncwh):', input_shape)

    input_crop = onnx_model.graph.input[0].name
    crop_shape = (16,)


    graph = onnx_model.graph
    print(graph.output)

    type_dict = {input_name: 'uint8', input_crop: 'int32'}
    shape_dict = {input_name: input_shape, input_crop: crop_shape}
    convert_config=None
    mod = relay.frontend.from_hmonnx(onnx_model, shape_dict, type_dict, resizer_attr=None)
    with relay.build_config(opt_level=3):
        graph, lib, params = relay.build(mod, 'hdpl --host=llvm')
    print('build model done.')

    #Load test image and run
    filename='tracking'
    tcim.store_model(filename, graph, params, lib)
    print('store model done.')

if __name__ == '__main__':
    gen_model()
