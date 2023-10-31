import argparse
import os

import numpy as np
import onnx
import tvm
import tvm.relay as relay
import tvm.tcim as tcim
from tvm.contrib import hdpl_graph_executor

local_path = 'apollo/bev'
bev3d_golden_path = local_path + '/fastbev3d_golden_0403'

def get_onnx_module_and_golden():
    if not os.path.exists(bev3d_golden_path):
        os.system('curl -upublic:Password@123 \
            http://10.10.1.53:8082/artifactory/hdpl_test_data/quant_models/apollo/bev/fastbev3d_golden_0403.zip -o bev3d.zip')
        if not os.path.exists(local_path):
            os.makedirs(local_path)
        os.system('unzip -o bev3d.zip -d %s'%(local_path))

def calc_cosine_distance(output, golden):
    output_ = output.reshape(-1).astype('int64')
    golden_ = golden.reshape(-1).astype('int64')
    cos_dst = np.dot(golden_, output_) / (np.linalg.norm(golden_)* np.linalg.norm(output_))
    return cos_dst

def gen_model(nchw_shape, compile_only):
    #Compile model
    get_onnx_module_and_golden()
    onnx_model = onnx.load(bev3d_golden_path + '/hmquant_fastbev3d_with_act.onnx')

    input_name = onnx_model.graph.input[0].name
    print('intput_name:',onnx_model.graph.input[0].name)

    type_dict = {input_name : 'int8'}
    input_shape = (1, 256, 128, 128) if (nchw_shape==[]) else nchw_shape
    shape_dict = {input_name : input_shape}
    layout_dict = {input_name, 'NHWC'}
    # convert_config={"transpose_axes": [0,1,3,2]}
    convert_config=None

    #Load test image and run
    filename='bev3d'
    mod = relay.frontend.from_hmonnx(onnx_model, shape_dict, type_dict, layout=layout_dict, resizer_attr=None, convert_config=convert_config)
    with relay.build_config(opt_level=3):
        graph, lib, params = relay.build(mod, 'hdpl --host=llvm')
    if compile_only:
        return True
    print('build model done.')
    print('======= compile fused model')
    rt_opt=''
    tcim.store_as_fusedop(filename, graph, params, shape_dict, lib, rt_opt, 1)
    return True


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='hm_onnx_apollo_bev3d')
    parser.add_argument('--nchw-shape', dest='nchw_shape', nargs='+', type=int, help='input nchw shape(eg. 1 3 224 224), default get from onnx model at nhwc order. if onnx model shape is nchw order, must set the param.', default=[])
    parser.add_argument('--compile_only', dest='compile_only', action='store_true',help='compile onnx only.')
    args = parser.parse_args()
    nchw_shape = args.nchw_shape
    compile_only = args.compile_only
    gen_model(nchw_shape, compile_only)
