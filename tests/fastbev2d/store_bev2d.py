import argparse
import os
import time
from typing import List

import numpy as np
import onnx
import tvm
import tvm.relay as relay
import tvm.tcim as tcim
from tvm.contrib import hdpl_graph_executor

local_path = 'apollo/bev'
bev2d_golden_path = local_path + '/fastbev2d_golden_0403'

def get_onnx_module_and_golden() -> None:
    if not os.path.exists(bev2d_golden_path):
        os.system('curl -upublic:Password@123 \
            http://10.10.1.53:8082/artifactory/hdpl_test_data/quant_models/apollo/bev/fastbev2d_golden_0403.zip -o bev2d.zip')
        if not os.path.exists(local_path):
            os.makedirs(local_path)
        os.system('unzip -o bev2d.zip -d %s'%(local_path))

def calc_cosine_distance(output, golden):
    output_ = output.reshape(-1).astype('int64')
    golden_ = golden.reshape(-1).astype('int64')
    cos_dst = np.dot(golden_, output_) / (np.linalg.norm(golden_)* np.linalg.norm(output_))
    return cos_dst

def run_model(batch: int, nchw_shape: List[int], compile_only: bool) -> bool:
    #Compile model
    get_onnx_module_and_golden()
    onnx_model = onnx.load(bev2d_golden_path + '/hmquant_fastbev2d_with_act.onnx')

    input_name = onnx_model.graph.input[0].name
    print('intput_name:',onnx_model.graph.input[0].name)

    type_dict = {input_name : 'uint8'}
    input_shape = (batch, 3, 256, 704) if (nchw_shape==[]) else nchw_shape
    shape_dict = {input_name : input_shape}
    layout_dict = {input_name, 'NHWC'}
    convert_config={'transpose_axes': [0,2,3,1]}

    #Load test image and run
    filename='bev2d'
    data_dir = bev2d_golden_path + '/hmquant_fastbev2d_with_act/'

    mod = relay.frontend.from_hmonnx(onnx_model, shape_dict, type_dict, layout=layout_dict, resizer_attr=None,convert_config=convert_config)
    with relay.build_config(opt_level=3):
        graph, lib, params = relay.build(mod, 'hdpl --host=llvm')
    if compile_only:
        return True
    print('build model done.')

    print('======= compile fused model')
    rt_opt=''
    if batch == 1:
      tcim.store_as_fusedop(filename, graph, params, shape_dict, lib, rt_opt, 1)
    else:
      tcim.store_as_fusedop(filename, graph, params, shape_dict, lib, rt_opt, 4)
    return True

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='hm_onnx_apollo_bev2d')
    parser.add_argument('--nchw-shape', dest='nchw_shape', nargs='+', type=int, help='input nchw shape(eg. 1 3 224 224), default get from onnx model at nhwc order. if onnx model shape is nchw order, must set the param.', default=[])
    parser.add_argument('--compile_only', dest='compile_only', action='store_true',help='compile onnx only.')
    parser.add_argument('--batch', dest='batch', type=int, default=1, help='batch size')
    args = parser.parse_args()
    batch = args.batch
    nchw_shape = args.nchw_shape
    compile_only = args.compile_only
    assert run_model(batch, nchw_shape, compile_only)
