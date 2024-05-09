#!/usr/bin/env python

import os
import sys
import numpy as np
import argparse
import time
import onnx
import onnx_graphsurgeon as gs
from prettytable import PrettyTable
import tvm
import tvm.relay as relay
import tvm.tcim as tcim
from tvm.relay.backend import Executor
from hmassist.utils.dist_metrics import cosine_distance
from hmassist.utils import logger
from hmassist.utils.utils import sanitize_name

def get_args():
    """Parse commandline"""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--model_name',
        dest='model_name',
        type=str,
        default='cruise_cutin',
        help='input image size',
    )
    parser.add_argument(
        '--batch',
        dest='batch',
        type=int,
        default=1,
        help='Set batch size for implicit batch houmo model',
    )
    parser.add_argument(
        '--mode',
        dest='mode',
        type=str,
        default='dichotomy',
        help='cut mode in [dichotomy, all], default is dichotomy',
    )
    args = parser.parse_args()
    return args

# Extract the model, the submodel includes all nodes from input_names and output names
def extract_model(input_path, output_path, input_names, output_names):
    onnx.utils.extract_model(input_path, output_path, input_names, output_names, check_model=True)


# Get the nodes list by topological order in the model
def get_nodes_by_topo_order(model):
    nodes_list = []
    graph = gs.import_onnx(model)
    # graph.toposort()
    for node in graph.nodes:
        nodes_list.append(node.name)

    print(nodes_list)
    return nodes_list


# Get all input node names in the model
def get_input_node_names_info(mode_path):
    inputs = {}
    for ins in model.graph.input:
        shape = str(ins.type.tensor_type.shape.dim)
        inputs[ins.name] = [int(s) for s in shape.split() if s.isdigit()]
    return inputs


def build_and_test(model_name, model_path):
    # build model
    image_format = 'YUV422SP'
    # batch = args.batch
    shape_dict = {}
    layout_dict = {}
    convert_config = {}
    convert_config["layout"] = "NHWC"

    onnx_model = onnx.load(model_path)
    inputs = onnx_model.graph.input
    model_dir = os.path.dirname(model_path)

    for input in inputs:
        dims = input.type.tensor_type.shape.dim
        input_shape = [dim.dim_value for dim in dims]
        print('input name:', input.name)
        print('input shape:', input_shape)
        shape_dict[input.name] = input_shape

    mod = relay.frontend.from_hmonnx(
        onnx_model, shape_dict, layout=layout_dict,
        resizer_attr=None, convert_config=convert_config,
    )
    executor = Executor('aot')
    # compile_config = {
    #     'tcim.fuse_strategy': 1,
    #     'tcim.codegen_pic': True,
    #     "tcim.for_benchmark": True
    # }
    # compile_config = {"tcim.fuse_strategy": 1, "tcim.gen_intrinsic": 0, "tcim.sync_strategy": 1, "tcim.mem_plan_strategy": "linearscan"}
    compile_config = {}

    target = tvm.target.Target('hdpl', host='c')
    with tvm.transform.PassContext(opt_level=4, config=compile_config):
        graph, lib, params = relay.build(
            mod, target, executor=executor, mod_name=model_name,
        )
    tcim.store_so(model_name, lib)
    print(model_name, 'saved as a aot model.')

    # compare with golden
    cosine_dist = 0.0
    result = False
    module = tcim.load_so(model_name)
    for input in inputs:
        input_file_name = 'hmquant_' + model_name + '_' + input.name + '_input.npy'
        input_data_path = os.path.join(model_dir, input_file_name)
        input_data = np.load(input_data_path).astype("int8")
        print("input[{}] shape = {}, dtype = {}".format(input.name, input_data.shape, input_data.dtype))
        module.set_input(input.name, input_data, image_format)

    module.run()

    output_num = module.get_num_outputs()
    assert output_num == 1
    output_name = module.get_output_name_by_index(0)
    output_data = module.get_output_by_name(output_name).numpy()
    print("output[{}] shape = {}, dtype = {}".format(output_name, output_data.shape, output_data.dtype))
    if len(output_data.shape) == 4:
        output_data = np.transpose(output_data, (0, 3, 1, 2))
    output_data.tofile("{}.txt".format(output_name), sep="\n")
    output_data_path = os.path.join(model_dir, 'hmquant_' + model_name + '_with_act', output_name + '.npy')
    if os.path.exists(output_data_path):
        golden_output = np.load(output_data_path, allow_pickle=True).item().get("output_tensor")
        golden_output.tofile("{}_golden.txt".format(output_name), sep="\n")
    else:
        print("[warning] compare canceled while golden data not found -> {}".format(output_data_path))
    if golden_output.shape == output_data.shape:
        cosine_dist = cosine_distance(golden_output, output_data)
        is_match = (golden_output == output_data).all()
        print("[compare] golden output [{}] match={}, similarity={:.6f}"
                    .format(output_name, is_match, cosine_dist))
        if cosine_dist >= 0.99:
            result = True
        if len(output_data.flatten()) == 1:
            result = is_match
    else:
        print("[compare] golden output [{}] shape not match {} vs {}"
                        .format(output_name, golden_output.shape, output_data.shape))

    return result, cosine_dist, output_name, output_data.shape


# extract the submodel end with specified node index end_node_idx,
# build the submodel and compare the output with golden data
# return True if the result is 
def audit_submodel(model_name, mode_path, input_names, output_names, end_node_idx):
    # The model is run with incorrect output result compared with golden data
    output_path = os.path.join(os.path.dirname(mode_path), str(end_node_idx) + "_" + output_names + ".onnx")
    extract_model(mode_path, output_path, input_names, [output_names])
    result, cos_dist, name, shape = build_and_test(model_name, output_path)
    print("subgraph: {}, cos: {:.6f}, result: {}".format(output_path, cos_dist, result))
    return result, cos_dist, name, shape


# auto audit the source of the first inaccuracy node
def audit(model_name, mode="dichotomy"):
    '''
    Audit the first inaccuracy node if the inference output of the model is incorrect
    '''
    model_path = "./hmquant_" + model_name + "_with_act.onnx"
    model = onnx.load(model_path)
    input_names = []
    inputs = model.graph.input
    for input in inputs:
        input_names.append(input.name)
    nodes_list = get_nodes_by_topo_order(model)
    left = -1
    right = len(nodes_list) - 1
    history = {}

    output_names = sanitize_name(nodes_list[right])
    result, cos_dist, name, shape = audit_submodel(model_name, model_path, input_names, output_names, right)
    history[right] = (result, cos_dist, name, shape)
    print("cur: [{}, {}] history: {}".format(left, right, history))

    # locate the first node is incorrect in binary search order
    # The previous node is correct and the next node is incorrect,
    # the next node is the target node
    spot = right
    while (left <= right and right >= 0):
        if mode == "dichotomy":
            mid = left + (right - left) // 2
            # break the loop if the mid node is tested
            if history.get(mid) is not None:
                break
            output_names = sanitize_name(nodes_list[mid])
            result, cos_dist, name, shape = audit_submodel(model_name, model_path, input_names, output_names, mid)
            history[mid] = (result, cos_dist, name, shape)
            print("cur: [{}, {}] history: {}".format(left, right, history))
            if result:
                left = mid + 1
                spot = left
            else:
                right = mid - 1
        if mode == "all":
            result, cos_dist, name, shape = audit_submodel(model_name, model_path, input_names, nodes_list[right], right)
            history[right] = (result, cos_dist, name, shape)
            print("cur: [{}, {}] history: {}".format(left, right, history))
            if not result:
                spot = right
            right = right - 1

    print("\n[final] possible spot: id = {}, name = {}".format(spot, nodes_list[spot]))

    sorted_history = sorted(history.items())
    header = ["Id", "layer_name", "shape", "match", "similarity"]
    table = PrettyTable(header)
    for key, value in sorted_history:
        row = [key, value[2], value[3], value[0], value[1]]
        table.add_row(row)
    print(f"\n{table}")
    return left


if __name__ == "__main__":
    args = get_args()
    audit(args.model_name, args.mode)
