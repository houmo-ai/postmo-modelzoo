#!/usr/bin/env python

import os
import sys
import numpy as np
import argparse
import time
import onnx
import onnx_graphsurgeon as gs
from prettytable import PrettyTable
import tcim
from hmassist.utils.dist_metrics import cosine_distance
from hmassist.utils import logger
from hmassist.utils.utils import sanitize_name

def get_args():
    """Parse commandline"""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--model',
        dest='model',
        type=str,
        help='input image size',
    )
    parser.add_argument(
        '--batch',
        dest='batch',
        type=int,
        default=1,
        help='batch size',
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

def build_and_test(model_name, model_path):
    # build model
    onnx_model = onnx.load(model_path)
    compile_config = {
        "tcim.codegen_pic": True,
        "tcim.use_convadd": True,
        "tcim.fuse_strategy": 1,
        'tcim.special_model_name':'bino_fish_part2',
        "tcim.gen_intrinsic": 1,
        "tcim.mem_plan_strategy": "linearscan",
    }
    tcim.build.build_from_hmonnx(onnx_model, model_name=model_name, compiler_cfg=compile_config)
    print(model_name, 'build completed.')

    model_dir = os.path.dirname(model_path)
    module = tcim.runtime.load(model_name + ".hmm")

    # set input with golden
    input_num = module.get_num_inputs()
    for id in range(input_num):
        input_name = module.get_input_name(id)
        input_info = module.get_input_info(input_name)
        print("input_info[{}] shape = {}, dtype = {}, format = {}".format(input_name, input_info.shape,
                                                                        input_info.dtype, input_info.format.name))
        input_file_name = 'hmquant_' + model_name + '_' + input_name + '_input.npy'
        input_data_path = os.path.join(model_dir, input_file_name)
        input_data = np.load(input_data_path).astype(input_info.dtype)
        # input_data = np.concatenate([input_data for i in range(batch)], axis=0)
        print("golden input[{}] shape = {}, dtype = {}".format(input_name, input_data.shape, input_data.dtype))
        module.set_input(input_name, input_data)

    # compare with golden
    module.run()
    module.sync()

    # get output and compare with golden
    result_check = True
    output_num = module.get_num_outputs()
    for id in range(output_num):
        output_name = module.get_output_name(id)
        output_info = module.get_output_info(output_name, is_quanted=True)
        print("output_info[{}] shape = {}, dtype = {}, format = {}".format(output_name, output_info.shape,
                                                                            output_info.dtype, output_info.format.name))
        output_data = module.get_output(output_name, is_quanted=True)
        print("output[{}] shape = {}, dtype = {}".format(output_name, output_data.shape, output_data.dtype))
        if len(output_data.shape) == 4:
            output_data = np.transpose(output_data, (0, 3, 1, 2))
        print("output[{}] transpose to {}".format(output_name, output_data.shape))

        output_data_path = os.path.join(model_dir, 'hmquant_' + model_name + '_with_act', output_name + '.npy')
        if os.path.exists(output_data_path):
            golden_output = np.load(output_data_path, allow_pickle=True).item().get("output_tensor")
            # golden_output = np.concatenate([golden_output for i in range(batch)], axis=0)
        else:
            result_check = False
            print("[warning] compare canceled while golden data not found -> {}".format(output_data_path))
            continue
        if golden_output.shape == output_data.shape:
            cosine_dist = cosine_distance(golden_output, output_data)
            is_match = (golden_output == output_data).all()
            print("[compare] golden output [{}] match={}, similarity={:.6f}"
                .format(output_name, is_match, cosine_dist))
            if cosine_dist < 0.99:
                result_check = False
        else:
            result_check = False
            print("[compare] golden output [{}] shape not match {} vs {}"
                .format(output_name, golden_output.shape, output_data.shape))

    return result_check, cosine_dist, output_name, output_data.shape


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
    audit(args.model, args.mode)
