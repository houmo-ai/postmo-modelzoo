#!/usr/bin/python3
# -*- coding: utf-8 -*-

import os
import numpy as np
import onnx
import argparse
import tcim


def get_args() -> argparse.Namespace:
    """Parse commandline."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--model_dir',
        dest='model_dir',
        type=str,
        default='qwen2_decode_2block',
        help='path to the model dir',
    )
    parser.add_argument(
        '--model_name',
        dest='model_name',
        type=str,
        default='qwen2_decode_2block',
        help='output houmo model name',
    )
    parser.add_argument(
        '--batch',
        dest='batch',
        type=int,
        default=1,
        help='batch size',
    )
    parser.add_argument(
        '--core',
        dest='core',
        type=int,
        default=4,
        help='core number',
    )
    parser.add_argument(
        '--stage',
        dest='stage',
        type=str,
        default="all",
        help='build stage choise=["build", "test", "all"]',
    )
    args = parser.parse_args()
    return args


def build(args=None):
    """build and test houmo model."""
    model_name = args.model_name
    batch = args.batch
    core_num = args.core
    stage = args.stage
    model_dir = args.model_dir
    part_name = f"qwen2_decode_2block_{core_num}cores"
    quant_name = "qwen2_decode_2block"
    onnx_name = quant_name + ".onnx"
    model_path = os.path.join(model_dir, onnx_name)

    os.environ['AOT_COMPILE_NO_FORMAT'] = '1'

    # 1. build model
    if stage == 'build' or stage == 'all':
        print(f"\n===> {part_name} build start...")
        onnx_model = onnx.load(model_path)
        compile_config = {
            'tcim.fuse_strategy': 1,
            "tcim.gen_intrinsic": 1,
            "tcim.sync_strategy": 0,
            "tcim.special_model_name": "vit_small",
            "tcim.batch_num": 1,
            "tcim.codegen_pic": True,
            "tcim.mem_plan_strategy": "linearscan",
            "tcim.large_split_maxsize": True,
            "tcim.split_const" : True
        }
        if core_num == 4:
            compile_config["tcim.core_num"] = 4
            compile_config["tcim.batch_used_core_num"] = 4
            compile_config["tcim.1batch_4core"] = True
        elif core_num == 2:
            compile_config["tcim.core_num"] = 2
            compile_config["tcim.batch_used_core_num"] = 2
            compile_config["tcim.1batch_2core"] = True
        else:
            print("[error] not support core =", core_num)
            exit(-1)
        input_cfg = {}
        inputs = onnx_model.graph.input
        for i, input in enumerate(inputs):
            dims = input.type.tensor_type.shape.dim
            input_shape = [dim.dim_value for dim in dims]
            if i <= 2:
                input_cfg[input.name] = tcim.HMInput(shape=input_shape)
            else:
                input_cfg[input.name] = tcim.HMInput(shape=input_shape, layout='NCHW')
        weight_path = "2block_weight.npy"
        data_dict = np.load(weight_path, allow_pickle=True).item()
        kvcache_path = os.path.join(model_dir, '../kvcache.npy')
        if os.path.exists(kvcache_path):
            kvcache_data_dict = np.load(kvcache_path, allow_pickle=True).item()
            data_dict.update(kvcache_data_dict)
        constant_path = os.path.join(model_dir, '../constant.npy')
        if os.path.exists(constant_path):
            constant_data_dict = np.load(constant_path, allow_pickle=True).item()
            data_dict.update(constant_data_dict)
        tcim.build.build_from_hmonnx(onnx_model, weights=data_dict, model_name=part_name, compiler_cfg=compile_config,
                                     inputs=input_cfg, hdplcc_options=["-O2"], const_weight_prefix=part_name+"_")
        print(f"<=== {part_name} build success.")

    # 2. test model, prefill model should run first to make the correct result
    if stage == 'test' or stage == 'all':
        if core_num != 4 and core_num != 2:
            print("[warnning] not support core =", core_num)
            exit(0)

        print(f"\n===> {part_name} test start...")
        # 2.1 load model
        weight_manager = tcim.runtime.create_weight_manager()
        # module_prefill = tcim.runtime.load("qwen_prefill.hmm", weight_manager=weight_manager)
        module = tcim.runtime.load(part_name + ".hmm", weight_manager=weight_manager)
        current_length = 0

        input_num = module.get_num_inputs()
        for id in range(input_num):
            input_name = module.get_input_name(id)
            input_info = module.get_input_info(input_name)
            print("input_info[{}] shape = {}, dtype = {}, format = {}".format(input_name, input_info.shape,
                                                                              input_info.dtype, input_info.format.name))
            input_file_name = input_name + '.npy'
            input_data_path = os.path.join(model_dir, input_file_name)
            input_data = np.load(input_data_path).astype(input_info.dtype)
            if input_name == 'current_length':
                current_length = input_data[0]
            input_data = np.concatenate([input_data for i in range(batch)], axis=0)
            print("golden input[{}] shape = {}, dtype = {}".format(input_name, input_data.shape, input_data.dtype))
            module.set_input(input_name, input_data)

        # 2.3 infer model
        module.run()
        module.sync()

        # 2.4. get output and compare with golden
        result_check = True
        output_num = module.get_num_outputs()
        for id in range(output_num):
            output_name = module.get_output_name(id)
            output_info = module.get_output_info(output_name, is_quanted=True)
            print("output_info[{}] shape = {}, dtype = {}, format = {}".format(output_name, output_info.shape,
                                                                               output_info.dtype, output_info.format.name))
            output_data = module.get_output(output_name, is_quanted=True)
            print("output[{}] shape = {}, dtype = {}".format(output_name, output_data.shape, output_data.dtype))
            # only compare [1,current_length,4096]
            output_data = output_data[:1, :current_length, :]
            output_data_path = os.path.join(model_dir, output_name + '.npy')
            if os.path.exists(output_data_path):
                golden_output = np.load(output_data_path, allow_pickle=True).item().get("output_tensor")
                golden_output = golden_output[:1, :current_length, :]
                golden_output = np.concatenate([golden_output for i in range(batch)], axis=0)
            else:
                result_check = False
                print("[warning] compare canceled while golden data not found -> {}".format(output_data_path))
                continue
            if golden_output.shape == output_data.shape:
                from hmassist.utils.dist_metrics import cosine_distance
                cosine_dist = cosine_distance(golden_output, output_data)
                is_match = (golden_output == output_data).all()
                print("[compare] golden output [{}] match={}, similarity={:.6f}"
                      .format(output_name, is_match, cosine_dist))
                if cosine_dist < 0.999:
                    result_check = False
            else:
                result_check = False
                print("[compare] golden output [{}] shape not match {} vs {}"
                      .format(output_name, golden_output.shape, output_data.shape))
        if not result_check:
            print("[error] result check failed.")
            exit(-1)
        print(f"<=== {part_name} test success.")

if __name__ == '__main__':
    args = get_args()
    build(args)
