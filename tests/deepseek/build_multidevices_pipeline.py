# Copyright 2025 HOUMO AI
#
# File: build_multidevices_pipeline.py
# Description:
#   Build multi-devices pipeline for DeepSeek models.
#   This script implements a pipeline for splitting and building DeepSeek models across multiple devices.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# SPDX-License-Identifier: Apache-2.0

import os
import numpy as np
import time
import argparse
import onnx
import json
import logging

logging.basicConfig(level="INFO")

HOUMO_TARGET = os.getenv("HOUMO_TARGET")
assert HOUMO_TARGET in ["xh1", "xh2"], f"Unsupported HOUMO_TARGET: {HOUMO_TARGET}"


def sanitize_name(name: str) -> str:
    """
    Sanitize name by replacing special characters with underscores.

    Args:
        name (str): Original name to sanitize

    Returns:
        str: Sanitized name with special characters replaced
    """
    return name.replace(":", "_").replace("/", "_")


def get_value_info_by_name(onnx_model: onnx.ModelProto, name: str):
    """
    Get value info from ONNX model by name.

    Args:
        onnx_model (onnx.ModelProto): ONNX model to search in
        name (str): Name of the value to find

    Returns:
        ValueInfoProto or None: Found value info or None if not found
    """
    for input_ in onnx_model.graph.input:
        if input_.name == name:
            return input_
    for output in onnx_model.graph.output:
        if output.name == name:
            return output
    for value_info in onnx_model.graph.value_info:
        if value_info.name == name:
            return value_info
    return None


def get_tensor_from_initializer(onnx_model: onnx.ModelProto, name: str):
    """
    Get tensor data from ONNX model initializer by name.

    Args:
        onnx_model (onnx.ModelProto): ONNX model to search in
        name (str): Name of the initializer to find

    Returns:
        numpy.ndarray: Tensor data as numpy array, or empty array if not found
    """
    for init in onnx_model.graph.initializer:
        if init.name == name:
            return onnx.numpy_helper.to_array(init)
    return np.array([])


def get_shape_by_name(onnx_model: onnx.ModelProto, name: str):
    """
    Get tensor shape by name from ONNX model.

    Args:
        onnx_model (onnx.ModelProto): ONNX model to search in
        name (str): Name of the tensor to find shape for

    Returns:
        list: Shape of the tensor as a list of integers
    """
    value_info = get_value_info_by_name(onnx_model, name)
    if value_info is not None:
        shape = [
            d.dim_value if d.dim_value > 0 else 1
            for d in value_info.type.tensor_type.shape.dim
        ]
        return shape
    tensor = get_tensor_from_initializer(onnx_model, name)
    shape = list(tensor.shape)
    return shape


def get_node_by_output(onnx_model: onnx.ModelProto, name: str):
    """
    Get ONNX node by output name.

    Args:
        onnx_model (onnx.ModelProto): ONNX model to search in
        name (str): Name of the output to find

    Returns:
        NodeProto or None: Found node or None if not found
    """
    for node in onnx_model.graph.node:
        if name in node.output:
            return node
    return None


def get_attribute(node, attr_name, default_value=None):
    """
    Get attribute value from ONNX node by attribute name.

    Args:
        node: ONNX node to search in
        attr_name (str): Name of the attribute to find
        default_value: Default value to return if attribute not found

    Returns:
        Attribute value or default value if not found
    """
    found = [attr for attr in node.attribute if attr.name == attr_name]
    if found:
        return onnx.helper.get_attribute_value(found[0])
    return default_value


def cosine_distance(data1, data2) -> float:
    """
    Calculate cosine distance between two data arrays.

    Args:
        data1: First data array
        data2: Second data array

    Returns:
        float: Cosine distance between the two arrays (-1 if shapes don't match or result is NaN)
    """
    if data1.shape != data2.shape:
        print(f"[error] shape not equal {data1.shape} vs {data2.shape}")
        return -1
    v1_d = data1.flatten().astype("float64")
    v2_d = data2.flatten().astype("float64")
    v1_d[v1_d == np.inf] = np.finfo(np.float16).max
    v2_d[v2_d == np.inf] = np.finfo(np.float16).max
    v1_d[v1_d == -np.inf] = np.finfo(np.float16).min
    v2_d[v2_d == -np.inf] = np.finfo(np.float16).min
    v1_norm = v1_d / np.linalg.norm(v1_d)
    v2_norm = v2_d / np.linalg.norm(v2_d)
    cosine_dist = np.dot(v1_norm, v2_norm)
    if np.isnan(cosine_dist):
        return -1
    return cosine_dist


def save_submodel_golden(model_dir: str, model_name: str, output_names: list):
    """
    Save golden data for submodel testing.

    Args:
        model_dir (str): Directory containing the model files
        model_name (str): Name of the model
        output_names (list): List of output names to process
    """
    for name in output_names:
        file_path = os.path.join(model_dir, f"hmquant_{model_name}_with_act/{name}.npy")
        print(file_path)
        if os.path.exists(file_path):
            data = np.load(file_path, allow_pickle=True).item().get("output_tensor")
            save_path1 = os.path.join(
                model_dir, f"hmquant_{model_name}_{name}_input.npy"
            )
            save_path2 = os.path.join(
                model_dir, f"hmquant_{model_name}_{name}_output.npy"
            )
            np.save(save_path1, data)
            np.save(save_path2, data)
            print(
                f"{os.path.basename(save_path1)} saved in {os.path.dirname(save_path1)}"
            )
            print(
                f"{os.path.basename(save_path2)} saved in {os.path.dirname(save_path2)}"
            )


def extract_model(src, dest, input_names, output_names, metadata_entrys=None):
    """
    Extract a sub-model from a larger ONNX model.

    Args:
        src (str): Path to the source ONNX model
        dest (str): Path to save the extracted model
        input_names (list): List of input names for the sub-model
        output_names (list): List of output names for the sub-model
        metadata_entrys (dict): Optional metadata entries to preserve

    Returns:
        dict: Updated metadata entries after extraction
    """
    import onnx
    import onnx_graphsurgeon

    model = onnx.load(src)
    graph = onnx_graphsurgeon.import_onnx(model)
    tensors = graph.tensors()
    print(f"inputs: {input_names}")
    print(f"outputs: {output_names}")
    graph.inputs = [tensors[in_t.strip()] for in_t in input_names]
    graph.outputs = [tensors[out_t.strip()] for out_t in output_names]
    graph.cleanup()
    dst_model = onnx_graphsurgeon.export_onnx(graph)

    if metadata_entrys is None:
        metadata_entrys = {}
        for dst_out in dst_model.graph.output:
            block_shape = get_shape_by_name(dst_model, dst_out.name)
            pre_node = get_node_by_output(dst_model, dst_out.name)
            scale = get_attribute(pre_node, "output_scale")
            assert scale is not None, f"scale not find, this is unreasonable!"
            zero_point = get_attribute(pre_node, "output_zero_point")
            assert zero_point is not None, f"zero not find, this is unreasonable!"
            src_dtype = "float32"
            dst_dtype = get_attribute(pre_node, "output_dtype").decode("utf-8")
            assert dst_dtype in [
                "int32",
                "int16",
                "int8",
            ], f"dst_dtype should be int32, int16 or int8, but is {dst_dtype}"
            dst_min = 2 ** (int(dst_dtype[3:]) - 1) * -1
            dst_max = 2 ** (int(dst_dtype[3:]) - 1) - 1
            entry_info = {
                "block_shape": block_shape,
                "scale": scale,
                "zero_point": zero_point,
                "src_dtype": src_dtype,
                "dst_dtype": dst_dtype,
                "dst_min": dst_min,
                "dst_max": dst_max,
            }
            entry_info_json = json.dumps(entry_info)
            metadata_entry = onnx.StringStringEntryProto(
                key="houmo.quant.info", value=entry_info_json
            )
            metadata_entrys[dst_out.name] = metadata_entry

    # add input metadata_props
    for dst_input in dst_model.graph.input:
        finish_flag = False
        for src_input in model.graph.input:
            if src_input.name != dst_input.name:
                continue
            for metadata_prop in src_input.metadata_props:
                dst_input.metadata_props.append(metadata_prop)
            finish_flag = True
            break
        if not finish_flag:
            assert (
                metadata_entrys is not None
            ), f"Process input:{dst_input.name}, metadata_entrys not should is None!"
            if dst_input.name in list(metadata_entrys.keys()):
                dst_input.metadata_props.append(metadata_entrys[dst_input.name])
            else:
                raise ValueError(
                    f"ERR: this model input:{dst_input.name} not found metadata_prop info!"
                )

    # add output metadata_props
    for dst_output in dst_model.graph.output:
        finish_flag = False
        for src_output in model.graph.output:
            if src_output.name != dst_output.name:
                continue
            for metadata_prop in src_output.metadata_props:
                dst_output.metadata_props.append(metadata_prop)
            finish_flag = True
            break
        if not finish_flag:
            assert (
                metadata_entrys is not None
            ), f"Process output:{dst_output.name}, metadata_entrys not should is None!"
            if dst_output.name in list(metadata_entrys.keys()):
                dst_output.metadata_props.append(metadata_entrys[dst_output.name])
            else:
                raise ValueError(
                    f"ERR: this model output:{dst_output.name} not found metadata_prop info!"
                )

    onnx.save(dst_model, dest)

    print(f"extracted model saved in", dest)
    return metadata_entrys


def get_args() -> argparse.Namespace:
    """Parse command line arguments for the multi-device pipeline builder."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_dir",
        dest="model_dir",
        type=str,
        default=os.path.join("output", HOUMO_TARGET, "hmquant"),
        help="path to the model dir",
    )
    parser.add_argument(
        "--model_name",
        dest="model_name",
        type=str,
        default="deepseek",
        help="output houmo model name",
    )
    parser.add_argument(
        "--batch",
        dest="batch",
        type=int,
        default=1,
        help="batch size",
    )
    parser.add_argument(
        "--ncore",
        dest="ncore",
        type=int,
        default=4,
        help="core number",
    )
    parser.add_argument(
        "--nblocks",
        dest="nblocks",
        type=int,
        default=48,
        help="block number",
    )
    parser.add_argument(
        "--stage",
        dest="stage",
        type=str,
        default="build",
        help='build stage choise=["build", "test", "all"]',
    )
    parser.add_argument(
        "--output_dir",
        dest="output_dir",
        type=str,
        default=os.path.join("output", HOUMO_TARGET),
        help="build output dir",
    )
    args = parser.parse_args()
    return args


def clip(raw_path, part1_path, part2_path, nblocks):
    """
    Split a model into two parts at a specific layer boundary.

    Args:
        raw_path (str): Path to the original ONNX model
        part1_path (str): Path to save the first part of the model
        part2_path (str): Path to save the second part of the model
        nblocks (int): Total number of blocks in the model
    """
    dir_name = os.path.dirname(raw_path)
    mid_layer_id = 25
    mid_layer_name = f"model_layers_{mid_layer_id}_resadd2"
    save_submodel_golden(dir_name, "deepseek", [mid_layer_name])
    input_names = ["input_1", "valid_length", "current_length"]
    for i in range(mid_layer_id + 1):
        input_names.append(f"model_layers_{i}_self_attn_kcache_input")
        input_names.append(f"model_layers_{i}_self_attn_vcache_input")
        input_names.append(f"model_layers_{i}_self_attn_kcache_history_sum")
    # onnx.utils.extract_model(raw_path, part1_path, input_names=input_names,
    #                          output_names=[mid_layer_name], check_model=True)
    metadata_entrys = extract_model(
        raw_path,
        part1_path,
        input_names=input_names,
        output_names=[mid_layer_name],
        metadata_entrys=None,
    )
    input_names = [mid_layer_name, "valid_length", "current_length"]
    for i in range(mid_layer_id + 1, nblocks):
        input_names.append(f"model_layers_{i}_self_attn_kcache_input")
        input_names.append(f"model_layers_{i}_self_attn_vcache_input")
        input_names.append(f"model_layers_{i}_self_attn_kcache_history_sum")
    # onnx.utils.extract_model(raw_path, part2_path, input_names=input_names,
    #                          output_names=['Output_lm_head_add_list_1'], check_model=True)
    extract_model(
        raw_path,
        part2_path,
        input_names=input_names,
        output_names=["Output_lm_head_add_list_1"],
        metadata_entrys=metadata_entrys,
    )


def build(model_name, model_dir, model_path, output_dir, profile, ncore=1):
    """
    Build a model using the TCIM compiler.

    Args:
        model_name (str): Name of the model to build
        model_dir (str): Directory containing the model files
        model_path (str): Path to the ONNX model file
        output_dir (str): Directory to save the compiled model
        profile (dict): Dictionary to store timing information
        ncore (int): Number of cores to use for compilation
    """
    import tcim

    start = time.time()
    print(f"\n===> {model_name} build start...")
    decode_model = os.path.join(model_dir, model_path)
    tcim.build_from_hmonnx(
        decode_model,
        weights=os.path.join(model_dir, "weight.npy"),
        output_name=model_name,
        ncore=ncore,
        llm_opt=True,
        output_dir=output_dir,
        work_dir=os.path.join(output_dir, "tcim"),
    )
    profile["build"] = time.time() - start
    print(f'{model_name} build completed in {profile["build"]:.3f} s.', flush=True)


def test(model_name, model_dir, output_dir, profile, batch=1, prefix=None):
    """
    Test a compiled model and compare outputs with golden data.

    Args:
        model_name (str): Name of the model to test
        model_dir (str): Directory containing input/output data
        output_dir (str): Directory containing the compiled model
        profile (dict): Dictionary to store timing information
        batch (int): Batch size for testing
        prefix (str): Prefix for naming files (defaults to model_name)
    """
    import tcim_lite

    print(f"\n===> {model_name} test start...")
    # load model
    model_path = os.path.join(output_dir, f"{model_name}.hmm")
    start = time.time()
    option = tcim_lite.runtime.Option(0)
    module = tcim_lite.runtime.load(model_path, option)
    profile["load"] = time.time() - start
    print(f'{model_name} load completed in {profile["load"]:.3f} s.', flush=True)

    # set input
    current_length = 0
    profile["set_input"] = 0
    if prefix is None:
        prefix = model_name
    input_num = module.get_num_inputs()
    for id in range(input_num):
        input_name = module.get_input_name(id)
        input_info = module.get_input_info(input_name)
        print(
            f"input_info[{input_name}] shape = {input_info.shape}, dtype = {input_info.dtype}, format = {input_info.format.name}"
        )
        input_data_path = os.path.join(
            model_dir, f"hmquant_{prefix}_{sanitize_name(input_name)}_input.npy"
        )
        input_data = np.load(input_data_path).astype(input_info.dtype)
        if input_name == "current_length":
            current_length = input_data[0]
            print("current_length is", current_length)
        input_data = np.concatenate([input_data for i in range(batch)], axis=0)
        print(
            f"golden input[{input_name}] shape = {input_data.shape}, dtype = {input_data.dtype}"
        )
        start = time.time()
        module.set_input(input_name, input_data)
        profile["set_input"] += time.time() - start
    print(
        f'{model_name} set {input_num} inputs completed in {profile["set_input"]*1000:.3f} ms.'
    )

    # infer model
    start = time.time()
    module.run()
    module.sync()
    profile["infer"] = time.time() - start
    print(f'{model_name} infer completed in {profile["infer"]*1000:.3f} ms.')

    # get output and compare with golden
    profile["get_output"] = 0
    result_check = True
    output_num = module.get_num_outputs()
    for id in range(output_num):
        output_name = module.get_output_name(id)
        output_info = module.get_output_info(output_name)
        print(
            f"output_info[{output_name}] shape = {output_info.shape}, dtype = {output_info.dtype}, format = {output_info.format.name}"
        )
        start = time.time()
        output_data = module.get_output(output_name).numpy()
        if len(output_data.shape) == 3:
            output_data = output_data[:1, :current_length, :]
        profile["get_output"] += time.time() - start
        print(
            f"output[{output_name}] shape = {output_data.shape}, dtype = {output_data.dtype}"
        )
        output_data_path = os.path.join(
            model_dir, f"hmquant_{prefix}_{sanitize_name(output_name)}_output.npy"
        )
        if os.path.exists(output_data_path):
            golden_output = np.load(output_data_path)
            if len(golden_output.shape) == 3:
                golden_output = golden_output[:1, :current_length, :]
            golden_output = np.concatenate(
                [golden_output for i in range(batch)], axis=0
            )
        else:
            result_check = False
            print(
                f"[warning] compare canceled while golden data not found -> {output_data_path}"
            )
            continue
        if golden_output.shape == output_data.shape:
            cosine_dist = cosine_distance(golden_output, output_data)
            is_match = (golden_output == output_data).all()
            print(
                f"[compare] golden output [{output_name}] match={is_match}, similarity={cosine_dist:.6f}"
            )
            if is_match:
                continue
            if cosine_dist < 0.999:
                result_check = False
        else:
            result_check = False
            print(
                f"[compare] golden output [{output_name}] shape not match {golden_output.shape} vs {output_data.shape}"
            )
    print(
        f'{model_name} get {output_num} ouputs completed in {profile["get_output"]*1000:.3f} ms.'
    )
    if not result_check:
        print("[error] result check failed.")
        exit(-1)
    print(f"<=== {model_name} test success.")


if __name__ == "__main__":
    args = get_args()
    curdir = os.getcwd()
    model_dir = args.model_dir
    model_name = args.model_name
    nblocks = args.nblocks
    output_dir = args.output_dir
    ncore = args.ncore
    batch = args.batch
    profile = {}

    # clip model to 2 parts
    raw_path = os.path.join(model_dir, "prefill/hmquant_deepseek_with_act.onnx")
    part1_path = os.path.join(model_dir, "prefill/hmquant_deepseek_part1_with_act.onnx")
    part2_path = os.path.join(model_dir, "prefill/hmquant_deepseek_part2_with_act.onnx")
    clip(raw_path, part1_path, part2_path, nblocks)
    raw_path = os.path.join(model_dir, "decoder/hmquant_deepseek_with_act.onnx")
    part1_path = os.path.join(model_dir, "decoder/hmquant_deepseek_part1_with_act.onnx")
    part2_path = os.path.join(model_dir, "decoder/hmquant_deepseek_part2_with_act.onnx")
    clip(raw_path, part1_path, part2_path, nblocks)

    # build model
    if args.stage == "build" or args.stage == "all":
        model_path = f"prefill/hmquant_{model_name}_part1_with_act.onnx"
        build(
            "deepseek_prefill_part1", model_dir, model_path, output_dir, profile, ncore
        )
        model_path = f"prefill/hmquant_{model_name}_part2_with_act.onnx"
        build(
            "deepseek_prefill_part2", model_dir, model_path, output_dir, profile, ncore
        )
        model_path = f"decoder/hmquant_{model_name}_part1_with_act.onnx"
        build(
            "deepseek_decode_part1", model_dir, model_path, output_dir, profile, ncore
        )
        model_path = f"decoder/hmquant_{model_name}_part2_with_act.onnx"
        build(
            "deepseek_decode_part2", model_dir, model_path, output_dir, profile, ncore
        )

    # test model
    if args.stage == "test" or args.stage == "all":
        part_dir = os.path.join(model_dir, "prefill")
        test("deepseek_prefill_part1", part_dir, output_dir, profile, prefix=model_name)
        test("deepseek_prefill_part2", part_dir, output_dir, profile, prefix=model_name)
        part_dir = os.path.join(model_dir, "decoder")
        test("deepseek_decode_part1", part_dir, output_dir, profile, prefix=model_name)
        test("deepseek_decode_part2", part_dir, output_dir, profile, prefix=model_name)
