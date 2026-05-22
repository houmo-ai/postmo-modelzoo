# Copyright (c) 2025 HOUMO AI
#
# File: ptq.py
# Description:
#  Fun-CosyVoice3-0.5B-2512 Model PTQ quantization script.
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
import json
import onnx
import torch
import torch
import torch.nn as nn
from torch import Tensor
import onnxsim
import onnxruntime as ort
import numpy as np
import argparse
import shutil
import glob
import time
from pathlib import Path
from loguru import logger
from onnx import helper, TensorProto, shape_inference, numpy_helper

# convert hmonnx
from xhquant.api import (
    Config,
    ConfigDict,
    convert_onnx_to_hmonnx,
    QuantScheme,
    create_quant_config,
    DeviceType,
    QTensor,
    set_random_seed,
    get_root_logger,
    convert_fx_model_to_quanted_model,
    HMONNXGoldenInference,
)
import xhquant.utils.suppress_printing
from xh_model_zoo.xh_llm.models.builder import MODELS
from xh_model_zoo.xh_llm.models.base_llm_model import LLMBaseModel
from xh_model_zoo.xh_llm.models.cosyvoice3 import XHQwen2LegacyModel
from xh_model_zoo_develop.utils.cpu_gpu_utils import print_gpu_info
from xh_model_zoo.utils.time_profiler import time_profiler
from xh_model_zoo.xh_llm.models.eval_model_type import EvalModelType
from xh_model_zoo.xh_llm.utils import decode_next_token

HOUMO_TARGET = os.getenv("HOUMO_TARGET")
assert HOUMO_TARGET in ["xh2"], f"Unsupported HOUMO_TARGET: {HOUMO_TARGET}"

script_dir = os.path.dirname(os.path.abspath(__file__))


def get_args() -> argparse.Namespace:
    """Parse commandline."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_dir",
        dest="model_dir",
        type=str,
        default="./Fun-CosyVoice3-0.5B-2512",
        help="input hf model path",
    )
    parser.add_argument(
        "--model_name",
        dest="model_name",
        type=str,
        default="cosyvoice3",
        help="output hmonnx model name",
    )
    parser.add_argument(
        "--output_dir",
        dest="output_dir",
        type=str,
        default=os.path.join("output", HOUMO_TARGET, "hmquant"),
        help="output directory",
    )
    parser.add_argument(
        "--work_dir",
        dest="work_dir",
        type=str,
        default="./work_dirs",
        help="working directory",
    )
    parser.add_argument(
        "--context_length",
        dest="context_length",
        type=int,
        default=2048,
        help="max sequence length",
    )
    parser.add_argument(
        "--input_sequence_length",
        dest="input_sequence_length",
        type=int,
        default=256,
        help="input sequence length",
    )
    parser.add_argument(
        "--quant_type",
        dest="quant_type",
        type=str,
        default="w8a16h1_sefp",
        help="quant type, default is w8a16h1_sefp",
    )
    parser.add_argument(
        "--debug",
        dest="debug",
        action="store_true",
        help="debug mode",
    )
    parser.add_argument("--seed", type=int, default=1024)
    parser.add_argument("--valid", action="store_true", help="validate the model")
    args = parser.parse_args()
    return args


def simplify_model(model, output_path=""):
    slimmed_model, check = onnxsim.simplify(model)
    assert check, "Simplified model is invalid!"
    if output_path:
        onnx.save(slimmed_model, output_path)
        logger.success(f"saved simplified model to {output_path}")
    return slimmed_model


def onnx_fix_shape(fixed_dims, model_path, output_path=""):
    # load model and logger.info input/output node information
    model = onnx.load(model_path)
    providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
    session = ort.InferenceSession(model_path, providers=providers)
    logger.info("Model input nodes:")
    for input_node in session.get_inputs():
        logger.info(
            f"  {input_node.name}: {input_node.type}, shape: {input_node.shape}"
        )
    logger.info("Model output nodes:")
    for output_node in session.get_outputs():
        logger.info(
            f"  {output_node.name}: {output_node.type}, shape: {output_node.shape}"
        )

    # change input shape according to fixed_dims
    for input_node in model.graph.input:
        dims = [
            d.dim_value if d.dim_value != 0 else d.dim_param
            for d in input_node.type.tensor_type.shape.dim
        ]
        # replace symbolic dims with fixed values from fixed_dims dict
        new_dims = []
        for dim in dims:
            if dim in fixed_dims:
                # replace with fixed value
                new_dims.append(fixed_dims[dim])
            else:
                # if it's not in fixed_dims, keep it as is
                new_dims.append(dim if isinstance(dim, int) else 0)
        # update input_node shape
        input_node.type.tensor_type.shape.ClearField("dim")
        for dim_val in new_dims:
            input_node.type.tensor_type.shape.dim.add(dim_value=dim_val)

    slimmed_model = simplify_model(model, output_path)
    logger.success("✅ Completed fixing input shapes and simplified the model.")
    return slimmed_model


def dump_golden_data(hmonnx_path, input_args, golden_dir):
    model = HMONNXGoldenInference(hmonnx_path)
    model.save_golden = True
    model.exec_device = torch.device("cuda:0")
    model.golden_dir = golden_dir
    with torch.no_grad():
        model.forward(*input_args)


def move_golden_data(golden_dir, output_dir):
    output_golden_dir = os.path.join(output_dir, "step_0")
    os.makedirs(output_golden_dir, exist_ok=True)

    for npy_file in glob.glob(os.path.join(golden_dir, "*.npy")):
        dst_file = os.path.join(output_golden_dir, os.path.basename(npy_file))
        if os.path.exists(dst_file):
            os.remove(dst_file)
        shutil.copy2(npy_file, dst_file)


def quantize_campplus(
    model_name,
    model_dir,
    root_work_dir,
    root_output_dir,
    quant_type,
    batch_size=1,
    sequence_length=1000,
):
    campplus_onnx = f"{model_dir}/campplus.onnx"
    work_dir = os.path.join(root_work_dir, "campplus")
    output_dir = os.path.join(root_output_dir, "campplus")
    golden_dir = os.path.join(work_dir, "step_0")
    os.makedirs(work_dir, exist_ok=True)
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(golden_dir, exist_ok=True)

    simplified_campplus = f"{work_dir}/campplus_simplify.onnx"

    # Fix shape
    fixed_dims = {
        "batch_size": batch_size,
        "sequence_length": sequence_length,
    }

    # Load the ONNX model and logger.info input/output node information
    model = onnx_fix_shape(fixed_dims, campplus_onnx, simplified_campplus)

    hmonnx_name = f"hmquant_{HOUMO_TARGET}_{model_name}_{quant_type}_{batch_size}x{sequence_length}_campplus.onnx"
    model_input = torch.randn(
        fixed_dims["batch_size"], fixed_dims["sequence_length"], 80
    )
    quant_scheme = QuantScheme(target_device=DeviceType.XH2a, quant_type=quant_type)
    quant_config = create_quant_config(quant_scheme)
    hmonnx_path = os.path.join(output_dir, hmonnx_name)
    convert_onnx_to_hmonnx(
        simplified_campplus,
        (model_input,),
        out_hmonnx_file=hmonnx_path,
        device_type="XH2A",
        quant_config=quant_config,
    )

    # Dump golden data
    model_input = model_input.to(torch.float16)
    dump_golden_data(hmonnx_path, (model_input,), golden_dir)
    move_golden_data(golden_dir, output_dir)


def convert_speech_tokenizer(
    speech_tokenizer_onnx, work_dir, fixed_feat_length, mask_shape, mask1_shape
):
    output_onnx = f"{work_dir}/speech_tokenizer_v3_{fixed_feat_length}_1.onnx"
    output_onnx_mask = f"{work_dir}/speech_tokenizer_v3_{fixed_feat_length}_mask.onnx"
    converted_onnx = f"{work_dir}/speech_tokenizer_v3_{fixed_feat_length}.onnx"

    feat_length_name = "feats_length"
    fixed_dims = {"T": fixed_feat_length}
    # mask1 parameters
    block_ids = list(range(6))
    mask1_input_name = "mask1"
    mask1_dtype = TensorProto.FLOAT

    # ---------------------- step 1: fix input shapes ----------------------
    # Load the original ONNX model and logger.info input/output node information
    model = onnx_fix_shape(fixed_dims, speech_tokenizer_onnx)

    # ---------------------- step 2: replace feat_length with constant ----------------------
    graph = model.graph

    # remove feat_length input
    feat_length_input = next(inp for inp in graph.input if inp.name == feat_length_name)
    graph.input.remove(feat_length_input)

    # create Constant node to replace input
    const_node = helper.make_node(
        op_type="Constant",
        inputs=[],
        outputs=[feat_length_name],
        value=helper.make_tensor(
            name="fixed_feat_val",
            data_type=TensorProto.INT32,
            dims=[1],
            vals=[fixed_feat_length],
        ),
    )
    graph.node.insert(0, const_node)

    # simplify model
    simplified_model, check = onnxsim.simplify(
        model, check_n=0, skip_fuse_bn=False, dynamic_input_shape=False
    )
    assert check, "Simplified model is invalid!"
    onnx.save(simplified_model, output_onnx)
    logger.success(
        "✅ Finished fixing input shapes and replacing feat_length with constant."
    )

    # ---------------------- step 3: add mask input and insert Add nodes ----------------------
    def add_mask_input(model, shape):
        input_names = [i.name for i in model.graph.input]
        if "mask" in input_names:
            logger.warning("mask input already exists.")
            return
        mask_input = helper.make_tensor_value_info("mask", TensorProto.FLOAT, shape)
        model.graph.input.append(mask_input)
        logger.info(f"Added mask input with shape {shape}")

    def insert_add_before_softmax(model):
        softmax_nodes = [n for n in model.graph.node if n.op_type == "Softmax"]
        count = 0
        for node in softmax_nodes:
            original_in = node.input[0]
            new_in = original_in + "_masked"
            add_node = helper.make_node(
                "Add",
                inputs=[original_in, "mask"],
                outputs=[new_in],
                name=f"Add_mask_{count}",
            )
            node.input[0] = new_in
            idx = list(model.graph.node).index(node)
            model.graph.node.insert(idx, add_node)
            count += 1
        logger.info(f"Inserted {count} Add nodes before Softmax ops.")

        return model

    def simplify_model(model):
        logger.info("Running onnx-simplify ...")
        model_simp, check = onnxsim.simplify(model)
        assert check, "onnx-simplify check failed"
        logger.info("onnx-simplify done")

        return model_simp

    def process_model(model_in, mask_shape):
        """Receive an ONNX model in memory, add mask input and insert Add nodes."""
        model = model_in
        add_mask_input(model, shape=mask_shape)
        model = insert_add_before_softmax(model)
        model = simplify_model(model)

        return model

    masked_model = process_model(simplified_model, mask_shape)
    onnx.save(masked_model, output_onnx_mask)
    logger.success("✅ Finished adding mask input.")

    # ---------------------- step 4: add mask1 input and insert Mul nodes ----------------------
    def add_mask_input_if_missing(graph, name, dtype, shape):
        existing_names = {i.name for i in graph.input}
        if name in existing_names:
            logger.warning(f"Model already has input '{name}', skip adding.")
            return
        vi = helper.make_tensor_value_info(name, dtype, shape)
        graph.input.append(vi)
        logger.info(f"Added model input '{name}' with shape {shape}.")

    def find_target_add_nodes(graph, block_id):
        name1 = f"/blocks.{block_id}/attn/value/Add"
        name2 = f"/blocks.{block_id}/attn/Add"
        return [node for node in graph.node if node.name in (name1, name2)]

    def insert_mul_after_node(graph, target_node, mask_name):
        if len(target_node.output) == 0:
            raise RuntimeError(f"Target node {target_node.name} has no outputs")
        old_output = target_node.output[0]
        downstream_nodes = [node for node in graph.node if old_output in node.input]
        mul_inserted = False
        for node in downstream_nodes:
            new_mul_output = old_output + f"_mul_{node.name}"
            mul_name = f"{target_node.name}_Mul_{node.name}"
            mul_node = helper.make_node(
                "Mul",
                inputs=[old_output, mask_name],
                outputs=[new_mul_output],
                name=mul_name,
            )
            try:
                idx = list(graph.node).index(target_node)
            except ValueError:
                graph.node.append(mul_node)
            else:
                graph.node.insert(idx + 1, mul_node)
            for i, inp in enumerate(node.input):
                if inp == old_output:
                    node.input[i] = new_mul_output
            mul_inserted = True
            logger.info(f"Inserted Mul '{mul_name}' for branch '{node.name}'.")
        if mul_inserted:
            for out_vi in graph.output:
                if out_vi.name == old_output:
                    out_vi.name = old_output + "_mul"
        return mul_inserted

    def fix_reducemean_axes_to_input(graph):
        for node in graph.node:
            if node.op_type == "ReduceMean":
                axes_attr = next((a for a in node.attribute if a.name == "axes"), None)
                if axes_attr is None:
                    continue
                axes_name = node.name + "_axes"
                axes_tensor = helper.make_tensor(
                    name=axes_name,
                    data_type=TensorProto.INT64,
                    dims=[len(axes_attr.ints)],
                    vals=np.array(axes_attr.ints, dtype=np.int64),
                )
                graph.initializer.append(axes_tensor)
                node.input.append(axes_name)
                node.attribute.remove(axes_attr)
                logger.info(f"Fixed ReduceMean '{node.name}': axes attr -> input")

    def process_model_final(
        model_in, output_path, block_ids, mask1_input_name, mask1_shape, mask1_dtype
    ):
        model = model_in
        graph = model.graph
        add_mask_input_if_missing(graph, mask1_input_name, mask1_dtype, mask1_shape)
        total_inserted = 0
        for bid in block_ids:
            matched = find_target_add_nodes(graph, bid)
            logger.info(f"[block {bid}] found {len(matched)} target nodes")
            for node in matched:
                insert_mul_after_node(graph, node, mask1_input_name)
                total_inserted += 1
        logger.info(f"Total Mul nodes inserted: {total_inserted}")

        opset_imports = [helper.make_operatorsetid("", 18)]
        tmp_model = helper.make_model(
            graph, producer_name="mask_inserter", opset_imports=opset_imports
        )
        tmp_model.ir_version = model.ir_version
        inferred_model = shape_inference.infer_shapes(tmp_model)
        fix_reducemean_axes_to_input(inferred_model.graph)
        logger.info("Running onnx-simplify ...")
        try:
            simplified_model, check = onnxsim.simplify(inferred_model)
        except Exception as e:
            tmp_path = output_path.replace(".onnx", ".pre_simplify.onnx")
            onnx.save(inferred_model, tmp_path)
            logger.error(
                f"onnx-simplify error: {e}, saved pre-simplify model to {tmp_path}"
            )
            raise
        if not check:
            failed_path = output_path.replace(".onnx", ".simplify_failed.onnx")
            onnx.save(simplified_model, failed_path)
            logger.error(f"onnx-simplify check failed, saved to {failed_path}")
            raise RuntimeError(f"onnx-simplify check failed, saved to {failed_path}")

        onnx.save(simplified_model, output_path)
        logger.success(f"Done! save model to {output_path}")

    process_model_final(
        masked_model,
        converted_onnx,
        block_ids,
        mask1_input_name,
        mask1_shape,
        mask1_dtype,
    )

    return converted_onnx


def test_simplified_model(original_onnx, converted_onnx):
    """Test the original and converted ONNX models with dummy inputs to verify correctness.

    Args:
        original_onnx (str): Path to the original ONNX model.
        converted_onnx (str): Path to the converted ONNX model with mask inputs.
    """
    np.random.seed(42)
    # generate dummy inputs
    input1 = np.random.rand(1, 128, 348).astype(np.float32)
    input2 = np.array([348], dtype=np.int32)

    logger.info("\n===== Original model inference =====")
    session = ort.InferenceSession(original_onnx, providers=["CPUExecutionProvider"])
    input_names = [inp.name for inp in session.get_inputs()]
    inputs = {input_names[0]: input1, input_names[1]: input2}
    outputs = session.run(None, inputs)
    for i, out in enumerate(outputs):
        logger.info(f"Output {i} shape: {out.shape}")

    logger.info("\n===== Converted model inference =====")
    seq_len = 348
    target_len = 3000
    padded_input1 = np.zeros((1, 128, target_len), dtype=np.float32)
    padded_input1[:, :, :seq_len] = input1

    # construct mask inputs
    mask_shape = (1, 20, 750, 750)
    mask = np.full(mask_shape, -1e9, dtype=np.float32)
    mask[:, :, :, :87] = 0  # valid region

    mask1 = np.zeros((1, 750, 1280), dtype=np.float32)
    mask1[:, 0:87, :] = 1.0

    # load the final model
    session = ort.InferenceSession(converted_onnx, providers=["CPUExecutionProvider"])
    input_names = [inp.name for inp in session.get_inputs()]
    logger.info(f"Converted model input names: {input_names}")

    # construct input dictionary
    inputs = {
        input_names[0]: padded_input1,
        input_names[1]: mask,
        input_names[2]: mask1,
    }

    # run inference and print results
    outputs = session.run(None, inputs)
    for i, out in enumerate(outputs):
        logger.info(f"Output {i} shape: {out.shape}")


def quantize_speech_tokenizer(
    model_name,
    model_dir,
    root_work_dir,
    root_output_dir,
    quant_type,
    fixed_feat_length=3000,
):
    speech_tokenizer_onnx = f"{model_dir}/speech_tokenizer_v3.onnx"
    work_dir = os.path.join(root_work_dir, "speech_tokenizer")
    output_dir = os.path.join(root_output_dir, "speech_tokenizer")
    golden_dir = os.path.join(work_dir, "step_0")
    os.makedirs(work_dir, exist_ok=True)
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(golden_dir, exist_ok=True)

    mask_shape = [1, 20, 750, 750]
    mask1_shape = [1, 750, 1280]
    converted_onnx = convert_speech_tokenizer(
        speech_tokenizer_onnx,
        work_dir,
        fixed_feat_length=fixed_feat_length,
        mask_shape=mask_shape,
        mask1_shape=mask1_shape,
    )
    test_simplified_model(speech_tokenizer_onnx, converted_onnx)

    hmonnx_name = f"hmquant_{HOUMO_TARGET}_{model_name}_{quant_type}_{fixed_feat_length}_speech_tokenizer.onnx"
    model_input = torch.randn(1, 128, fixed_feat_length)
    mask = torch.randn(*mask_shape)
    mask1 = torch.randn(*mask1_shape)
    quant_scheme = QuantScheme(target_device=DeviceType.XH2a, quant_type=quant_type)
    quant_config = create_quant_config(quant_scheme)
    hmonnx_path = os.path.join(output_dir, hmonnx_name)
    convert_onnx_to_hmonnx(
        converted_onnx,
        (model_input, mask, mask1),
        out_hmonnx_file=hmonnx_path,
        device_type="XH2A",
        quant_config=quant_config,
    )

    # Dump golden data
    model_input = model_input.to(torch.float16)
    mask = mask.to(torch.float16)
    mask1 = mask1.to(torch.float16)
    input_args = (model_input, mask, mask1)
    dump_golden_data(hmonnx_path, input_args, golden_dir)
    move_golden_data(golden_dir, output_dir)


def to_device(inputs, device):
    if isinstance(inputs, Tensor):
        return inputs.to(device)
    elif isinstance(inputs, (list, tuple)):
        return type(inputs)([to_device(x, device) for x in inputs])
    elif isinstance(inputs, dict):
        return {k: to_device(v, device) for k, v in inputs.items()}
    elif isinstance(inputs, QTensor):
        return inputs.to(device)
    else:
        return inputs


def xhmodel_export_onnx(
    xh_model: LLMBaseModel,
    tokenizer,
    data_batch,
    onnx_output_dir: str,
    hmonnx_name,
    device,
    dtype,
    logger,
    valid: bool = True,
):
    logger.info("************* Start Exported Graph *************")
    xh_model.to("cpu")  # switch to CPU for model export
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    print_gpu_info(logger)
    xh_model.convert_to_export_graph(data_batch)
    logger.info("************* End Exported Graph *************")

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    xh_model.change_eval_type(EvalModelType.EXPORTED)

    if valid:
        xh_model.to(device)
        xh_model.to(dtype)
        data_batch["input_ids"] = data_batch["input_ids"].to(device)
        with torch.no_grad():
            outs = xh_model.test_step(data_batch)
            exported_logits = outs.logits.detach()

            exported_logits = exported_logits.squeeze(1)
            next_tokens = torch.argmax(exported_logits, dim=-1)
            next_tokens = next_tokens.unsqueeze(0)
            next_token_str = tokenizer.batch_decode(
                next_tokens, skip_special_tokens=True
            )[0]
        logger.info(f"Exported model next token: {next_tokens} {next_token_str}")

    xh_model.to("cpu")  # switch to CPU for model export
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    print_gpu_info(logger)
    logger.info("*************** Start exporting onnx ***************")
    onnx_file = xh_model.to_export_onnx(data_batch, onnx_output_dir, hmonnx_name)[0]
    return onnx_file


def quantize_llm_qwen2(
    model_name,
    model_dir,
    work_dir,
    output_dir,
    quant_type,
    context_length,
    input_sequence_length,
    debug=False,
    seed=1024,
    valid=False,
):
    quant_cfg_path = f"{script_dir}/quant_cfg_qwen2_05b_instruct_xh2a_2k.py"
    llm_qwen2_work_dir = os.path.join(work_dir, "llm_qwen2")
    os.makedirs(llm_qwen2_work_dir, exist_ok=True)
    hmonnx_prefix = f"hmquant_{HOUMO_TARGET}_{model_name}_{quant_type}"

    # create ONNX output directories
    prefill_onnx_dir = os.path.join(output_dir, "llm_prefill")
    decode_onnx_dir = os.path.join(output_dir, "llm_decode")
    os.makedirs(prefill_onnx_dir, exist_ok=True)
    os.makedirs(decode_onnx_dir, exist_ok=True)

    hf_model_dir = f"{model_dir}/CosyVoice-BlankEN"

    cfg = Config.fromfile(quant_cfg_path)
    cfg.hf_model_dir = hf_model_dir
    cfg.model.hf_model = hf_model_dir
    cfg.work_dir = llm_qwen2_work_dir
    cfg.device = "cuda" if torch.cuda.is_available() else "cpu"
    cfg.exec_device = "cuda" if torch.cuda.is_available() else "cpu"
    cfg.dtype = "float16"
    cfg.debug = debug
    if debug:
        os.makedirs(f"{llm_qwen2_work_dir}/debug", exist_ok=True)
    for key in ["quarot", "gptq"]:
        if key not in cfg:
            cfg[key] = False
    if cfg.quarot or cfg.gptq:
        assert (
            cfg.resume_from and Path(cfg.resume_from).exists()
        ), "resume_from must be valid path"
    cfg.model.wrap_cfg.max_sequence_length = context_length
    cfg.model.wrap_cfg.input_sequence_length = input_sequence_length

    # initialize logging and random seed
    set_random_seed(seed)
    quant_logger = get_root_logger()
    quant_logger.info(f"Config:\n{cfg.pretty_text}")
    cfg.dump(f"{llm_qwen2_work_dir}/quant_config_llm_qwen2.py")

    xhquant.utils.suppress_printing.disable_printing = True

    # device and dtype settings
    device = torch.device(cfg.device)
    exec_device = torch.device(cfg.exec_device)
    dtype = getattr(torch, cfg.dtype)

    # initialize meta information
    meta_info = ConfigDict(
        {
            "create_time": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()),
            "config": str(Path(quant_cfg_path).name),
            "model_name": hmonnx_prefix,
            "wrap_cfg": cfg.model.wrap_cfg.to_dict(),
        }
    )

    # load model and tokenizer
    meta_info.hf_model = hf_model_dir
    xh_model: XHQwen2LegacyModel = MODELS.build(cfg.model)
    tokenizer = xh_model.get_tokenizer()
    native_model = xh_model.get_hf_model("cpu")

    # copy HF configuration files
    hf_config_dir = os.path.join(llm_qwen2_work_dir, "hf_config")
    os.makedirs(hf_config_dir, exist_ok=True)
    for cfg_file in [
        "config.json",
        "generation_config.json",
        "tokenizer_config.json",
        "vocab.json",
        "tokenizer.json",
        "chat_template.jinja",
        "added_tokens.json",
    ]:
        src = f"{hf_model_dir}/{cfg_file}"
        if os.path.exists(src):
            shutil.copyfile(src, f"{hf_config_dir}/{cfg_file}")
    meta_info.hf_config = hf_config_dir

    # prepare input data
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "你多大了？用中文回答。"},
    ]
    text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    input_ids = tokenizer([text], return_tensors="pt").input_ids.to(device)

    # load model weights
    archive_file = f"{model_dir}/llm.pt"
    xh_model.load_wraped_model_state_dict_prefix(native_model, archive_file)
    resume_from = cfg.get("resume_from", None)
    if resume_from is not None:
        xh_model.load_wraped_model_state_dict(native_model, cfg.resume_from)

    # model wrapping and embedding saving
    xh_model.init_wrap_model(native_model)
    native_model = None  # release memory
    token_embedding_file = f"{output_dir}/quant_embedding.pt"
    torch.save(xh_model.token_embedding.state_dict(), token_embedding_file)
    meta_info.token_embedding_file = token_embedding_file

    # kv cache meta information
    if xh_model.past_key_caches and len(xh_model.past_key_caches) > 0:
        meta_info.update(
            {
                "use_cache": True,
                "kv_cache_shape": xh_model.past_key_caches[0].shape,
                "num_hidden_layers": len(xh_model.past_key_caches),
            }
        )

    xh_model.change_eval_type(EvalModelType.WRAPED)
    wraped_model: nn.Module = xh_model.wrap_model

    def pre_hook(module, inputs):
        if isinstance(module, nn.Linear):
            module.to(exec_device)
            return to_device(inputs, exec_device)
        return inputs

    def post_hook(module, inputs, outputs):
        if isinstance(module, nn.Linear):
            module.to("cpu")
            return to_device(outputs, "cpu")
        return outputs

    if device != exec_device:
        for module in wraped_model.modules():
            if isinstance(module, nn.Linear):
                module.register_forward_pre_hook(pre_hook)
                module.register_forward_hook(post_hook)
    else:
        wraped_model.to(device)

    xh_model.to(device).to(dtype)
    data_batch = {"input_ids": input_ids, "past_seq_length": 0}
    inputs = xh_model.prepare_inputs_for_graph(data_batch)

    quant_scheme = QuantScheme(target_device=DeviceType.XH2a, quant_type=quant_type)
    quant_config = ConfigDict(create_quant_config(quant_scheme))
    xh_model._quanted_model = convert_fx_model_to_quanted_model(
        xh_model._wrap_model, inputs, cfg.target_device, quant_config=quant_config
    )
    xh_model.change_eval_type(EvalModelType.QUANTED_ALIGNED)
    xh_model.to(device)
    xh_model.to(dtype)

    print_gpu_info(quant_logger)
    if valid:
        with torch.no_grad():
            with time_profiler() as t:
                outs = xh_model.test_step(data_batch)
            quant_logger.info(f"QUANTED_ALIGNED: {t():.04f}")
            quanted_aligned_logits = outs.logits.detach()

        prefill_next_token_id, prefill_next_token_text = decode_next_token(
            tokenizer, quanted_aligned_logits
        )
        quant_logger.info(
            f"Prefill Quanted Model next token: {prefill_next_token_id} {prefill_next_token_text}"
        )
        xh_model.quanted_model.dump_quant_info_to_onnx(
            f"{cfg.work_dir}/{hmonnx_prefix}_quant_info.onnx"
        )
    else:
        prefill_next_token_id = None

    # export prefill model
    xh_model = xh_model.to("cpu")
    data_batch["input_ids"] = data_batch["input_ids"].to("cpu")
    quant_logger.info("*************** Start exporting prefill model ***************")

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    prefill_onnx_file = xhmodel_export_onnx(
        xh_model,
        tokenizer,
        data_batch,
        prefill_onnx_dir,
        f"{hmonnx_prefix}_prefill",
        device,
        dtype,
        quant_logger,
        valid,
    )
    meta_info.prefill_onnx_file = f"{hmonnx_prefix}_prefill.onnx"
    # clear exported model to avoid affecting subsequent exports
    xh_model.release_exported_model()
    logger.info(f"save prefill onnx model to {prefill_onnx_file}")
    logger.info("*************** Finished exporting prefill model ***************")
    print_gpu_info(quant_logger)

    # export decode model
    xh_model.change_eval_type(EvalModelType.QUANTED_ALIGNED)
    xh_model.to(device)
    xh_model.to(dtype)
    data_batch["input_ids"] = data_batch["input_ids"].to(device)
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    xh_model.set_input_sequence_length(1)

    past_seq_len = input_ids.shape[-1]
    if prefill_next_token_id is None:
        prefill_next_token_id = input_ids[:, :1]
    input_ids = prefill_next_token_id
    logger.info(f"past_seq_len: {past_seq_len}")

    data_batch = {
        "input_ids": input_ids.to(device),
        "past_seq_length": past_seq_len,
    }
    if valid:
        with torch.no_grad():
            outs = xh_model.test_step(data_batch)
            decode_logits = outs.logits.detach()
        decode_token_id, decode_token_text = decode_next_token(tokenizer, decode_logits)
        logger.info(
            f"Decode on Quanted Model next token: {decode_token_id} {decode_token_text}"
        )

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    logger.info("*************** Start exporting decode model ***************")
    xh_model = xh_model.to("cpu")
    data_batch["input_ids"] = data_batch["input_ids"].to("cpu")
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    decode_onnx_file = xhmodel_export_onnx(
        xh_model,
        tokenizer,
        data_batch,
        decode_onnx_dir,
        f"{hmonnx_prefix}_decode",
        device,
        dtype,
        quant_logger,
        valid,
    )

    meta_info.decode_onnx_file = f"{hmonnx_prefix}_decode.onnx"
    # clear exported model to avoid affecting subsequent exports
    xh_model.release_exported_model()
    logger.info(f"save decode onnx model to {decode_onnx_file}")
    json.dump(meta_info, open(f"{llm_qwen2_work_dir}/meta_info.json", "w"), indent=4)
    logger.info("*************** Finished exporting decode model ***************")


def quantize_llm_decoder(
    model_name,
    root_work_dir,
    root_output_dir,
    quant_type,
    llm_input_size=896,
):
    model_path = os.path.join(
        script_dir, "cosyvoice3_raw_files", "onnx", "llm_decoder.onnx"
    )
    golden_dir = os.path.join(root_work_dir, "llm_decoder", "step_0")
    output_dir = os.path.join(root_output_dir, "llm_decoder")
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(golden_dir, exist_ok=True)

    hmonnx_name = f"hmquant_{HOUMO_TARGET}_{model_name}_{quant_type}_{llm_input_size}_llm_decoder.onnx"
    model_input = torch.randn(1, llm_input_size)
    quant_scheme = QuantScheme(target_device=DeviceType.XH2a, quant_type=quant_type)
    quant_config = create_quant_config(quant_scheme)
    hmonnx_path = os.path.join(output_dir, hmonnx_name)
    convert_onnx_to_hmonnx(
        model_path,
        (model_input,),
        out_hmonnx_file=hmonnx_path,
        device_type="XH2A",
        quant_config=quant_config,
    )

    # Dump golden data
    model_input = model_input.to(torch.float16)
    dump_golden_data(hmonnx_path, (model_input,), golden_dir)
    move_golden_data(golden_dir, output_dir)


def quantize_flow_spk_embed_affine_layer(
    model_name,
    root_work_dir,
    root_output_dir,
    quant_type,
    spk_embed_dim=192,
):
    model_path = os.path.join(
        script_dir, "cosyvoice3_raw_files", "onnx", "spk_embed_affine_layer.onnx"
    )
    output_dir = os.path.join(root_output_dir, "flow_spk")
    golden_dir = os.path.join(root_work_dir, "flow_spk", "step_0")
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(golden_dir, exist_ok=True)

    hmonnx_name = f"hmquant_{HOUMO_TARGET}_{model_name}_{quant_type}_{spk_embed_dim}_flow_spk.onnx"
    model_input = torch.randn(1, spk_embed_dim)
    quant_scheme = QuantScheme(target_device=DeviceType.XH2a, quant_type=quant_type)
    quant_config = create_quant_config(quant_scheme)
    hmonnx_path = os.path.join(output_dir, hmonnx_name)
    convert_onnx_to_hmonnx(
        model_path,
        (model_input,),
        out_hmonnx_file=hmonnx_path,
        device_type="XH2A",
        quant_config=quant_config,
    )

    model_input = model_input.to(torch.float16)
    dump_golden_data(hmonnx_path, (model_input,), golden_dir)
    move_golden_data(golden_dir, output_dir)


def quantize_flow_encoder(
    model_name,
    root_work_dir,
    root_output_dir,
    quant_type,
    in_channels=80,
    channels=1024,
):
    """Quantize flow encoder model.

    Args:
        model_name: The name of the model.
        model_dir: The directory where the original ONNX model is located.
        root_output_dir: The root directory where the quantized model will be saved.
        quant_type: The type of quantization to apply (e.g., "w8a16h1_sefp").
        in_channels: The number of input channels for the flow encoder.
        channels: The number of channels for the flow encoder.
    """
    model_path = os.path.join(
        script_dir, "cosyvoice3_raw_files", "onnx", "pre_lookahead_layer.onnx"
    )
    output_dir = os.path.join(root_output_dir, "flow_encoder")
    golden_dir = os.path.join(root_work_dir, "flow_encoder", "step_0")
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(golden_dir, exist_ok=True)

    hmonnx_name = f"hmquant_{HOUMO_TARGET}_{model_name}_{quant_type}_{channels}x{in_channels}_flow_encoder.onnx"
    model_input = torch.randn(1, channels, in_channels)
    quant_scheme = QuantScheme(target_device=DeviceType.XH2a, quant_type=quant_type)
    quant_config = create_quant_config(quant_scheme)
    hmonnx_path = os.path.join(output_dir, hmonnx_name)
    convert_onnx_to_hmonnx(
        model_path,
        (model_input,),
        out_hmonnx_file=hmonnx_path,
        device_type="XH2A",
        quant_config=quant_config,
    )

    model_input = model_input.to(torch.float16)
    dump_golden_data(hmonnx_path, (model_input,), golden_dir)
    move_golden_data(golden_dir, output_dir)


def quantize_flow_decoder(
    model_name,
    model_dir,
    root_work_dir,
    root_output_dir,
    quant_type,
    batch_size=2,
    seq_len=2048,
    out_channels=80,
):
    model_path = f"{model_dir}/flow.decoder.estimator.fp32.onnx"
    work_dir = os.path.join(root_work_dir, "flow_decoder")
    output_dir = os.path.join(root_output_dir, "flow_decoder")
    golden_dir = os.path.join(work_dir, "step_0")
    os.makedirs(work_dir, exist_ok=True)
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(golden_dir, exist_ok=True)
    model_path_simplify = f"{work_dir}/flow_decoder_simplify.onnx"

    def get_dummy_input(batch_size, seq_len, out_channels):
        device = "cpu"
        x = torch.rand(
            (batch_size, out_channels, seq_len), dtype=torch.float32, device=device
        )
        mask = torch.ones((batch_size, 1, seq_len), dtype=torch.float32, device=device)
        mu = torch.rand(
            (batch_size, out_channels, seq_len), dtype=torch.float32, device=device
        )
        t = torch.rand((batch_size), dtype=torch.float32, device=device)
        spks = torch.rand(
            (batch_size, out_channels), dtype=torch.float32, device=device
        )
        cond = torch.rand(
            (batch_size, out_channels, seq_len), dtype=torch.float32, device=device
        )
        return x, mask, mu, t, spks, cond

    # fix shape
    fixed_dims = {
        "seq_len": seq_len,
    }

    model = onnx_fix_shape(fixed_dims, model_path, model_path_simplify)

    hmonnx_name = f"hmquant_{HOUMO_TARGET}_{model_name}_{quant_type}_{batch_size}x{seq_len}x{out_channels}_flow_decoder.onnx"
    x, mask, mu, t, spks, cond = get_dummy_input(batch_size, seq_len, out_channels)
    quant_scheme = QuantScheme(target_device=DeviceType.XH2a, quant_type=quant_type)
    quant_config = create_quant_config(quant_scheme)
    hmonnx_path = os.path.join(output_dir, hmonnx_name)
    convert_onnx_to_hmonnx(
        model_path_simplify,
        (x, mask, mu, t, spks, cond),
        out_hmonnx_file=hmonnx_path,
        device_type="XH2A",
        quant_config=quant_config,
    )

    # Dump golden data
    x = x.to(torch.float16)
    mask = mask.to(torch.float16)
    mu = mu.to(torch.float16)
    t = t.to(torch.float16)
    spks = spks.to(torch.float16)
    cond = cond.to(torch.float16)
    input_args = (x, mask, mu, t, spks, cond)
    dump_golden_data(hmonnx_path, input_args, golden_dir)
    move_golden_data(golden_dir, output_dir)


def quantize_hift(
    model_name,
    root_work_dir,
    root_output_dir,
    quant_type,
    batch_size=1,
    seq_len=1024,
):
    model_path = os.path.join(script_dir, "cosyvoice3_raw_files", "onnx", "hift.onnx")
    work_dir = os.path.join(root_work_dir, "hift")
    output_dir = os.path.join(root_output_dir, "hift")
    golden_dir = os.path.join(work_dir, "step_0")
    os.makedirs(work_dir, exist_ok=True)
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(golden_dir, exist_ok=True)

    model_path_simplify = os.path.join(work_dir, "hift_simplify.onnx")
    reflect_constant_path = os.path.join(
        work_dir, "hift_simplify_reflect_replaced_constant.onnx"
    )
    final_onnx_path = os.path.join(work_dir, "hift_simplify_final.onnx")
    final_onnx_scatter_path = os.path.join(work_dir, "hift_simplify_final_1.onnx")

    fixed_dims = {
        "batch_size": batch_size,
        "seq_len": seq_len,
    }

    def replace_resize_with_sizes(model):
        """Replace Resize node's size input with a constant initializer named "sizes" """
        graph = model.graph

        sizes = numpy_helper.from_array(
            np.array([1, 9, 1024], dtype=np.int64),
            name="sizes",
        )

        graph.initializer.append(sizes)

        for node in graph.node:
            if node.name == "/m_source/l_sin_gen/Resize":
                node.input[2] = ""
                node.input.append("sizes")

    def transform_scatter_add(model_path, output_path, win=16, hop=4):
        """
        Decompose ScatterElements nodes with reduction=add into multiple ScatterElements + ReduceSum
        to adapt to backends that do not support add reduction.
        """

        model = onnx.load(model_path)
        graph = model.graph
        K = win // hop

        const_map = {
            init.name: numpy_helper.to_array(init) for init in graph.initializer
        }

        new_graph_nodes = []
        transform_count = 0

        for node in graph.node:
            if node.op_type == "ScatterElements":
                reduction = "none"
                axis = 0
                for attr in node.attribute:
                    if attr.name == "reduction":
                        reduction = attr.s.decode() if attr.s else "none"
                    elif attr.name == "axis":
                        axis = attr.i

                if reduction == "add":
                    logger.info(f"\n Decompose node: {node.name}")
                    transform_count += 1

                    data = node.input[0]
                    indices_name = node.input[1]
                    updates = node.input[2]
                    output = node.output[0]

                    logger.info(f"  data: {data}")
                    logger.info(f"  indices: {indices_name}")
                    logger.info(f"  updates: {updates}")

                    # load indices constant value
                    if indices_name in const_map:
                        indices = const_map[indices_name]
                        is_indices_const = True
                    else:
                        logger.warning(
                            "indices is not constant, attempting to trace..."
                        )
                        # if indices is not constant, we cannot decompose this node
                        new_graph_nodes.append(node)
                        continue

                    logger.info(f"  indices shape: {indices.shape}")
                    logger.info(f"  indices dtype: {indices.dtype}")

                    # process flattened indices of shape [1, total] or [total]
                    if indices.ndim == 2 and indices.shape[0] == 1:
                        indices = indices[0]  # remove leading dim
                        logger.info(f"  Flattened shape: {indices.shape}")

                    total_len = len(indices)
                    num_frames = total_len // win

                    logger.info(f"  Total length: {total_len}")
                    logger.info(f"  Frame length (win): {win}")
                    logger.info(f"  Number of frames: {num_frames}")
                    logger.info(f"  Number of groups K: {K}")

                    # Create a zero tensor of the same shape as data to scatter into
                    zero_tensor = f"{output}_zero"
                    shape_name = f"{output}_shape"

                    new_graph_nodes.append(
                        helper.make_node(
                            "Shape", [data], [shape_name], f"{output}_Shape"
                        )
                    )

                    zero_value = numpy_helper.from_array(
                        np.array([0.0], dtype=np.float32)
                    )
                    const_of_shape_node = helper.make_node(
                        "ConstantOfShape",
                        [shape_name],
                        [zero_tensor],
                        f"{output}_ConstShape",
                    )
                    const_of_shape_node.attribute.append(
                        helper.make_attribute("value", zero_value)
                    )
                    new_graph_nodes.append(const_of_shape_node)

                    # Process in groups by q
                    outs = []

                    for q in range(K):
                        logger.info(f"  Processing group q={q}")

                        # Generate new indices_q
                        indices_q_list = []
                        for t in range(num_frames):
                            for r in range(hop):
                                idx = (t + q) * hop + r
                                indices_q_list.append(idx)

                        indices_q = np.array(indices_q_list, dtype=np.int64)
                        # Keep the same dimensions as the original indices (2D)
                        indices_q = indices_q.reshape(1, -1)
                        indices_q_name = f"{output}_indices_q{q}"
                        graph.initializer.append(
                            numpy_helper.from_array(indices_q, name=indices_q_name)
                        )

                        updates_q_name = f"{output}_updates_q{q}"

                        # Slice parameters
                        starts = f"{output}_starts_q{q}"
                        ends = f"{output}_ends_q{q}"
                        axes = f"{output}_axes_q{q}"
                        steps = f"{output}_steps_q{q}"

                        reshape_name = f"{output}_reshape_q{q}"
                        reshape_shape = f"{output}_reshape_shape_q{q}"

                        graph.initializer.append(
                            helper.make_tensor(
                                reshape_shape, TensorProto.INT64, [2], [num_frames, win]
                            )
                        )

                        new_graph_nodes.append(
                            helper.make_node(
                                "Reshape",
                                [updates, reshape_shape],
                                [reshape_name],
                                f"{output}_Reshape_q{q}",
                            )
                        )

                        slice_out_name = f"{output}_slice_q{q}"
                        graph.initializer.extend(
                            [
                                helper.make_tensor(
                                    f"{output}_s{q}_0",
                                    TensorProto.INT64,
                                    [2],
                                    [0, q * hop],
                                ),
                                helper.make_tensor(
                                    f"{output}_e{q}_0",
                                    TensorProto.INT64,
                                    [2],
                                    [num_frames, (q + 1) * hop],
                                ),
                                helper.make_tensor(
                                    f"{output}_a{q}_0", TensorProto.INT64, [2], [0, 1]
                                ),
                            ]
                        )

                        new_graph_nodes.append(
                            helper.make_node(
                                "Slice",
                                [
                                    reshape_name,
                                    f"{output}_s{q}_0",
                                    f"{output}_e{q}_0",
                                    f"{output}_a{q}_0",
                                ],
                                [slice_out_name],
                                f"{output}_Slice_q{q}",
                            )
                        )

                        shape_flat_name = f"{output}_shape_flat_q{q}"
                        graph.initializer.append(
                            helper.make_tensor(
                                shape_flat_name, TensorProto.INT64, [2], [1, -1]
                            )
                        )

                        new_graph_nodes.append(
                            helper.make_node(
                                "Reshape",
                                [slice_out_name, shape_flat_name],
                                [updates_q_name],
                                f"{output}_Reshape_back_q{q}",
                            )
                        )

                        out_q_name = f"{output}_out_q{q}"
                        scatter = helper.make_node(
                            "ScatterElements",
                            [zero_tensor, indices_q_name, updates_q_name],
                            [out_q_name],
                            f"{output}_Sct_q{q}",
                            axis=axis,
                        )
                        scatter.attribute.append(
                            helper.make_attribute("reduction", "none")
                        )
                        new_graph_nodes.append(scatter)
                        outs.append(out_q_name)

                    # === Accumulate all groups ===
                    if K == 1:
                        new_graph_nodes.append(
                            helper.make_node("Identity", outs, [output], f"{output}_Id")
                        )
                    elif K == 2:
                        new_graph_nodes.append(
                            helper.make_node("Add", outs, [output], f"{output}_Add")
                        )
                    else:
                        current = outs[0]
                        for i in range(1, len(outs)):
                            next_out = f"{output}_add_{i}"
                            new_graph_nodes.append(
                                helper.make_node(
                                    "Add",
                                    [current, outs[i]],
                                    [next_out],
                                    f"{output}_Add_{i}",
                                )
                            )
                            current = next_out
                        new_graph_nodes[-1].output[0] = output

                else:
                    new_graph_nodes.append(node)
            else:
                new_graph_nodes.append(node)

        logger.info(f"Total processed {transform_count} ScatterElements nodes")
        # Replace nodes
        graph.ClearField("node")
        graph.node.extend(new_graph_nodes)

        onnx.save(model, output_path)

        try:
            onnx.checker.check_model(output_path)
            logger.info("✓ Model check passed")
        except Exception as e:
            logger.info(f"✗ Model check failed: {e}")
            import traceback

            traceback.logger.info_exc()

        return output_path

    def convert_hmonnx(
        current_model_path,
        current_output_path,
        current_golden_dir,
        current_quant_type,
        current_batch_size,
        current_seq_len,
        current_model_name,
    ):
        inp = torch.randn(current_batch_size, 80, current_seq_len)
        scheme = QuantScheme(
            target_device=DeviceType.XH2a,
            quant_type=current_quant_type,
        )
        config = create_quant_config(scheme)
        hmonnx_name = (
            f"hmquant_{HOUMO_TARGET}_{current_model_name}_{current_quant_type}_"
            f"{current_batch_size}x{current_seq_len}_hift.onnx"
        )
        hmonnx_path = os.path.join(current_output_path, hmonnx_name)
        convert_onnx_to_hmonnx(
            current_model_path,
            (inp,),
            out_hmonnx_file=hmonnx_path,
            device_type="XH2A",
            quant_config=config,
        )

        inp = inp.to(torch.float16)
        dump_golden_data(hmonnx_path, (inp,), current_golden_dir)
        move_golden_data(current_golden_dir, current_output_path)

        return hmonnx_path

    model = onnx_fix_shape(fixed_dims, model_path, model_path_simplify)

    logger.info("Convert Pad to constant Pad...")
    model = onnx.load(model_path_simplify)
    for node in model.graph.node:
        if node.op_type == "Pad":
            for a in node.attribute:
                if a.name == "mode":
                    a.s = b"constant"
    simplify_model(model, reflect_constant_path)

    logger.info("replace_resize_with_sizes...")
    model = onnx.load(reflect_constant_path)
    replace_resize_with_sizes(model)
    simplify_model(model, final_onnx_path)

    logger.info("transform_scatter_add...")
    transform_scatter_add(final_onnx_path, final_onnx_scatter_path)

    logger.info("convert_hmonnx...")
    hmonnx_path = convert_hmonnx(
        final_onnx_scatter_path,
        output_dir,
        golden_dir,
        quant_type,
        batch_size,
        seq_len,
        model_name,
    )

    hift_part1_onnx = f"{hmonnx_path[:-5]}_part1.onnx"
    hift_part2_onnx = f"{hmonnx_path[:-5]}_part2.onnx"
    onnx.utils.extract_model(
        hmonnx_path,
        hift_part1_onnx,
        input_names=["speech_feat"],
        output_names=["reshape_2"],
        check_model=True,
        infer_shapes=False,
    )
    onnx.utils.extract_model(
        hmonnx_path,
        hift_part2_onnx,
        input_names=["stft", "speech_feat"],
        output_names=["generated_speech"],
        check_model=True,
        infer_shapes=False,
    )


def move_embeddings(output_dir):
    embedding_dir = os.path.join(script_dir, "cosyvoice3_raw_files", "embedding")
    if not os.path.exists(embedding_dir):
        logger.error(f"Embedding directory not found: {embedding_dir}")
        return
    os.makedirs(output_dir, exist_ok=True)

    pt_files = glob.glob(f"{embedding_dir}/*.pt")
    if not pt_files:
        logger.warning(
            f"No embedding files found in embedding directory: {embedding_dir}"
        )
        return

    for file_path in pt_files:
        file_name = os.path.basename(file_path)
        dst = os.path.join(output_dir, file_name)
        if os.path.isfile(file_path):
            shutil.copyfile(file_path, dst)
            logger.info(f"Copied {file_path} to {dst}")


def main(args):
    model_dir = args.model_dir
    model_name = args.model_name
    work_dir = args.work_dir
    output_dir = args.output_dir
    quant_type = args.quant_type
    context_length = args.context_length
    input_sequence_length = args.input_sequence_length

    os.makedirs(work_dir, exist_ok=True)
    os.makedirs(output_dir, exist_ok=True)

    quantize_campplus(
        model_name,
        model_dir,
        work_dir,
        output_dir,
        quant_type=quant_type,
    )
    quantize_speech_tokenizer(
        model_name,
        model_dir,
        work_dir,
        output_dir,
        quant_type=quant_type,
    )
    quantize_llm_qwen2(
        model_name,
        model_dir,
        work_dir,
        output_dir,
        quant_type=quant_type,
        context_length=context_length,
        input_sequence_length=input_sequence_length,
        debug=args.debug,
        seed=args.seed,
        valid=args.valid,
    )
    quantize_llm_decoder(
        model_name,
        work_dir,
        output_dir,
        quant_type=quant_type,
    )
    quantize_flow_spk_embed_affine_layer(
        model_name,
        work_dir,
        output_dir,
        quant_type=quant_type,
    )
    quantize_flow_encoder(
        model_name,
        work_dir,
        output_dir,
        quant_type=quant_type,
    )
    quantize_flow_decoder(
        model_name,
        model_dir,
        work_dir,
        output_dir,
        quant_type=quant_type,
    )
    quantize_hift(
        model_name,
        work_dir,
        output_dir,
        quant_type=quant_type,
    )
    move_embeddings(output_dir)

    shutil.rmtree(work_dir, ignore_errors=True)


if __name__ == "__main__":
    args = get_args()
    logger.info(f"Arguments: {args}")

    main(args)
