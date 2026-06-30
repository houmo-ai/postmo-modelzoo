# Copyright (c) 2025 HOUMO AI
#
# File: ptq.py
# Description:
#   Post-Training Quantization Tool - Python script for quantizing
# bge-m3 and bge-reranker-m3-v2 embedding models using post-training quantization techniques.
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
import argparse
import gc
import shutil
import onnx
import onnxsim
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
from transformers import AutoModel, AutoModelForSequenceClassification
from typing import Optional
from onnx import TensorProto, numpy_helper, shape_inference

from loguru import logger

from xhquant.api import (
    DeviceType,
    HMONNXGoldenInference,
    QuantScheme,
    convert_onnx_to_hmonnx,
    create_quant_config,
)
from hmatc.utils.monitor import ProcessMemoryMonitor
from hmatc.utils.utils import check_gpu, first_not_none, get_model_configs, parse_context_length

HOUMO_TARGET = os.getenv("HOUMO_TARGET")
assert HOUMO_TARGET == "xh2", "Only support HOUMO_TARGET: xh2."
DEFAULT_CONFIG_PATH = os.path.join(os.path.dirname(__file__), "config.yaml")

def get_default_model_dir(model_config: dict, idx: int) -> str:
    repo_ids = model_config.get("modelscope_repo", [])
    if len(repo_ids) >= idx:
        return repo_ids[idx].rsplit("/", maxsplit=1)[-1]
    model_name = model_config.get("model_name", "bge-m3").upper()
    model_size = model_config.get("model_size", "0.5b").upper()
    return f"{model_name}-{model_size}"


def msg_output_format(title):
    padding_str = "*" * 10
    title = f"{padding_str} {title} {padding_str}"
    return title


def cleanup_cuda():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()


def cleanup_cpu():
    gc.collect()
    try:
        import ctypes

        ctypes.CDLL("libc.so.6").malloc_trim(0)
    except Exception:
        pass

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config", type=str, default=DEFAULT_CONFIG_PATH, help="path to config.yaml"
    )
    parser.add_argument(
        "--embedder_model", type=str, default=None, help="path of embedder hf model"
    )
    parser.add_argument(
        "--reranker_model", type=str, default=None, help="path of reranker hf model"
    )
    parser.add_argument(
        "--work_dir", type=str, default="work_dir", help="path of onnx model"
    )
    parser.add_argument(
        "--out_dir", type=str, default="output/{}".format(HOUMO_TARGET), help="output save path"
    )
    parser.add_argument(
        "--model_name", type=str, default=None, help="output hmonnx model name"
    )
    parser.add_argument(
        "--embedder_name", type=str, default=None, help="embedder model name"
    )
    parser.add_argument(
        "--reranker_name", type=str, default=None, help="reranker model name"
    )
    parser.add_argument(
        "--model_size", dest="model_size", type=str, default=None, help="model size"
    )
    parser.add_argument(
        "--context_length", type=int, default=None, help="context length"
    )
    parser.add_argument(
        "--batch_size", type=int, default=None, help="batch size"
    )
    parser.add_argument("--output_path", default=f"output/{HOUMO_TARGET}", type=str)
    parser.add_argument(
        "--quant_type",
        type=str,
        default=None,
        help="quant precision, xh2 support w8a8_sefp, w4a8_ssfp or w8a16_sefp",
    )
    parser.add_argument(
        "--avoid_mem_export", action="store_true", help="Use when exporting multiple batches with a large context_length",
    )
    parser.add_argument(
        "--dump_golden", action="store_true", help="If need dump golden, please ON!",
    )
    args = parser.parse_args()

    default_model_size, default_model_name, model_configs = get_model_configs(
        args.config
    )
    model_config = model_configs.get(args.model_name, {}).get(args.model_size, {})
    args.model_name = first_not_none(args.model_name, default_model_name)
    args.embedder_name = first_not_none(args.embedder_name, model_config.get("embedder_name", "bge-m3"))
    args.reranker_name = first_not_none(args.reranker_name, model_config.get("reranker_name", "bge-reranker-v2-m3"))
    args.model_size = first_not_none(args.model_size, default_model_size)
    model_config = model_configs.get(args.model_name, {}).get(args.model_size, {})
    args.quant_type = first_not_none(
        args.quant_type, model_config.get("quant_type", "w8a8_sefp")
    )
    args.embedder_model = first_not_none(args.embedder_model, get_default_model_dir(model_config, 1))
    args.reranker_model = first_not_none(args.reranker_model, get_default_model_dir(model_config, 0))
    args.batch_size = first_not_none(args.batch_size, model_config.get("batch", 1))
    args.context_length = first_not_none(
        args.context_length,
        parse_context_length(model_config.get("context_length", "4k")),
    )
    return args

class WrappedBGE(nn.Module):
    """Unified wrapper for bge embedder and reranker models.

    Fixes the forward parameter order by using keyword arguments.
    Embedder post-processing: CLS pooling + L2 norm (per demo.py HmBGEM3).
    Reranker: returns last_hidden_state directly (per demo.py HmBGEReranker).
    """
    def __init__(self, model: nn.Module, model_type: str = "embedder"):
        super().__init__()
        self.model = model
        self.model_type = model_type

    def forward(self, input_ids, token_type_ids, attention_mask):
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
        )
        last_hidden = outputs[0]  # [batch, seq_len, hidden]
        if self.model_type == "embedder":
            # CLS pooling + L2 normalization (matches HmBGEM3.embedder)
            # cls_embedding = last_hidden[:, 0, :]  # [batch, hidden]
            # return torch.nn.functional.normalize(cls_embedding, p=2, dim=-1)
            return last_hidden
        else:
            # Reranker: classification logits (Linear→Tanh→Linear head) → [batch, 1]
            return outputs.logits


def houmo_export_model(args, hf_model_path, model_name, model_type):
    work_dir = Path(args.work_dir) / f"{model_name}-{args.quant_type}"
    work_dir.mkdir(exist_ok=True, parents=True)

    if model_type == "embedder":
        native_model = AutoModel.from_pretrained(hf_model_path, device_map="cpu", torch_dtype=torch.float16)
    else:
        native_model = AutoModelForSequenceClassification.from_pretrained(hf_model_path, device_map="cpu", torch_dtype=torch.float16)
    native_model.eval()

    wrapped_model = WrappedBGE(native_model, model_type)
    wrapped_model.eval()

    bs = args.batch_size if not args.avoid_mem_export else 1
    context_length = args.context_length

    onnx_dir = work_dir / "onnx_model"
    onnx_dir.mkdir(exist_ok=True, parents=True)
    temp_dir = onnx_dir / "temp_model"
    temp_dir.mkdir(exist_ok=True, parents=True)
    temp_onnx_file = str(temp_dir / f"{model_name}_{bs}x{context_length}_temp.onnx")
    onnx_file = str(onnx_dir / f"{model_name}_{bs}x{context_length}.onnx")

    device = next(wrapped_model.parameters()).device
    input_ids = torch.randint(0, 151645, (bs, context_length), dtype=torch.int64, device=device)
    attention_mask = torch.ones(bs, context_length, dtype=torch.int16, device=device)
    token_type_ids = torch.zeros(bs, context_length, dtype=torch.int32, device=device)

    output_names = ["hidden_state"]

    logger.info(f"Export onnx model to {temp_onnx_file}")
    torch.onnx.export(
        wrapped_model,
        (input_ids, token_type_ids, attention_mask),
        temp_onnx_file,
        input_names=["input_ids", "token_type_ids", "attention_mask"],
        output_names=output_names,
    )

    temp_onnx_model = onnx.load(temp_onnx_file)

    onnx_model_sim, checked = onnxsim.simplify(temp_onnx_model)
    if checked:
        os.system(f"rm -rf {temp_dir}")
        temp_onnx_model = onnx_model_sim

    if not os.path.exists(onnx_file):
        onnx.save(
            temp_onnx_model,
            onnx_file,
            save_as_external_data=True,
            all_tensors_to_one_file=True,
            location=f"{Path(onnx_file).stem}_external_data",
        )
    logger.info("Export onnx model finished!")

    if args.avoid_mem_export:
        new_onnx_file = str(onnx_dir / f"{model_name}_{args.batch_size}x{context_length}.onnx")
        # Patch batch_size from 1 -> args.batch_size without reloading weights
        patch_onnx_batch_size(onnx_file, batch_size=args.batch_size, output_path=new_onnx_file)
        onnx_file = new_onnx_file
        # Recreate dummy inputs with the target batch size for hmonnx export
        input_ids = torch.randint(0, 151645, (args.batch_size, context_length), dtype=torch.int64, device=device)
        attention_mask = torch.ones(args.batch_size, context_length, dtype=torch.int16, device=device)
        token_type_ids = torch.zeros(args.batch_size, context_length, dtype=torch.int32, device=device)

    hmonnx_dir = work_dir / "hmonnx"
    hmonnx_dir.mkdir(exist_ok=True, parents=True)
    # hmonnx always uses the target batch size (may differ from export bs when avoid_mem_export)
    prefix = f"{model_name}_{args.quant_type}_{args.batch_size}x{context_length}"
    hmonnx_file = str(hmonnx_dir / f"{prefix}.onnx")
    quant_scheme = QuantScheme(target_device=DeviceType.XH2a, quant_type=args.quant_type)
    quant_config = create_quant_config(quant_scheme)
    logger.info(f"convert onnx model to hmonnx model .....")
    if not Path(hmonnx_file).exists():
        convert_onnx_to_hmonnx(
            onnx_file,
            (input_ids.cpu(), token_type_ids.cpu(), attention_mask.cpu()),
            device_type=DeviceType.XH2a,
            out_hmonnx_file=hmonnx_file,
            quant_config=quant_config,
            input_names=["input_ids", "token_type_ids", "attention_mask"],
            output_names=output_names,
        )
    logger.info(f"HMONNX model export to {hmonnx_file}")

    if args.dump_golden:
        session_device = "cuda" if torch.cuda.is_available() else "cpu"
        golden_dir = work_dir / "golden"
        golden_dir.mkdir(exist_ok=True, parents=True)
        session = HMONNXGoldenInference(hmonnx_file)
        session.to(session_device)
        session.save_golden = True
        session.golden_dir = str(golden_dir)
        session(
            input_ids.to(torch.int32).to(session_device),
            token_type_ids.to(torch.int32).to(session_device),
            attention_mask.to(torch.int16).to(session_device),
        )
        logger.info(f"Export golden data to {golden_dir}")

    cleanup_cpu()
    cleanup_cuda()

# ---------------------------------------------------------------------------
# ONNX batch_size patching (ported from patch_onnx_batch_size.py)
# ---------------------------------------------------------------------------


def _tensor_to_array(tensor: TensorProto) -> Optional[np.ndarray]:
    try:
        return numpy_helper.to_array(tensor)
    except Exception:
        return None


def _array_to_tensor(arr: np.ndarray, name: str) -> TensorProto:
    return numpy_helper.from_array(arr.astype(arr.dtype, copy=False), name=name)


def _set_first_dim(value_info, batch_size: int) -> bool:
    shape = value_info.type.tensor_type.shape
    if len(shape.dim) == 0:
        return False
    old = shape.dim[0].dim_value or shape.dim[0].dim_param
    shape.dim[0].ClearField("dim_param")
    shape.dim[0].dim_value = batch_size
    return old != batch_size


def _set_first_dim_if_old_batch(value_info, old_batch: int, new_batch: int) -> bool:
    shape = value_info.type.tensor_type.shape
    if len(shape.dim) == 0:
        return False
    dim = shape.dim[0]
    if dim.dim_value != old_batch:
        return False
    dim.ClearField("dim_param")
    dim.dim_value = new_batch
    return True


def _patch_shape_array(arr: np.ndarray, old_batch: int, new_batch: int) -> tuple[np.ndarray, bool]:
    if arr.ndim != 1 or arr.size == 0 or arr.size > 8:
        return arr, False
    if arr.dtype.kind not in {"i", "u"}:
        return arr, False
    patched = arr.copy()
    changed = False
    for i, v in enumerate(patched.tolist()):
        if int(v) == old_batch:
            patched[i] = new_batch
            changed = True
    return patched, changed


def _initializer_map(graph) -> dict[str, TensorProto]:
    return {init.name: init for init in graph.initializer}


def _constant_output_map(graph) -> dict[str, TensorProto]:
    mapping: dict[str, TensorProto] = {}
    for node in graph.node:
        if node.op_type != "Constant" or not node.output:
            continue
        for attr in node.attribute:
            if attr.name == "value" and attr.HasField("t"):
                mapping[node.output[0]] = attr.t
                break
    return mapping


def _shape_tensor_names_by_consumers(graph, op_types: tuple[str, ...]) -> set[str]:
    names: set[str] = set()
    for node in graph.node:
        if node.op_type not in op_types:
            continue
        if len(node.input) < 2:
            continue
        shape_input = node.input[1]
        if shape_input:
            names.add(shape_input)
    return names


def _patch_tensor_proto_inplace(tensor: TensorProto, old_batch: int, new_batch: int) -> bool:
    arr = _tensor_to_array(tensor)
    if arr is None:
        return False
    patched, ok = _patch_shape_array(arr, old_batch, new_batch)
    if ok:
        tensor.CopyFrom(_array_to_tensor(patched, tensor.name))
    return ok


def _patch_reshape_shape_inputs(graph, old_batch: int, new_batch: int) -> tuple[int, int]:
    init_map = _initializer_map(graph)
    const_map = _constant_output_map(graph)
    total = 0
    changed = 0
    for node in graph.node:
        if node.op_type != "Reshape":
            continue
        total += 1
        if len(node.input) < 2:
            continue
        shape_input = node.input[1]
        tensor = init_map.get(shape_input) or const_map.get(shape_input)
        if tensor is None:
            continue
        if _patch_tensor_proto_inplace(tensor, old_batch, new_batch):
            changed += 1
    return total, changed


def _patch_named_tensors(
    graph,
    tensor_names: set[str],
    old_batch: int,
    new_batch: int,
) -> tuple[int, int]:
    init_map = _initializer_map(graph)
    const_map = _constant_output_map(graph)
    total = 0
    changed = 0
    for tensor_name in tensor_names:
        tensor = init_map.get(tensor_name) or const_map.get(tensor_name)
        if tensor is None:
            continue
        total += 1
        if _patch_tensor_proto_inplace(tensor, old_batch, new_batch):
            changed += 1
    return total, changed


def _get_external_data_locations(model) -> list[str]:
    locations: list[str] = []
    for init in model.graph.initializer:
        for item in init.external_data:
            if item.key == "location" and item.value not in locations:
                locations.append(item.value)
    return locations


def _set_external_data_location(model, location: str) -> None:
    for init in model.graph.initializer:
        if not init.external_data:
            continue
        found = False
        for item in init.external_data:
            if item.key == "location":
                item.value = location
                found = True
                break
        if not found:
            entry = init.external_data.add()
            entry.key = "location"
            entry.value = location


def _external_data_name(output_path: Path) -> str:
    stem = output_path.name
    if stem.endswith(".onnx"):
        stem = stem[:-5]
    return f"{stem}_external_data"


def patch_onnx_batch_size(
    input_path: str,
    batch_size: int,
    output_path: Optional[str] = None,
    old_batch: int = 1,
    infer_shapes: bool = True,
) -> str:
    """Modify a static-batch ONNX graph's batch dimension without loading external_data.

    Args:
        input_path: Path to the source ONNX file.
        batch_size: Target batch size.
        output_path: Output path; if None, auto-generate with _bs{batch_size} suffix.
        old_batch: Original batch size in the graph (default 1).
        infer_shapes: Run ONNX shape inference after patching.

    Returns:
        Path to the patched ONNX file.
    """
    input_file = Path(input_path).resolve()
    if output_path is None:
        output_file = input_file.with_name(f"{input_file.stem}_bs{batch_size}.onnx")
    else:
        output_file = Path(output_path).resolve()
    output_file.parent.mkdir(parents=True, exist_ok=True)

    logger.info(f"Patching ONNX batch_size: {input_file.name} -> {batch_size}")

    # Load only protobuf metadata, skip external_data weights
    model = onnx.load(str(input_file), load_external_data=False)
    original_ir_version = model.ir_version
    original_opset_import = [opset for opset in model.opset_import]

    changed_io = 0
    for vi in list(model.graph.input) + list(model.graph.output):
        if _set_first_dim(vi, batch_size):
            changed_io += 1

    changed_value_info = 0
    for vi in model.graph.value_info:
        if _set_first_dim_if_old_batch(vi, old_batch, batch_size):
            changed_value_info += 1

    reshape_total, reshape_changed = _patch_reshape_shape_inputs(
        model.graph, old_batch=old_batch, new_batch=batch_size,
    )

    shape_tensor_names = _shape_tensor_names_by_consumers(model.graph, ("Reshape",))
    shape_total, shape_changed = _patch_named_tensors(
        model.graph,
        shape_tensor_names,
        old_batch=old_batch,
        new_batch=batch_size,
    )

    logger.info(f"  IO dims changed: {changed_io}, value_info: {changed_value_info}")
    logger.info(f"  Reshape nodes: {reshape_total}, patched: {reshape_changed}")
    logger.info(f"  Shape tensors by consumer: {shape_changed}/{shape_total}")

    if infer_shapes:
        try:
            model = shape_inference.infer_shapes(model)
            logger.info("  shape inference: OK")
        except Exception as exc:
            logger.warning(f"  shape inference: {exc}")

    if not model.opset_import:
        model.opset_import.extend(original_opset_import)
    if model.ir_version == 0:
        model.ir_version = original_ir_version

    src_locations = _get_external_data_locations(model)
    external_data = _external_data_name(output_file)
    if len(src_locations) == 1:
        src_external = input_file.parent / src_locations[0]
        dst_external = output_file.parent / external_data
        if src_external.resolve() != dst_external.resolve():
            shutil.copyfile(src_external, dst_external)
        _set_external_data_location(model, external_data)
    elif len(src_locations) > 1:
        logger.warning(f"multiple external_data: {src_locations}, keeping original references")

    onnx.save(
        model,
        str(output_file),
        save_as_external_data=True,
        all_tensors_to_one_file=True,
        location=external_data,
    )
    logger.info(f"Patched ONNX saved to {output_file}")
    return str(output_file)


def move_models(
    work_dir: Path,
    source: str = "bge-m3",
    model: str = "bge-m3",
    target_name: str = "hmquant_bge-m3_with_act.onnx",
):
    source_dir = work_dir / "hmquant/{}".format(source)
    matched_files = list(source_dir.glob("*{}*.onnx".format(model)))

    if not matched_files:
        raise FileNotFoundError(f"No matching ONNX files found in {source_dir}")

    target_path = source_dir / target_name
    if target_path.exists():
        target_path.unlink()

    shutil.move(matched_files[0], target_path)
    return target_path


def format_number(n):
    if n >= 1024 * 1024:
        return f"{n // (1024 * 1024)}m"
    elif n >= 1024:
        return f"{n // 1024}k"
    else:
        return "0k"

def move_hmonnx(args):
    work_dir = Path(args.work_dir)
    dest_dir = Path(args.out_dir)
    embedder_model_name = args.embedder_name
    reranker_model_name = args.reranker_name
    hm_model_name = "hmquant_{}_with_act.onnx".format(args.embedder_name)
    START_MOVE_MSG = "Start move from {} to {}"
    ### embedder ###
    hmm_model_dir = "{}-{}".format(
        embedder_model_name, args.quant_type
    )
    logger.info(
        msg_output_format(START_MOVE_MSG).format(
            work_dir / hmm_model_dir, dest_dir
        )
    )
    embedder_dst_dir = dest_dir / f"hmquant/{embedder_model_name}"
    embedder_dst_dir.mkdir(parents=True, exist_ok=True)
    os.system("mv {}/* {}".format(str(work_dir / hmm_model_dir / "hmonnx"), str(embedder_dst_dir)))
    move_models(dest_dir, embedder_model_name, embedder_model_name, target_name=hm_model_name)
 
    ### reranker ###
    hm_model_name = "hmquant_{}_with_act.onnx".format(args.reranker_name)
    hmm_model_dir = "{}-{}".format(
        reranker_model_name, args.quant_type
    )
    logger.info(
        msg_output_format(START_MOVE_MSG).format(
            work_dir / hmm_model_dir, dest_dir
        )
    )
    reranker_dst_dir = dest_dir / f"hmquant/{reranker_model_name}"
    reranker_dst_dir.mkdir(parents=True, exist_ok=True)
    os.system("mv {}/* {}".format(str(work_dir / hmm_model_dir / "hmonnx"), str(reranker_dst_dir)))
    move_models(dest_dir, reranker_model_name, reranker_model_name, target_name=hm_model_name)

    logger.info(msg_output_format("Start remove work_dir: {}".format(work_dir)))
    shutil.rmtree(work_dir, ignore_errors=True)

if __name__ == "__main__":
    assert check_gpu() is True, "Error: Not found GPU device."

    args = parse_args()
    print(args)

    with ProcessMemoryMonitor(interval=2, quiet=True) as monitor:
        # Quantize bge-m3 embedder
        houmo_export_model(args, args.embedder_model, args.embedder_name, model_type="embedder")
        # Quantize bge-reranker-v2-m3
        houmo_export_model(args, args.reranker_model, args.reranker_name, model_type="reranker")

        move_hmonnx(args)
    print(
        f"\n=== Quantization completed. Peak memory: {monitor.peak_memory_mb:.2f} MB ==="
    )
    
