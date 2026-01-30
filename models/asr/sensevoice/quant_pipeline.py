# Copyright (c) 2025 HOUMO AI
#
# File: quant_pipline.py
# Description:
#   Quantization Pipeline Module - Python script implementing the
# quantization pipeline for SenseVoiceSmall ASR models.
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
import argparse
import os
from pathlib import Path
import torch
from funasr import AutoModel
from quant_meta import rebuild_for_onnx
from typing import Any, Dict, List, Optional, Sequence, Tuple
from xhquant.api import (
    DeviceType,
    QuantScheme,
    HMONNXGoldenInference,
    create_quant_config,
    convert_onnx_to_hmonnx,
    get_root_logger,
    xhquant_init
)
from loguru import logger
import shutil


def msg_output_format(title):
    padding_str = "*" * 10
    title = f"{padding_str} {title} {padding_str}"
    return title

class ScaledLayerNorm(torch.nn.Module):
    def __init__(self, original_ln, scale=32.0):
        super().__init__()
        self.ln = original_ln
        self.register_buffer('inv_scale', torch.tensor(1.0 / scale))

    def forward(self, x):
        return self.ln(x * self.inv_scale)

def replace_layernorm_with_scaled(model, scale=32.0):
    for name, module in model.named_children():
        if isinstance(module, torch.nn.LayerNorm):
            setattr(model, name, ScaledLayerNorm(module, scale))
        else:
            replace_layernorm_with_scaled(module, scale)

def generate_golden_data(hmonnx_file: Path, golden_path: Path, hm_inputs: List[torch.Tensor]):
    if not golden_path.exists():
        session = HMONNXGoldenInference(hmonnx_file)
        session.to("cuda")
        session.save_golden = True
        session.golden_dir = golden_path
        session.step = 0
        session(*hm_inputs)

def _get_static_input_shapes(onnx_path: Path) -> dict[str, tuple[int, ...]]:
    import onnx
    # Load only structure for speed
    model = onnx.load(str(onnx_path), load_external_data=False)
    shapes = {}
    for i in model.graph.input:
        shape = []
        is_static = True
        for d in i.type.tensor_type.shape.dim:
            if d.HasField("dim_value"):
                shape.append(d.dim_value)
            else:
                is_static = False
                break
        if is_static and shape:
            shapes[i.name] = tuple(shape)
    return shapes

def _onnx_io_names(onnx_path: Path) -> tuple[list[str], list[str]]:
    import onnx

    model = onnx.load(str(onnx_path), load_external_data=False)
    g = model.graph
    init_names = {i.name for i in g.initializer}
    inputs = [vi.name for vi in g.input if vi.name not in init_names]
    outputs = [vi.name for vi in g.output]
    return inputs, outputs

def _make_calib_data_from_dataset(args, input_names: List[str], target_shapes: dict[str, tuple[int, ...]]) -> List["object"]:
    import torch
    import torch.nn.functional as F
    import quant_common as sc
    from tqdm import tqdm

    print(f"Loading calibration data from HF dataset: {args.hf_dataset} (split={args.hf_split}, limit={args.calib_samples})")

    # Load samples
    samples = sc.load_hf_dataset(
        dataset=args.hf_dataset,
        config=args.hf_config,
        split=args.hf_split,
        limit=args.calib_samples,
        streaming=args.hf_streaming,
        audio_field=args.hf_audio_field
    )

    if not samples:
        raise ValueError("No samples found in dataset!")

    # Build frontend
    model_dir = Path(args.model).expanduser().resolve()
    frontend = sc.build_frontend(model_dir)
    target_sr = int(frontend.cfg.fs)

    # Process all samples
    # Result: list of dicts (inputs)
    all_inputs_dict = {name: [] for name in input_names}

    print("Processing audio samples...")
    for s in tqdm(samples):
        try:
            wav = sc.load_audio_any(s, target_sr=target_sr)
            feat, feat_len = sc.extract_features(frontend, wav)
            # Make inputs (returns dict of tensors with batch=1)
            inputs = sc.make_inputs_for_sample(feat, feat_len, s.language, s.textnorm)

            # Align with target shapes if needed (e.g. padding)
            for name, t in inputs.items():
                if name not in all_inputs_dict:
                    continue

                t = torch.as_tensor(t)
                if name == "speech":
                    t = t.to(torch.float32)
                else:
                    t = t.to(torch.int32)

                if name in target_shapes:
                    target_shape = target_shapes[name]
                    if name == "speech" and t.ndim == 3 and len(target_shape) == 3:
                        cur_t = t.shape[1]
                        tgt_t = target_shape[1]
                        if cur_t != tgt_t:
                            if cur_t < tgt_t:
                                t = F.pad(t, (0, 0, 0, tgt_t - cur_t))
                            else:
                                t = t[:, :tgt_t, :]

                all_inputs_dict[name].append(t)

        except Exception as e:
            print(f"Warning: Failed to process sample {s.audio_id}: {e}")
            continue

    # Stack into a single batch tensor for each input
    # Assuming the first dimension is batch size 1, we cat them to make batch size N
    final_inputs = []
    batch_size = len(all_inputs_dict[input_names[0]])
    print(f"Collected {batch_size} valid calibration samples.")

    for name in input_names:
        t_list = all_inputs_dict[name]
        if not t_list:
             raise ValueError(f"No data for input: {name}")

        # Handle variable length inputs (like speech) by padding to max length in the batch
        if name == "speech" and len(t_list) > 0 and t_list[0].ndim == 3:
            max_t = max(t.shape[1] for t in t_list)
            padded_list = []
            for t in t_list:
                cur_t = t.shape[1]
                if cur_t < max_t:
                    # Pad time dimension (dim 1) for (1, T, D) tensor
                    # F.pad arg order: (last_dim_left, last_dim_right, 2nd_last_left, 2nd_last_right, ...)
                    t = F.pad(t, (0, 0, 0, max_t - cur_t))
                padded_list.append(t)
            t_list = padded_list

        # Cat along dim 0
        batched = torch.cat(t_list, dim=0)
        final_inputs.append(batched)

    return final_inputs

def quant_llm(args):
    model_dir = str(Path(args.model).expanduser().resolve())
    work_dirs = Path(args.work_dirs).expanduser().resolve()
    onnx_dir = work_dirs / "onnx"
    onnx_dir.mkdir(parents=True, exist_ok=True)

    model, kwargs = AutoModel.build_model(model=model_dir, device=args.device, trust_remote_code=False)
    model.eval()

    print("Applying Input Scaling to LayerNorms for FP16 stability...")
    replace_layernorm_with_scaled(model, scale=32.0)

    model_file = onnx_dir / f"{args.model_name}.onnx"
    rebuilt_model = rebuild_for_onnx(
        model,
        max_seq_len=int(args.max_seq_len),
        device=str(args.device),
    )

    static_inputs = rebuilt_model.export_inputs()
    if not os.path.exists(model_file):
        import torch
        torch.onnx.export(
            rebuilt_model,
            static_inputs,
            str(model_file),
            verbose=bool(args.verbose),
            opset_version=int(args.opset),
            input_names=rebuilt_model.export_input_names(),
            output_names=rebuilt_model.export_output_names(),
            dynamic_axes=None,
        )

    import onnx
    model_proto = onnx.load(str(model_file))
    try:
        from onnxsim import simplify
        print("Simplifying ONNX model for static shape...")
        # Use onnxsim to fold constants and remove dynamic ops
        model_sim, check = simplify(model_proto)
        if check:
            onnx.save(model_sim, str(model_file))
            print(f"Simplified model saved to {model_file}")
        else:
            print("Simplification check failed! Keeping original model.")
    except ImportError:
        print("onnxsim not installed, skipping simplification. Install with: pip install onnxsim")
    except Exception as e:
        print(f"Simplification failed with error: {e}")

    meta = {
        "model_dir": model_dir,
        "device": args.device,
        "max_seq_len": args.max_seq_len,
        "opset": args.opset,
        "onnx": str(model_file),
    }
    (onnx_dir / "export_meta.json").write_text(
        __import__("json").dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    print(f"Exported FP32 ONNX: {model_file}")

    input_names, output_names = _onnx_io_names(model_file)
    target_shapes = _get_static_input_shapes(model_file)

    log_file = work_dirs / "convert.log"
    xhquant_init(str(log_file), debug=bool(args.debug))

    quant_scheme = QuantScheme(target_device=DeviceType.XH2a, quant_type=args.quant_type)
    quant_config = create_quant_config(quant_scheme)

    if "w_cfg" in quant_config and "quantizer" in quant_config["w_cfg"]:
        # Move calib_metric to outer level of quantizer config, not inside qspec
        quant_config["w_cfg"]["quantizer"]["calib_metric"] = args.calib_metric

    if "i_cfg" in quant_config and "quantizer" in quant_config["i_cfg"]:
        quant_config["i_cfg"]["quantizer"]["calib_metric"] = args.calib_metric


    prefix = f"{args.model_name}-{DeviceType.XH2a}-{args.quant_type}"
    work_dirs = Path(args.work_dirs).expanduser().resolve() / prefix
    work_dirs.mkdir(parents=True, exist_ok=True)
    out_hmonnx = work_dirs / f"{args.model_name}.onnx"

    if args.calib:
        calib_path = Path(args.calib).expanduser().resolve()
        input_list = _make_input_list(calib_path, input_names, target_shapes)
    else:
        # Use HF dataset
        input_list = _make_calib_data_from_dataset(args, input_names, target_shapes)

    batch_size = 1
    if len(input_list) > 0 and hasattr(input_list[0], "shape"):
        batch_size = input_list[0].shape[0]

    if batch_size > 1:
        print(f"Detected multiple calibration samples (batch={batch_size}). Using multi-batch calibration.")

        # Import lower-level APIs
        from xhquant.api.ptq_export_hmonnx import (
            _convert_model_to_quanted_model,
            convert_quanted_model_to_hmonnx,
        )
        from xhquant.common.types import FrontendType, PrecisionMode
        from xhquant.quantization import ptq_quantize
        import torch

        # 1. Convert to quanted graph (without running PTQ yet)
        # Use the first sample for graph construction and shape inference
        first_sample_args = [t[0:1] if isinstance(t, torch.Tensor) else t for t in input_list]

        quanted_graph_module = _convert_model_to_quanted_model(
            str(model_file),
            FrontendType.ONNX,
            first_sample_args,
            DeviceType.XH2a,
            quant_config,
            use_ptq=False,
            input_names=input_names,
        )

        # 2. Prepare calibration data list (List[List[Tensor]])
        calib_data = []
        for i in range(batch_size):
            sample_args = []
            for t in input_list:
                if isinstance(t, torch.Tensor):
                    # Slice the batch dimension, keeping it 1
                    sample_args.append(t[i:i+1].cpu())
                else:
                    sample_args.append(t)
            calib_data.append(sample_args)

        execution_device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

        # 3. Run PTQ with multiple samples
        ptq_quantize(
            quanted_graph_module,
            calib_data,
            PrecisionMode.ALIGNED,
            execution_device
        )

        # 4. Export to HMONNX
        convert_quanted_model_to_hmonnx(
            quanted_graph_module,
            first_sample_args, # Use first sample for tracing
            str(out_hmonnx),
            input_names,
            output_names,
        )
    else:
        convert_onnx_to_hmonnx(
            str(model_file),
            input_list,
            DeviceType.XH2a,
            str(out_hmonnx),
            quant_config=quant_config,
            input_names=input_names,
            output_names=output_names,
        )

    logger = get_root_logger()
    logger.info(f"Converted to HMONNX: {out_hmonnx}")
    print(f"Converted to HMONNX: {out_hmonnx}")

    args.calib_samples = 1
    hm_input_list = _make_calib_data_from_dataset(args, input_names, target_shapes)
    hm_inputs = []
    for input in hm_input_list:
        if input.dtype == torch.float32:
            hm_inputs.append(input.to(torch.float16))
        else:
            hm_inputs.append(input)
    generate_golden_data(out_hmonnx, work_dirs / "golden", hm_inputs)


def move_llm(args):
    work_dir = Path(args.work_dirs)
    dest_dir = Path(args.out_dir)
    model_name = os.path.basename(args.model_name)
    hm_model_name = "hmquant_{}_with_act.onnx".format(args.model_name)
    hmm_model_dir = "{}-XH2a-{}".format(
        model_name, args.quant_type
    )
    logger.info(
        msg_output_format("Start move from {} to {}").format(
            work_dir / hmm_model_dir / "golden" / "*" , dest_dir
        )
    )
    src_golden_dir = work_dir / hmm_model_dir / "golden"
    target_dir = dest_dir / "hmquant" / f"{args.model_name}"
    target_dir.mkdir(parents=True, exist_ok=True)
    os.remove(os.path.join(src_golden_dir, f"step_0/hmquant_{model_name}_with_act.onnx"))
    os.remove(os.path.join(src_golden_dir, f"step_0/{model_name}_external_data"))
    for file in os.listdir(src_golden_dir):
        shutil.move(src_golden_dir / file, target_dir / file)
    shutil.move(
        dest_dir / "hmquant" / f"{args.model_name}" / "step_0", dest_dir / "hmquant" / f"{args.model_name}" / "golden"
    )
    logger.info(msg_output_format("Start remove work_dir: {}".format(work_dir)))
    shutil.rmtree(work_dir, ignore_errors=True)
