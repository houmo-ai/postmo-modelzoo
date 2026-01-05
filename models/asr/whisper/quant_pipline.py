# Copyright (c) 2025 HOUMO AI
#
# File: quant_pipline.py
# Description:
#   Quantization Pipeline Module - Python script implementing the
# quantization pipeline for Whisper ASR models.
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
import onnx
import onnxsim
import torch
import os
import os.path as osp
from pathlib import Path
import tempfile
from copy import deepcopy
import torch.nn as nn
from typing import List, Tuple, Dict, Optional
from dataclasses import dataclass
from datasets import load_dataset
from transformers import WhisperForConditionalGeneration, WhisperProcessor
from xhquant.api import (
    DeviceType,
    HMONNXGoldenInference,
    QuantScheme,
    convert_onnx_to_hmonnx,
    create_quant_config,
    ptq_quantize,
    to_frontend_graph,
    to_quant_graph,
)
from xhquant.api.ptq_export_hmonnx import (
    convert_quanted_model_to_hmonnx,
)
from xhquant.common.types import PrecisionMode
from xhquant.core.datatype_mapping import TORCH_DTYPE_TO_FAKE_DTYPE
from xhquant.frontend.convert import to_frontend_graph
from xhquant.patch.core import RewriterContext
from xhquant.utils.config import ConfigDict

from xh_model_zoo.xh_llm.models.builder import wrap_llm_model
from xh_model_zoo.xh_llm.models.whisper._model_opt import (
    whisper_encoder_forward_v2,
    whisper_decoder_forward_v2,
    whisper_decoder_layer_forward_v2,
    _Whisper_attention,
    eager_attention_forward_cus,
    register_wrap_modules,
)


class Decoder(nn.Module):
    def __init__(self, model, proj_out, config=None):
        super().__init__()
        self.config = config
        self.model = model
        self.proj_out = proj_out

    def forward(
        self,
        decoder_input_ids,
        cache_position,
        past_len,
        current_len,
        mask_atten=None,
        k_cache_list=None,
        v_cache_list=None,
        k_list=None,
        v_list=None,
    ):
        hidden_state, k_cache_list, v_cache_list = self.model.decoder(
            input_ids=decoder_input_ids,
            k_list=k_list,
            v_list=v_list,
            position_ids=cache_position,
            k_cache=k_cache_list,
            v_cache=v_cache_list,
            past_len=past_len,
            current_len=current_len,
            mask_atten=mask_atten,
        )
        output = self.proj_out(hidden_state)
        return output, k_cache_list, v_cache_list


def process_encoder(args):
    out_dir = Path(args.out_dir)
    hf_model_dir = args.model
    model = WhisperForConditionalGeneration.from_pretrained(hf_model_dir)
    model.config.forced_decoder_ids = None
    model.config._attn_implementation = "eager"
    model.model.encoder.decoder_m = model.model.decoder
    
    work_dir = out_dir / "hmquant" / "encoder"
    work_dir.mkdir(exist_ok=True, parents=True)
    onnx_file = work_dir / "whisper_meduim.onnx"

    quant_scheme = QuantScheme(target_device=DeviceType.XH2a, quant_type=args.quant_type)
    quant_config = create_quant_config(quant_scheme)
    hmonnx_file = work_dir / f"hmquant_{args.model_name}_with_act.onnx"
    golden_path = work_dir / "golden"
    
    input_features = torch.randn(1, 80, 3000)
    
    if not onnx_file.exists():
        with tempfile.TemporaryDirectory() as tmp_dir:
            with RewriterContext(None, backend="onnxruntime"):
                temp_onnx = str(Path(tmp_dir) / onnx_file.name)
                torch.onnx.export(
                    model.model.encoder,
                    input_features,
                    temp_onnx,
                    input_names=["input_features"],
                    output_names=["hidden_state"]
                )
                onnx_model = onnx.load(temp_onnx)
                onnx_model_sim, checked = onnxsim.simplify(onnx_model)
                if checked:
                    onnx_model = onnx_model_sim
    else:
        onnx_model = onnx.load(onnx_file)
        
    if not onnx_file.exists():
        onnx.save(
            onnx_model,
            onnx_file,
            save_as_external_data=True,
            all_tensors_to_one_file=True,
            location=f"{onnx_file.stem}_external_data",
        )
        
    output_names = []
    for i in range(24):
        output_names.append(f"key_state_{i}")
    for i in range(24):
        output_names.append(f"value_state_{i}")

    if not hmonnx_file.exists():
        convert_onnx_to_hmonnx(
            str(onnx_file),
            [input_features],
            DeviceType.XH2a,
            hmonnx_file,
            quant_config=quant_config,
            input_names=["input_features"],
            output_names=output_names,
        )
    
    if args.gen_golden and not golden_path.exists():
        session = HMONNXGoldenInference(hmonnx_file)
        session.to("cuda")
        session.save_golden = True
        session.golden_dir = golden_path
        session.step = 0
        session(input_features.half().to("cuda"))


# Constants
NUM_LAYERS = 24
MAX_CACHE_SIZE = 1024
MAX_SEQ_LENGTH = 1500
DEFAULT_DTYPE = torch.float16
PAD_VALUE = -65504


@dataclass
class ProcessConfig:
    out_dir: Path
    model_dir: str
    model_name: str
    quant_type: str
    gen_golden: bool
    process_type: str  # "prefill" or "decoder"


def create_default_tensors(process_type: str) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if process_type == "prefill":
        input_ids = torch.tensor([[50258, 50259, 50359, 50363]])
        cache_pos = torch.tensor([[0, 1, 2, 3]])
    else:
        input_ids = torch.tensor([[50258]])
        cache_pos = torch.tensor([[0]])
    
    past_len = torch.tensor([0])
    return input_ids, cache_pos, past_len


def create_cache_tensors() -> Tuple[List[torch.Tensor], List[torch.Tensor], List[torch.Tensor], List[torch.Tensor]]:
    k_cache = [
        torch.ones([1, 16, MAX_CACHE_SIZE, 64], dtype=DEFAULT_DTYPE) * PAD_VALUE
        for _ in range(NUM_LAYERS)
    ]
    v_cache = [t.clone() for t in k_cache]
    k_list = [
        torch.ones([1, 16, MAX_SEQ_LENGTH, 64], dtype=DEFAULT_DTYPE) * PAD_VALUE
        for _ in range(NUM_LAYERS)
    ]
    v_list = [t.clone() for t in k_list]
    return k_cache, v_cache, k_list, v_list


def get_input_output_names() -> Tuple[List[str], List[str], List[str]]:
    base_inputs = [
        "decoder_input_ids",
        "cache_position",
        "past_len",
        "current_len",
        "mask_atten",
    ]

    cache_inputs = []
    for i in range(NUM_LAYERS):
        cache_inputs.extend([f"k_cache_{i}", f"v_cache_{i}"])

    kv_states = []
    for i in range(NUM_LAYERS):
        kv_states.extend([f"key_state_{i}", f"value_state_{i}"])

    full_inputs = base_inputs + cache_inputs + kv_states

    outputs = ["logits"]
    for i in range(NUM_LAYERS):
        outputs.extend([f"newk_cache_{i}", f"newv_cache_{i}"])
    return full_inputs, kv_states, outputs


def create_attention_mask(cache_len: int, past_len: torch.Tensor) -> torch.Tensor:
    mask_shape = [1, 16, cache_len, MAX_CACHE_SIZE]
    mask = torch.ones(mask_shape, dtype=DEFAULT_DTYPE)
    mask[:, :, :, past_len.item() + cache_len:] *= PAD_VALUE
    return mask


def build_quant_config(fronted_graph_module, warp_inp) -> ConfigDict:
    quant_config = ConfigDict()
    input_args = []
    for arg in warp_inp:
        input_args.extend(arg) if isinstance(arg, (list, tuple)) else input_args.append(arg)
    
    input_names = fronted_graph_module.get_input_names()
    assert len(input_names) == len(input_args), \
        f"Input names count ({len(input_names)}) != args count ({len(input_args)})"

    quant_config.inputs = ConfigDict()
    for name, arg in zip(input_names, input_args):
        qconfig = ConfigDict(dict(quantizer=dict(qspec=dict())))
        if isinstance(arg, torch.Tensor):
            if arg.dtype in TORCH_DTYPE_TO_FAKE_DTYPE:
                qconfig.quantizer.qspec.fake_dtype = TORCH_DTYPE_TO_FAKE_DTYPE[arg.dtype]
            else:
                raise ValueError(f"Unsupported dtype: {arg.dtype}")
        quant_config.inputs[name] = qconfig
    return quant_config


def generate_golden_data(hmonnx_file: Path, golden_path: Path, hm_inputs: List[torch.Tensor]):
    if not golden_path.exists():
        session = HMONNXGoldenInference(hmonnx_file)
        session.to("cuda")
        session.save_golden = True
        session.golden_dir = golden_path
        session.step = 0
        session(*hm_inputs)


def process_whisper(config: ProcessConfig):
    # Init paths
    work_dir = config.out_dir / "hmquant" / config.process_type
    work_dir.mkdir(exist_ok=True, parents=True)
    hmonnx_file = work_dir / f"hmquant_{config.model_name}_with_act.onnx"
    golden_path = work_dir / "golden"

    # Load model
    model = WhisperForConditionalGeneration.from_pretrained(config.model_dir)
    model.config.forced_decoder_ids = None
    model.config._attn_implementation = "eager"
    model_cus = Decoder(model.model, model.proj_out, config=model.config)

    # Create inputs and caches
    input_ids, cache_pos, past_len = create_default_tensors(config.process_type)
    k_cache, v_cache, k_list, v_list = create_cache_tensors()
    cache_len = input_ids.shape[0]
    current_len = torch.tensor([cache_len])
    mask_atten = create_attention_mask(cache_len, past_len)

    # Build warp input
    warp_inp = (input_ids, cache_pos, past_len, current_len, mask_atten, k_cache, v_cache, k_list, v_list)

    # Convert to frontend graph
    with RewriterContext(None, backend="onnxruntime"):
        warp_model = wrap_llm_model(model_cus)
        warp_model = warp_model.half()
        fronted_graph = to_frontend_graph(warp_model, "DynamoFX", warp_inp)

    # Build HM inputs
    hm_inputs = [
        input_ids.to(torch.int32),
        cache_pos.to(torch.int32),
        past_len.to(torch.int32),
        current_len.to(torch.int32),
        mask_atten,
    ]
    hm_inputs.extend(k_cache)
    hm_inputs.extend(v_cache)
    hm_inputs.extend(k_list)
    hm_inputs.extend(v_list)

    # Get IO names
    input_names, _, output_names = get_input_output_names()

    # Quantize and convert to HMONNX
    if not hmonnx_file.exists():
        quant_scheme = QuantScheme(target_device=DeviceType.XH2a, quant_type=config.quant_type)
        base_quant_cfg = create_quant_config(quant_scheme) or ConfigDict()
        full_quant_cfg = build_quant_config(fronted_graph, warp_inp)
        full_quant_cfg.update(base_quant_cfg)

        quanted_graph = to_quant_graph(fronted_graph, DeviceType.XH2a.name, full_quant_cfg)
        device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        ptq_quantize(quanted_graph, [warp_inp], PrecisionMode.ALIGNED, device)

        convert_quanted_model_to_hmonnx(
            quanted_graph,
            warp_inp,
            hmonnx_file,
            input_names,
            output_names,
        )

    # Generate golden data if needed
    if config.gen_golden:
        generate_golden_data(hmonnx_file, golden_path, hm_inputs)


def process_prefill(args):
    config = ProcessConfig(
        out_dir=Path(args.out_dir),
        model_dir=args.model,
        model_name=args.model_name,
        quant_type=args.quant_type,
        gen_golden=args.gen_golden,
        process_type="prefill"
    )
    process_whisper(config)


def process_decoder(args):
    config = ProcessConfig(
        out_dir=Path(args.out_dir),
        model_dir=args.model,
        model_name=args.model_name,
        quant_type=args.quant_type,
        gen_golden=args.gen_golden,
        process_type="decoder"
    )
    process_whisper(config)


def quant_and_export_llm(args):
    process_encoder(args)
    process_decoder(args)
    process_prefill(args)