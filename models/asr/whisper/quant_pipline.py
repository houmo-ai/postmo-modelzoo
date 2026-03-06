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
import argparse
import json
import os
import tempfile
from copy import deepcopy
from pathlib import Path
from typing import List, Tuple, Dict, Optional
from dataclasses import dataclass
import onnx
import onnxsim
import torch
import torch.nn as nn
from datasets import load_dataset
from transformers import WhisperForConditionalGeneration, WhisperProcessor
from transformers.pipelines.audio_utils import ffmpeg_read
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
from xh_model_zoo.xh_llm.models.whisper._model_opt import *
GB = int(2**30)
_LARGE_MODEL_SIZE_THRESHOLD = int(2**30 * 1.8)


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
        encoder_attention_mask=None,
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
            encoder_attention_mask=encoder_attention_mask,
        )
        output = self.proj_out(hidden_state)
        return output, k_cache_list, v_cache_list



def process_encoder(args):
    out_dir = Path(args.out_dir)
    model_dir = os.path.normpath(args.model)
    model_name = Path(model_dir).stem
    processor = WhisperProcessor.from_pretrained(model_dir)
    model = WhisperForConditionalGeneration.from_pretrained(model_dir)
    model.config.forced_decoder_ids = None
    model.config._attn_implementation = "eager"
    model.model.encoder.decoder_m = model.model.decoder

    name = "encoder"
    work_dir = out_dir / "hmquant" / name
    work_dir.mkdir(exist_ok=True, parents=True)
    onnx_file = work_dir / f"{model_name}_{name}.onnx"

    head_dim = model.model.decoder.layers[0].self_attn.head_dim
    num_heads = model.model.decoder.layers[0].self_attn.num_heads
    embed_dim = model.model.decoder.layers[0].self_attn.embed_dim
    max_source_positions = model.config.max_source_positions
    num_decode_layers = model.config.decoder_layers
    meta_info = {}
    meta_info_file = work_dir / "meta_info.json"
    if meta_info_file.exists():
        with open(meta_info_file, "r", encoding="utf-8") as f:
            meta_info = json.load(f)
    meta_info["hf_model"] = model_dir
    meta_info["model_cfg"] = {
        "head_dim": head_dim,
        "num_heads": num_heads,
        "embed_dim": embed_dim,
        "max_source_positions": max_source_positions,
        "num_decode_layers": num_decode_layers,
    }

    quant_scheme = QuantScheme(target_device=DeviceType.XH2a, quant_type=args.quant_type)
    quant_config = create_quant_config(quant_scheme)
    hmonnx_file = work_dir / f"hmquant_{args.model_name}_with_act.onnx"
    golden_path = work_dir / "golden"

    input_features = torch.randn(1, 80, 3000)


    meta_info["encoder"] = str(hmonnx_file.relative_to(work_dir))
    # input_features = torch.randn(1, 80, 3000)
    audo_file = str(Path(__file__).parent / "audio.mp3")
    sampling_rate = 16000
    with open(audo_file, "rb") as f:
        inputs = f.read()
    if isinstance(inputs, bytes):
        inputs = ffmpeg_read(inputs, sampling_rate)
    sample = {
        "array": inputs,
        "sampling_rate": sampling_rate,
    }
    input_features = processor(
        sample["array"], sampling_rate=sample["sampling_rate"], return_tensors="pt"
    ).input_features

    if not Path(onnx_file).exists():
        with tempfile.TemporaryDirectory() as tmp_dir:
            with RewriterContext(None, backend="onnxruntime"):
                temp_onnx_file = str(Path(tmp_dir) / Path(onnx_file).name)
                torch.onnx.export(
                    model.model.encoder,
                    input_features,  # inputs[0], #
                    temp_onnx_file,
                    input_names=["input_features"],
                    output_names=[
                        "hidden_state",
                    ],
                    # dynamo=True,
                )
                onnx_model = onnx.load(temp_onnx_file)
                model_byte_size = onnx_model.ByteSize()
                if model_byte_size <= _LARGE_MODEL_SIZE_THRESHOLD:
                    onnx_model_sim, checked = onnxsim.simplify(
                        onnx_model,
                        skipped_optimizers=[
                            "fuse_pad_into_conv",
                            "fuse_consecutive_slices",
                            "eliminate_common_subexpression",
                            "fuse_qkv",
                        ],
                    )
                else:
                    from xhquant.utils.onnxsim_large_model import simplify_large_onnx
                    onnx_model_sim, checked = simplify_large_onnx(
                        onnx_model,
                        skipped_optimizers=[
                            "fuse_pad_into_conv",
                            "fuse_consecutive_slices",
                            "eliminate_common_subexpression",
                            "fuse_qkv",
                        ],
                    )
                if checked:
                    onnx_model = onnx_model_sim
    else:
        onnx_model = onnx.load(onnx_file)
    if not os.path.exists(onnx_file):
        onnx.save(
            onnx_model,
            onnx_file,
            save_as_external_data=True,
            all_tensors_to_one_file=True,
            location=f"{Path(onnx_file).stem}_external_data",
        )
    # 2. create output names
    output_names = []
    for i in range(num_decode_layers):
        output_names.append(f"key_state_{i}")
    for i in range(num_decode_layers):
        output_names.append(f"value_state_{i}")
    # 3. convert onnx to hmonnx
    if not Path(hmonnx_file).exists():
        convert_onnx_to_hmonnx(
            str(onnx_file),
            [input_features],
            DeviceType.XH2a,
            hmonnx_file,
            quant_config=quant_config,
            input_names=["input_features"],
            output_names=output_names,
        )
    # generate golden
    if args.gen_golden and not Path(golden_path).exists():
        session = HMONNXGoldenInference(hmonnx_file)
        session.to("cuda")
        session.save_golden = True
        session.golden_dir = str(encoder_work_dir / "hmonnx/golden")
        session.step = 0
        session(input_features.half().to("cuda"))

def process_whisper(args, name: str):
    out_dir = Path(args.out_dir)
    model_dir = os.path.normpath(args.model)
    model_name = Path(model_dir).stem
    processor = WhisperProcessor.from_pretrained(model_dir)
    model = WhisperForConditionalGeneration.from_pretrained(model_dir)
    model.config.forced_decoder_ids = None
    model.config._attn_implementation = "eager"
    model.model.encoder.decoder_m = model.model.decoder

    work_dir = out_dir / "hmquant" / name
    work_dir.mkdir(exist_ok=True, parents=True)

    head_dim = model.model.decoder.layers[0].self_attn.head_dim
    num_heads = model.model.decoder.layers[0].self_attn.num_heads
    embed_dim = model.model.decoder.layers[0].self_attn.embed_dim
    max_source_positions = model.config.max_source_positions
    num_decode_layers = model.config.decoder_layers
    meta_info = {}
    meta_info_file = work_dir / "meta_info.json"
    if meta_info_file.exists():
        with open(meta_info_file, "r", encoding="utf-8") as f:
            meta_info = json.load(f)
    meta_info["hf_model"] = model_dir
    meta_info["model_cfg"] = {
        "head_dim": head_dim,
        "num_heads": num_heads,
        "embed_dim": embed_dim,
        "max_source_positions": max_source_positions,
        "num_decode_layers": num_decode_layers,
    }

    quant_scheme = QuantScheme(target_device=DeviceType.XH2a, quant_type=args.quant_type)
    quant_config = create_quant_config(quant_scheme)
    hmonnx_file = work_dir / f"hmquant_{args.model_name}_with_act.onnx"
    golden_path = work_dir / "golden"

    meta_info[name] = str(hmonnx_file.relative_to(work_dir))

    decoder_input_ids = torch.tensor([[50258, 50259, 50359, 50363]])
    cache_position = torch.tensor([[0, 1, 2, 3]])
    if name == "decoder":
        decoder_input_ids = torch.tensor([[2221]])
        cache_position = torch.tensor([[4]])
    past_len = torch.tensor([0])
    k_cache = [
        torch.ones([1, num_heads, embed_dim, head_dim], dtype=torch.float16) * (-65504)
        for i in range(num_decode_layers)
    ]
    v_cache = [
        torch.ones([1, num_heads, embed_dim, head_dim], dtype=torch.float16) * (-65504)
        for i in range(num_decode_layers)
    ]
    k_list = [
        torch.ones([1, num_heads, max_source_positions, head_dim], dtype=torch.float16)
        * (-65504)
        for i in range(num_decode_layers)
    ]
    v_list = [
        torch.ones([1, num_heads, max_source_positions, head_dim], dtype=torch.float16)
        * (-65504)
        for i in range(num_decode_layers)
    ]

    encoder_attention_mask = torch.zeros(
        (1, 1, 1, max_source_positions),
        dtype=torch.float16,
        device="cpu" # target_device
    )

    inputs_names = [
        "decoder_input_ids",
        "cache_position",
        "past_len",
        "current_len",
        "mask_atten",
        "encoder_attention_mask",
    ]  # , "past_key_length"
    for i in range(num_decode_layers):
        inputs_names.append(f"k_cache_{i}")
    for i in range(num_decode_layers):
        inputs_names.append(f"v_cache_{i}")

    output_names = []
    for i in range(num_decode_layers):
        output_names.append(f"key_state_{i}")
    for i in range(num_decode_layers):
        output_names.append(f"value_state_{i}")

    inputs_names += output_names
    model_cus = Decoder(model.model, model.proj_out, config=model.config)
    output_names = ["logits"]
    for i in range(num_decode_layers):
        output_names.append(f"newk_cache_{i}")
    for i in range(num_decode_layers):
        output_names.append(f"newv_cache_{i}")

    cache_len = decoder_input_ids.shape[1]
    mask_atten = torch.ones(([1, num_heads, cache_len, embed_dim])).half()
    mask_atten[:, :, :, past_len + cache_len :] *= -65504
    current_len = torch.tensor([cache_len])
    # warp
    warp_inp = (
        decoder_input_ids,
        cache_position,
        past_len,
        current_len,
        mask_atten,
        encoder_attention_mask,
        k_cache,
        v_cache,
        k_list,
        v_list,
    )

    with RewriterContext(None, backend="onnxruntime"):
        warp_model_cus = wrap_llm_model(model_cus)
        warp_model_cus = warp_model_cus.half()
        fronted_graph_module = to_frontend_graph(warp_model_cus, "DynamoFX", warp_inp)

    hm_inputs = [
        decoder_input_ids.to(torch.int32),
        cache_position.to(torch.int32),
        past_len.to(torch.int32),
        current_len.to(torch.int32),
        mask_atten,
        encoder_attention_mask,
    ]
    for i in range(num_decode_layers):
        hm_inputs.append(k_cache[i])
    for i in range(num_decode_layers):
        hm_inputs.append(v_cache[i])
    for i in range(num_decode_layers):
        hm_inputs.append(k_list[i])
        hm_inputs.append(v_list[i])
    if not Path(hmonnx_file).exists():
        _input_names = fronted_graph_module.get_input_names()
        if quant_config is None:
            quant_config = ConfigDict()
        if isinstance(quant_config, dict):
            quant_config = ConfigDict(quant_config)
        if "inputs" not in quant_config:
            quant_config.inputs = ConfigDict()

        input_args = []
        for arg in warp_inp:
            if isinstance(arg, (list, tuple)):
                input_args.extend(arg)
            else:
                input_args.append(arg)
        assert len(_input_names) == len(input_args), (
            f"input_names: {len(_input_names)}, input_args: {len(input_args)}"
        )
        for input_name, input_arg in zip(_input_names, input_args):
            input_qconfig = ConfigDict(
                dict(
                    quantizer=dict(
                        qspec=dict(),
                    ),
                )
            )
            if isinstance(input_arg, torch.Tensor):
                if input_arg.dtype in TORCH_DTYPE_TO_FAKE_DTYPE:
                    input_qconfig.quantizer.qspec.fake_dtype = (
                        TORCH_DTYPE_TO_FAKE_DTYPE[input_arg.dtype]
                    )
                else:
                    raise ValueError(f"Unsupported dtype: {input_arg.dtype}")
            quant_config.inputs[input_name] = input_qconfig
        quanted_graph_module = to_quant_graph(
            fronted_graph_module, DeviceType.XH2a.name, quant_config
        )
        execution_devce = (
            torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        )
        ptq_quantize(
            quanted_graph_module, [input_args], PrecisionMode.ALIGNED, execution_devce
        )
        convert_quanted_model_to_hmonnx(
            quanted_graph_module,
            warp_inp,
            hmonnx_file,
            inputs_names,
            output_names,
        )
    with open(meta_info_file, "w", encoding="utf-8") as f:
        json.dump(meta_info, f, ensure_ascii=False, indent=4)
    if args.gen_golden and not Path(golden_path).exists():
        session = HMONNXGoldenInference(hmonnx_file)
        session.to("cuda")
        session.save_golden = True
        session.golden_dir = str(work_dir / "hmonnx/golden")
        session.step = 0
        session(*hm_inputs)


def quant_and_export_llm(args):
    process_encoder(args)
    process_whisper(args, "prefill")
    process_whisper(args, "decoder")