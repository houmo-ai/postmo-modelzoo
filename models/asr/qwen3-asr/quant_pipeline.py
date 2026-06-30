# Copyright (c) 2025 HOUMO AI
#
# File: quant_pipline.py
# Description:
#   Quantization Pipeline Module - Python script implementing the
# quantization pipeline for Qwen3-ASR models.
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
import stat
import shutil
import sys
import json
import time
from typing import List, Tuple

import librosa
import argparse
import tempfile

import onnx
import onnxsim
import importlib.util

import torch
import torch.nn as nn

from pathlib import Path

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

from onnx import TensorProto, helper, numpy_helper
from xhquant.api.ptq_export_hmonnx import (
    convert_quanted_model_to_hmonnx,
)
from xhquant.common.types import PrecisionMode
from xhquant.core.datatype_mapping import TORCH_DTYPE_TO_FAKE_DTYPE
from xhquant.frontend.convert import to_frontend_graph
from xhquant.patch.core import RewriterContext
from xhquant.utils.config import ConfigDict
from xh_model_zoo.xh_llm.models.builder import wrap_llm_model
from xh_model_zoo.xh_llm.models.builder import MODELS
from xh_model_zoo.xh_llm.models.eval_model_type import EvalModelType
from xh_model_zoo.xh_llm.models.qwen3_asr import (
    XHQwen3ASRLLMModel,
    XHQwen3ASRHMONNXModel,
)
from xhquant.api import Config, ConfigDict, PrecisionMode, get_root_logger, ptq_quantize

from qwen_asr.core.transformers_backend import Qwen3ASRConfig, Qwen3ASRProcessor

from xh_model_zoo.xh_llm.models.qwen3_asr import Qwen3ASRForConditionalGeneration

import numpy as np

GB = int(2**30)
_LARGE_MODEL_SIZE_THRESHOLD = int(2**30 * 1.8)

def generate_golden_data(
    hmonnx_file: Path, golden_path: Path, hm_inputs: List[torch.Tensor]
):
    session = HMONNXGoldenInference(hmonnx_file)
    session.to("cuda")
    session.save_golden = True
    session.golden_dir = golden_path
    session.step = 0
    session(*hm_inputs)


def flatten_model_inputs(model_inputs):
    flattened_inputs = []
    for model_input in model_inputs:
        if isinstance(model_input, (List, Tuple)):
            flattened_inputs.extend(model_input)
        else:
            flattened_inputs.append(model_input)
    return flattened_inputs


def get_audio_export_shape(model_cfg):
    audio_config = model_cfg.thinker_config.audio_config
    num_mel_bins = getattr(audio_config, "num_mel_bins", None)
    max_source_positions = getattr(audio_config, "max_source_positions", None)
    if num_mel_bins is None or max_source_positions is None:
        raise ValueError(
            "audio_config must provide num_mel_bins and max_source_positions"
        )
    return int(num_mel_bins), int(max_source_positions) * 2


def get_text_export_shape(xh_model, hf_model):
    text_config = hf_model.config.thinker_config.text_config
    input_sequence_length = getattr(xh_model.wrap_cfg, "input_sequence_length", None)
    if input_sequence_length is None:
        input_sequence_length = getattr(text_config, "max_position_embeddings", None)
    if input_sequence_length is None:
        raise ValueError(
            "Unable to resolve input_sequence_length from wrap_cfg or text_config"
        )
    return ConfigDict(
        dict(
            full_seq_len=int(input_sequence_length),
            hidden_size=int(text_config.hidden_size),
            num_hidden_layers=int(text_config.num_hidden_layers),
        )
    )


def dump_stage_config(cfg, stage: str):
    config_file = Path(cfg.work_dir) / stage / f"{stage}.py"
    config_file.parent.mkdir(exist_ok=True, parents=True)
    print(f"Saving config to {config_file}")
    cfg.dump(config_file)
    return config_file


def xhmodel_export_onnx(
    xh_model,
    tokenizer,
    data_batch,
    onnx_output_dir: str,
    cfg_name,
    device,
    dtype,
    logger,
    valid: bool = True,
):
    logger.info("Start exporting...")
    xh_model.to("cpu")
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    xh_model.convert_to_export_graph(data_batch)
    logger.info("Finish exporting...")

    logger.info(f"************* Start Exported Graph *************")

    logger.info(f"************* End Exported Graph *************")
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    xh_model.change_eval_type(EvalModelType.EXPORTED)

    xh_model.to("cpu")
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    logger.info("*************** Start exporting onnx ***************")
    onnx_file = xh_model.to_export_onnx(data_batch, onnx_output_dir, cfg_name)[0]
    return onnx_file


def quant_encode(args):
    target_device = "XH2a"
    model_dir = os.path.normpath(args.model)

    model = Qwen3ASRForConditionalGeneration.from_pretrained(model_dir)
    cfg = Qwen3ASRConfig.from_pretrained(model_dir)

    model.eval()
    model.thinker.audio_tower.eval()

    model.config.forced_decoder_ids = None
    model.config._attn_implementation = "eager"

    model_name = args.model_name

    work_dir = Path(args.out_dir)
    work_dir.mkdir(exist_ok=True, parents=True)

    head_dim = cfg.thinker_config.text_config.head_dim
    num_heads = cfg.thinker_config.text_config.num_attention_heads
    num_key_value_heads = cfg.thinker_config.text_config.num_key_value_heads
    embed_dim = cfg.thinker_config.text_config.hidden_size
    num_decode_layers = cfg.thinker_config.text_config.num_hidden_layers

    max_source_positions = cfg.thinker_config.audio_config.max_source_positions
    # Manually specified fixed audio length, used to fix the Encoder input time dimension T during ONNX/HMONNX export
    max_audio_length = int(args.max_audio_length)
    num_mel_bins = model.config.thinker_config.audio_config.num_mel_bins

    meta_info = {}
    meta_info_file = work_dir / "meta_info.json"
    if meta_info_file.exists():
        with open(meta_info_file, "r", encoding="utf-8") as f:
            meta_info = json.load(f)
    meta_info["hf_model"] = model_dir
    meta_info["model_cfg"] = {
        "head_dim": head_dim,
        "num_heads": num_heads,
        "num_key_value_heads": num_key_value_heads,
        "embed_dim": embed_dim,
        "max_source_positions": max_source_positions,
        "num_decode_layers": num_decode_layers,
        "num_mel_bins": num_mel_bins,
        "fixed_max_audio_length": max_audio_length,
    }

    name = "encode"
    onnx_file = work_dir / name / f"{model_name}_{target_device}.onnx"
    onnx_file.parent.mkdir(exist_ok=True, parents=True)
    quant_type = args.quant_type
    quant_scheme = QuantScheme(target_device=DeviceType.XH2a, quant_type=quant_type)
    quant_config = create_quant_config(quant_scheme)
    hmonnx_file = work_dir / name / f"hmquant_{model_name}_with_act.onnx"
    golden_path = work_dir / name
    meta_info["encoder"] = str(hmonnx_file.relative_to(work_dir))

    # Use manually specified max_audio_length to fix the Encoder input time dimension T, using mel dimension from config
    input_features = (
        torch.randn(1, num_mel_bins, max_audio_length).to(model.device).to(model.dtype)
    )
    # The corresponding length tensor must be consistent with T to ensure fixed graph input shape after export
    feature_lens = torch.tensor([max_audio_length], dtype=torch.int32).to(model.device)

    if not Path(onnx_file).exists():
        with tempfile.TemporaryDirectory() as tmp_dir:
            with RewriterContext(None, backend="onnxruntime"):
                temp_onnx_file = str(Path(tmp_dir) / Path(onnx_file).name)
                torch.onnx.export(
                    model.thinker.audio_tower,
                    (
                        input_features,
                        feature_lens,
                    ),  # Pass input_features and feature_lens
                    temp_onnx_file,
                    input_names=["input_features", "feature_lens"],
                    output_names=["hidden_state"],
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

    output_names = []
    output_names.append("hidden_state")

    if not Path(hmonnx_file).exists():
        convert_onnx_to_hmonnx(
            str(onnx_file),
            [input_features, feature_lens],
            DeviceType.XH2a,
            hmonnx_file,
            quant_config=quant_config,
            input_names=["input_features", "feature_lens"],
            output_names=output_names,
        )

    # Generate golden
    if args.gen_golden:
        session = HMONNXGoldenInference(hmonnx_file)
        session.to("cuda")
        session.save_golden = True
        session.golden_dir = golden_path
        session.step = 0
        session(input_features.half().to("cuda"), feature_lens.to("cuda"))

    os.remove(onnx_file)
    os.remove(str(onnx_file).replace(".onnx", "_external_data"))


def quant_forcealigner_encode(args):
    target_device = "XH2a"
    model_dir = os.path.normpath(args.model)

    model = Qwen3ASRForConditionalGeneration.from_pretrained(model_dir)
    cfg = Qwen3ASRConfig.from_pretrained(model_dir)

    model.eval()
    model.thinker.audio_tower.eval()

    model.config.forced_decoder_ids = None
    model.config._attn_implementation = "eager"

    model_name = args.model_name

    work_dir = Path(args.out_dir)
    work_dir.mkdir(exist_ok=True, parents=True)

    head_dim = cfg.thinker_config.text_config.head_dim
    num_heads = cfg.thinker_config.text_config.num_attention_heads
    num_key_value_heads = cfg.thinker_config.text_config.num_key_value_heads
    embed_dim = cfg.thinker_config.text_config.hidden_size
    num_decode_layers = cfg.thinker_config.text_config.num_hidden_layers

    max_source_positions = cfg.thinker_config.audio_config.max_source_positions
    num_mel_bins, encoder_input_length = get_audio_export_shape(cfg)

    meta_info = {}
    meta_info_file = work_dir / "meta_info.json"
    if meta_info_file.exists():
        with open(meta_info_file, "r", encoding="utf-8") as f:
            meta_info = json.load(f)
    meta_info["hf_model"] = model_dir
    meta_info["model_cfg"] = {
        "head_dim": head_dim,
        "num_heads": num_heads,
        "num_key_value_heads": num_key_value_heads,
        "embed_dim": embed_dim,
        "max_source_positions": max_source_positions,
        "num_decode_layers": num_decode_layers,
    }

    name = "encode"
    onnx_file = work_dir / name / f"{model_name}_{target_device}.onnx"
    onnx_file.parent.mkdir(exist_ok=True, parents=True)
    quant_type = args.quant_type
    quant_scheme = QuantScheme(target_device=DeviceType.XH2a, quant_type=quant_type)
    quant_config = create_quant_config(quant_scheme)
    hmonnx_file = work_dir / name / f"hmquant_{model_name}_with_act.onnx"
    golden_path = work_dir / name
    meta_info["encoder"] = str(hmonnx_file.relative_to(work_dir))

    input_features = (
        torch.randn(1, num_mel_bins, encoder_input_length)
        .to(model.device)
        .to(model.dtype)
    )
    feature_lens = torch.tensor([encoder_input_length], dtype=torch.int32).to(
        model.device
    )

    if not Path(onnx_file).exists():
        with tempfile.TemporaryDirectory() as tmp_dir:
            with RewriterContext(None, backend="onnxruntime"):
                temp_onnx_file = str(Path(tmp_dir) / Path(onnx_file).name)
                torch.onnx.export(
                    model.thinker.audio_tower,
                    (
                        input_features,
                        feature_lens,
                    ),  # Pass input_features and feature_lens
                    temp_onnx_file,
                    input_names=["input_features", "feature_lens"],
                    output_names=["hidden_state"],
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

    output_names = []
    output_names.append("hidden_state")

    if not Path(hmonnx_file).exists():
        convert_onnx_to_hmonnx(
            str(onnx_file),
            [input_features, feature_lens],
            DeviceType.XH2a,
            hmonnx_file,
            quant_config=quant_config,
            input_names=["input_features", "feature_lens"],
            output_names=output_names,
        )

    # Generate golden
    if args.gen_golden:
        session = HMONNXGoldenInference(hmonnx_file)
        session.to("cuda")
        session.save_golden = True
        session.golden_dir = golden_path
        session.step = 0
        session(input_features.half().to("cuda"), feature_lens.to("cuda"))

    os.remove(onnx_file)
    os.remove(str(onnx_file).replace(".onnx", "_external_data"))


def _get_feat_extract_output_lengths(input_lengths):
    """
    Compute the output length of convolutional layers and audio encoder output length.
    Consistent with the implementation in xh2modelzoo.
    """
    # 8 = [100, ... 100]
    input_lengths_leave = input_lengths % 100  # [0, 0, ..., 0]
    feat_lengths = (input_lengths_leave - 1) // 2 + 1  # 0
    output_lengths = (
        ((feat_lengths - 1) // 2 + 1 - 1) // 2 + 1 + (input_lengths // 100) * 13
    )
    # output_lengths: tensor([13, 13, 13, 13, 13, 13, 13, 13], device='cuda:0')
    return output_lengths


def quant_model(args, stages: Tuple[str, ...] = ("prefill", "decode")):

    MODEL_PATH = os.path.expanduser(args.model)

    model_name = os.path.basename(MODEL_PATH)
    target_device = "XH2a"

    cfg = Config.fromfile(args.config)
    cfg_name = f"{model_name}_{target_device}"
    print(cfg_name)
    cfg_name = f"{cfg_name}"
    cfg.work_dir = str(Path(args.out_dir))
    print(f"Work dir: {cfg.work_dir}")
    log_file = Path(cfg.work_dir) / f"{cfg_name}.log"
    Path(cfg.work_dir).mkdir(exist_ok=True, parents=True)
    cfg.device = "cuda" if torch.cuda.is_available() else "cpu"
    cfg.exec_device = "cuda" if torch.cuda.is_available() else "cpu"

    cfg.dtype = "float16"
    logger = get_root_logger()

    cfg.config_dir = MODEL_PATH
    cfg.hf_model_dir = cfg.config_dir
    cfg.model.hf_model = cfg.hf_model_dir

    device = torch.device(cfg.device)
    exec_device = torch.device(cfg.exec_device)
    dtype = getattr(torch, cfg.dtype)

    xh_model = MODELS.build(cfg.model)

    model = xh_model.get_hf_model()
    assert isinstance(
        xh_model, XHQwen3ASRLLMModel
    ), f"Model must be XHQwen3ASRLLMModel, but got {type(xh_model)}"

    xh_model.init_wrap_model(model.thinker.model)

    xh_model.wrap_model.lm_head = model.thinker.lm_head
    xh_model.wrap_model.lm_head.to(device)
    xh_model.wrap_model.lm_head.to(dtype)

    processor = xh_model.get_processor()

    meta_info = ConfigDict(
        dict(
            create_time=time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()),
            stage_configs={},
        )
    )
    meta_info["wrap_cfg"] = xh_model.wrap_cfg.to_dict()
    hf_model_dir = cfg.hf_model_dir
    meta_info.hf_model = hf_model_dir

    hf_model_config_dir = cfg.config_dir

    hf_config_dir = Path(cfg.work_dir) / "hf_config"
    hf_config_dir.mkdir(exist_ok=True, parents=True)
    hf_config_files = [
        "chat_template.json",
        "config.json",
        "tokenizer_config.json",
        "vocab.json",
        "configuration.json",
        "generation_config.json",
        "merges.txt",
        "preprocessor_config.json",
    ]

    for cfg_file in hf_config_files:
        shutil.copyfile(
            Path(hf_model_config_dir) / cfg_file,
            Path(hf_config_dir) / cfg_file,
        )
    meta_info.hf_config = str(hf_config_dir.relative_to(cfg.work_dir))

    token_embedding = xh_model.token_embedding
    token_embedding_file = Path(cfg.work_dir) / "quant_embedding.pt"
    torch.save(token_embedding.state_dict(), str(token_embedding_file))
    meta_info.token_embedding_file = str(token_embedding_file.relative_to(cfg.work_dir))

    if xh_model.past_key_caches is not None and len(xh_model.past_key_caches) > 0:
        meta_info.use_cache = True
        meta_info.kv_cache_shape = xh_model.past_key_caches[0].shape
        meta_info.num_hidden_layers = len(xh_model.past_key_caches)

    # wrapped decoder
    xh_model.to(device)
    xh_model.to(dtype)
    xh_model.change_eval_type(eval_type=EvalModelType.WRAPED)
    # entire model
    model.to(device)

    model.config.forced_decoder_ids = None
    model.config._attn_implementation = "eager"
    text_export_shape = get_text_export_shape(xh_model, model)
    hidden_size = text_export_shape.hidden_size
    num_decode_layers = text_export_shape.num_hidden_layers

    # Compute the sequence length after embedding based on max_audio_length
    max_audio_length = args.max_audio_length
    embed_lengths = _get_feat_extract_output_lengths(max_audio_length)
    text_embed_lengths = embed_lengths + 21  # 21 is the text prompt length

    tokenizer = processor.tokenizer
    final_inputs_embeds = torch.randn(
        (1, text_embed_lengths, hidden_size), device=device, dtype=dtype
    )
    print(f"final_inputs_embeds.shape: {final_inputs_embeds.shape}")

    seq_len = final_inputs_embeds.shape[1]

    # Pad the second dimension to text_embed_lengths
    final_inputs_embeds = torch.cat(
        [
            final_inputs_embeds,
            torch.zeros(
                (1, text_embed_lengths - seq_len, final_inputs_embeds.shape[2]),
                dtype=dtype,
                device=device,
            ),
        ],
        dim=1,
    )

    data_batch = {"input_embeds": final_inputs_embeds.half(), "past_seq_length": [0]}

    # Set the input sequence length
    xh_model.set_input_sequence_length(text_embed_lengths)

    xh_model.interactive_mode = True
    logger.info("************* convert to frontend graph *************")

    xh_model.convert_to_fronted_graph(data_batch)
    logger.info(f"************* Start Frontend Graph *************")

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    logger.info("************* convert to quanted graph *************")

    xh_model.convert_to_quant_graph(target_device)

    xh_model.change_eval_type(EvalModelType.CALIBRATION)
    xh_model.enable_calibration()
    xh_model.to(dtype)
    xh_model.to(device)

    logger.info("*************** Start PTQ Quantize ***************")

    calib_data = xh_model.prepare_inputs(data_batch)
    new_args = []
    for arg in calib_data:
        if isinstance(arg, (List, Tuple)):
            new_args.extend(arg)
        else:
            new_args.append(arg)
    calib_data = new_args
    ptq_quantize(
        xh_model.quanted_model, [calib_data], PrecisionMode.ALIGNED, [exec_device]
    )
    logger.info("*************** Finished PTQ Quantize **************")

    xh_model.change_eval_type(EvalModelType.QUANTED_ALIGNED)
    xh_model.to(device)
    xh_model.to(dtype)

    xh_model = xh_model.to("cpu")

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    # export_cfg: Expand past_key_cache and past_value_cache inputs into multiple inputs for easier alignment
    num_hidden_layers = num_decode_layers
    base_inputs = ["input_embeds", "past_seq_length", "current_input_length"]
    key_names = [f"past_key_cache_{i}" for i in range(num_hidden_layers)]
    value_names = [f"past_value_cache_{i}" for i in range(num_hidden_layers)]
    input_names = base_inputs + key_names + value_names
    xh_model.export_cfg = ConfigDict(
        dict(input_names=input_names, output_names=["last_hidden_state"])
    )

    stage_batches = {
        "prefill": {
            "input_embeds": final_inputs_embeds.half(),
            "past_seq_length": [0],
        },
        "decode": {
            "input_embeds": final_inputs_embeds[:, -1:, :].to(device),
            "past_seq_length": [final_inputs_embeds.shape[1]],
        },
    }
    stage_sequence_lengths = {
        "prefill": text_embed_lengths,
        "decode": 1,
    }

    for stage in stages:
        if stage not in stage_batches:
            raise ValueError(f"Unsupported export stage: {stage}")

        onnx_dir = Path(cfg.work_dir) / stage
        onnx_dir.mkdir(exist_ok=True, parents=True)
        config_file = dump_stage_config(cfg, stage)
        meta_info["stage_configs"][stage] = str(config_file.relative_to(cfg.work_dir))

        if stage == "decode":
            xh_model.change_eval_type(EvalModelType.QUANTED_ALIGNED)
            xh_model.to(device)
            xh_model.to(dtype)
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            logger.info(f"past_seq_len: {stage_batches[stage]['past_seq_length'][0]}")
            logger.info(
                f"*************** Start exporting {stage} model ***************"
            )

        xh_model.set_input_sequence_length(stage_sequence_lengths[stage])
        xh_model = xh_model.to("cpu")

        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()

        onnx_file = xhmodel_export_onnx(
            xh_model,
            tokenizer,
            stage_batches[stage],
            str(onnx_dir),
            f"hmquant_{args.model_name}_with_act",
            "cpu",
            dtype,
            logger,
            False,
        )

        logger.info(f"save {stage} onnx model to {onnx_file}")
        logger.info(
            "*************** Finished exporting " + stage + " model ***************"
        )
        meta_info[stage + "_onnx_file"] = str(Path(onnx_file).relative_to(cfg.work_dir))

        if args.gen_golden:
            golden_inputs = flatten_model_inputs(
                xh_model.prepare_inputs(stage_batches[stage])
            )
            generate_golden_data(onnx_file, onnx_dir, golden_inputs)

        xh_model.release_exported_model()

    with open(Path(cfg.work_dir) / "export_meta_info.json", "w", encoding="utf-8") as f:
        json.dump(meta_info, f, indent=4)


def quant_forcealigner(args):
    from xh_model_zoo.xh_llm.models.qwen3_forcealigner import XHQwen3ASRLLMModel

    DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    MODEL_PATH = os.path.normpath(args.model)

    hf_model = Qwen3ASRForConditionalGeneration.from_pretrained(
        MODEL_PATH,
        dtype=torch.float16,
        device_map=DEVICE,
    )
    hf_model.eval()

    model_name = args.model_name
    target_device = "XH2a"  # Quantization target device

    cfg = Config.fromfile(args.config)
    cfg_name = f"{model_name}_{target_device}"
    print(cfg_name)
    cfg_name = f"{cfg_name}"
    cfg.work_dir = str(Path(args.out_dir))
    print(f"Work dir: {cfg.work_dir}")
    log_file = Path(cfg.work_dir) / f"{cfg_name}.log"
    Path(cfg.work_dir).mkdir(exist_ok=True, parents=True)
    cfg.device = "cuda" if torch.cuda.is_available() else "cpu"
    cfg.exec_device = "cuda" if torch.cuda.is_available() else "cpu"

    cfg.dtype = "float16"
    logger = get_root_logger()

    cfg.config_dir = MODEL_PATH
    cfg.hf_model_dir = cfg.config_dir
    cfg.model.hf_model = cfg.hf_model_dir

    logger.info(f"\nConfig:\n{cfg.pretty_text}")
    config_file = Path(cfg.work_dir) / Path(args.config).name
    cfg.dump(config_file)

    device = torch.device(cfg.device)
    exec_device = torch.device(cfg.exec_device)
    dtype = getattr(torch, cfg.dtype)

    xh_model = MODELS.build(cfg.model)
    model = xh_model.get_hf_model()
    assert isinstance(xh_model, XHQwen3ASRLLMModel)

    # =============== Key: Override wrap_cfg before init_wrap_model ===============
    full_seq_len = getattr(xh_model.wrap_cfg, "input_sequence_length", None)
    if full_seq_len is None:
        full_seq_len = getattr(
            model.config.thinker_config.text_config, "max_position_embeddings", None
        )
    if full_seq_len is None:
        raise ValueError(
            "Unable to resolve input_sequence_length for forcealigner export"
        )
    xh_model.wrap_cfg.num_logits_to_keep = 0
    xh_model.wrap_cfg.only_first_block = False
    xh_model.wrap_cfg.input_sequence_length = full_seq_len

    # wrap text model
    xh_model.init_wrap_model(model.thinker.model)

    xh_model.wrap_model.lm_head = model.thinker.lm_head
    xh_model.wrap_model.lm_head.to(device)
    xh_model.wrap_model.lm_head.to(dtype)

    processor = xh_model.get_processor()
    tokenizer = processor.tokenizer

    # ===== Model settings =====
    xh_model.to(device)
    xh_model.to(dtype)
    xh_model.change_eval_type(eval_type=EvalModelType.WRAPED)

    model.to(device)
    model.config.forced_decoder_ids = None
    model.config._attn_implementation = "eager"

    text_config = model.config.thinker_config.text_config
    hidden_size = text_config.hidden_size
    num_decode_layers = text_config.num_hidden_layers

    logger.info(f"text_config.hidden_size={hidden_size}, layers={num_decode_layers}")

    meta_info = ConfigDict(
        dict(
            create_time=time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()),
            config=str(config_file.relative_to(cfg.work_dir)),
        )
    )
    meta_info["wrap_cfg"] = xh_model.wrap_cfg.to_dict()
    hf_model_dir = cfg.hf_model_dir
    meta_info.hf_model = hf_model_dir

    hf_model_config_dir = cfg.config_dir

    hf_config_dir = Path(str(args.model))
    hf_config_dir.mkdir(exist_ok=True, parents=True)
    hf_config_files = [
        "chat_template.json",
        "config.json",
        "tokenizer_config.json",
        "vocab.json",
        "configuration.json",
        "generation_config.json",
        "merges.txt",
        "preprocessor_config.json",
    ]
    meta_info.hf_config = str(hf_config_dir)

    token_embedding = xh_model.token_embedding
    token_embedding_file = Path(cfg.work_dir) / "quant_embedding.pt"
    torch.save(token_embedding.state_dict(), str(token_embedding_file))
    meta_info.token_embedding_file = str(token_embedding_file.relative_to(cfg.work_dir))

    # ===== Construct prefill input =====
    final_inputs_embeds = torch.randn(
        (1, full_seq_len, hidden_size), device=device, dtype=torch.float16
    )

    data_batch = {
        "input_embeds": final_inputs_embeds.half(),
        "past_seq_length": [0],
        "current_input_length": [full_seq_len],
    }

    # Debug: the exported input shape should match wrap_cfg.input_sequence_length and hidden_size.
    with torch.no_grad():
        outs = xh_model.test_step(data_batch)

    # ===== Quantization =====
    xh_model.interactive_mode = True
    logger.info("************* convert to frontend graph *************")
    xh_model.convert_to_fronted_graph(data_batch)

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    logger.info("************* convert to quanted graph *************")
    xh_model.convert_to_quant_graph(target_device)

    xh_model.change_eval_type(EvalModelType.CALIBRATION)
    xh_model.enable_calibration()
    xh_model.to(dtype)
    xh_model.to(device)

    logger.info("*************** Start PTQ Quantize ***************")
    calib_data = xh_model.prepare_inputs(data_batch)
    new_args = []
    for arg in calib_data:
        if isinstance(arg, (List, Tuple)):
            new_args.extend(arg)
        else:
            new_args.append(arg)
    ptq_quantize(
        xh_model.quanted_model, [new_args], PrecisionMode.ALIGNED, [exec_device]
    )
    logger.info("*************** Finished PTQ Quantize **************")

    xh_model.change_eval_type(EvalModelType.QUANTED_ALIGNED)
    xh_model.to(device)
    xh_model.to(dtype)

    # ===== Prefill export =====
    xh_model = xh_model.to("cpu")

    work_dir = Path(args.out_dir)
    work_dir.mkdir(exist_ok=True, parents=True)
    prefill_onnx_dir = work_dir / "prefill"
    prefill_golden_path = prefill_onnx_dir / "hmonnx/golden"
    prefill_onnx_dir.mkdir(exist_ok=True, parents=True)

    num_hidden_layers = num_decode_layers
    base_inputs = ["input_embeds", "past_seq_length", "current_input_length"]
    key_names = [f"past_key_cache_{i}" for i in range(num_hidden_layers)]
    value_names = [f"past_value_cache_{i}" for i in range(num_hidden_layers)]
    input_names = base_inputs + key_names + value_names

    xh_model.export_cfg = ConfigDict(
        dict(
            input_names=input_names,
            output_names=["hidden_states"],  # Name doesn't matter, key is output shape
        )
    )

    xh_model.set_input_sequence_length(full_seq_len)

    prefill_onnx_file = xhmodel_export_onnx(
        xh_model,
        tokenizer,
        data_batch,
        str(prefill_onnx_dir),
        f"hmquant_{args.model_name}_with_act",
        "cpu",
        dtype,
        logger,
        False,
    )

    if args.gen_golden:
        session = HMONNXGoldenInference(prefill_onnx_file)
        session.to("cuda")
        session.save_golden = True
        session.golden_dir = str(prefill_onnx_dir / "hmonnx/golden")
        session.step = 0
        session(*calib_data)

    xh_model.release_exported_model()
    logger.info(f"save prefill onnx model to {prefill_onnx_file}")


def quant_asr(args):
    model_basename = os.path.basename(os.path.normpath(args.model))
    if model_basename != "Qwen3-ForcedAligner-0.6B":
        quant_encode(args)
        quant_model(args)
    else:
        quant_forcealigner_encode(args)
        quant_forcealigner(args)
