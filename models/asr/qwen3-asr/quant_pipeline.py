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
from xh_model_zoo.xh_llm.models.qwen3_asr import XHQwen3ASRLLMModel, XHQwen3ASRHMONNXModel
from xhquant.api import Config, ConfigDict, PrecisionMode, get_root_logger, ptq_quantize

from qwen_asr.core.transformers_backend import (
    Qwen3ASRConfig,
    Qwen3ASRProcessor
)

from xh_model_zoo.xh_llm.models.qwen3_asr import (
    Qwen3ASRForConditionalGeneration
)


GB = int(2**30)
_LARGE_MODEL_SIZE_THRESHOLD = int(2**30 * 1.8)


def generate_golden_data(hmonnx_file: Path, golden_path: Path, hm_inputs: List[torch.Tensor]):
    session = HMONNXGoldenInference(hmonnx_file)
    session.to("cuda")
    session.save_golden = True
    session.golden_dir = golden_path
    session.step = 0
    session(*hm_inputs)

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
    processor = Qwen3ASRProcessor.from_pretrained(model_dir)

    model.eval()
    model.thinker.audio_tower.eval()

    model.config.forced_decoder_ids = None
    model.config._attn_implementation = "eager"

    model_name = args.model_name
    cfg_name = f"{model_name}_{target_device}"
    work_dir = Path(args.out_dir) / "hmquant"
    work_dir.mkdir(exist_ok=True, parents=True)

    head_dim = cfg.thinker_config.text_config.head_dim
    num_heads = cfg.thinker_config.text_config.num_attention_heads
    num_key_value_heads = cfg.thinker_config.text_config.num_key_value_heads
    embed_dim = cfg.thinker_config.text_config.hidden_size
    num_decode_layers = cfg.thinker_config.text_config.num_hidden_layers

    max_source_positions = cfg.thinker_config.audio_config.max_source_positions


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
    num_mel_bins = model.config.thinker_config.audio_config.num_mel_bins

    input_features = torch.randn(1, 128, 3000).to(model.device).to(model.dtype)
    feature_lens = torch.tensor([3000], dtype=torch.int32).to(model.device)

    if not Path(onnx_file).exists():
        with tempfile.TemporaryDirectory() as tmp_dir:
            with RewriterContext(None, backend="onnxruntime"):
                temp_onnx_file = str(Path(tmp_dir) / Path(onnx_file).name)
                torch.onnx.export(
                    model.thinker.audio_tower,
                    (input_features, feature_lens),  # 传入 input_features 和 feature_lens
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

    convert_onnx_to_hmonnx(
        str(onnx_file),
        [input_features, feature_lens],
        # [input_features],
        DeviceType.XH2a,
        hmonnx_file,
        quant_config=quant_config,
        input_names=["input_features", "feature_lens"],
        # input_names=["input_features"],
        output_names=output_names,
    )


    generate_golden_data(hmonnx_file, golden_path, [input_features.half().to("cuda"), feature_lens.to("cuda")])
def quant_model(args, stage : str):

    DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    MODEL_PATH = os.path.expanduser(args.model)
    hf_model = Qwen3ASRForConditionalGeneration.from_pretrained(
        MODEL_PATH,
        dtype=torch.float16,
        device_map=DEVICE,
    )
    hf_model.eval()

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

    cfg.config_dir = str(os.path.basename(Path(args.model)))
    cfg.hf_model_dir = cfg.config_dir
    cfg.model.hf_model = cfg.hf_model_dir

    config_file = Path(cfg.work_dir) / "hmquant" / stage / (stage + ".py")
    config_file.parent.mkdir(exist_ok=True, parents=True)
    print(f"Saving config to {config_file}")
    cfg.dump(config_file)


    device = torch.device(cfg.device)
    exec_device = torch.device(cfg.exec_device)
    dtype = getattr(torch, cfg.dtype)

    xh_model = MODELS.build(cfg.model)

    model = xh_model.get_hf_model()
    assert isinstance(xh_model, XHQwen3ASRLLMModel), f"Model must be XHQwen3ASRLLMModel, but got {type(xh_model)}"

    xh_model.init_wrap_model(model.thinker.model)

    xh_model.wrap_model.lm_head = model.thinker.lm_head
    xh_model.wrap_model.lm_head.to(device)
    xh_model.wrap_model.lm_head.to(dtype)

    processor = xh_model.get_processor()

    onnx_dir = Path(cfg.work_dir) / "hmquant" /stage
    onnx_dir.mkdir(exist_ok=True, parents=True)

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
        "preprocessor_config.json"
    ]

    for cfg_file in hf_config_files:
        shutil.copyfile(
            Path(hf_model_config_dir) / cfg_file,
            Path(hf_config_dir) / cfg_file,
        )
    meta_info.hf_config = str(hf_config_dir.relative_to(cfg.work_dir))

    token_embedding = xh_model.token_embedding
    token_embedding_file = Path(cfg.work_dir) / "hmquant" / "quant_embedding.pt"
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
    text_config = model.config.thinker_config.text_config

    cache_len = 1500
    hidden_size = text_config.hidden_size
    head_dim = text_config.hidden_size // text_config.num_key_value_heads # 128
    num_key_value_heads = text_config.num_key_value_heads # 8
    num_decode_layers = text_config.num_hidden_layers # 28
    full_seq_len = 411
    tokenizer = processor.tokenizer
    if "0.6B" in args.model:
        final_inputs_embeds = torch.randn((1, full_seq_len, 1024), device=device, dtype=torch.float16)
    else:
        final_inputs_embeds = torch.randn((1, full_seq_len, 2048), device=device, dtype=torch.float16)
    print(f"final_inputs_embeds.shape: {final_inputs_embeds.shape}") # torch.Size([1, 411, 2048])

    seq_len = final_inputs_embeds.shape[1]

    final_inputs_embeds = torch.cat([final_inputs_embeds, torch.zeros((1, 411 - seq_len, final_inputs_embeds.shape[2]), dtype=torch.float16, device=device)], dim=1)

    
    data_batch = {
        "input_embeds": final_inputs_embeds.half(),
        "past_seq_length": [0]
    }


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
    ptq_quantize(xh_model.quanted_model, [calib_data], PrecisionMode.ALIGNED, [exec_device])
    logger.info("*************** Finished PTQ Quantize **************")


    xh_model.change_eval_type(EvalModelType.QUANTED_ALIGNED)
    xh_model.to(device)
    xh_model.to(dtype)

    xh_model = xh_model.to("cpu")


    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    # export_cfg 展开 past_key_cache 和 past_value_cache 的输入，变成多个输入，方便后续对齐
    num_hidden_layers = 28
    base_inputs = ["input_embeds", "past_seq_length", "current_input_length"]
    key_names = [f"past_key_cache_{i}" for i in range(num_hidden_layers)]
    value_names = [f"past_value_cache_{i}" for i in range(num_hidden_layers)]
    input_names = base_inputs + key_names + value_names
    xh_model.export_cfg = ConfigDict(dict(input_names=input_names, output_names=["last_hidden_state"]))


    xh_model.set_input_sequence_length(full_seq_len)

    if stage == "prefill":
        onnx_file = xhmodel_export_onnx(
            xh_model,
            tokenizer,
            data_batch,
            str(onnx_dir),
            f"hmquant_{args.model_name}_with_act",
            "cpu",
            dtype,
            logger,
            False,
        )
        logger.info(f"save {stage} onnx model to {onnx_file}")
        logger.info("*************** Finished exporting " + stage + " model ***************")
        meta_info[stage + "_onnx_file"] = str(Path(onnx_file).relative_to(cfg.work_dir))

    xh_model.release_exported_model()



    if stage == "decode":
        xh_model.change_eval_type(EvalModelType.QUANTED_ALIGNED)
        xh_model.to(device)
        xh_model.to(dtype)

        data_batch["input_embeds"] = data_batch["input_embeds"].to(device)

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        xh_model.set_input_sequence_length(1)

        past_seq_len = final_inputs_embeds.shape[1]
        prefill_next_token_embeds = final_inputs_embeds[:, -1:, :]
        final_inputs_embeds = prefill_next_token_embeds
        logger.info(f"past_seq_len: {past_seq_len}")

        data_batch = {
            "input_embeds": final_inputs_embeds.to(device),
            "past_seq_length": [past_seq_len],
        }

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        logger.info(f"*************** Start exporting {stage} model ***************")

        xh_model = xh_model.to("cpu")

        cpu_inputs_embeds = data_batch["input_embeds"].to("cpu")
        if cpu_inputs_embeds.dim() == 2:
            cpu_inputs_embeds = cpu_inputs_embeds.unsqueeze(0)

        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()

        onnx_file = xhmodel_export_onnx(
            xh_model,
            tokenizer,
            data_batch,
            str(onnx_dir),
            f"hmquant_{args.model_name}_with_act",
            "cpu",
            dtype,
            logger,
            False,
        )

        logger.info(f"save {stage} onnx model to {onnx_file}")
        logger.info("*************** Finished exporting " + stage + " model ***************")
        meta_info[stage + "_onnx_file"] = str(Path(onnx_file).relative_to(cfg.work_dir))

    json.dump(meta_info, open(Path(cfg.work_dir) / "export_meta_info.json", "w"), indent=4)

    decode_inputs = xh_model.prepare_inputs(data_batch)
    decode_calib_data = []
    for arg in decode_inputs:
        if isinstance(arg, (List, Tuple)):
            decode_calib_data.extend(arg)
        else:
            decode_calib_data.append(arg)

    generate_golden_data(onnx_file, onnx_dir, decode_calib_data)


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

    model_name = os.path.basename(MODEL_PATH)
    target_device = "XH2a"  # 量化目标设备

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

    cfg.config_dir = str(os.path.basename(Path(args.model)))
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

    # =============== 关键：在 init_wrap_model 前覆盖 wrap_cfg ===============
    full_seq_len = 411
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

    # ===== 模型设置 =====
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
        "preprocessor_config.json"
    ]
    meta_info.hf_config = str(hf_config_dir)

    token_embedding = xh_model.token_embedding
    token_embedding_file = Path(cfg.work_dir) / "hmquant" /"quant_embedding.pt"
    torch.save(token_embedding.state_dict(), str(token_embedding_file))
    meta_info.token_embedding_file = str(token_embedding_file.relative_to(cfg.work_dir))

    # ===== 构造 prefill 输入 =====
    final_inputs_embeds = torch.randn((1, full_seq_len, hidden_size), device=device, dtype=torch.float16)

    data_batch = {
        "input_embeds": final_inputs_embeds.half(),
        "past_seq_length": [0],
        "current_input_length": [full_seq_len],
    }

    # Debug: 这里必须看到 (1,411,1024)
    with torch.no_grad():
        outs = xh_model.test_step(data_batch)

    # ===== 量化 =====
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
    ptq_quantize(xh_model.quanted_model, [new_args], PrecisionMode.ALIGNED, [exec_device])
    logger.info("*************** Finished PTQ Quantize **************")

    xh_model.change_eval_type(EvalModelType.QUANTED_ALIGNED)
    xh_model.to(device)
    xh_model.to(dtype)

    # ===== prefill 导出 =====
    xh_model = xh_model.to("cpu")

    work_dir = Path(args.out_dir) / "hmquant"
    work_dir.mkdir(exist_ok=True, parents=True)
    prefill_onnx_dir = work_dir / "prefill"
    prefill_golden_path = prefill_onnx_dir / "hmonnx/golden"
    prefill_onnx_dir.mkdir(exist_ok=True, parents=True)

    num_hidden_layers = num_decode_layers
    base_inputs = ["input_embeds", "past_seq_length", "current_input_length"]
    key_names = [f"past_key_cache_{i}" for i in range(num_hidden_layers)]
    value_names = [f"past_value_cache_{i}" for i in range(num_hidden_layers)]
    input_names = base_inputs + key_names + value_names

    xh_model.export_cfg = ConfigDict(dict(
        input_names=input_names,
        output_names=["hidden_states"],  # 名字不重要，关键是输出 shape
    ))

    xh_model.set_input_sequence_length(full_seq_len)

    prefill_onnx_file = xhmodel_export_onnx(
        xh_model, tokenizer, data_batch,
        str(prefill_onnx_dir),
        f"hmquant_{args.model_name}_with_act",
        "cpu", dtype, logger, False
    )

    if args.gen_golden and not Path(prefill_golden_path).exists():
        session = HMONNXGoldenInference(prefill_onnx_file)
        session.to("cuda")
        session.save_golden = True
        session.golden_dir = str(prefill_onnx_dir / "hmonnx/golden")
        session.step = 0
        session(*calib_data)

    xh_model.release_exported_model()
    logger.info(f"save prefill onnx model to {prefill_onnx_file}")

def quant_asr(args):
    # quant_encode(args)
    if args.model != "Qwen3-ForcedAligner-0.6B":
        quant_model(args, "prefill")
        quant_model(args, "decode")
    else:
        quant_forcealigner(args)