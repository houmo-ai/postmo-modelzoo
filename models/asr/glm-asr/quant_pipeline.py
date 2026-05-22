# Copyright 2025 HOUMO AI
#
# File: quant_pipeline.py
# Description: Quantization Pipeline Module for GLM-ASR model
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
import time
import shutil
import tempfile
from pathlib import Path
from typing import List, Tuple

import onnx
import onnxsim
import torch
import torch.nn as nn

from xhquant.api import (
    DeviceType,
    HMONNXGoldenInference,
    QuantScheme,
    convert_onnx_to_hmonnx,
    create_quant_config,
    ptq_quantize,
    Config,
    PrecisionMode,
    get_root_logger,
)
from xhquant.utils.config import ConfigDict
from xhquant.patch.core import RewriterContext

from xh_model_zoo.xh_llm.models.builder import MODELS
from xh_model_zoo.xh_llm.models.eval_model_type import EvalModelType
from xh_model_zoo.xh_llm.models.glm_asr import (
    GlmAsrForConditionalGeneration,
    XHGlmAsrLLMModel,
)

GB = int(2**30)
_LARGE_MODEL_SIZE_THRESHOLD = int(2**30 * 1.8)
FILE_DIR = Path(__file__).resolve().parent
WORK_DIR_ROOT = FILE_DIR / "work_dirs"
INTERNAL_LLM_CONFIG_NAME = "glm_asr_decode_xh2a.py"


def build_llm_export_config(model_path: str) -> Config:
    model_path = os.path.normpath(model_path)
    frontend_type = "TorchFX"
    return Config(
        dict(
            target_device="XH2a",
            frontend_type=frontend_type,
            hf_model_dir=model_path,
            config_dir=model_path,
            quant_config=dict(),
            model=dict(
                type="XHGlmAsrLLMModel",
                hf_model=model_path,
                wrap_cfg=dict(
                    max_sequence_length=2048,
                    input_sequence_length=411,
                    use_cache=True,
                    num_logits_to_keep=1,
                    kv_cache=dict(cache_axis=2),
                ),
                quant_config=dict(),
                frontend_type=frontend_type,
                export_cfg=dict(
                    input_names=[
                        "inputs_embeds",
                        "past_seq_length",
                        "current_input_length",
                        "past_key_cache",
                        "past_value_cache",
                    ],
                    output_names=["last_hidden_state"],
                ),
            ),
        ),
        format_python_code=False,
    )


def build_runtime_model_config(model_dir: Path) -> dict:
    return dict(
        glm_asr=dict(
            type="XHGlmAsrHMONNXModel",
            model_dir=str(model_dir),
        )
    )


def write_final_export_metadata(
    work_dir: Path, final_hmquant_dir: Path, output_model_name: str
):
    meta_info_file = work_dir / "export_meta_info.json"
    if not meta_info_file.exists():
        return

    with open(meta_info_file, "r", encoding="utf-8") as f:
        meta_info = json.load(f)

    final_meta_info = dict(meta_info)
    final_meta_info["encoder"] = str(
        Path("encode") / f"hmquant_{output_model_name}_with_act.onnx"
    )
    final_meta_info["prefill_onnx_file"] = str(
        Path("prefill") / f"hmquant_{output_model_name}_with_act.onnx"
    )
    final_meta_info["decode_onnx_file"] = str(
        Path("decode") / f"hmquant_{output_model_name}_with_act.onnx"
    )
    final_meta_info["token_embedding_file"] = "quant_embedding.pt"
    final_meta_info["model"] = build_runtime_model_config(Path("."))

    with open(final_hmquant_dir / "export_meta_info.json", "w", encoding="utf-8") as f:
        json.dump(final_meta_info, f, indent=4)


def dump_final_config_snapshot(args, final_hmquant_dir: Path):
    cfg = build_llm_export_config(args.model)
    cfg.work_dir = str(final_hmquant_dir)
    cfg.device = "cuda" if torch.cuda.is_available() else "cpu"
    cfg.exec_device = "cuda" if torch.cuda.is_available() else "cpu"
    cfg.dtype = "float16"
    cfg.dump(final_hmquant_dir / INTERNAL_LLM_CONFIG_NAME)


class AudioEncoderWithProjector(nn.Module):
    """Wraps audio_tower + reshape + multi_modal_projector into a single exportable module.

    Input:  input_features (B, num_mel_bins, T)
    Output: audio_embeds   (B, T_out, text_hidden_size)
    """

    def __init__(self, audio_tower, multi_modal_projector, intermediate_size):
        super().__init__()
        self.audio_tower = audio_tower
        self.multi_modal_projector = multi_modal_projector
        self.intermediate_size = intermediate_size

    def forward(self, input_features):
        audio_outputs = self.audio_tower(input_features, return_dict=True)
        hidden_states = audio_outputs.last_hidden_state
        # Merge every 4 time-steps: (B, T, encoder_hidden) -> (B, T/4, intermediate_size)
        hidden_states = hidden_states.reshape(
            input_features.shape[0], -1, self.intermediate_size
        )
        audio_embeds = self.multi_modal_projector(hidden_states)
        return audio_embeds


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
    xh_model.to("cpu")  # Switch to CPU for model export
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    xh_model.convert_to_export_graph(data_batch)
    logger.info("Finish converting to export graph...")

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    xh_model.change_eval_type(EvalModelType.EXPORTED)

    xh_model.to("cpu")
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    logger.info("*************** Start exporting onnx ***************")
    onnx_file = xh_model.to_export_onnx(data_batch, onnx_output_dir, cfg_name)[0]
    return onnx_file


def get_work_dir(args, model_name, target_device):
    cfg_name = f"{model_name}_{target_device}"
    return WORK_DIR_ROOT / cfg_name


def log_move(src_path, dst_path):
    print(f"[MOVE] {src_path} -> {dst_path}")


def log_delete(path):
    print(f"[DELETE] {path}")


def ensure_clean_dir(dst_dir):
    if dst_dir.exists():
        log_delete(dst_dir)
        shutil.rmtree(dst_dir)
    dst_dir.parent.mkdir(exist_ok=True, parents=True)


def move_export_dir(src_dir, dst_dir):
    if not src_dir.exists():
        return False
    ensure_clean_dir(dst_dir)
    log_move(src_dir, dst_dir)
    shutil.move(str(src_dir), str(dst_dir))
    return True


def move_export_file(src_file, dst_file):
    if not src_file.exists():
        return False
    dst_file.parent.mkdir(exist_ok=True, parents=True)
    if dst_file.exists():
        log_delete(dst_file)
        dst_file.unlink()
    log_move(src_file, dst_file)
    shutil.move(str(src_file), str(dst_file))
    return True


def rename_onnx_file(dst_dir, new_name):
    onnx_files = list(dst_dir.glob("*.onnx"))
    if not onnx_files:
        return
    src_file = onnx_files[0]
    dst_file = dst_dir / new_name
    if src_file != dst_file:
        if dst_file.exists():
            log_delete(dst_file)
            dst_file.unlink()
        log_move(src_file, dst_file)
        src_file.rename(dst_file)


def process_encoder(args):
    target_device = "XH2a"
    model_dir = os.path.normpath(args.model)
    model_name = os.path.basename(model_dir)

    model = GlmAsrForConditionalGeneration.from_pretrained(model_dir)
    # processor = AutoProcessor.from_pretrained(model_dir)

    model.eval()
    model.audio_tower.eval()
    model.config.forced_decoder_ids = None
    model.config._attn_implementation = "eager"

    work_dir = get_work_dir(args, model_name, target_device)
    work_dir.mkdir(exist_ok=True, parents=True)

    # Get audio config
    head_dim = model.config.audio_config.head_dim
    num_heads = model.config.audio_config.num_attention_heads
    num_key_value_heads = model.config.audio_config.num_key_value_heads
    embed_dim = model.config.audio_config.hidden_size
    num_decode_layers = model.config.audio_config.num_hidden_layers
    max_source_positions = model.config.audio_config.max_position_embeddings

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

    # encoder processing =======================================================================================
    name = "Encoder"
    encoder_work_dir = work_dir / name
    encoder_work_dir.mkdir(exist_ok=True, parents=True)
    onnx_file = encoder_work_dir / f"{model_name}_{name}.onnx"
    quant_type = args.quant_type
    quant_scheme = QuantScheme(target_device=DeviceType.XH2a, quant_type=quant_type)
    quant_config = create_quant_config(quant_scheme)
    hmonnx_file = (
        encoder_work_dir / "hmonnx" / f"{model_name}_{name}_xh2a_{quant_type}.onnx"
    )
    golden_path = encoder_work_dir / "hmonnx/golden"
    meta_info["encoder"] = str(hmonnx_file.relative_to(work_dir))
    num_mel_bins = model.config.audio_config.num_mel_bins

    # Fixed at T=3000 seq_lens
    model = model.to(torch.float32)
    input_features = torch.randn(1, num_mel_bins, 3000).to(model.device).to(model.dtype)

    # Build combined module: audio_tower + reshape + multi_modal_projector
    intermediate_size = model.config.audio_config.intermediate_size
    encoder_with_proj = AudioEncoderWithProjector(
        model.audio_tower, model.multi_modal_projector, intermediate_size
    )
    encoder_with_proj.eval()

    # 1. Export ONNX
    if not Path(onnx_file).exists():
        with tempfile.TemporaryDirectory() as tmp_dir:
            with RewriterContext(None, backend="onnxruntime"):
                temp_onnx_file = str(Path(tmp_dir) / Path(onnx_file).name)
                torch.onnx.export(
                    encoder_with_proj,
                    input_features,
                    temp_onnx_file,
                    input_names=["input_features"],
                    output_names=["audio_embeds"],
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
        if not os.path.exists(onnx_file):
            onnx.save(
                onnx_model,
                onnx_file,
                save_as_external_data=True,
                all_tensors_to_one_file=True,
                location=f"{Path(onnx_file).stem}_external_data",
            )
        print(
            f"ONNX model saved: {onnx_file}, size: {onnx_model.ByteSize() / GB:.2f} GB"
        )
    else:
        print(f"File {onnx_file} already exists, skipping ONNX export.")

    # 2. Construct output names
    output_names = ["audio_embeds"]

    # 3. Convert to HMONNX
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

    # Generate golden data
    if args.gen_golden and not Path(golden_path).exists():
        session = HMONNXGoldenInference(hmonnx_file)
        session.to("cuda")
        session.save_golden = True
        session.golden_dir = str(encoder_work_dir / "hmonnx/golden")
        session.step = 0
        session(input_features.half().to("cuda"))


def process_prefill_decode(args):
    # ============================================================ Config and Initialization ============================================================
    DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    MODEL_PATH = os.path.normpath(args.model)

    hf_model = GlmAsrForConditionalGeneration.from_pretrained(
        MODEL_PATH,
        dtype=torch.float16,
        device_map=DEVICE,
    )
    hf_model.eval()

    model_name = os.path.basename(MODEL_PATH)
    target_device = "XH2a"  # Target quantization device

    cfg = build_llm_export_config(MODEL_PATH)
    cfg_name = f"{model_name}_{target_device}"

    cfg.work_dir = str(get_work_dir(args, model_name, target_device))
    Path(cfg.work_dir).mkdir(exist_ok=True, parents=True)
    cfg.device = "cuda" if torch.cuda.is_available() else "cpu"
    cfg.exec_device = "cuda" if torch.cuda.is_available() else "cpu"

    cfg.dtype = "float16"
    logger = get_root_logger()
    logger.info(f"\nConfig:\n{cfg.pretty_text}")
    config_file = Path(cfg.work_dir) / INTERNAL_LLM_CONFIG_NAME
    cfg.dump(config_file)

    device = torch.device(cfg.device)
    exec_device = torch.device(cfg.exec_device)
    dtype = getattr(torch, cfg.dtype)

    xh_model = MODELS.build(cfg.model)

    model = xh_model.get_hf_model()
    assert isinstance(
        xh_model, XHGlmAsrLLMModel
    ), f"Model must be XHGlmAsrLLMModel, but got {type(xh_model)}"

    # For GLM-ASR, the LLM is accessed via language_model, not thinker.model
    xh_model.init_wrap_model(hf_model.language_model)

    xh_model.wrap_model.lm_head = hf_model.language_model.lm_head
    xh_model.wrap_model.lm_head.to(device)
    xh_model.wrap_model.lm_head.to(dtype)

    processor = xh_model.get_processor()

    prefill_onnx_dir = Path(cfg.work_dir) / "Prefill"
    prefill_golden_path = prefill_onnx_dir / "hmonnx/golden"
    decode_onnx_dir = Path(cfg.work_dir) / "Decoder"
    decode_golden_path = decode_onnx_dir / "hmonnx/golden"
    prefill_onnx_dir.mkdir(exist_ok=True, parents=True)
    decode_onnx_dir.mkdir(exist_ok=True, parents=True)

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

    hf_config_dir = Path(cfg.work_dir) / "ConfigFiles"
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
        src_file = Path(hf_model_config_dir) / cfg_file
        if src_file.exists():
            shutil.copyfile(src_file, Path(hf_config_dir) / cfg_file)
    meta_info.hf_config = str(hf_config_dir.relative_to(cfg.work_dir))

    token_embedding = xh_model.token_embedding
    token_embedding_file = Path(cfg.work_dir) / "token_embedding.pt"
    torch.save(token_embedding.state_dict(), str(token_embedding_file))
    meta_info.token_embedding_file = str(token_embedding_file.relative_to(cfg.work_dir))

    # xh_model.past_key_caches is already wrapped as list in init_wrap_model
    if xh_model.past_key_caches is not None and len(xh_model.past_key_caches) > 0:
        meta_info.use_cache = True
        meta_info.kv_cache_shape = xh_model.past_key_caches[0].shape
        meta_info.num_hidden_layers = len(xh_model.past_key_caches)

    # ============================================================ Move model to device ============================================================
    # wrapped decoder
    xh_model.to(device)
    xh_model.to(dtype)
    xh_model.change_eval_type(eval_type=EvalModelType.WRAPED)
    # entire model - since wrap_llm_model modifies in-place, moving the original
    # HF language_model also moves the _wrap_model
    hf_model.language_model.to(device)

    hf_model.config.forced_decoder_ids = None
    hf_model.config._attn_implementation = "eager"
    text_config = hf_model.config.text_config

    # ============================================================ Audio Text Preprocessing and Feature Fusion ============================================================
    tokenizer = processor.tokenizer

    hidden_size = text_config.hidden_size

    # Create dummy input embeds (simulating fused audio + text features)
    final_inputs_embeds = torch.randn(
        (1, 411, hidden_size), device=device, dtype=torch.float16
    )
    print(f"final_inputs_embeds.shape: {final_inputs_embeds.shape}")

    # ============================================================ Construct Inputs ============================================================
    seq_len = final_inputs_embeds.shape[1]
    # Pad second dimension to 411
    if seq_len < 411:
        final_inputs_embeds = torch.cat(
            [
                final_inputs_embeds,
                torch.zeros(
                    (1, 411 - seq_len, final_inputs_embeds.shape[2]),
                    dtype=torch.float16,
                    device=device,
                ),
            ],
            dim=1,
        )

    data_batch = {"input_embeds": final_inputs_embeds.half(), "past_seq_length": [0]}

    with torch.no_grad():
        outs = xh_model.test_step(data_batch)

    # ============================================================ Quantization ============================================================
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

    # ============================================================ Prefill Export ============================================================
    xh_model = xh_model.to("cpu")
    full_seq_len = 411

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    # export_cfg: expand past_key_cache and past_value_cache inputs into multiple inputs for alignment
    num_hidden_layers = text_config.num_hidden_layers
    base_inputs = ["input_embeds", "past_seq_length", "current_input_length"]
    key_names = [f"past_key_cache_{i}" for i in range(num_hidden_layers)]
    value_names = [f"past_value_cache_{i}" for i in range(num_hidden_layers)]
    input_names = base_inputs + key_names + value_names
    xh_model.export_cfg = ConfigDict(
        dict(input_names=input_names, output_names=["last_hidden_state"])
    )

    xh_model.set_input_sequence_length(full_seq_len)

    prefill_onnx_file = xhmodel_export_onnx(
        xh_model,
        tokenizer,
        data_batch,
        str(prefill_onnx_dir),
        f"{cfg_name}_prefill",
        "cpu",
        dtype,
        logger,
        False,
    )

    if args.gen_golden and not Path(prefill_golden_path).exists():
        from xhquant.api import HMONNXGoldenInference

        session = HMONNXGoldenInference(prefill_onnx_file)
        session.to("cuda")
        session.save_golden = True
        session.golden_dir = str(prefill_onnx_dir / "hmonnx/golden")
        session.step = 0
        session(*calib_data)

    xh_model.release_exported_model()
    logger.info(f"save prefill onnx model to {prefill_onnx_file}")
    logger.info("*************** Finished exporting prefill model ***************")
    meta_info.prefill_onnx_file = str(Path(prefill_onnx_file).relative_to(cfg.work_dir))

    # ============================================================ Decode Export ============================================================
    xh_model.change_eval_type(EvalModelType.QUANTED_ALIGNED)
    xh_model.to(device)
    xh_model.to(dtype)

    data_batch["input_embeds"] = data_batch["input_embeds"].to(device)

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # Set input sequence length to 1 (Decode mode)
    xh_model.set_input_sequence_length(1)

    # Prepare Decode stage Embedding and data batch
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

    logger.info("*************** Start exporting decode model ***************")

    # Switch to CPU for export preparation
    xh_model = xh_model.to("cpu")

    decode_cpu_inputs_embeds = data_batch["input_embeds"].to("cpu")
    if decode_cpu_inputs_embeds.dim() == 2:
        decode_cpu_inputs_embeds = decode_cpu_inputs_embeds.unsqueeze(0)

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    # Execute export
    decode_onnx_file = xhmodel_export_onnx(
        xh_model,
        tokenizer,
        data_batch,
        str(decode_onnx_dir),
        f"{cfg_name}_decode",
        "cpu",
        dtype,
        logger,
        False,
    )

    meta_info.decode_onnx_file = str(Path(decode_onnx_file).relative_to(cfg.work_dir))
    json.dump(
        meta_info, open(Path(cfg.work_dir) / "export_meta_info.json", "w"), indent=4
    )

    decode_inputs = xh_model.prepare_inputs(data_batch)
    decode_calib_data = []
    for arg in decode_inputs:
        if isinstance(arg, (List, Tuple)):
            decode_calib_data.extend(arg)
        else:
            decode_calib_data.append(arg)

    if args.gen_golden and not Path(decode_golden_path).exists():
        from xhquant.api import HMONNXGoldenInference

        session = HMONNXGoldenInference(decode_onnx_file)
        session.to("cuda")
        session.save_golden = True
        session.golden_dir = str(decode_onnx_dir / "hmonnx/golden")
        session.step = 0
        session(*decode_calib_data)


def quant_and_export_llm(args):
    print("=================== Starting Encoder Export ===================")
    process_encoder(args)
    print("=================== Finished Encoder Export ===================")

    print("=================== Starting Prefill/Decode Export ===================")
    process_prefill_decode(args)
    print("=================== Finished Prefill/Decode Export ===================")
    print(
        "=================== Organizing and Renaming Exported Models ==================="
    )

    model_name = os.path.basename(os.path.normpath(args.model))
    target_device = "XH2a"
    work_dir = get_work_dir(args, model_name, target_device)
    final_hmquant_dir = Path(args.out_dir)
    final_hmquant_dir.mkdir(exist_ok=True, parents=True)

    encoder_src_dir = work_dir / "Encoder" / "hmonnx"
    encoder_dst_dir = final_hmquant_dir / "encode"
    if move_export_dir(encoder_src_dir, encoder_dst_dir):
        rename_onnx_file(encoder_dst_dir, f"hmquant_{args.model_name}_with_act.onnx")

    prefill_src_dir = work_dir / "Prefill"
    prefill_dst_dir = final_hmquant_dir / "prefill"
    if move_export_dir(prefill_src_dir, prefill_dst_dir):
        rename_onnx_file(prefill_dst_dir, f"hmquant_{args.model_name}_with_act.onnx")

    decoder_src_dir = work_dir / "Decoder"
    decoder_dst_dir = final_hmquant_dir / "decode"
    if move_export_dir(decoder_src_dir, decoder_dst_dir):
        rename_onnx_file(decoder_dst_dir, f"hmquant_{args.model_name}_with_act.onnx")

    embedding_src_file = work_dir / "token_embedding.pt"
    embedding_dst_file = final_hmquant_dir / "quant_embedding.pt"
    move_export_file(embedding_src_file, embedding_dst_file)

    write_final_export_metadata(work_dir, final_hmquant_dir, args.model_name)
    dump_final_config_snapshot(args, final_hmquant_dir)

    print("=================== Cleaning up intermediate files ===================")
    if work_dir.exists():
        log_delete(work_dir)
        shutil.rmtree(work_dir)

    if WORK_DIR_ROOT.exists():
        log_delete(WORK_DIR_ROOT)
        shutil.rmtree(WORK_DIR_ROOT, ignore_errors=True)

    print(f"[INFO] Final hmquant directory: {final_hmquant_dir}")
    print("=================== Finished Organizing Models ===================")
