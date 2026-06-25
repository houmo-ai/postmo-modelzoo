#!/usr/bin/python3
# -*- coding: utf-8 -*-
# Copyright 2026 HOUMO AI
#
# File: ptq.py
# Description:
#   Post-training quantization of the qwen3omni model.
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
import shutil
import os.path as osp
from pathlib import Path
import time
import psutil
import threading
import numpy as np

import torch
from transformers import (
    AutoConfig,
    AutoTokenizer,
    AutoProcessor,
    Qwen3OmniMoeForConditionalGeneration, 
    Qwen3OmniMoeProcessor)
import gc
from loguru import logger
import onnx
from typing import Any, cast
from tqdm import tqdm


script_dir = osp.dirname(osp.abspath(__file__))
DEFAULT_SAMPLE_DIR = f"{script_dir}/sample_data"


def _ensure_xh_model_zoo_on_path():
    """Called on ImportError to locate hmodel/xh2 when env.sh was not sourced."""
    import sys as _sys
    _root = os.environ.get("HOUMO_EXAMPLES_PATH")
    if not _root:
        _c = Path(script_dir).resolve()
        for _p in (_c, *_c.parents):
            if (_p / "env.sh").is_file() and (_p / "hmodel" / "xh2").is_dir():
                _root = str(_p)
                os.environ["HOUMO_EXAMPLES_PATH"] = _root
                break
    if not _root:
        raise RuntimeError(
            "HOUMO_EXAMPLES_PATH is not set and cannot be inferred "
            "from script location. Source env.sh before running ptq.py."
        )
    _xh2 = osp.join(_root, "hmodel", "xh2")
    if osp.isdir(_xh2) and _xh2 not in _sys.path:
        _sys.path.insert(0, _xh2)


from base_utils import parse_quant_types

try:
    from xh_model_zoo.xh_llm import LLMConverter
except ImportError:
    _ensure_xh_model_zoo_on_path()
    from xh_model_zoo.xh_llm import LLMConverter

from xhquant.api import (
    CacheTensor,
    DeviceType, 
    QuantScheme, 
    Config,
    ConfigDict,
    convert_fx_model_to_quanted_model,
    convert_quanted_model_to_hmonnx,
    xhquant_init, 
    create_quant_config,
    convert_onnx_to_hmonnx,
    convert_fx_model_to_hmonnx,
)
from xhquant.utils.onnxsim_large_model.simplify_large_onnx import simplify_large_onnx
from xh_model_zoo.utils.memory_tracker import MemoryTracker  # isort:skip
from xh_model_zoo.utils.time_profiler import TimeProfiler  # isort:skip
from xh_model_zoo.xh_llm.models.builder import wrap_llm_model
from xh_model_zoo.xh_llm.models.base_converter import BaseConverter

from hmatc.utils.utils import (
    first_not_none,
    get_model_configs,
    parse_context_length,
)

from _hmonnx_pipeline import (
    _patch_inputs_embeds_generation_device,
    _patch_runtime_device_property,
    _force_eager_moe_implementation,
    save_json,
)


HOUMO_DATASETS_PATH = os.getenv('HOUMO_DATASETS_PATH', '')
HOUMO_TARGET = os.getenv('HOUMO_TARGET', 'xh2')
DEFAULT_CONFIG_PATH = os.path.join(os.path.dirname(__file__), "config.yaml")
CONFIG_BRIDGE_ATTRS = ("bos_token_id", "eos_token_id", "pad_token_id", "vocab_size")
OPTIONAL_TEXT_ASSET_FILES = (
    "chat_template.json",
    "tokenizer_config.json",
    "special_tokens_map.json",
    "merges.txt",
    "vocab.json",
)
STATIC_TALKER_SEQ_LEN = 256


imodelzoo_models_path = os.getenv("IMODELZOO_MODELS_PATH")
MODEL_FOLDER = (
    os.path.join(imodelzoo_models_path, os.path.basename(script_dir))
    if imodelzoo_models_path
    else "."
)
print(MODEL_FOLDER)


def _find_examples_root_from_script(start_dir: str) -> str | None:
    current = Path(start_dir).resolve()
    for candidate in (current, *current.parents):
        if (candidate / "env.sh").is_file() and (candidate / "hmodel" / "gptqmodel").is_dir():
            return str(candidate)
    return None


def _resolve_gptqmodel_repo_path() -> str:
    examples_path = os.environ.get("HOUMO_EXAMPLES_PATH")
    candidates = []
    if examples_path:
        candidates.append(Path(examples_path).resolve())

    inferred_examples_path = _find_examples_root_from_script(script_dir)
    if inferred_examples_path:
        inferred_path = Path(inferred_examples_path).resolve()
        if inferred_path not in candidates:
            candidates.append(inferred_path)

    for candidate in candidates:
        gptq_path = candidate / "hmodel" / "gptqmodel"
        if gptq_path.is_dir() and any(gptq_path.iterdir()):
            if not examples_path:
                os.environ["HOUMO_EXAMPLES_PATH"] = str(candidate)
                logger.warning(
                    "HOUMO_EXAMPLES_PATH is not set; falling back to inferred repo root: {}",
                    candidate,
                )
            return str(gptq_path)

    searched = ", ".join(str(path) for path in candidates) or script_dir
    raise RuntimeError(
        "Unable to locate a valid in-repo gptqmodel directory. "
        f"Searched from: {searched}. Check HOUMO_EXAMPLES_PATH / env.sh."
    )


def resolve_accept_hidden_layer(model_dir: str):
    config_path = Path(model_dir) / "config.json"
    if not config_path.exists():
        return None

    try:
        config_payload = json.loads(config_path.read_text(encoding="utf-8"))
    except Exception:
        return None

    talker_config = config_payload.get("talker_config")
    if not isinstance(talker_config, dict):
        return None

    accept_hidden_layer = talker_config.get("accept_hidden_layer")
    if accept_hidden_layer is None:
        return None
    return int(accept_hidden_layer)

def get_jsonl_texts(path, nsamples, text_key="text"):
    samples = []
    with open(path, "r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            text = record.get(text_key)
            if not isinstance(text, str) or not text.strip():
                raise ValueError(f"Invalid `{text_key}` at {path}:{line_no}")
            samples.append(text)
            if len(samples) >= nsamples:
                break
    if len(samples) == 0:
        raise ValueError(f"No calibration samples found in {path}")
    return samples


def get_wikitext2(nsamples, seqlen, local_dir=None, tokenizer=None):
    if local_dir is None:
        from datasets import load_dataset

        traindata = load_dataset("wikitext", "wikitext-2-raw-v1", split="train").filter(
            lambda x: len(x["text"]) >= seqlen
        )
        return [example["text"] for example in traindata.select(range(nsamples))]
    else:
        from datasets import load_from_disk
        import random

        train_data = load_from_disk(local_dir)["train"]
        if tokenizer is None:
            ValueError("tokenizer must be provided when local_dir is specified")
        trainenc = tokenizer("\n\n".join(train_data["text"]))
        random.seed(0)
        train_samples = []
        input_ids = trainenc.input_ids
        data_seq_len = len(input_ids)
        nsamples = min(nsamples, data_seq_len // seqlen)
        for i in tqdm(range(nsamples)):
            start = i * seqlen
            end = start + seqlen
            inp = trainenc.input_ids[start:end]
            inp_mask = trainenc["attention_mask"][start:end]
            train_samples.append(
                {
                    "input_ids": inp,
                    "attention_mask": inp_mask,
                }
            )
        return train_samples

class ProcessMemoryMonitor:
    """
    Monitors the memory usage of the current Python process in real-time using psutil.
    """
    def __init__(self, interval=2, log_file=None):
        """
        Initializes the monitor.
        Args:
            interval (int): Time between measurements in seconds.
            log_file (str, optional): Path to a file to log results. If None, prints to console.
        """
        self.process = psutil.Process(os.getpid())
        self.interval = interval
        self.log_file = log_file
        self.is_monitoring = False
        self.peak_memory_mb = 0
        self.include_children = True

    def get_memory_info(self):
        """
        Gets current memory usage information.
        Returns:
            dict: A dictionary containing memory usage data.
        """
        rss = self.process.memory_info().rss
        if self.include_children:
            for child in self.process.children(recursive=True):
                try:
                    rss += child.memory_info().rss
                except psutil.NoSuchProcess:
                    continue
        rss_mb = rss / (1024 * 1024)  # Resident Set Size in MB
        percent = self.process.memory_percent()   # Percentage of system memory
        return {'rss_mb': rss_mb, 'percent': percent}

    def start(self):
        """Starts the monitoring loop in a separate daemon thread."""
        self.is_monitoring = True
        self.peak_memory_mb = 0
        self.monitor_thread = threading.Thread(target=self._monitor_loop)
        self.monitor_thread.daemon = True  # Thread will exit when main program does
        self.monitor_thread.start()
        print(f"Memory monitoring started (interval: {self.interval}s)")

    def _monitor_loop(self):
        """The internal loop that runs in the thread."""
        while self.is_monitoring:
            mem_info = self.get_memory_info()
            self.peak_memory_mb = max(self.peak_memory_mb, mem_info['rss_mb'])

            timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
            log_message = f"{timestamp} - RSS: {mem_info['rss_mb']:.2f} MB, System%: {mem_info['percent']:.2f}%"
            # Output to console or file
            if self.log_file:
                with open(self.log_file, 'a') as f:
                    f.write(log_message + '\n')

            time.sleep(self.interval)
        timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
        logger.info(f"{timestamp} - Max RSS: {self.peak_memory_mb:.2f} MB, System%: {self.process.memory_percent():.2f}%")

    def stop(self):
        """Stops the monitoring loop and prints peak usage."""
        self.is_monitoring = False
        if hasattr(self, 'monitor_thread'):
            self.monitor_thread.join(timeout=1) # Wait a moment for the thread to finish
        print(f"[Monitoring stopped. Peak RSS: {self.peak_memory_mb:.2f} MB]")

def check_gpu():
    import subprocess

    try:
        result = subprocess.run(
            "nvidia-smi --query-gpu=count --format=csv,noheader,nounits | wc -l",
            shell=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
            text=True
        )
        if result.returncode == 0 and int(result.stdout.strip()) > 0:
            return True
        return False
    except Exception as e:
        print(f"Not install GPU driver, error msg: {e}")
        return False

def str2bool(v):
    if isinstance(v, bool):
       return v
    if v.lower() in ('yes', 'true', 't', 'y', '1',""):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

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

def export_vision_model(sample_path, native_model, model_name, work_dir, quant_type, vision_size=448):
    from xh_model_zoo.xh_llm.models.qwen3_omni._vision_model import (
        register_wrap_modules as vision_register_wrap_modules,
    )
    from PIL import Image

    # Fixed square input resolution the vision encoder is exported at.
    # vision_size/patch16/merge2 tokens per side; e.g. 448 -> 14*14 = 196 vision
    # tokens (vs 49 at 224x224), preserving more visual detail at inference time.
    vision_size = int(vision_size)

    # Load real image for actual accuracy (instead of random data)
    def load_sample_image():
        sample_image_path = os.path.join(sample_path, "demo.jpg")
        if Path(sample_image_path).exists():
            logger.info(f"Loading sample image from {sample_image_path}")
            img = Image.open(sample_image_path).convert('RGB')
            img = img.resize((vision_size, vision_size), Image.Resampling.LANCZOS)
            img_tensor = torch.from_numpy(np.array(img)).float() / 255.0
            img_tensor = img_tensor.permute(2, 0, 1).unsqueeze(0)  # [1, 3, H, W]
            img_3d = img_tensor.unsqueeze(2).repeat(1, 1, 2, 1, 1)  # [1, 3, T, H, W]
            return img_3d.to(torch.float16)
        else:
            logger.warning(f"Sample image not found or not provided, falling back to random data")
            return torch.randn(1, 3, 2, vision_size, vision_size, dtype=torch.float16)

    visual = native_model.thinker.visual
    if visual is None:
        raise RuntimeError("Model does not have visual encoder.")
    
    vision_register_wrap_modules()
    visual = visual.to(torch.float16)
    vision_wrap_cfg = Config(
        dict(
            max_size_w=vision_size,
            max_size_h=vision_size,
            max_size_t=2,
            temporal_patch_size=2,
            patch_size=16,
            only_first_block=False,
        )
    )
    wrapped_visual = wrap_llm_model(visual, vision_wrap_cfg)
    dummy_pixels = load_sample_image()
    logger.info(f"Input pixels shape: {dummy_pixels.shape}")
    
    inputs = (dummy_pixels,)
    input_names = ["pixel_values"]
    output_names = ["vision_embeds", "deepstack_0", "deepstack_1", "deepstack_2"]

    vision_dir = work_dir / "hmonnx"
    vision_dir.mkdir(exist_ok=True, parents=True)
    vision_hmonnx_file = vision_dir / f"{model_name}-vision-{quant_type}.onnx"
    vision_tmp_onnx_file = work_dir / f"{model_name}_vision_tmp.onnx"

    logger.info(f"Exporting vision onnx to {vision_hmonnx_file}")
    compatible_names = BaseConverter.xh1_hmonnx_compatible(input_names)

    torch.onnx.export(
        wrapped_visual,
        dummy_pixels,
        vision_tmp_onnx_file,
        export_params=True,
        opset_version=18,
        do_constant_folding=True,
        input_names=compatible_names,
        output_names=output_names,
        verbose=False,
    )

    onnx_model = onnx.load(vision_tmp_onnx_file)
    onnx_model, check = simplify_large_onnx(onnx_model)
    if not check:
        logger.warning(f"Simplified ONNX model failed check: {check}")
    
    quant_scheme = QuantScheme(target_device=DeviceType.XH2a, quant_type=quant_type)
    quant_config = create_quant_config(quant_scheme)
    
    convert_onnx_to_hmonnx(
        onnx_model,
        inputs,
        device_type=DeviceType.XH2a,
        quant_config=quant_config,
        out_hmonnx_file=str(vision_hmonnx_file),
        input_names=compatible_names,
        output_names=output_names,
    )
    logger.info(f"Vision model export successful: {vision_hmonnx_file}")

    meta_info = {
        "create_time": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()),
        "module": "vision_encoder",
        "model_name": model_name,
        "vision_encoder_onnx": str(vision_hmonnx_file.relative_to(work_dir)),
        "vision_patch_size": 16,
        "vision_input_size": [vision_size, vision_size],
        "vision_channels": 3,
        "vision_temporal_patch_size": 2,
        "vision_max_size_t": 2,
        "export_method": "two_stage_onnx_simplify",  # Mark as using improved method
    }
    meta_file = work_dir / "meta_vision.json"
    save_json(meta_file, meta_info)
    logger.info(f"Vision export complete. Meta saved to {meta_file}")

def export_audio_model(native_model, model_name, work_dir, quant_type):
    from xh_model_zoo.xh_llm.models.qwen3_omni._audio_model import (
        register_wrap_modules as audio_register_wrap_modules,
    )

    audio_tower = native_model.thinker.audio_tower
    if audio_tower is None:
        raise RuntimeError("Model does not have audio tower.")
    
    audio_register_wrap_modules()

    audio_tower = audio_tower.to(torch.float16)
    wrapped_audio = wrap_llm_model(audio_tower, Config(dict()))

    mel_bins = int(getattr(audio_tower.config, "num_mel_bins", 128))
    dummy_features = torch.randn(1, mel_bins, 100, dtype=torch.float16)  # [1, mel_bins, mel_length]
    dummy_cu = torch.tensor([0, 13], dtype=torch.int32)  # [0, cnn_steps]

    inputs = (dummy_features, dummy_cu)
    input_names = ["padded_feature", "cu_seqlens"]
    output_names = ["audio_embeds"]

    audio_dir = work_dir / "hmonnx"
    audio_dir.mkdir(exist_ok=True, parents=True)
    audio_hmonnx_file = audio_dir / f"{model_name}-audio-{quant_type}.onnx"

    quant_scheme = QuantScheme(target_device=DeviceType.XH2a, quant_type=quant_type)
    quant_config = create_quant_config(quant_scheme)
    
    logger.info(f"Exporting audio model to {audio_hmonnx_file}")
    compatible_names = BaseConverter.xh1_hmonnx_compatible(input_names)
    convert_fx_model_to_hmonnx(
        wrapped_audio,
        inputs,
        device_type=DeviceType.XH2a,
        out_hmonnx_file=str(audio_hmonnx_file),
        quant_config=quant_config,
        input_names=compatible_names,
        output_names=output_names,
    )
    logger.info(f"Audio model export successful: {audio_hmonnx_file}")

    meta_info = {
        "create_time": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()),
        "module": "audio_encoder",
        "model_name": model_name,
        "audio_encoder_onnx": str(audio_hmonnx_file.relative_to(work_dir)),
        "audio_mel_dim": mel_bins,
        "audio_max_length": 100, #mel_length
        "audio_batch_size": 1,
    }
    meta_file = work_dir / "meta_audio.json"
    save_json(meta_file, meta_info)
    logger.info(f"Audio export complete. Meta saved to {meta_file}")

def export_talker_projection(current_model, model_name, sub_part, hidden_size, work_dir, quant_type):
    htp_dir = work_dir / "hmonnx"
    htp_dir.mkdir(exist_ok=True, parents=True)
    htp_hmonnx_file = htp_dir / f"{model_name}-{sub_part}-{quant_type}.onnx"

    dummy_input = torch.rand(1, STATIC_TALKER_SEQ_LEN, hidden_size, dtype=torch.float16)

    quant_scheme = QuantScheme(target_device=DeviceType.XH2a, quant_type=quant_type)
    quant_config = create_quant_config(quant_scheme)

    logger.info(f"Exporting  talker.{sub_part} ...")
    torch.onnx.export(
        current_model,
        (dummy_input,),
        str(work_dir / f"{model_name}_{sub_part}_tmp.onnx"),
        export_params=True,
        opset_version=18,
        do_constant_folding=True,
        input_names=["input"],
        output_names=["output"],
    )
    convert_onnx_to_hmonnx(
        str(work_dir / f"{model_name}_{sub_part}_tmp.onnx"),
        [dummy_input],
        device_type=DeviceType.XH2a,
        quant_config=quant_config,
        out_hmonnx_file=htp_hmonnx_file,
        input_names=["input"],
    )
    logger.info(f"{sub_part} export successful: {htp_hmonnx_file}")
    meta_info = {
        "create_time": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()),
        "module": f"talker.{sub_part}",
        "model_name": model_name,
        "hidden_projection_hmonnx": str(htp_hmonnx_file.relative_to(work_dir)),
        "hidden_size": hidden_size,
        "quant_type": quant_type,
    }
    meta_file = work_dir / f"meta_{sub_part}.json"
    save_json(meta_file, meta_info)
    logger.info(f"Projection export complete. Meta saved to {meta_file}")

def export_codec_lm_head(current_model, model_name, work_dir, quant_type):
    quant_scheme = QuantScheme(target_device=DeviceType.XH2a, quant_type=quant_type)
    quant_config = create_quant_config(quant_scheme)
    
    codec_dir = work_dir / "codec_embedding"
    codec_dir.mkdir(exist_ok=True, parents=True)
    for i, emb in enumerate(current_model.get_input_embeddings()):
        torch.save(emb.weight, codec_dir / f"token_embeding_{i}.pt")
    logger.info(f"Saved codec embeddings to {codec_dir}")

    lm_head_hmonnx_dir = work_dir / "hmonnx"
    lm_head_hmonnx_dir.mkdir(exist_ok=True, parents=True)
    hidden_size = current_model.config.hidden_size if hasattr(current_model, "config") else 1024
    
    for i, head in enumerate(current_model.lm_head):
        head_hmonnx_path = lm_head_hmonnx_dir / f"{model_name}-lm_head_{i}-{quant_type}.onnx"
        head_tmp_onnx_path = work_dir / f"{model_name}_lm_head_{i}.onnx"
        dummy_input = torch.rand(1, STATIC_TALKER_SEQ_LEN, hidden_size).half()
        torch.onnx.export(
            head,
            (dummy_input,),
            head_tmp_onnx_path,
            export_params=True,
            opset_version=18,
            do_constant_folding=True,
            input_names=[f"input_{i}"],
            output_names=[f"logits_{i}"],
        )
        convert_onnx_to_hmonnx(
            head_tmp_onnx_path,
            [dummy_input],
            device_type=DeviceType.XH2a,
            quant_config=quant_config,
            out_hmonnx_file=str(head_hmonnx_path),
            input_names=[f"input_{i}"],    
        )
        logger.info(f"Exported lm_head_{i} to {head_hmonnx_path}")
    logger.info("Export talker.code_predictor.lm_head successfully.")

    meta_info = {
        "create_time": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()),
        "module": "codec_lm_head",
        "model_name": model_name,
        "codec_embedding_dir": str(codec_dir.relative_to(work_dir)),
        "lm_head_dir": str(lm_head_hmonnx_dir.relative_to(work_dir)),
        "hidden_size": hidden_size,
    }
    meta_file = work_dir / "meta_codec_lm_head.json"
    save_json(meta_file, meta_info)
    logger.info(f"Predictor export complete. Meta saved to {meta_file}")

def export_talker_prediction(args, native_model, hf_model_path, model_name, work_dir, quant_type):
    from copy import deepcopy

    def _save_predictor_assets(native_model, asset_file: Path):
        codec_embeddings = [emb.weight.detach().cpu() for emb in native_model.talker.code_predictor.get_input_embeddings()]
        lm_head_weights = [head.weight.detach().cpu() for head in native_model.talker.code_predictor.lm_head]
        asset_payload = {
            "codec_embeddings": codec_embeddings,
            "lm_head_weights": lm_head_weights,
            "num_codec_embeddings": len(codec_embeddings),
            "num_lm_heads": len(lm_head_weights),
        }
        torch.save(asset_payload, asset_file)
        logger.info(f"Saved talker prediction asset bundle to {asset_file}")
        return asset_payload

    def _clone_capture_value(value):
        if isinstance(value, torch.Tensor):
            return value.detach().cpu().clone()
        return deepcopy(value)
    
    def _capture_predictor_inputs(args, native_model, processor, work_dir):
        """Run a full generate to capture code_predictor.model.forward inputs, or load from cache."""
        from qwen_omni_utils import process_mm_info
        import types

        device = next(native_model.parameters()).device
        dtype = next(native_model.parameters()).dtype

        capture_contract_version = 2

        class _PredictorInputsCaptured(RuntimeError):
            pass

        capture_path = work_dir / "talker_prediction_inputs.pth"
        if capture_path.exists():
            logger.info(f"Loading cached predictor inputs from {capture_path}")
            cached = torch.load(capture_path, map_location="cpu", weights_only=False)
            cached_entry = cached[0] if cached else None
            if (
                isinstance(cached_entry, dict)
                and int(cached_entry.get("capture_contract_version", 0)) >= capture_contract_version
            ):
                return cached
            logger.info("Cached predictor inputs use a stale capture contract, recapturing with current processor kwargs")

        logger.info("Running full generate to capture predictor inputs ...")
        image_path = str(Path(args.sample_path) / "cars.jpg")
        audio_path = str(Path(args.sample_path) / "cough.wav")
        conversation = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image_path},
                    {"type": "audio", "audio": audio_path},
                    {"type": "text", "text": "What can you see and hear? Please answer in four complete sentences, with enough detail to make the synthesized speech noticeably longer."},
                ],
            },
        ]
        text = processor.apply_chat_template(conversation, add_generation_prompt=True, tokenize=False)
        audios, images, videos = process_mm_info(conversation, use_audio_in_video=True)
        inputs = processor(
            text=text, 
            audio=audios, 
            images=images, 
            videos=videos or None,
            return_tensors="pt", 
            padding=True, 
            seconds_per_chunk=2.0,
            position_id_per_seconds=13,
            use_audio_in_video=True,
        )
        inputs = inputs.to(device).to(dtype)

        # Skip talker kwarg validation
        def _skip_validate(self, model_kwargs):
            return

        native_model.talker.code_predictor.model._validate_model_kwargs = types.MethodType(
            _skip_validate, native_model.talker.code_predictor.model
        )

        original_forward = native_model.talker.code_predictor.model.forward
        captured = []

        def forward_hook(*args, **kwargs):
            if not captured:
                save_kwargs = {k: _clone_capture_value(v) for k, v in kwargs.items()}
                save_kwargs["capture_contract_version"] = capture_contract_version
                captured.append(save_kwargs)
                raise _PredictorInputsCaptured()
            return original_forward(*args, **kwargs)

        native_model.talker.code_predictor.model.forward = forward_hook

        try:
            with torch.no_grad():
                native_model.generate(
                    **inputs,
                    speaker="Ethan",
                    thinker_return_dict_in_generate=True,
                    use_audio_in_video=True,
                )
        except _PredictorInputsCaptured:
            logger.info("Captured first predictor forward inputs, stopping generate early")
        finally:
            native_model.talker.code_predictor.model.forward = original_forward

        if not captured:
            raise RuntimeError("Failed to capture predictor inputs")

        torch.save(captured, capture_path)
        logger.info(f"Captured {len(captured)} predictor forward calls, saved to {capture_path}")
        return captured

    processor = Qwen3OmniMoeProcessor.from_pretrained(hf_model_path)
    asset_payload = _save_predictor_assets(native_model, work_dir / "talker_prediction_assets.pth")
    codec_embeddings_file = work_dir / "codec_embeddings.pt"
    codec_embeddings_payload = {
        "codec_embeddings": asset_payload["codec_embeddings"],
        "num_codec_embeddings": asset_payload["num_codec_embeddings"],
    }
    torch.save(codec_embeddings_payload, codec_embeddings_file)
    logger.info(f"Saved codec embeddings payload to {codec_embeddings_file}")
    
    captured = _capture_predictor_inputs(args, native_model, processor, work_dir)

    from xh_model_zoo.xh_llm.models.qwen3_omni._talker_prediction import (
        register_wrap_modules as pred_register_wrap_modules,
    )

    pred_register_wrap_modules()

    code_predictor_dir = work_dir / "hmonnx"
    code_predictor_prefill_hmonnx_file = code_predictor_dir / f"{model_name}-talker_prediction_prefill-{quant_type}.onnx"
    code_predictor_decode_hmonnx_file = code_predictor_dir / f"{model_name}-talker_prediction_decode-{quant_type}.onnx"

    quant_scheme = QuantScheme(target_device=DeviceType.XH2a, quant_type=quant_type)
    quant_config = ConfigDict(create_quant_config(quant_scheme))

    code_predictor = native_model.talker.code_predictor.to(torch.float16)

    context_length = args.context_length
    input_sequence_length = captured[0]["inputs_embeds"].shape[1]
    
    wrap_cfg = Config(
        dict(
            batch_size=1,
            max_sequence_length=context_length,
            input_sequence_length=input_sequence_length,
            use_cache=True,
            num_logits_to_keep=1,
            accept_hidden_layer=0,
            kv_cache=dict(cache_axis=2),
        )
    )

    wrapped_model = wrap_llm_model(code_predictor, wrap_cfg)

    num_hidden_layers = wrapped_model.model.config.num_hidden_layers
    head_dim = wrapped_model.model.layers[0].self_attn.head_dim
    num_key_value_heads = wrapped_model.model.config.num_key_value_heads

    kv_cache_shape = [1, num_key_value_heads, context_length, head_dim]
    past_key_caches = [CacheTensor(torch.zeros(kv_cache_shape, dtype=torch.float16)) for _ in range(num_hidden_layers)]
    past_value_caches = [CacheTensor(torch.zeros(kv_cache_shape, dtype=torch.float16)) for _ in range(num_hidden_layers)]

    inputs_embeds = captured[0]["inputs_embeds"].to(torch.float16)
    if inputs_embeds.shape[1] > input_sequence_length:
        inputs_embeds = inputs_embeds[:, :input_sequence_length, :]
    
    past_seq_length_t = torch.tensor([0], dtype=torch.int32)
    curren_input_length_t = torch.tensor([int(inputs_embeds.shape[1])], dtype=torch.int32)

    num_lm_heads = int(asset_payload["num_lm_heads"])
    batch = int(inputs_embeds.shape[0])
    prefill_seq = int(inputs_embeds.shape[1])
    prefill_step = max(0, min(prefill_seq - 2, num_lm_heads - 1))  # Ensure at least 2 tokens for decode and not exceed lm_head count
    head_mask_prefill = torch.zeros(batch, prefill_seq, num_lm_heads, 1, dtype=torch.float16)
    head_mask_prefill[:, :, prefill_step, 0] = 1.0
    head_mask_decode = torch.zeros(batch, 1, num_lm_heads, 1, dtype=torch.float16)
    head_mask_decode[0, 0, 0, 0] = 1.0

    prefill_inputs = (inputs_embeds, head_mask_prefill, past_seq_length_t, curren_input_length_t, past_key_caches, past_value_caches)

    input_names = ["inputs_embeds", "head_mask", "past_seq_length", "current_input_length"]
    for i in range(num_hidden_layers):
        input_names.append(f"past_key_cache_{i}")
    for i in range(num_hidden_layers):
        input_names.append(f"past_value_cache_{i}")
    output_names = ["logits", "hidden_states"]

    logger.info(f"Exporting talker.code_predictor prefill to {code_predictor_prefill_hmonnx_file}")

    quanted_model = convert_fx_model_to_quanted_model(wrapped_model, prefill_inputs, DeviceType.XH2a, quant_config,)
    compatible_names = BaseConverter.xh1_hmonnx_compatible(input_names)
    convert_quanted_model_to_hmonnx(quanted_model, prefill_inputs, str(code_predictor_prefill_hmonnx_file), compatible_names, output_names,)
    logger.info(f"Talker prediction prefill export successful: {code_predictor_prefill_hmonnx_file}")

    decode_inputs = (
        inputs_embeds[:, -1:, :],
        head_mask_decode,
        past_seq_length_t,
        torch.ones_like(curren_input_length_t),
        past_key_caches,
        past_value_caches,
    )
    wrap_cfg.input_sequence_length = 1
    quanted_model.update_cfg(wrap_cfg)

    logger.info(f"Exporting talker.code_predictor decode to {code_predictor_decode_hmonnx_file}")
    compatible_names = BaseConverter.xh1_hmonnx_compatible(input_names)
    convert_quanted_model_to_hmonnx(quanted_model, decode_inputs, str(code_predictor_decode_hmonnx_file), compatible_names, output_names,)
    logger.info(f"Talker prediction decode export successful: {code_predictor_decode_hmonnx_file}")

    meta_info = {
        "create_time": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()),
        "module": "talker_prediction",
        "model_name": model_name,
        "talker_prediction_prefill_onnx": str(code_predictor_prefill_hmonnx_file.relative_to(work_dir)),
        "talker_prediction_decode_onnx": str(code_predictor_decode_hmonnx_file.relative_to(work_dir)),
        "output_names": output_names,
        "hidden_states_output_contract": "predictor_input_embeds_for_talker_residual_sum",
        "codec_embedding_count": int(asset_payload["num_codec_embeddings"]),
        "lm_head_count": num_lm_heads,
        "talker_prediction_kv_cache": {"shape": kv_cache_shape, "num_decoder_layers": num_hidden_layers},
        "talker_prediction_hidden_size": int(inputs_embeds.shape[-1]),
        "talker_prediction_input_sequence_length": int(input_sequence_length),
    }
    meta_file = work_dir / "meta_talker_prediction.json"
    save_json(meta_file, meta_info)
    logger.info(f"Predictor export complete. Meta saved to {meta_file}")

def export_talker_model(args, native_model, hf_model_path, work_dir, model_name, quant_type):
    from copy import deepcopy

    def _pad_prefill_tensor(tensor: torch.Tensor, target_seq_len: int, fill_value: float = 0.0) -> torch.Tensor:
        current_seq_len = int(tensor.shape[1])
        if current_seq_len >= target_seq_len:
            return tensor[:, :target_seq_len, ...]

        pad_shape = (tensor.shape[0], target_seq_len - current_seq_len, *tensor.shape[2:])
        pad = torch.full(pad_shape, fill_value, dtype=tensor.dtype)
        return torch.cat([tensor, pad], dim=1) 
    
    def _build_prefill_fused_inputs(captured_entry, inputs_embeds: torch.Tensor, meta_info):
        seq_len = STATIC_TALKER_SEQ_LEN # int(inputs_embeds.shape[1])
        batch = int(inputs_embeds.shape[0])
        hidden_state_size = int(
            meta_info.get(
                "talker_hidden_state_size",
                meta_info.get("talker_thinker_hidden_size", inputs_embeds.shape[-1]),
            )
        )

        if captured_entry is not None:
            hidden_state = captured_entry.get("hidden_state")
            role_mask = captured_entry.get("role_mask")
            bypass_embeds = captured_entry.get("bypass_embeds")
            bypass_mask = captured_entry.get("bypass_mask")
            if all(isinstance(item, torch.Tensor) for item in (hidden_state, role_mask, bypass_embeds, bypass_mask)):
                hidden_state = hidden_state.to(torch.float16).cpu()
                role_mask = role_mask.to(torch.float16).cpu()
                bypass_embeds = bypass_embeds.to(torch.float16).cpu()
                bypass_mask = bypass_mask.to(torch.float16).cpu()
                return (
                    _pad_prefill_tensor(hidden_state, seq_len, fill_value=0.0),
                    _pad_prefill_tensor(role_mask, seq_len, fill_value=0.0),
                    _pad_prefill_tensor(bypass_embeds, seq_len, fill_value=0.0),
                    _pad_prefill_tensor(bypass_mask, seq_len, fill_value=0.0),
                )

        return (
            torch.zeros(batch, seq_len, hidden_state_size, dtype=torch.float16),
            torch.zeros(batch, seq_len, 1, dtype=torch.float16),
            inputs_embeds,
            torch.ones(batch, seq_len, 1, dtype=torch.float16),
        )

    def _clone_capture_value(value):
        if isinstance(value, torch.Tensor):
            return value.detach().cpu().clone()
        return deepcopy(value)
    
    def _compute_talker_prefill_static_length(captured_input_sequence_length: int, context_length: int) -> int:
        captured_input_sequence_length = int(captured_input_sequence_length)
        context_length = int(context_length)
        # Add some buffer to captured length, but still keep it within context length
        return min(context_length, captured_input_sequence_length + 32)
    
    def _capture_talker_inputs(args, native_model, processor, device, dtype, work_dir):
        """Run a full generate to capture talker.forward inputs, or load from cache."""
        from qwen_omni_utils import process_mm_info
        import types

        capture_contract_version = 3

        class _TalkerInputsCaptured(RuntimeError):
            pass

        capture_path = work_dir / "talker_model_inputs.pth"
        if capture_path.exists():
            logger.info(f"Loading cached talker inputs from {capture_path}")
            cached = torch.load(capture_path, map_location="cpu", weights_only=False)
            cached_entry = cached[0] if cached else None
            if isinstance(cached_entry, dict) and int(cached_entry.get("capture_contract_version", 0)) >= capture_contract_version:
                return cached
            logger.info("Cached talker inputs use a stale capture contract, recapturing with current processor kwargs")

        logger.info("Running full generate to capture talker inputs ...")
        image_path = str(Path(args.sample_path) / "cars.jpg")
        audio_path = str(Path(args.sample_path) / "cough.wav")

        conversation = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image_path},
                    {"type": "audio", "audio": audio_path},
                    {"type": "text", "text": "What can you see and hear? Please answer in four complete sentences, with enough detail to make the synthesized speech noticeably longer."},
                ],
            },
        ]
        text = processor.apply_chat_template(conversation, add_generation_prompt=True, tokenize=False)
        audios, images, videos = process_mm_info(conversation, use_audio_in_video=True)
        inputs = processor(
            text=text,
            audio=audios,
            images=images or None,
            videos=videos,
            return_tensors="pt",
            padding=True,
            seconds_per_chunk=2.0,
            position_id_per_seconds=13,
            use_audio_in_video=True,
        )
        inputs = inputs.to(device).to(dtype)

        # Monkey-patch talker._validate_model_kwargs to skip custom kwarg validation
        def _skip_validate(self, model_kwargs):
            return

        native_model.talker._validate_model_kwargs = types.MethodType(_skip_validate, native_model.talker)

        capture_ctx = {"generate_kwargs": {}, "segments": []}

        original_generate = native_model.talker.generate
        original_get_user_parts = native_model._get_talker_user_parts
        original_get_assistant_parts = native_model._get_talker_assistant_parts

        def generate_hook(self, *args, **kwargs):
            if not capture_ctx["generate_kwargs"]:
                for key in ("inputs_embeds", "trailing_text_hidden", "tts_pad_embed", "talker_input_ids"):
                    if key in kwargs:
                        capture_ctx["generate_kwargs"][key] = _clone_capture_value(kwargs[key])
            return original_generate(*args, **kwargs)

        def user_parts_hook(self, im_start_index, segment_end_index, multimodal_mask, thinker_hidden, thinker_embed):
            user_talker_part = original_get_user_parts(
                im_start_index,
                segment_end_index,
                multimodal_mask,
                thinker_hidden,
                thinker_embed,
            )
            user_mm_mask = multimodal_mask[:, im_start_index:segment_end_index]
            user_source = thinker_embed[:, im_start_index:segment_end_index].clone()
            if user_mm_mask.any():
                user_source[user_mm_mask] = thinker_hidden[:, im_start_index:segment_end_index][user_mm_mask]
            capture_ctx["segments"].append(
                {
                    "hidden_state": _clone_capture_value(user_source),
                    "role_mask": _clone_capture_value((~user_mm_mask).unsqueeze(-1).to(user_talker_part.dtype)),
                    "bypass_embeds": _clone_capture_value(torch.zeros_like(user_talker_part)),
                    "bypass_mask": _clone_capture_value(
                        torch.zeros(*user_talker_part.shape[:2], 1, dtype=user_talker_part.dtype, device=user_talker_part.device)
                    ),
                }
            )
            return user_talker_part

        def assistant_parts_hook(
            self,
            im_start_index,
            segment_end_index,
            speaker_id,
            thinker_embed,
            tts_pad_embed,
            tts_bos_embed,
            tts_eos_embed,
        ):
            input_embeds, input_ids, trailing_text_hidden = original_get_assistant_parts(
                im_start_index,
                segment_end_index,
                speaker_id,
                thinker_embed,
                tts_pad_embed,
                tts_bos_embed,
                tts_eos_embed,
            )
            assistant_source = torch.zeros(
                input_embeds.shape[0],
                input_embeds.shape[1],
                thinker_embed.shape[-1],
                dtype=thinker_embed.dtype,
                device=thinker_embed.device,
            )
            assistant_role_mask = torch.ones(
                input_embeds.shape[0], input_embeds.shape[1], 1, dtype=input_embeds.dtype, device=input_embeds.device
            )
            assistant_bypass_embeds = input_embeds.clone()
            assistant_bypass_mask = torch.ones(
                input_embeds.shape[0], input_embeds.shape[1], 1, dtype=input_embeds.dtype, device=input_embeds.device
            )

            projected_prefix = min(3, max(segment_end_index - im_start_index, 0))
            if projected_prefix > 0:
                assistant_source[:, :projected_prefix, :] = thinker_embed[
                    :, im_start_index : im_start_index + projected_prefix, :
                ]
                assistant_bypass_embeds[:, :projected_prefix, :] = 0
                assistant_bypass_mask[:, :projected_prefix, :] = 0

            capture_ctx["segments"].append(
                {
                    "hidden_state": _clone_capture_value(assistant_source),
                    "role_mask": _clone_capture_value(assistant_role_mask),
                    "bypass_embeds": _clone_capture_value(assistant_bypass_embeds),
                    "bypass_mask": _clone_capture_value(assistant_bypass_mask),
                    "trailing_text_hidden": _clone_capture_value(trailing_text_hidden),
                    "assistant_input_ids": _clone_capture_value(input_ids),
                }
            )
            return input_embeds, input_ids, trailing_text_hidden

        native_model.talker.generate = types.MethodType(generate_hook, native_model.talker)
        native_model._get_talker_user_parts = types.MethodType(user_parts_hook, native_model)
        native_model._get_talker_assistant_parts = types.MethodType(assistant_parts_hook, native_model)

        # Hook talker.forward to capture inputs
        original_forward = native_model.talker.forward
        captured = []

        def forward_hook(*args, **kwargs):
            # Construct attention_mask if not provided (talker.generate doesn't pass it)
            if kwargs.get("attention_mask") is None and "inputs_embeds" in kwargs:
                ie = kwargs["inputs_embeds"]
                kwargs["attention_mask"] = torch.ones(ie.shape[:2], dtype=torch.long, device=ie.device)
            # Only capture the first (prefill) call; skip large past_key_values
            if not captured:
                save_kwargs = {}
                for k, v in kwargs.items():
                    if k in ("past_key_values", "cache_position"):
                        continue
                    save_kwargs[k] = _clone_capture_value(v)

                for key, value in capture_ctx["generate_kwargs"].items():
                    save_kwargs.setdefault(key, value)

                if capture_ctx["segments"]:
                    hidden_state = torch.cat([segment["hidden_state"] for segment in capture_ctx["segments"]], dim=1)
                    role_mask = torch.cat([segment["role_mask"] for segment in capture_ctx["segments"]], dim=1)
                    bypass_embeds = torch.cat([segment["bypass_embeds"] for segment in capture_ctx["segments"]], dim=1)
                    bypass_mask = torch.cat([segment["bypass_mask"] for segment in capture_ctx["segments"]], dim=1)
                    if int(hidden_state.shape[1]) == int(save_kwargs["inputs_embeds"].shape[1]):
                        save_kwargs["hidden_state"] = hidden_state
                        save_kwargs["role_mask"] = role_mask
                        save_kwargs["bypass_embeds"] = bypass_embeds
                        save_kwargs["bypass_mask"] = bypass_mask

                    trailing_text_hidden = next(
                        (
                            segment.get("trailing_text_hidden")
                            for segment in capture_ctx["segments"]
                            if isinstance(segment.get("trailing_text_hidden"), torch.Tensor)
                        ),
                        None,
                    )
                    if trailing_text_hidden is not None:
                        save_kwargs.setdefault("trailing_text_hidden", trailing_text_hidden)

                save_kwargs["capture_contract_version"] = capture_contract_version
                captured.append(save_kwargs)
                raise _TalkerInputsCaptured()
            return original_forward(*args, **kwargs)

        native_model.talker.forward = forward_hook

        try:
            with torch.no_grad():
                native_model.generate(
                    **inputs,
                    speaker="Ethan",
                    thinker_return_dict_in_generate=True,
                    use_audio_in_video=True,
                )
        except _TalkerInputsCaptured:
            logger.info("Captured first talker forward inputs, stopping generate early")
        finally:
            native_model.talker.forward = original_forward
            native_model.talker.generate = original_generate
            native_model._get_talker_user_parts = original_get_user_parts
            native_model._get_talker_assistant_parts = original_get_assistant_parts

        if not captured:
            raise RuntimeError("Failed to capture talker inputs — generate produced no talker calls")

        torch.save(captured, capture_path)
        logger.info(f"Captured {len(captured)} talker forward calls, saved to {capture_path}")
        return captured
    _force_eager_moe_implementation(native_model, logger)
    native_model.eval()
    _patch_inputs_embeds_generation_device(native_model.talker, "talker", logger)
    _patch_inputs_embeds_generation_device(native_model.talker.hidden_projection, "talker.code_predictor", logger)
    if hasattr(native_model, "code2wav"):
        _patch_runtime_device_property(native_model.code2wav, "code2wav", logger)

    processor = Qwen3OmniMoeProcessor.from_pretrained(hf_model_path)

    device = next(native_model.parameters()).device
    dtype = next(native_model.parameters()).dtype

    captured = _capture_talker_inputs(args, native_model, processor, device, dtype, work_dir)
    captured_entry = captured[0]

    from xh_model_zoo.xh_llm.models.qwen3_omni._talker_model import (
        register_wrap_modules as talker_register_wrap_modules,
    )

    talker_register_wrap_modules()

    talker = native_model.talker.to(torch.float16)

    context_length = args.context_length
    actual_input_sequence_length = int(captured[0]["inputs_embeds"].shape[1])
    # Pin the talker prefill static length to STATIC_TALKER_SEQ_LEN so the whole
    # talker family (except talker prediction) shares one seq dim. This must match
    # the fused-input seq_len in _build_prefill_fused_inputs — it bakes into the
    # in-graph slice / llm_gather / cos-sin DynamicSlice via wrap_cfg.input_sequence_length.
    input_sequence_length = STATIC_TALKER_SEQ_LEN
    if actual_input_sequence_length > input_sequence_length:
        raise ValueError(
            f"Captured talker prefill length {actual_input_sequence_length} exceeds "
            f"STATIC_TALKER_SEQ_LEN={STATIC_TALKER_SEQ_LEN}; increase the constant."
        )

    talker_embedding = talker.model.get_input_embeddings()
    
    projection_in_features = int(talker.hidden_projection.linear_fc1.in_features)

    wrap_cfg = Config(
        dict(
            batch_size=1,
            max_sequence_length=context_length,
            input_sequence_length=int(input_sequence_length),
            use_cache=True,
            num_logits_to_keep=1,
            accept_hidden_layer=0,
            kv_cache=dict(cache_axis=2),
        )
    )

    wrapped_talker = wrap_llm_model(talker, wrap_cfg)

    num_hidden_layers = wrapped_talker.model.config.num_hidden_layers
    head_dim = wrapped_talker.model.layers[0].self_attn.head_dim
    num_key_value_heads = wrapped_talker.model.config.num_key_value_heads

    kv_cache_shape = [1, num_key_value_heads, context_length, head_dim]
    past_key_caches = [CacheTensor(torch.zeros(kv_cache_shape, dtype=torch.float16)) for _ in range(num_hidden_layers)]
    past_value_caches = [CacheTensor(torch.zeros(kv_cache_shape, dtype=torch.float16)) for _ in range(num_hidden_layers)]

    inputs_embeds = captured[0]["inputs_embeds"].to(torch.float16)
    if inputs_embeds.shape[1] > actual_input_sequence_length:
            inputs_embeds = inputs_embeds[:, :actual_input_sequence_length, :]
    inputs_embeds = _pad_prefill_tensor(inputs_embeds, input_sequence_length, fill_value=0.0)

    past_seq_length_t = torch.tensor([0], dtype=torch.int32)
    current_input_length_t = torch.tensor([actual_input_sequence_length], dtype=torch.int32)

    source_batch = int(inputs_embeds.shape[0])
    thinker_hs = int(talker.hidden_projection.linear_fc1.in_features)
    
    source_prefill, role_mask_prefill, bypass_embeds_prefill, bypass_mask_prefill = _build_prefill_fused_inputs(
            captured_entry,
            inputs_embeds,
            {
                "talker_hidden_state_size": thinker_hs,
                "talker_thinker_hidden_size": thinker_hs,
            },
        )

    prefill_inputs = (
        source_prefill,
        role_mask_prefill,
        bypass_embeds_prefill,
        bypass_mask_prefill,
        past_seq_length_t, 
        current_input_length_t, 
        past_key_caches, 
        past_value_caches,
    )
    input_names = ["hidden_state", "role_mask", "bypass_embeds", "bypass_mask", "past_seq_length", "current_input_length"]
    for i in range(num_hidden_layers):
        input_names.append(f"past_key_cache_{i}")
    for i in range(num_hidden_layers):
        input_names.append(f"past_value_cache_{i}")
    output_names = ["logits", "hidden_states"]

    quant_scheme = QuantScheme(
        target_device=DeviceType.XH2a,
        quant_type=quant_type,
        nodes={
            "hidden_projection_linear_fc1": args.projection_quant_type,
            "hidden_projection_linear_fc2": args.projection_quant_type,
            "text_projection_linear_fc1": args.projection_quant_type,
            "text_projection_linear_fc2": args.projection_quant_type,
        },
    )
    quant_config = ConfigDict(create_quant_config(quant_scheme))

    from xhquant.api.quant_type import parse_quant_format
    for _node_name, _node_quant_type in quant_scheme.nodes.items():
        _fmt = parse_quant_format(_node_quant_type)
        quant_config["nodes_cfg"][_node_name] = ConfigDict(dict(
            w_cfg=dict(
                quantizer=dict(
                    qspec=dict(
                        fp_mode=_fmt.fp_mode,
                        hidden_bit=_fmt.hidden_bit,
                        man_bit=_fmt.weight_bit,
                    ),
                ),
            ),
            i_cfg=dict(
                quantizer=dict(
                    qspec=dict(
                        hidden_bit=True,
                        man_bit=_fmt.act_bit,
                    ),
                ),
            ),
        ))

    talker_hmonnx_dir = work_dir / "hmonnx"
    talker_hmonnx_dir.mkdir(exist_ok=True, parents=True)

    talker_prefill_hmonnx_file = talker_hmonnx_dir / f"{model_name}-talker_prefill-{quant_type}.onnx"
    talker_decode_hmonnx_file = talker_hmonnx_dir / f"{model_name}-talker_decode-{quant_type}.onnx"
    
    logger.info(f"Exporting talker model prefill to {talker_prefill_hmonnx_file}")
    quanted_talker = convert_fx_model_to_quanted_model(wrapped_talker, prefill_inputs, device_type=DeviceType.XH2a, quant_config=quant_config)
    compatible_names = BaseConverter.xh1_hmonnx_compatible(input_names)
    convert_quanted_model_to_hmonnx(quanted_talker, prefill_inputs, str(talker_prefill_hmonnx_file), compatible_names, output_names,)
    logger.info(f"Talker model prefill export successful: {talker_prefill_hmonnx_file}")

    source_decode = torch.zeros(source_batch, 1, thinker_hs, dtype=torch.float16)
    role_mask_decode = torch.zeros(source_batch, 1, 1, dtype=torch.float16)
    bypass_embeds_decode = inputs_embeds[:, -1:, :]
    bypass_mask_decode = torch.ones(source_batch, 1, 1, dtype=torch.float16)

    decode_inputs = (
        source_decode,
        role_mask_decode,
        bypass_embeds_decode,
        bypass_mask_decode,
        past_seq_length_t,
        torch.ones_like(current_input_length_t),
        past_key_caches,
        past_value_caches,
    )
    wrap_cfg.input_sequence_length = 1
    quanted_talker.update_cfg(wrap_cfg)

    logger.info(f"Exporting talker model decode to {talker_decode_hmonnx_file}")
    compatible_names = BaseConverter.xh1_hmonnx_compatible(input_names)
    convert_quanted_model_to_hmonnx(quanted_talker, decode_inputs, str(talker_decode_hmonnx_file), compatible_names, output_names,)
    logger.info(f"Talker model decode export successful: {talker_decode_hmonnx_file}")

    embed_file = work_dir / "talker_embedding.pt"
    torch.save(talker_embedding.weight, embed_file)
    logger.info(f"Talker embedding saved to {embed_file}")

    meta_info = {
        "create_time": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()),
        "module": "talker_model",
        "model_name": model_name,
        "talker_prefill_onnx": str(talker_prefill_hmonnx_file.relative_to(work_dir)),
        "talker_decode_onnx": str(talker_decode_hmonnx_file.relative_to(work_dir)),
        "output_names": output_names,
        "quant_type": quant_type,
        "projection_quant_type": args.projection_quant_type,
        "talker_projection_in_features": int(projection_in_features),
        "talker_embedding_file": str(embed_file.relative_to(work_dir)),
        "talker_kv_cache": {"shape": kv_cache_shape, "num_decoder_layers": num_hidden_layers},
        "talker_hidden_size": int(inputs_embeds.shape[-1]),
        "talker_thinker_hidden_size": int(thinker_hs),
        "talker_input_sequence_length": int(input_sequence_length),
        "talker_prefill_guidance_inputs": ["hidden_state", "role_mask", "bypass_embeds", "bypass_mask"],
        "fused_projection": True,
    }
    trailing_text_hidden = captured_entry.get("trailing_text_hidden") if captured_entry is not None else None
    if isinstance(trailing_text_hidden, torch.Tensor):
        meta_info["talker_style_guidance_hidden_size"] = int(trailing_text_hidden.shape[-1])
    meta_file = work_dir / "meta_talker_model.json"
    save_json(meta_file, meta_info)
    logger.info(f"Talker export complete. Meta saved to {meta_file}")

def export_code2wav(native_model, model_name, work_dir, quant_type):
    def _dedupe_onnx_node_outputs(onnx_path: Path):
        model = onnx.load(str(onnx_path))
        producers = {}
        for node_idx, node in enumerate(model.graph.node):
            for output_idx, output_name in enumerate(node.output):
                if not output_name:
                    continue
                producers.setdefault(output_name, []).append((node_idx, output_idx))

        for tensor_name, occurrences in producers.items():
            if len(occurrences) <= 1:
                continue

            for dup_idx, (producer_idx, output_idx) in enumerate(occurrences[:-1]):
                next_producer_idx = occurrences[dup_idx + 1][0]
                new_name = f"{tensor_name}__dedup_{dup_idx}"
                model.graph.node[producer_idx].output[output_idx] = new_name

                for node in model.graph.node[producer_idx + 1 : next_producer_idx]:
                    for input_idx, input_name in enumerate(node.input):
                        if input_name == tensor_name:
                            node.input[input_idx] = new_name

                for value_info in model.graph.value_info:
                    if value_info.name == tensor_name:
                        value_info.name = new_name

        onnx.save(model, str(onnx_path))

    dummy_codes = torch.randint(100, (1, 16, STATIC_TALKER_SEQ_LEN), dtype=torch.int32)
    code2wav_onnx_file = work_dir / f"{model_name}-code2wav.onnx"
    code2wav_dir = work_dir / "hmonnx"
    code2wav_dir.mkdir(exist_ok=True, parents=True)
    code2wav_hmonnx_file = code2wav_dir / f"{model_name}-code2wav-{quant_type}.onnx"

    quant_scheme = QuantScheme(target_device=DeviceType.XH2a, quant_type=quant_type)
    quant_config = create_quant_config(quant_scheme)

    logger.info(f"Exporting code2wav with static_code_len = {STATIC_TALKER_SEQ_LEN} ...")
    torch.onnx.export(
        native_model.code2wav,
        dummy_codes,
        code2wav_onnx_file,
        opset_version=18,
        dynamo=True,
        do_constant_folding=True,
        input_names=["codes"],
        output_names=["embedding"]
    )
    _dedupe_onnx_node_outputs(code2wav_onnx_file)
    convert_onnx_to_hmonnx(
        code2wav_onnx_file,
        [dummy_codes],
        device_type=DeviceType.XH2a,
        quant_config=quant_config,
        out_hmonnx_file=str(code2wav_hmonnx_file),
        input_names=["codes"],
    )
    logger.info(f"code2wav export successful: {code2wav_hmonnx_file}")

    meta_info = {
        "create_time": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()),
        "module": "code2wav",
        "model_name": model_name,
        "code2wav_hmonnx": str(code2wav_hmonnx_file.relative_to(work_dir)),
        "static_code_len": STATIC_TALKER_SEQ_LEN,
        "quant_type": quant_type,
    }
    meta_file = work_dir / "meta_code2wav.json"
    save_json(meta_file, meta_info)
    logger.info(f"code2wav export complete. Meta saved to {meta_file}")

def gptq_quant_text(args):

    # Force the in-repo gptqmodel ($HOUMO_EXAMPLES_PATH/hmodel/gptqmodel, on
    # PYTHONPATH via env.sh) to win over the gptqmodel preinstalled in
    # /opt/venv/houmo. site-packages can otherwise shadow PYTHONPATH (editable
    # .pth entries, venv launch order), so we prepend the repo path explicitly
    # and assert the resolved module actually comes from there.
    import sys
    import importlib

    gptq_path = _resolve_gptqmodel_repo_path()
    if sys.path[0] != gptq_path:
        sys.path.insert(0, gptq_path)
    # Drop any gptqmodel already imported from the wrong location.
    for mod in [m for m in sys.modules if m == "gptqmodel" or m.startswith("gptqmodel.")]:
        del sys.modules[mod]

    import gptqmodel as _gptqmodel
    resolved = osp.realpath(osp.dirname(_gptqmodel.__file__))
    if osp.realpath(gptq_path) not in resolved:
        raise RuntimeError(
            f"gptqmodel resolved to {resolved}, expected under {gptq_path}. "
            "Check PYTHONPATH / env.sh.")
    logger.info(f"Using gptqmodel from {resolved}")

    from gptqmodel import GPTQModel, QuantizeConfig

    def patch_qwen3_omni_config_bridge() -> None:
        import transformers.models.qwen3_omni_moe.configuration_qwen3_omni_moe as qwen3_omni_config

        if getattr(qwen3_omni_config, "_gptqmodel_qwen3_omni_cfg_bridge", False):
            return

        def _bridge_property(name: str, text_config_getter):
            def _get(instance):
                value = instance.__dict__.get(name, None)
                if value is not None:
                    return value

                text_cfg = text_config_getter(instance)
                return getattr(text_cfg, name, None) if text_cfg is not None else None

            def _set(instance, value):
                instance.__dict__[name] = value
                text_cfg = text_config_getter(instance)
                if text_cfg is not None and hasattr(text_cfg, name):
                    setattr(text_cfg, name, value)

            return property(_get, _set)

        root_cfg_cls = qwen3_omni_config.Qwen3OmniMoeConfig
        thinker_cfg_cls = qwen3_omni_config.Qwen3OmniMoeThinkerConfig

        root_text_config = lambda instance: getattr(getattr(instance, "thinker_config", None), "text_config", None)
        thinker_text_config = lambda instance: getattr(instance, "text_config", None)

        for attr in CONFIG_BRIDGE_ATTRS:
            setattr(root_cfg_cls, attr, _bridge_property(attr, root_text_config))
            setattr(thinker_cfg_cls, attr, _bridge_property(attr, thinker_text_config))

        setattr(qwen3_omni_config, "_gptqmodel_qwen3_omni_cfg_bridge", True)

    modelscope_name = os.path.basename(args.model)
    quant_path = os.path.join(
        args.work_dir, "{}_gptqmodel_4bit".format(modelscope_name)
    )

    if args.calib_data:
        if args.calib_data.endswith(".jsonl"):
            calibration_dataset = get_jsonl_texts(args.calib_data, nsamples=256)
        elif os.path.isdir(args.calib_data) and "wikitext" in args.calib_data.lower():
            tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
            calibration_dataset = get_wikitext2(
                nsamples=256,
                seqlen=1024,
                local_dir=args.calib_data,
                tokenizer=tokenizer,
            )
        else:
            raise ValueError(f"Unsupported calibration data format: {args.calib_data}")
    else:
        # Auto-detect local wikitext zip downloaded by get_model.py --type raw
        local_zip = os.path.join(os.path.dirname(args.model), "wikitext-2-raw-v1.zip")
        if os.path.exists(local_zip):
            # Extract to a fixed cache dir under work_dir (only once)
            calib_dir = os.path.join(args.work_dir, "_calib_data", "wikitext-2-raw-v1")
            if not os.path.isdir(calib_dir):
                import zipfile
                logger.info(f"Extracting calibration data from {local_zip} to {calib_dir}")
                os.makedirs(calib_dir, exist_ok=True)
                with zipfile.ZipFile(local_zip, 'r') as zf:
                    zf.extractall(os.path.join(args.work_dir, "_calib_data"))
                # zip wraps content in "wikitext-2-raw-v1/" subdir, already matches calib_dir
            tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
            calibration_dataset = get_wikitext2(
                nsamples=256, seqlen=1024, local_dir=calib_dir, tokenizer=tokenizer,
            )
        else:
            logger.warning(f"Local wikitext zip not found at {local_zip}, falling back to HF download.")
            calibration_dataset = get_wikitext2(nsamples=256, seqlen=1024)

    patch_qwen3_omni_config_bridge()

    quant_config = QuantizeConfig(
        bits=4,
        group_size=64,
        rotation=None,
        offload_to_disk=False,
    )

    load_kwargs = dict(trust_remote_code=False, device_map="auto")

    model = GPTQModel.load(args.model, quant_config, **load_kwargs)

    # increase `batch_size` to match gpu/vram specs to speed up quantization
    model.quantize(cast(Any, calibration_dataset), batch_size=1)

    model.save(quant_path)
    try:
        tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=False)
    except Exception:
        processor = AutoProcessor.from_pretrained(args.model, trust_remote_code=False)
        tokenizer = getattr(processor, "tokenizer", None)
        if tokenizer is None:
            raise RuntimeError("Failed to load tokenizer from both AutoTokenizer and AutoProcessor")
    tokenizer.save_pretrained(quant_path)
    for file_name in OPTIONAL_TEXT_ASSET_FILES:
        source_file = os.path.join(args.model, file_name)
        if os.path.isfile(source_file):
            shutil.copy2(source_file, os.path.join(quant_path, file_name))

    del model
    del tokenizer
    cleanup_cuda()
    cleanup_cpu()

def houmo_export_text_llm(args):
    from xh_model_zoo.xh_llm.models.qwen3_omni import Qwen3OmniMoeConvertConfig
    from _thinker_gptq_view import (
        is_qwen3_omni_checkpoint,
        is_qwen3_omni_gptq_checkpoint,
        prepare_qwen3_omni_thinker_text_view,
    )

    hf_model_path = osp.normpath(osp.abspath(args.model))
    modelscope_name = Path(hf_model_path).name
    model_name = args.model_name
    quant_type = args.llm_quant_type
    part_module = "text_llm"
    if args.gptqmodel:
        quant_path = os.path.join(
            args.work_dir, "{}_gptqmodel_4bit".format(modelscope_name)
        )
        hf_model_path = osp.normpath(osp.abspath(quant_path))
        quant_type = "w4a8h0_ssfp" if "w4a8" not in args.llm_quant_type else args.llm_quant_type

    accept_hidden_layer = resolve_accept_hidden_layer(hf_model_path)
    use_qwen3omni_text_view = is_qwen3_omni_checkpoint(hf_model_path)
    use_qwen3omni_gptq_view = is_qwen3_omni_gptq_checkpoint(hf_model_path)
    if use_qwen3omni_text_view and accept_hidden_layer is None:
        raise NotImplementedError("Failed to resolve accept_hidden_layer from model config!")
    
    prefix = f"{model_name}-{part_module}-{quant_type}"
    work_dir = Path(args.work_dir) / prefix
    work_dir.mkdir(exist_ok=True, parents=True)
    log_file = work_dir / f"{part_module}-convert.log"
    xhquant_init(log_file, debug=args.debug)

    quant_scheme = QuantScheme(target_device=DeviceType.XH2a, quant_type=quant_type)
    if use_qwen3omni_text_view:
        text_view_dir = prepare_qwen3_omni_thinker_text_view(
            hf_model_path,
            work_dir / "_thinker_text_view" / model_name,
        )
        export_model_path = str(text_view_dir)
        export_architecture = "Qwen3OmniMoeThinkerTextForCausalLM"
        view_kind = "GPTQ thinker view" if use_qwen3omni_gptq_view else "HF thinker text view"
        config = Qwen3OmniMoeConvertConfig(
            batch_size=1,
            context_length=args.context_length,
            input_sequence_length=args.input_sequence_length,
            quant_scheme=quant_scheme,
            quant_weight=None,
            accept_hidden_layer=accept_hidden_layer,
            num_logits_to_keep=1,
        )
        logger.info(
            f"Exporting thinker text module from {view_kind} {export_model_path}. "
            "This avoids FX tracing the full multimodal Qwen3OmniMoeForConditionalGeneration forward path."
        )
    else:
        export_model_path = hf_model_path
        export_architecture = "Qwen3OmniMoeForConditionalGeneration"
        config = Qwen3OmniMoeConvertConfig(
            batch_size=1,
            context_length=args.context_length,
            input_sequence_length=args.input_sequence_length,
            quant_scheme=quant_scheme,
            quant_weight=None,
            accept_hidden_layer=accept_hidden_layer,
            num_logits_to_keep=1,
            export_audio_encoder=False,
            export_vision_encoder=False,
            export_talker_model=False,
            export_talker_prediction=False,
        )
        logger.info(f"Exporting thinker text module from {hf_model_path}")

    text_hmonnx_dir = work_dir / "hmonnx"
    text_hmonnx_dir.mkdir(exist_ok=True, parents=True)

    text_prefill_hmonnx_file = work_dir / f"{model_name}-text_prefill-{quant_type}.onnx"
    text_decode_hmonnx_file = work_dir / f"{model_name}-text_decode-{quant_type}.onnx"

    try:
        with TimeProfiler("convert_text", logger), MemoryTracker("cuda:0", "convert_text", logger):
            LLMConverter.from_pretrained(export_model_path, export_architecture, config, str(work_dir))
        logger.info(f"Text module export successful: {text_prefill_hmonnx_file}, {text_decode_hmonnx_file}")
    finally:
        cleanup_cuda()
        cleanup_cpu()

def houmo_export_talker(args):
    hf_model_path = osp.normpath(osp.abspath(args.model))
    model_name = args.model_name
    quant_type = args.quant_type

    try:
        for part_module in ["talker", "talker_prediction"]:
            prefix = f"{model_name}-{part_module}-{quant_type}"
            work_dir = Path(args.work_dir) / prefix
            work_dir.mkdir(exist_ok=True, parents=True)
            log_file = work_dir / f"{part_module}-convert.log"
            xhquant_init(log_file, debug=args.debug)

            native_model = Qwen3OmniMoeForConditionalGeneration.from_pretrained(hf_model_path, 
                                                                        torch_dtype=torch.float16,
                                                                        device_map="cpu",
                                                                        attn_implementation="eager",
                                                                        trust_remote_code=True)
            native_model.eval()

            if part_module == "talker":
                export_talker_model(args, native_model, hf_model_path, work_dir, model_name, quant_type)
            elif part_module == "talker_prediction":
                export_talker_prediction(args, native_model, hf_model_path, model_name, work_dir, quant_type)
            del native_model
    finally:
        cleanup_cuda()
        cleanup_cpu()

def houmo_export_other(args):
    hf_model_path = osp.normpath(osp.abspath(args.model))
    model_name = args.model_name
    quant_type = args.quant_type

    native_model = Qwen3OmniMoeForConditionalGeneration.from_pretrained(hf_model_path, 
                                                                        torch_dtype=torch.float16,
                                                                        device_map="cpu",
                                                                        attn_implementation="eager",
                                                                        trust_remote_code=True)
    native_model.eval()

    try:
        for part_module in ["vision", "audio", "text_projection", "code2wav"]:
            prefix = f"{model_name}-{part_module}-{quant_type if "projection" not in part_module else args.projection_quant_type}"
            work_dir = Path(args.work_dir) / prefix
            work_dir.mkdir(exist_ok=True, parents=True)
            log_file = work_dir / f"{part_module}-convert.log"
            xhquant_init(log_file, debug=args.debug)

            if part_module == "vision":
                export_vision_model(args.sample_path,
                                    native_model,
                                    model_name,
                                    work_dir,
                                    quant_type,
                                    vision_size=args.vision_size)
            elif part_module == "audio":
                export_audio_model(native_model,
                                    model_name,
                                    work_dir,
                                    quant_type)
            elif part_module == "text_projection":
                export_talker_projection(native_model.talker.text_projection,
                                            model_name,
                                            "text_projection",
                                            native_model.talker.text_projection.linear_fc1.in_features,
                                            work_dir,
                                            args.projection_quant_type)
            elif part_module == "code2wav":
                export_code2wav(
                    native_model,
                    model_name,
                    work_dir,
                    quant_type,
                )
    finally:
        del native_model
        cleanup_cuda()
        cleanup_cpu()

def move_models(
    work_dir: Path,
    source: str = "prefill",
    model: str = "prefill",
    target_name: str = "hmquant_qwen3-omni_with_act.onnx",
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
    model_name = args.model_name
    hm_model_name = "hmquant_{}_with_act.onnx".format(args.model_name)
    START_MOVE_MSG = "Start move from {} to {}"
    ### visual ###
    hmm_model_dir = "{}-vision-{}".format(
        model_name, args.quant_type
    )
    logger.info(
        msg_output_format(START_MOVE_MSG).format(
            work_dir / hmm_model_dir, dest_dir
        )
    )
    visual_dst_dir = dest_dir / "hmquant/visual"
    visual_dst_dir.mkdir(parents=True, exist_ok=True)
    os.system("mv {}/* {}".format(str(work_dir / hmm_model_dir / "hmonnx"), str(visual_dst_dir)))
    move_models(dest_dir, "visual", "vision", target_name=hm_model_name)
 
    ### audio ###
    hmm_model_dir = "{}-audio-{}".format(
        model_name, args.quant_type
    )
    logger.info(
        msg_output_format(START_MOVE_MSG).format(
            work_dir / hmm_model_dir, dest_dir
        )
    )
    audio_dst_dir = dest_dir / "hmquant/audio"
    audio_dst_dir.mkdir(parents=True, exist_ok=True)
    os.system("mv {}/* {}".format(str(work_dir / hmm_model_dir / "hmonnx"), str(audio_dst_dir)))
    move_models(dest_dir, "audio", "audio", target_name=hm_model_name)

    ### text llm ###
    hmm_model_dir = "{}-text_llm-{}".format(
        model_name, args.llm_quant_type
    )
    logger.info(
        msg_output_format(START_MOVE_MSG).format(
            work_dir / hmm_model_dir, dest_dir
        )
    )
    prefill_dst_dir = dest_dir / "hmquant/prefill"
    prefill_dst_dir.mkdir(parents=True, exist_ok=True)
    os.system("mv {}/* {}".format(str(work_dir / hmm_model_dir / "hmonnx/prefill"), str(prefill_dst_dir)))
    move_models(dest_dir, "prefill", "prefill", target_name=hm_model_name)
    decode_dst_dir = dest_dir / "hmquant/decode"
    decode_dst_dir.mkdir(parents=True, exist_ok=True)
    os.system("mv {}/* {}".format(str(work_dir / hmm_model_dir / "hmonnx/decode"), str(decode_dst_dir)))
    move_models(dest_dir, "decode", "decode", target_name=hm_model_name)
    shutil.move(
        work_dir / hmm_model_dir / "token_embedding.pt",
        dest_dir / "hmquant/quant_embedding.pt",
    )

    ### talker model ###
    hmm_model_dir = "{}-talker-{}".format(
        model_name, args.quant_type
    )
    logger.info(
        msg_output_format(START_MOVE_MSG).format(
            work_dir / hmm_model_dir, dest_dir
        )
    )
    talker_prefill_dst_dir = dest_dir / "hmquant/talker_prefill"
    talker_prefill_dst_dir.mkdir(parents=True, exist_ok=True)
    os.system("mv {}/*talker_prefill* {}".format(str(work_dir / hmm_model_dir / "hmonnx"), str(talker_prefill_dst_dir)))
    move_models(dest_dir, "talker_prefill", "talker_prefill", target_name=hm_model_name)
    talker_decode_dst_dir = dest_dir / "hmquant/talker_decode"
    talker_decode_dst_dir.mkdir(parents=True, exist_ok=True)
    os.system("mv {}/*talker_decode* {}".format(str(work_dir / hmm_model_dir / "hmonnx"), str(talker_decode_dst_dir)))
    move_models(dest_dir, "talker_decode", "talker_decode", target_name=hm_model_name)
    shutil.move(
        work_dir / hmm_model_dir / "talker_embedding.pt",
        dest_dir / "hmquant/quant_embedding_talker.pt",
    )
    
    ### talker prediction ###
    hmm_model_dir = "{}-talker_prediction-{}".format(
        model_name, args.quant_type
    )
    logger.info(
        msg_output_format(START_MOVE_MSG).format(
            work_dir / hmm_model_dir, dest_dir
        )
    )
    talker_prediction_prefill_dst_dir = dest_dir / "hmquant/talker_prediction_prefill"
    talker_prediction_prefill_dst_dir.mkdir(parents=True, exist_ok=True)
    os.system("mv {}/*prefill* {}".format(str(work_dir / hmm_model_dir / "hmonnx"), str(talker_prediction_prefill_dst_dir)))
    move_models(dest_dir, "talker_prediction_prefill", "prefill", target_name=hm_model_name)
    talker_prediction_decode_dst_dir = dest_dir / "hmquant/talker_prediction_decode"
    talker_prediction_decode_dst_dir.mkdir(parents=True, exist_ok=True)
    os.system("mv {}/*decode* {}".format(str(work_dir / hmm_model_dir / "hmonnx"), str(talker_prediction_decode_dst_dir)))
    move_models(dest_dir, "talker_prediction_decode", "decode", target_name=hm_model_name)
    shutil.move(
        work_dir / hmm_model_dir / "codec_embeddings.pt",
        dest_dir / "hmquant/quant_embedding_codec.pt",
    )

    ### move code2wav ###
    hmm_model_dir = "{}-code2wav-{}".format(
        model_name, args.quant_type
    )
    logger.info(
        msg_output_format(START_MOVE_MSG).format(
            work_dir / hmm_model_dir, dest_dir
        )
    )
    code2wav_dst_dir = dest_dir / "hmquant/code2wav"
    code2wav_dst_dir.mkdir(parents=True, exist_ok=True)
    os.system("mv {}/* {}".format(str(work_dir / hmm_model_dir / "hmonnx"), str(code2wav_dst_dir)))
    move_models(dest_dir, "code2wav", "code2wav", target_name=hm_model_name)

    ### text projection ###
    hmm_model_dir = "{}-text_projection-{}".format(model_name, args.projection_quant_type)
    logger.info(
        msg_output_format(START_MOVE_MSG).format(
            work_dir / hmm_model_dir, dest_dir
        )
    )
    text_projection_dst_dir = dest_dir / "hmquant/text_projection"
    text_projection_dst_dir.mkdir(parents=True)
    os.system("mv {}/* {}".format(str(work_dir / hmm_model_dir / "hmonnx"), str(text_projection_dst_dir)))
    move_models(dest_dir, "text_projection", "text_projection", target_name=hm_model_name)

    logger.info(msg_output_format("Start remove work_dir: {}".format(work_dir)))
    shutil.rmtree(work_dir, ignore_errors=True)


def parse_args():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--config", dest="config_path", type=str, default=DEFAULT_CONFIG_PATH, help="path to config.yaml")
    parser.add_argument("--model", type=str, default=None, help="input hf model path")
    parser.add_argument("--model-name", type=str, default=None, help="output hmonnx model name")
    parser.add_argument("--model-size", type=str, default=None, help="model size")
    parser.add_argument("--work-dir", type=str, default="work_dirs")
    parser.add_argument("--out-dir", type=str, default="output/{}".format(HOUMO_TARGET))
    parser.add_argument("--sample-path", type=str, default=DEFAULT_SAMPLE_DIR, help="sample data path for capture")
    parser.add_argument("--debug", action="store_true", help="debug mode")
    parser.add_argument("--context-length", type=str, default=None, help="max sequence length")
    parser.add_argument("--input-sequence-length", type=int, default=None, help="input sequence length")
    parser.add_argument("--quant-type", default=None, help="text-LLM / default quant type (config 'other')")
    parser.add_argument("--llm-quant-type", default=None, help="text-LLM quant type.")
    parser.add_argument("--projection-quant-type", default=None, help="talker projection quant type (config 'projection')")
    parser.add_argument("--vision-size", type=int, default=None,
                        help="fixed square input resolution for the vision encoder export; "
                                "must be a multiple of patch_size*merge_size (32). "
                                "448 -> 196 vision tokens, 224 -> 49")
    parser.add_argument("--gptqmodel", nargs="?", const=True, default=False, type=str2bool, help="use gptqmodel to quant, only text llm part supported.")
    parser.add_argument("--calib_data", type=str, default=None, help="calibration dataset path (default None -> auto-detect local wikitext-2-raw-v1.zip next to model dir)",)
    args = parser.parse_args()

    default_model_size, default_model_name, model_configs = get_model_configs(
        args.config_path
    )
    args.model_name = first_not_none(args.model_name, default_model_name)
    args.model_size = first_not_none(args.model_size, default_model_size)
    model_config = model_configs.get(args.model_name, {}).get(args.model_size, {})

    repo_ids = model_config.get("modelscope_repo", [])
    default_model_dir = (
        os.path.join(MODEL_FOLDER, repo_ids[0].rsplit("/", maxsplit=1)[-1])
        if repo_ids
        else os.path.join(MODEL_FOLDER, "Qwen3-Omni-30B-A3B-Instruct")
    )
    args.model = first_not_none(args.model, default_model_dir)

    args.context_length = parse_context_length(
        first_not_none(args.context_length, model_config.get("context_length", "4k"))
    )
    args.input_sequence_length = first_not_none(
        args.input_sequence_length, model_config.get("prefill_length", 256)
    )
    args.vision_size = first_not_none(
        args.vision_size, model_config.get("vision_size", 448)
    )

    quant_types = parse_quant_types(model_config)
    args.quant_type = first_not_none(args.quant_type, quant_types["other"])
    args.llm_quant_type = first_not_none(args.llm_quant_type, quant_types["llm"])
    args.projection_quant_type = first_not_none(
        args.projection_quant_type, quant_types["projection"]
    )
    return args

def main():
    args = parse_args()
    if args.gptqmodel:
        logger.warning("GPTQ quantization is enabled, but it is currently only supported for the text LLM part. "
                        "The exported talker model will still use the default quantization method.")
        gptq_quant_text(args)
    houmo_export_text_llm(args)
    houmo_export_other(args)
    houmo_export_talker(args)
    move_hmonnx(args)

if __name__ == "__main__":
    if not check_gpu():
        print("Error: Not found GPU device.")
        exit(-1)
    memory_monitor = ProcessMemoryMonitor(interval=2, log_file="./cpu_memory.log")
    memory_monitor.start()
    main()
    memory_monitor.stop()