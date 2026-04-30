# Copyright (c) 2025 HOUMO AI
#
# File: ptq.py
# Description:
#   Post-Training Quantization Tool - Python script for quantizing
# CoPaw-Flash-9B models using post-training quantization techniques.
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
import sys
import shutil
import os.path as osp
from pathlib import Path
import time
import psutil
import threading
import queue
import traceback
import multiprocessing as mp
import random
import numpy as np
from typing import Any, Dict

import torch
import json
import torch.nn as nn
import gc
from loguru import logger
from tqdm import tqdm
from transformers import AutoConfig, AutoModelForImageTextToText, AutoProcessor

from xhquant.api import (
    CacheTensor,
    Config,
    DeviceType,
    ConfigDict,
    HMONNXGoldenInference,
    PrecisionMode,
    ptq_quantize,
    set_random_seed,
    QuantScheme,
    xhquant_init,
    convert_onnx_to_hmonnx,
    create_quant_config,
)

from xh_model_zoo.xh_llm.models.builder import MODELS
from xh_model_zoo.xh_llm.models.eval_model_type import EvalModelType
from xh_model_zoo.xh_llm.models.qwen3_5 import (
    XHQwen3_5Model,
    XHQwen3_5VisionModel,
    Qwen3_5Processor,
)

HOUMO_DATASETS_PATH = os.getenv(
    "HOUMO_DATASETS_PATH",
    str(Path(__file__).resolve().parents[3] / "data" / "datasets"),
)
HOUMO_PIC_PATH = os.getenv(
    "HOUMO_PIC_PATH", str(Path(__file__).resolve().parents[3] / "data" / "pic")
)
HOUMO_TARGET = os.getenv("HOUMO_TARGET", "")


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
        percent = self.process.memory_percent()  # Percentage of system memory
        return {"rss_mb": rss_mb, "percent": percent}

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
            self.peak_memory_mb = max(self.peak_memory_mb, mem_info["rss_mb"])

            timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
            log_message = f"{timestamp} - RSS: {mem_info['rss_mb']:.2f} MB, System%: {mem_info['percent']:.2f}%"
            # Output to console or file
            if self.log_file:
                with open(self.log_file, "a") as f:
                    f.write(log_message + "\n")

            time.sleep(self.interval)
        timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
        logger.info(
            f"{timestamp} - Max RSS: {self.peak_memory_mb:.2f} MB, System%: {self.process.memory_percent():.2f}%"
        )

    def stop(self):
        """Stops the monitoring loop and prints peak usage."""
        self.is_monitoring = False
        if hasattr(self, "monitor_thread"):
            self.monitor_thread.join(
                timeout=1
            )  # Wait a moment for the thread to finish
        print(f"[Monitoring stopped. Peak RSS: {self.peak_memory_mb:.2f} MB]")


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def check_gpu():
    import subprocess

    try:
        result = subprocess.run(
            "nvidia-smi --query-gpu=count --format=csv,noheader,nounits | wc -l",
            shell=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
            text=True,
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
    if v.lower() in ("yes", "true", "t", "y", "1", ""):
        return True
    elif v.lower() in ("no", "false", "f", "n", "0"):
        return False
    else:
        raise argparse.ArgumentTypeError("Boolean value expected.")


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


def save_json(file_path: Path, data: Dict[str, Any]):
    with open(file_path, "w", encoding="utf-8") as file_obj:
        json.dump(data, file_obj, indent=4, ensure_ascii=False)


def _unwrap_model_output(output):
    if isinstance(output, torch.Tensor):
        return output
    if isinstance(output, (list, tuple)):
        for item in output:
            if isinstance(item, torch.Tensor):
                return item
    if isinstance(output, dict):
        for key in ("image_embeds", "last_hidden_state", "hidden_states", "logits"):
            value = output.get(key)
            if isinstance(value, torch.Tensor):
                return value
        for value in output.values():
            if isinstance(value, torch.Tensor):
                return value
    raise TypeError(f"Unsupported model output type: {type(output)}")


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
        trainenc = tokenizer("n\n".join(train_data["text"]))
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


def build_prompt_inputs(tokenizer, device, max_len: int):
    prompt = "你多大了？用中文回答。"
    if hasattr(tokenizer, "apply_chat_template"):
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt},
        ]
        text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        input_ids = tokenizer([text], return_tensors="pt").input_ids
    else:
        input_ids = tokenizer(prompt, return_tensors="pt").input_ids

    if input_ids.shape[1] > max_len:
        input_ids = input_ids[:, :max_len]
    return input_ids.to(device)


def gptq_quant_llm(args):
    sys.path.append(os.path.dirname(__file__))
    from gptqmodel import GPTQModel, QuantizeConfig

    modelscope_name = os.path.basename(args.model)
    quant_path = os.path.join(
        args.work_dir, "{}_gptqmodel_4bit".format(modelscope_name)
    )

    if args.calib_data:
        if args.calib_data.endswith(".jsonl"):
            calibration_dataset = get_jsonl_texts(args.calib_data, nsamples=256)
        elif os.path.isdir(args.calib_data) and "wikitext" in args.calib_data.lower():
            tokenizer = AutoConfig.from_pretrained(args.model)
            calibration_dataset = get_wikitext2(
                nsamples=256,
                seqlen=1024,
                local_dir=args.calib_data,
                tokenizer=tokenizer,
            )
        else:
            raise ValueError(f"Unsupported calibration data format: {args.calib_data}")
    else:
        calibration_dataset = get_wikitext2(nsamples=256, seqlen=1024)

    quant_config = QuantizeConfig(
        bits=4,
        group_size=64,
        hessian_mse=True,
        rotation="hadamard",
        offload_to_disk=False,
    )

    load_kwargs = dict(trust_remote_code=False, device_map="auto")

    model = GPTQModel.load(args.model, quant_config, **load_kwargs)

    # increase `batch_size` to match gpu/vram specs to speed up quantization
    model.quantize(calibration_dataset, batch_size=1)

    model.save(quant_path)

    del model
    cleanup_cuda()
    cleanup_cpu()


def export_prefill_decode(
    cfg, work_dir, input_ids, model_name, quant_type, mode="prefill"
):
    def _flatten_inputs(inputs):
        flat = []
        for arg in inputs:
            if isinstance(arg, (list, tuple)):
                flat.extend(arg)
            else:
                flat.append(arg)
        return flat

    copaw_flash_model: XHQwen3_5Model = MODELS.build(cfg.model)
    tokenizer = copaw_flash_model.get_tokenizer()
    native_model = copaw_flash_model.get_hf_model(
        device_map="cpu", use_safetensors=True, torch_dtype="float16"
    )
    native_model.eval()

    hmonnx_dir = Path(work_dir) / f"{mode}"
    hmonnx_dir.mkdir(exist_ok=True, parents=True)

    copaw_flash_model.init_wrap_model(native_model)

    del native_model

    current_input_seq_length = cfg.model.wrap_cfg.input_sequence_length

    copaw_flash_model.change_eval_type(EvalModelType.WRAPED)
    copaw_flash_model.set_exec_device(torch.device("cpu"))
    copaw_flash_model.to("cpu")
    copaw_flash_model.to(torch.float16)

    logger.info(f"Start export {mode} hmonnx ...")
    copaw_flash_model.set_linear_attention_mode(
        "chunk" if mode == "prefill" else "recurrent"
    )
    copaw_flash_model.set_input_sequence_length(
        current_input_seq_length if mode == "prefill" else 1
    )
    onnx_prefix = f"{model_name}_{mode}_{quant_type}"
    if mode == "prefill":
        input_ids_pad = torch.cat(
            [
                input_ids,
                torch.full(
                    (input_ids.shape[0], current_input_seq_length - input_ids.shape[1]),
                    tokenizer.pad_token_id,
                    dtype=torch.long,
                ),
            ],
            dim=-1,
        )
        data_batch = {
            "input_ids": input_ids_pad,
            "past_seq_length": [0],
            "current_input_length": [input_ids.shape[1]],
        }
    else:
        data_batch = {
            "input_ids": input_ids[:, -1:],
            "past_seq_length": [input_ids.shape[1]],
            "current_input_length": [1],
        }

    copaw_flash_model.interactive_mode = True
    copaw_flash_model.convert_to_fronted_graph(data_batch, release_wraped_model=True)
    copaw_flash_model.convert_to_quant_graph(DeviceType.XH2a)

    calib_data = _flatten_inputs(copaw_flash_model.prepare_inputs_for_graph(data_batch))

    ptq_quantize(
        copaw_flash_model.quanted_model,
        [calib_data],
        PrecisionMode.ALIGNED,
        [torch.device("cpu")],
        auto_release_unused_parameters=True,
    )

    copaw_flash_model.change_eval_type(EvalModelType.QUANTED_ALIGNED)

    hmonnx_dir = Path(work_dir) / f"{mode}"
    hmonnx_dir.mkdir(exist_ok=True, parents=True)

    copaw_flash_model.convert_to_export_graph(data_batch)
    copaw_flash_model.change_eval_type(EvalModelType.EXPORTED)

    onnx_file = copaw_flash_model.to_export_onnx(data_batch, hmonnx_dir, onnx_prefix)[0]

    logger.info(f"Exported {mode} hmonnx model successful, saved to {onnx_file}")

    copaw_flash_model.release_exported_model()
    copaw_flash_model.release_quanted_model()
    copaw_flash_model.release_frontend_model()
    copaw_flash_model.release_wraped_model()

    del copaw_flash_model
    cleanup_cuda()
    cleanup_cpu()

    return onnx_file


def get_prepare_context(cfg, work_dir):
    copaw_flash_model: XHQwen3_5Model = MODELS.build(cfg.model)
    tokenizer = copaw_flash_model.get_tokenizer()
    input_ids = build_prompt_inputs(
        tokenizer, "cpu", cfg.model.wrap_cfg.input_sequence_length
    )
    native_model = copaw_flash_model.get_hf_model(
        device_map="cpu", use_safetensors=True, torch_dtype="float16"
    )
    native_model.eval()

    embed_file = Path(work_dir) / f"token_embedding.pt"
    if hasattr(native_model, "language_model"):
        token_embedding = native_model.model.language_model.get_input_embeddings()
    else:
        token_embedding = native_model.model.get_input_embeddings()
    torch.save(token_embedding.state_dict(), str(embed_file))

    copaw_flash_model.init_wrap_model(native_model)

    del native_model

    copaw_flash_model.change_eval_type(EvalModelType.WRAPED)
    copaw_flash_model.set_exec_device(torch.device("cpu"))
    copaw_flash_model.to("cpu")
    copaw_flash_model.to(torch.float16)

    kv_cache_shape = list(copaw_flash_model.past_key_caches[0].shape)
    num_hidden_layers = len(copaw_flash_model.past_key_caches)

    copaw_flash_model.release_wraped_model()

    del copaw_flash_model
    cleanup_cuda()
    cleanup_cpu()

    return (
        input_ids,
        embed_file,
        kv_cache_shape,
        num_hidden_layers,
        token_embedding.weight.shape[-1],
    )


def rotate_fp_vl(args):
    from gptqmodel.models.definitions.qwen3_5 import Qwen3_5QModel
    from gptqmodel.models.definitions.qwen3_vl import Qwen3_VLQModel
    from gptqmodel.quantization.rotation.rotation import (
        fuse_layer_norms_qwen3_5,
        rotate_model_qwen3_5_vl,
    )
    from gptqmodel.utils.model import get_module_by_name_prefix

    def maybe_clone_lm_head(model, lm_head_name: str):
        if not getattr(model.config, "tie_word_embeddings", False):
            return

        model.config.tie_word_embeddings = False
        text_config = getattr(model.config, "text_config", None)
        if text_config is not None:
            text_config.tie_word_embeddings = False

        lm_head, _ = get_module_by_name_prefix(model, lm_head_name)
        lm_head.weight = torch.nn.Parameter(lm_head.weight.data.clone())

    modelscope_name = os.path.basename(args.model)
    rotate_path = os.path.join(args.work_dir, "{}_rotate_fp_vl".format(modelscope_name))

    Qwen3_5QModel.before_model_load(Qwen3_5QModel, load_quantized_model=False)
    processor = AutoProcessor.from_pretrained(args.model, trust_remote_code=False)
    model = AutoModelForImageTextToText.from_pretrained(
        args.model,
        trust_remote_code=False,
        device_map="cpu",
        torch_dtype="auto",
    )
    model.eval()

    logger.info("Start rotating vision model...")
    lm_head_name = (Qwen3_VLQModel.lm_head,)
    layers_node = Qwen3_VLQModel.extract_layers_node()
    pre_lm_head_norm_module_name = Qwen3_VLQModel.pre_lm_head_norm_module
    maybe_clone_lm_head(model, lm_head_name)

    model = fuse_layer_norms_qwen3_5(
        model=model,
        pre_lm_head_norm_module_name=pre_lm_head_norm_module_name,
        layers_node=layers_node,
        lm_head_name=lm_head_name,
    )

    model, _ = rotate_model_qwen3_5_vl(
        model=model,
        rotate_mode="hadamard",
        device="cuda" if torch.cuda.is_available() else "cpu",
        lm_head_name=lm_head_name,
        layers_node=layers_node,
    )
    model.eval()
    logger.info("Vision model rotation complete.")

    logger.info(f"Save rotating result to {rotate_path}")
    model.generation_config.do_sample = True
    rotate_path_dir = Path(rotate_path)
    rotate_path_dir.mkdir(exist_ok=True, parents=True)
    model.save_pretrained(rotate_path, safe_serialization=True, max_shard_size="5GB")
    processor.save_pretrained(rotate_path)
    logger.info("Model saved successfully.")

    del model
    cleanup_cuda()
    cleanup_cpu()


def houmo_export_llm(args):
    hf_model_path = osp.normpath(osp.abspath(args.model))
    modelscope_name = Path(hf_model_path).name

    cfg = Config.fromfile(args.llm_config)

    quant_scheme = QuantScheme(
        target_device=DeviceType.XH2a, quant_type=args.quant_type
    )
    QConfig = create_quant_config(quant_scheme)
    cfg.model.quant_config.w_schema.bits = QConfig.w_schema.man_bit
    cfg.model.quant_config.w_schema.fp_mode = QConfig.w_schema.fp_mode
    cfg.model.quant_config.w_schema.hidden_bit = QConfig.w_schema.hidden_bit
    cfg.model.quant_config.act_schema.bits = QConfig.act_schema.man_bit
    cfg.model.quant_config.act_schema.fp_mode = QConfig.act_schema.fp_mode
    cfg.model.quant_config.act_schema.hidden_bit = QConfig.act_schema.hidden_bit

    if args.gptqmodel:
        quant_path = os.path.join(
            args.work_dir, "{}_gptqmodel_4bit".format(modelscope_name)
        )
        hf_model_path = osp.normpath(osp.abspath(quant_path))
        cfg.model.quant_config.w_schema.bits = 4

    prefix = "{}-XH2a-{}".format(args.model_name, format_number(args.context_length))
    work_dir = Path(args.work_dir) / prefix
    work_dir.mkdir(exist_ok=True, parents=True)
    log_file = work_dir / "convert.log"
    xhquant_init(log_file, debug=args.debug)

    cfg.hf_model_dir = hf_model_path
    cfg.model.hf_model = hf_model_path
    cfg.model.type = "XHQwen3_5Model"
    cfg.model.wrap_cfg.max_sequence_length = args.context_length
    cfg.model.wrap_cfg.input_sequence_length = args.input_sequence_length

    input_ids, embed_file, kv_cache_shape, num_hidden_layers, hidden_size = (
        get_prepare_context(cfg, work_dir)
    )

    quant_bits_str = f"w{cfg.model.quant_config.w_schema.bits}a{cfg.model.quant_config.act_schema.bits}"
    quant_hidden_str = "h1" if cfg.model.quant_config.w_schema.hidden_bit else "h0"
    quant_type = (
        f"{quant_bits_str}{quant_hidden_str}_{cfg.model.quant_config.w_schema.fp_mode}"
    )
    prefill_hmonnx_file = export_prefill_decode(
        cfg, work_dir, input_ids, args.model_name, quant_type, mode="prefill"
    )
    decode_hmonnx_file = export_prefill_decode(
        cfg, work_dir, input_ids, args.model_name, quant_type, mode="decode"
    )

    meta_info = {
        "create_time": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()),
        "module": "text_llm",
        "model_name": args.model_name,
        "prefill_onnx": str(prefill_hmonnx_file),
        "decode_onnx": str(decode_hmonnx_file),
        "embedding_file": str(embed_file),
        "kv_cache": {"shape": kv_cache_shape, "num_decoder_layers": num_hidden_layers},
        "hidden_size": int(hidden_size),
        "input_sequence_length": int(input_ids.shape[-1]),
    }
    meta_file = work_dir / "meta.json"
    save_json(meta_file, meta_info)
    logger.info(f"Text LLM export complete. Meta saved to {meta_file}")


def houmo_export_vision(args):
    from qwen_vl_utils import process_vision_info
    import onnx

    from xhquant.utils.onnxsim_large_model.simplify_large_onnx import (
        simplify_large_onnx,
    )

    hf_model_path = osp.normpath(osp.abspath(args.model))
    image_path = osp.normpath(osp.abspath(args.image))
    modelscope_name = Path(hf_model_path).name

    cfg = Config.fromfile(args.vision_config)

    quant_scheme = QuantScheme(
        target_device=DeviceType.XH2a, quant_type=args.quant_type
    )
    QConfig = create_quant_config(quant_scheme)
    cfg.model.quant_config.w_schema.bits = QConfig.w_schema.man_bit
    cfg.model.quant_config.w_schema.fp_mode = QConfig.w_schema.fp_mode
    cfg.model.quant_config.w_schema.hidden_bit = QConfig.w_schema.hidden_bit
    cfg.model.quant_config.act_schema.bits = QConfig.act_schema.man_bit
    cfg.model.quant_config.act_schema.fp_mode = QConfig.act_schema.fp_mode
    cfg.model.quant_config.act_schema.hidden_bit = QConfig.act_schema.hidden_bit

    prefix = "{}-XH2a-{}".format(args.model_name, format_number(args.context_length))
    work_dir = Path(args.work_dir) / prefix
    work_dir.mkdir(exist_ok=True, parents=True)
    log_file = work_dir / "convert.log"
    xhquant_init(log_file, debug=args.debug)

    cfg.model.type = "XHQwen3_5VisionModel"

    if args.gptqmodel:
        rotate_path = os.path.join(
            args.work_dir, "{}_rotate_fp_vl".format(modelscope_name)
        )
        processor_config_path = os.path.join(hf_model_path, "processor_config.json")
        hf_model_path = osp.normpath(osp.abspath(rotate_path))
        if not os.path.exists(os.path.join(hf_model_path, "processor_config.json")):
            shutil.copy(
                processor_config_path,
                os.path.join(hf_model_path, "processor_config.json"),
            )

    cfg.model.hf_model = hf_model_path
    cfg.hf_model_dir = hf_model_path

    hf_config = AutoConfig.from_pretrained(hf_model_path, trust_remote_code=True)
    expected_hidden_size = int(
        getattr(getattr(hf_config, "vision_config", None), "out_hidden_size", 0)
        or getattr(getattr(hf_config, "text_config", None), "hidden_size", 0)
    )
    if expected_hidden_size <= 0:
        raise ValueError(
            "Failed to determine expected vision output hidden size from HF config"
        )

    copaw_flash_vision_model: XHQwen3_5VisionModel = MODELS.build(cfg.model)
    native_model = copaw_flash_vision_model.get_hf_model(device_map="cpu")

    processor = Qwen3_5Processor.from_pretrained(hf_model_path)
    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "image": image_path,
                    "resized_height": cfg.model.wrap_cfg.max_size_h,
                    "resized_width": cfg.model.wrap_cfg.max_size_w,
                },
                {"type": "text", "text": "清晰描述图片中的内容。"},
            ],
        }
    ]

    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    image_inputs, video_inputs = process_vision_info(
        messages, image_patch_size=cfg.model.wrap_cfg.patch_size
    )

    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    )

    inputs.to("cpu")

    assert (
        inputs["image_grid_thw"][0][-1]
        == cfg.model.wrap_cfg.max_size_w // cfg.model.wrap_cfg.patch_size
    ), (
        f"inputs['image_grid_thw'][0][-1] = {inputs['image_grid_thw'][0][-1]}, "
        f"max_size_w = {cfg.model.wrap_cfg.max_size_w}, patch_size = {cfg.model.wrap_cfg.patch_size}"
    )
    assert (
        inputs["image_grid_thw"][0][-2]
        == cfg.model.wrap_cfg.max_size_h // cfg.model.wrap_cfg.patch_size
    ), (
        f"inputs['image_grid_thw'][0][-2] = {inputs['image_grid_thw'][0][-2]}, "
        f"max_size_h = {cfg.model.wrap_cfg.max_size_h}, apatch_size = {cfg.model.wrap_cfg.patch_size}"
    )

    copaw_flash_vision_model.init_wrap_model(native_model)
    wraped_model = copaw_flash_vision_model.wrap_model

    merger = getattr(wraped_model, "merger", None)
    linear_fc2 = getattr(merger, "linear_fc2", None) if merger is not None else None
    visual_cfg_out_hidden = getattr(
        getattr(wraped_model, "config", None), "out_hidden_size", None
    )
    if linear_fc2 is not None:
        logger.info(
            "Vision merger/projector info: "
            f"config.out_hidden_size={visual_cfg_out_hidden}, "
            f"linear_fc2.in_features={linear_fc2.in_features}, "
            f"linear_fc2.out_features={linear_fc2.out_features}"
        )
        if int(linear_fc2.out_features) != expected_hidden_size:
            raise ValueError(
                "Visual merger projector mismatch before export: "
                f"linear_fc2.out_features={linear_fc2.out_features}, expected={expected_hidden_size}. "
                "This indicates the loaded HF visual tower itself is already inconsistent with the target text model."
            )

    pixel_values = (
        inputs["hm_pixel_values"][0].type(wraped_model.dtype).to(wraped_model.device)
    )

    with torch.no_grad():
        sample_output = wraped_model.float().eval().cpu()(pixel_values.float().cpu())
    sample_tensor = _unwrap_model_output(sample_output)
    sample_shape = tuple(sample_tensor.shape)
    if not sample_shape:
        raise ValueError("Visual wrap model returned empty shape")
    if int(sample_shape[-1]) != expected_hidden_size:
        raise ValueError(
            "Visual export validation failed: wrapped vision model output shape "
            f"{sample_shape}, but expected last dim {expected_hidden_size}. "
            "This indicates `wrap_model` is not producing the final projected `image_embeds`. "
            "Please check `XHQwen3_5VisionModel.init_wrap_model()` / `wrap_model` and export the final projector output."
        )

    logger.info("Start export vision onnx ...")

    vision_tmp_onnx_dir = work_dir / "vision_tmp"
    vision_tmp_onnx_dir.mkdir(exist_ok=True, parents=True)
    vision_tmp_onnx_file = str(vision_tmp_onnx_dir / f"{args.model_name}_visual.onnx")
    torch.onnx.export(
        wraped_model.float().eval().cpu(),
        (pixel_values.float().cpu(),),
        vision_tmp_onnx_file,
        export_params=True,
        opset_version=18,
        do_constant_folding=True,
        input_names=["pixel_values"],
        output_names=["image_embeds"],
        verbose=True,
    )
    onnx_model = onnx.load(vision_tmp_onnx_file, load_external_data=True)

    onnx_model, _ = simplify_large_onnx(onnx_model)

    vision_onnx_file = str(work_dir / f"{args.model_name}_visual.onnx")
    onnx.save(
        onnx_model,
        vision_onnx_file,
        save_as_external_data=True,
        all_tensors_to_one_file=True,
        location=f"{args.model_name}_visual_external_data",
        convert_attribute=True,
    )

    os.system("rm -rf {}".format(vision_tmp_onnx_dir))

    vision_hmonnx_dir = work_dir / "visual"
    vision_hmonnx_dir.mkdir(exist_ok=True, parents=True)
    vision_hmonnx_file = (
        vision_hmonnx_dir / f"{args.model_name}_visual_{args.quant_type}.onnx"
    )

    convert_onnx_to_hmonnx(
        vision_onnx_file,
        [pixel_values.float().cpu()],
        device_type=DeviceType.XH2a,
        out_hmonnx_file=vision_hmonnx_file,
        quant_config=cfg.model.quant_config,
    )
    logger.info(
        "Export vision model successful, saved to {}".format(vision_hmonnx_file)
    )

    meta_info = {
        "create_time": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()),
        "module": "vision",
        "model_name": args.model_name,
        "vision_encoder_onnx": str(vision_hmonnx_file.relative_to(work_dir)),
        "vision_output_shape": list(sample_shape),
        "vision_expected_hidden_size": expected_hidden_size,
        "vision_patch_size": cfg.model.wrap_cfg.patch_size,
        "vision_input_size": [
            cfg.model.wrap_cfg.max_size_h,
            cfg.model.wrap_cfg.max_size_w,
        ],
        "vision_channels": 3,
        "vision_temporal_patch_size": cfg.model.wrap_cfg.temporal_patch_size,
        "vision_max_size_t": cfg.model.wrap_cfg.max_size_t,
        "export_method": "two_stage_onnx_simplify",  # Mark as using improved method
    }
    meta_file = work_dir / "meta_vision.json"
    save_json(meta_file, meta_info)
    logger.info(f"Vision export complete. Meta saved to {meta_file}")


def move_models(
    work_dir: Path,
    source: str = "prefill",
    model: str = "prefill",
    target_name: str = "hmquant_copaw-flash_with_act.onnx",
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
        return f"0k"


def move_llm(args):
    work_dir = Path(args.work_dir)
    dest_dir = Path(args.out_dir)
    dest_dir.mkdir(parents=True, exist_ok=True)
    model_name = os.path.basename(args.model_name)
    hm_model_name = "hmquant_{}_with_act.onnx".format(model_name)
    hmm_model_dir = "{}-XH2a-{}".format(model_name, format_number(args.context_length))
    logger.info(
        msg_output_format("Start move from {} to {}").format(
            work_dir / hmm_model_dir, dest_dir
        )
    )
    shutil.move(work_dir / hmm_model_dir / "prefill", dest_dir / "hmquant/prefill")
    move_models(dest_dir, "prefill", target_name=hm_model_name)
    shutil.move(work_dir / hmm_model_dir / "decode", dest_dir / "hmquant/decode")
    move_models(dest_dir, "decode", "decode", target_name=hm_model_name)
    shutil.move(
        work_dir / hmm_model_dir / "token_embedding.pt",
        dest_dir / "hmquant/quant_embedding.pt",
    )
    shutil.move(work_dir / hmm_model_dir / "visual", dest_dir / "hmquant/visual")
    move_models(dest_dir, "visual", "visual", target_name=hm_model_name)
    logger.info(msg_output_format("Start remove work_dir: {}".format(work_dir)))
    shutil.rmtree(work_dir, ignore_errors=True)


def _run_isolated_step(step_name: str, args_dict: Dict[str, Any], result_queue):
    child_args = argparse.Namespace(**args_dict)
    try:
        set_seed(42)
        if step_name == "gptq_quant_llm":
            gptq_quant_llm(child_args)
        elif step_name == "rotate_fp_vl":
            rotate_fp_vl(child_args)
        else:
            raise ValueError(f"Unsupported isolated step: {step_name}")
        cleanup_cuda()
        cleanup_cpu()
        result_queue.put({"ok": True})
    except Exception as exc:
        cleanup_cuda()
        cleanup_cpu()
        result_queue.put(
            {
                "ok": False,
                "error": str(exc),
                "traceback": traceback.format_exc(),
            }
        )


def run_step_in_fresh_process(step_name: str, args: argparse.Namespace):
    logger.info(f"Start isolated step in fresh process: {step_name}")
    ctx = mp.get_context("spawn")
    result_queue = ctx.Queue()
    process = ctx.Process(
        target=_run_isolated_step,
        args=(step_name, vars(args).copy(), result_queue),
        name=step_name,
    )
    try:
        process.start()
        process.join()

        exitcode = process.exitcode

        result = None
        try:
            result = result_queue.get(timeout=1)
        except queue.Empty:
            result = None

        if result is not None and not result.get("ok", False):
            raise RuntimeError(
                f"Isolated step `{step_name}` failed: {result['error']}\n{result['traceback']}"
            )

        if exitcode != 0:
            raise RuntimeError(
                f"Isolated step `{step_name}` exited unexpectedly with code {exitcode}"
            )

        logger.info(f"Isolated step finished successfully: {step_name}")
    finally:
        result_queue.close()
        result_queue.join_thread()
        process.close()


if HOUMO_TARGET == "xh2":

    def parse_args():
        parser = argparse.ArgumentParser(
            formatter_class=argparse.ArgumentDefaultsHelpFormatter
        )
        parser.add_argument("--model", type=str, default="CoPaw-Flash-9B")
        parser.add_argument(
            "--model-name",
            type=str,
            default="copaw-flash",
            help="output hmonnx model name",
        )
        parser.add_argument(
            "--calib_data",
            type=str,
            default="qwen3_8b_gen_data.jsonl",
            help="calibration dataset choose",
        )
        parser.add_argument(
            "--image",
            type=str,
            default=f"{HOUMO_PIC_PATH}/beach.jpeg",
            help="vision input image",
        )
        parser.add_argument("--work-dir", type=str, default="work_dirs/")
        parser.add_argument(
            "--out-dir", type=str, default="output/{}".format(HOUMO_TARGET)
        )
        parser.add_argument("--llm_config", type=str, default="llm_config.py")
        parser.add_argument("--vision_config", type=str, default="vision_config.py")
        parser.add_argument("--debug", action="store_true", help="debug mode")
        parser.add_argument(
            "--context-length", type=int, default=2048, help="max sequence length"
        )
        parser.add_argument(
            "--input-sequence-length",
            type=int,
            default=256,
            help="input sequence length",
        )
        parser.add_argument(
            "--quant-type",
            default="w8a8h0_ssfp",
            help="quant type, default is w8a8h0_ssfp",
        )
        parser.add_argument(
            "--gptqmodel", action="store_true", help="use gptqmodel to quant"
        )
        args = parser.parse_args()
        return args

    def main():
        args = parse_args()
        set_seed(42)
        if args.gptqmodel:
            run_step_in_fresh_process("rotate_fp_vl", args)
            run_step_in_fresh_process("gptq_quant_llm", args)
        houmo_export_llm(args)
        houmo_export_vision(args)
        move_llm(args)


if __name__ == "__main__":
    if not check_gpu():
        print("Error: Not found GPU device.")
        exit(-1)
    memory_monitor = ProcessMemoryMonitor(interval=2, log_file="./cpu_memory.log")
    memory_monitor.start()
    main()
    memory_monitor.stop()
