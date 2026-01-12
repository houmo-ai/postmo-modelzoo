# Copyright (c) 2025 HOUMO AI
#
# File: quant_pipeline.py
# Description:
#   Quantization Pipeline Module - Python script implementing the
# quantization pipeline for Qwen2.5-VL models.
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
import os.path as osp
import shutil
from pathlib import Path

import torch
import torch.nn as nn
import transformers
from loguru import logger
from qwen_vl_utils import process_vision_info
from safetensors.torch import load_file as load_safetensors_file
from safetensors.torch import save_file as save_safetensors_file
from tqdm import tqdm
from transformers import AutoConfig, AutoProcessor

from xh_model_zoo.xh_llm import LLMConverter
from xh_model_zoo.xh_llm.models.qwen2_5_vl import Qwen2_5_VLConvertConfig, VisualConfig

from xhquant.api import (
    DeviceType,
    xhquant_init,
    QuantScheme,
    get_root_logger,
)  # isort:skip
from xh_model_zoo.utils.memory_tracker import MemoryTracker  # isort:skip
from xh_model_zoo.utils.time_profiler import TimeProfiler  # isort:skip


def msg_output_format(title):
    padding_str = "*" * 10
    title = f"{padding_str} {title} {padding_str}"
    return title


def robust_file_copy(src_file, dst_path):
    try:
        # Check if source file exists
        if not os.path.exists(src_file):
            raise FileNotFoundError(f"Error: Source file '{src_file}' does not exist.")

        dst_dir = os.path.dirname(dst_path)
        os.makedirs(
            dst_dir, exist_ok=True
        )  # Create target directory if it doesn't exist

        # Perform copy operation
        shutil.copy2(src_file, dst_path)
        print(f"Operation successful! File copied to: {dst_path}")

    except PermissionError:
        print(
            f"Error: Permission denied, cannot write to target location '{dst_path}'."
        )
    except OSError as e:
        print(f"Error: Operating system error - {e}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")


def demo(model, processor):
    from accelerate import dispatch_model, infer_auto_device_map
    from accelerate.utils import get_balanced_memory

    from xh_model_zoo.xh_llm.quarot import utils

    raw_device = next(model.parameters()).device
    # model.to(utils.DEV)

    no_split_module_classes = [
        "LlamaDecoderLayer",
        "QuantDecoderLayer",
        "RotateModule",
        "SmoothModule",
        "Qwen2DecoderLayer",
        "Qwen2_5_VLDecoderLayer",
    ]
    max_memory = get_balanced_memory(
        model, no_split_module_classes=no_split_module_classes
    )
    device_map = infer_auto_device_map(
        model, max_memory=max_memory, no_split_module_classes=no_split_module_classes
    )
    dispatch_model(
        model,
        device_map=device_map,
        offload_buffers=True,
        offload_dir="offload",
        state_dict=model.state_dict(),
    )

    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "image": "https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen-VL/assets/demo.jpeg",
                },
                {"type": "text", "text": "Describe this image."},
            ],
        }
    ]

    # Preparation for inference
    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    )
    inputs = inputs.to("cuda")

    # Inference: Generation of the output
    generated_ids = model.generate(**inputs, max_new_tokens=512)
    generated_ids_trimmed = [
        out_ids[len(in_ids) :]
        for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    output_text = processor.batch_decode(
        generated_ids_trimmed,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )
    print(output_text)
    from accelerate.hooks import remove_hook_from_module

    remove_hook_from_module(model)
    model.to(raw_device)
    utils.cleanup_memory()


def quant_llm(args):
    from transformers import Qwen2_5_VLForConditionalGeneration

    out_dir = Path(args.work_dir)
    hf_model_dir = args.model

    hf_model_path = osp.normpath(osp.abspath(args.model))
    model_name = Path(hf_model_path).name
    cfg_name = model_name
    if not args.skip_quarot:
        cfg_name += "_quarot"
    if not args.skip_gptq:
        cfg_name += "_gptq"

    # INSERT_YOUR_CODE
    # 增加transformers版本信息到cfg_name
    cfg_name += f"_transformers-{transformers.__version__}"

    work_dir = Path(out_dir) / cfg_name
    work_dir.mkdir(exist_ok=True, parents=True)
    config = AutoConfig.from_pretrained(hf_model_dir)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16

    native_model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        hf_model_dir,
        torch_dtype=dtype,
        device_map="cpu",
        # config=config,
        trust_remote_code=True,
        attn_implementation="flash_attention_2",
    )

    if native_model.config.tie_word_embeddings:
        old_torchscript = native_model.config.torchscript
        native_model.config.torchscript = True
        native_model.tie_weights()
        native_model.config.tie_word_embeddings = False
        native_model.config.torchscript = old_torchscript

    native_model.eval()
    native_model.to(dtype)

    processor = AutoProcessor.from_pretrained(hf_model_dir)

    if args.validate:
        demo(native_model, processor)

    quant_methods = []
    if not args.skip_quarot:
        torch.cuda.reset_peak_memory_stats()
        quant_methods.append("quarot")
        quant_name = "_".join(quant_methods)
        filename = work_dir / f"{quant_name}-state-dict.safetensors"
        if not os.path.exists(filename):
            from xh_model_zoo.xh_llm.quarot.quantizer_utils import quarot

            logger.info(msg_output_format("Start quarot quantization"))
            native_model = quarot(native_model, device=device)
            logger.info(msg_output_format("End quarot quantization"))
            native_model.to(torch.float16)
            state_dict = native_model.state_dict()
            logger.info(msg_output_format(f"Saving checkpoint to: {filename}"))
            # torch.save(state_dict, filename)
            save_safetensors_file(state_dict, filename)
            logger.info(f"Save checkpoint to: {filename}")
        else:
            state_dict = load_safetensors_file(filename)
            native_model.to(torch.float16)
            native_model.load_state_dict(state_dict)
            logger.info(msg_output_format(f"Load state_dict from {filename}"))

    if args.validate:
        demo(native_model.to(torch.float32), processor)
    consumption = torch.cuda.max_memory_allocated()
    unit = "B"
    if consumption > 1024:
        consumption = consumption / 1024
        unit = "k"
        if consumption > 1024:
            consumption = consumption / 1024
            unit = "M"
        consumption = round(consumption, 2)
    logger.info(f"GPU memory cost for export {consumption}{unit}")

    if not args.skip_gptq:
        from xh_model_zoo.xh_llm.quarot.quantizer_utils import gptq

        gptq_config = dict(
            calib_dataset=args.calib_dataset,
            calib_samples=args.calib_samples,
            seqlen=2048,
            w_clip=True,
            w_bits=args.w_bits,
            w_asym=False,
            w_groupsize=64,
            percdamp=0.01,
            act_order=False,
            int8_down_proj=False,
            heading_gptq=True,
            w_head_bits=args.w_head_bits
        )

        torch.cuda.reset_peak_memory_stats()
        logger.info(msg_output_format("Start gptq quantization"))
        quant_methods.append("gptq")
        layers_cache_dir = work_dir / "layers_cache"
        layers_cache_dir.mkdir(exist_ok=True, parents=True)

        native_model = gptq(
            native_model,
            args=args,
            model_name=hf_model_dir,
            **gptq_config,
            device=device,
            layers_cache_dir=str(layers_cache_dir),
            is_qwen2_5_vl=True,
            processor=processor,
            data_files=args.data_files,
        )
        logger.info(msg_output_format("End gptq quantization"))

        consumption = torch.cuda.max_memory_allocated()
        unit = "B"
        if consumption > 1024:
            consumption = consumption / 1024
            unit = "k"
            if consumption > 1024:
                consumption = consumption / 1024
                unit = "M"
            consumption = round(consumption, 2)
        logger.info(f"GPU memory cost for export {consumption}{unit}")

    if len(quant_methods) != 0:
        quant_name = "_".join(quant_methods)
        filename = work_dir / f"{quant_name}-state-dict.safetensors"
        state_dict = native_model.state_dict()
        # del native_model
        for k in tqdm(state_dict):
            paths = k.split(".")
            v = state_dict[k]
            if paths[-1] == "quant_weight":
                if v.min().item() >= -pow(2, 7) and v.max().item() <= pow(2, 7) - 1:
                    v = v.to(torch.int8)
                elif v.min().item() >= -pow(2, 15) and v.max().item() <= pow(2, 15) - 1:
                    v = v.to(torch.int16)
                else:
                    v = v.to(torch.float32)
            else:
                v = v.to(torch.float16)

            state_dict[k] = v
        logger.info(msg_output_format(f"Saving checkpoint to: {filename}"))
        save_safetensors_file(state_dict, filename)
        logger.info(f"Save checkpoint to: {filename}")


def export_llm(args):
    from xh_model_zoo.xh_llm.models.qwen2_5_vl.modeling_qwen2_5_vl import (
        Qwen2_5_VLForConditionalGeneration,
    )

    robust_file_copy("../../../data/pic/beach.jpeg", "data/images/qwen2_vl_demo.jpeg")
    hf_model_path = osp.normpath(osp.abspath(args.model))
    model_name = Path(hf_model_path).name
    target_device = DeviceType.XH2a
    quant_type = args.quant_type
    ops=dict(MatMul=dict(
            act_scheme=dict(
                bits=8,
                fp_mode="sefp",
            ),
            act_schema_2=dict(
                bits=16,
                fp_mode="sefp",
            ),))
    quant_scheme = QuantScheme(target_device=DeviceType.XH2a, quant_type=quant_type, ops=ops)
    model_name = os.path.basename(args.model)
    model_dir = os.path.join(args.work_dir, "{}_quarot_gptq".format(model_name))
    quant_weight = os.path.join(
        "{}_transformers-{}".format(model_dir, transformers.__version__),
        "quarot_gptq-state-dict.safetensors",
    )

    native_model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        hf_model_path,
        torch_dtype=torch.bfloat16,
        device_map="cpu",
        # config=config,
        trust_remote_code=True,
        attn_implementation="sdpa",
    )

    if native_model.config.tie_word_embeddings:
        old_torchscript = native_model.config.torchscript
        native_model.config.torchscript = True
        native_model.tie_weights()
        native_model.config.tie_word_embeddings = False
        native_model.config.torchscript = old_torchscript

    native_model.eval()
    native_model.to(torch.bfloat16)

    processor = AutoProcessor.from_pretrained(hf_model_path)

    # demo(native_model, processor)
    config = Qwen2_5_VLConvertConfig(
        batch_size=args.batch_size,
        context_length=args.context_length,
        quant_scheme=quant_scheme,
        quant_weight=quant_weight,
        gptqmodel_cfg=args.use_gptqmodel,
        max_pe_length=args.max_pe_length,
        visual_config=VisualConfig(
            image_max_size_h=args.image_max_size_h,
            image_max_size_w=args.image_max_size_w,
            image_max_size_t=args.image_max_size_t,
            temporal_patch_size=args.temporal_patch_size,
            patch_size=args.patch_size,
            sample_image_path=args.sample_image_path,
        ),
    )

    prefix = f"{model_name}-{target_device}"
    work_dir = Path("work_dirs") / prefix
    work_dir.mkdir(exist_ok=True, parents=True)
    log_file = work_dir / "convert.log"
    xhquant_init(log_file, debug=args.debug)
    logger = get_root_logger()
    with TimeProfiler("convert", logger), MemoryTracker("cuda:0", "convert", logger):
        LLMConverter.from_pretrained(
            hf_model_path, "Qwen2_5_VLForConditionalGeneration", config, work_dir
        )


def simple_batch_rename(directory, old_prefix, new_prefix):
    """
    Simple batch rename without dry-run mode [2,6](@ref)
    """
    for filename in os.listdir(directory):
        if filename.startswith(old_prefix):
            old_path = os.path.join(directory, filename)
            # Only process files, not directories [7](@ref)
            if os.path.isfile(old_path):
                new_filename = new_prefix + filename[len(old_prefix) :]
                new_path = os.path.join(directory, new_filename)

                try:
                    os.rename(old_path, new_path)
                    print(f"Renamed: {filename} -> {new_filename}")
                except Exception as e:
                    print(f"Error renaming {filename}: {e}")


def clean_directory(directory):
    """Remove all subfolders but keep files in the directory."""
    for item in os.listdir(directory):
        path = os.path.join(directory, item)
        if os.path.isdir(path):
            shutil.rmtree(path)

def ensure_and_clean_dir(dir_path):
    if dir_path.exists():
        for item in dir_path.iterdir():
            if item.is_file():
                item.unlink()
            elif item.is_dir():
                shutil.rmtree(item)
        print(f"Cleared existing directory: {dir_path}")
    else:
        dir_path.mkdir(parents=True, exist_ok=True)
        print(f"Created directory: {dir_path}")

def move_llm(args):
    work_dir = Path(args.work_dir)
    dest_dir = Path(args.out_dir)
    model_name = os.path.basename(args.model)
    hmm_model_dir = "{}-XH2a".format(model_name)
    hmm_model_prefix = hmm_model_dir + "-{}".format(args.quant_type)
    logger.info(
        msg_output_format("Start move from {} to {}").format(
            work_dir / hmm_model_dir, dest_dir
        )
    )
    ensure_and_clean_dir(dest_dir / "hmquant")
    shutil.move(
        work_dir / hmm_model_dir / "token_embedding.pt",
        dest_dir / "hmquant/quant_embedding.pt",
    )
    shutil.move(
        work_dir / hmm_model_dir / "golden/{}_vision".format(hmm_model_prefix),
        dest_dir / "hmquant/visual",
    )
    clean_directory("{}/hmquant/visual".format(dest_dir))
    simple_batch_rename(
        "{}/hmquant/visual".format(dest_dir),
        "hmquant_{}_vision".format(hmm_model_prefix),
        "hmquant_{}".format(args.model_name),
    )
    shutil.move(
        work_dir / hmm_model_dir / "golden/{}-llm-prefill".format(hmm_model_prefix),
        dest_dir / "hmquant/prefill",
    )
    clean_directory("{}/hmquant/prefill".format(dest_dir))
    simple_batch_rename(
        "{}/hmquant/prefill".format(dest_dir),
        "hmquant_{}-llm-prefill".format(hmm_model_prefix),
        "hmquant_{}".format(args.model_name),
    )
    shutil.move(
        work_dir / hmm_model_dir / "golden/{}-llm-decode".format(hmm_model_prefix),
        dest_dir / "hmquant/decoder",
    )
    clean_directory("{}/hmquant/decoder".format(dest_dir))
    simple_batch_rename(
        "{}/hmquant/decoder".format(dest_dir),
        "hmquant_{}-llm-decode".format(hmm_model_prefix),
        "hmquant_{}".format(args.model_name),
    )
    logger.info(msg_output_format("Start remove work_dir: {}".format(work_dir)))
    shutil.rmtree(work_dir, ignore_errors=True)
    shutil.rmtree("data", ignore_errors=True)
