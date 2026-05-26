#!/usr/bin/python3
# -*- coding: utf-8 -*-
# Copyright 2025 HOUMO AI
#
# File: ptq.py
# Description:
#   Post-training quantization of the minicpmo model.
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
import shutil
import os.path as osp
from pathlib import Path

import torch
import gc
from loguru import logger

from hmatc.utils.monitor import ProcessMemoryMonitor
from hmatc.utils.utils import (
    check_gpu,
    first_not_none,
    get_model_configs,
    parse_context_length,
)

from xh_model_zoo.xh_llm import LLMConverter

from xhquant.api import (
    DeviceType,
    QuantScheme,
    get_root_logger,
    xhquant_init,
)  # isort:skip
from xh_model_zoo.utils.memory_tracker import MemoryTracker  # isort:skip
from xh_model_zoo.utils.time_profiler import TimeProfiler  # isort:skip

HOUMO_TARGET = os.getenv("HOUMO_TARGET")
assert HOUMO_TARGET in ["xh2"], f"Unsupported HOUMO_TARGET: {HOUMO_TARGET}"

HOUMO_DATASETS_PATH = os.getenv("HOUMO_DATASETS_PATH", "")
DEFAULT_CONFIG_PATH = os.path.join(os.path.dirname(__file__), "config.yaml")


def msg_output_format(title):
    padding_str = "*" * 10
    title = f"{padding_str} {title} {padding_str}"
    return title


def cleanup_cuda():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()


def houmo_export_llm(args, component="vision"):
    hf_model_path = osp.normpath(osp.abspath(args.model))
    model_name = args.model_name
    audio_path = os.path.join(args.model, "assets/demo.wav")
    video_path = os.path.join(args.model, "assets/Skiing.mp4")
    quant_type = args.quant_type
    quant_scheme = QuantScheme(target_device=DeviceType.XH2a, quant_type=quant_type)
    if component == "vision":
        from xh_model_zoo.xh_llm.models.minicpmo.minicpmo_vision_convert_config import (
            MinicpmoVisionConvertConfig,
        )

        config = MinicpmoVisionConvertConfig(
            quant_scheme=quant_scheme,
            video=video_path,
            audio=audio_path,
            debug=args.debug,
            valid=args.valid,
            image_slice_max_size=[40, 40],
        )
        architecture = "MiniCPMOVisionEncoder"
    elif component == "audio":
        from xh_model_zoo.xh_llm.models.minicpmo.minicpmo_audio_convert_config import (
            MinicpmoAudioConvertConfig,
        )

        config = MinicpmoAudioConvertConfig(
            quant_scheme=quant_scheme,
            video=video_path,
            audio=audio_path,
            debug=args.debug,
            valid=args.valid,
            image_slice_max_size=[40, 40],
        )
        architecture = "MiniCPMWhisperEncoder"
    elif component == "llm":
        from xh_model_zoo.xh_llm.models.minicpmo.minicpmo_llm_convert_config import (
            MinicpmoLLMConvertConfig,
        )
        llm_quant_scheme = QuantScheme(
            target_device=DeviceType.XH2a, 
            quant_type=quant_type, 
            ops=dict(Normalize=dict(force_fp32=True)))
        config = MinicpmoLLMConvertConfig(
            quant_scheme=llm_quant_scheme,
            video=video_path,
            audio=audio_path,
            debug=args.debug,
            valid=args.valid,
            context_length=args.context_length,
            input_sequence_length=args.input_sequence_length,
            image_slice_max_size=[40, 40],
        )
        architecture = "MiniCPMOLLMEncoder"
    elif component == "tts":
        from xh_model_zoo.xh_llm.models.minicpmo.minicpmo_tts_convert_config import (
            MinicpmoTTSConvertConfig,
        )

        config = MinicpmoTTSConvertConfig(
            quant_scheme=quant_scheme,
            video=video_path,
            audio=audio_path,
            debug=args.debug,
            valid=args.valid,
            context_length=args.context_length,
            input_sequence_length=args.input_sequence_length,
            image_slice_max_size=[40, 40],
        )
        architecture = "MiniCPMOTTS"
    elif component == "tts_dvae":
        from xh_model_zoo.xh_llm.models.minicpmo.minicpmo_tts_dvae_convert_config import (
            MinicpmoTTSDVAEConvertConfig,
        )

        config = MinicpmoTTSDVAEConvertConfig(
            quant_scheme=quant_scheme,
            video=video_path,
            audio=audio_path,
            debug=args.debug,
            valid=args.valid,
            image_slice_max_size=[40, 40],
        )
        architecture = "MiniCPMOTTSDVAEEncoder"
    elif component == "tts_vocos":
        from xh_model_zoo.xh_llm.models.minicpmo.minicpmo_tts_vocos_convert_config import (
            MinicpmoTTSVocosConvertConfig,
        )

        config = MinicpmoTTSVocosConvertConfig(
            quant_scheme=quant_scheme,
            video=video_path,
            audio=audio_path,
            debug=args.debug,
            valid=args.valid,
            image_slice_max_size=[40, 40],
        )
        architecture = "MiniCPMOTTSVOCOS"
    else:
        raise ValueError(f"Unsupported component: {component}")

    prefix = f"{model_name}-{component}-{quant_type}"
    work_dir = Path("work_dirs") / prefix
    work_dir.mkdir(exist_ok=True, parents=True)
    log_file = work_dir / f"{component}-convert.log"
    xhquant_init(log_file, debug=args.debug)
    logger = get_root_logger()
    with TimeProfiler("convert", logger), MemoryTracker("cuda:0", "convert", logger):
        LLMConverter.from_pretrained(hf_model_path, architecture, config, str(work_dir))

    cleanup_cuda()


def move_models(
    work_dir: Path,
    source: str = "prefill",
    model: str = "prefill",
    target_name: str = "hmquant_minicpmo_with_act.onnx",
):
    source_dir = work_dir / "hmquant/{}".format(source)
    matched_files = list(source_dir.glob("*{}.onnx".format(model)))

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
    def remove_meta_info(info_path: Path):
        if info_path.exists():
            info_path.unlink()

    work_dir = Path(args.work_dir)
    dest_dir = Path(args.out_dir)
    model_name = args.model_name
    hm_model_name = "hmquant_{}_with_act.onnx".format(args.model_name)
    START_MOVE_MSG = "Start move from {} to {}"
    ### visual ###
    hmm_model_dir = "{}-vision-{}".format(model_name, args.quant_type)
    logger.info(
        msg_output_format(START_MOVE_MSG).format(work_dir / hmm_model_dir, dest_dir)
    )
    shutil.move(work_dir / hmm_model_dir / "hmonnx/vision", dest_dir / "hmquant/visual")
    move_models(dest_dir, "visual", "vision", target_name=hm_model_name)
    remove_meta_info(Path(dest_dir / "hmquant/visual/meta_info.json"))
    ### audio ###
    hmm_model_dir = "{}-audio-{}".format(model_name, args.quant_type)
    logger.info(
        msg_output_format(START_MOVE_MSG).format(work_dir / hmm_model_dir, dest_dir)
    )
    shutil.move(work_dir / hmm_model_dir / "hmonnx/audio", dest_dir / "hmquant/audio")
    move_models(dest_dir, "audio", "audio", target_name=hm_model_name)
    remove_meta_info(Path(dest_dir / "hmquant/audio/meta_info.json"))
    ### llm ###
    hmm_model_dir = "{}-llm-{}".format(model_name, args.quant_type)
    logger.info(
        msg_output_format(START_MOVE_MSG).format(work_dir / hmm_model_dir, dest_dir)
    )
    shutil.move(
        work_dir / hmm_model_dir / "hmonnx/llm/prefill_onnx",
        dest_dir / "hmquant/prefill",
    )
    move_models(dest_dir, "prefill", "llm_prefill", target_name=hm_model_name)
    shutil.move(
        work_dir / hmm_model_dir / "hmonnx/llm/decode_onnx",
        dest_dir / "hmquant/decoder",
    )
    move_models(dest_dir, "decoder", "llm_decode", target_name=hm_model_name)
    shutil.move(
        work_dir / hmm_model_dir / "token_embedding.pt",
        dest_dir / "hmquant/quant_embedding.pt",
    )
    ### tts ###
    hmm_model_dir = "{}-tts-{}".format(model_name, args.quant_type)
    logger.info(
        msg_output_format(START_MOVE_MSG).format(work_dir / hmm_model_dir, dest_dir)
    )
    shutil.move(
        work_dir / hmm_model_dir / "hmonnx/tts/prefill_onnx",
        dest_dir / "hmquant/tts_prefill",
    )
    move_models(dest_dir, "tts_prefill", "tts_prefill", target_name=hm_model_name)
    shutil.move(
        work_dir / hmm_model_dir / "hmonnx/tts/decode_onnx",
        dest_dir / "hmquant/tts_decoder",
    )
    move_models(dest_dir, "tts_decoder", "tts_decode", target_name=hm_model_name)
    shutil.move(
        work_dir / hmm_model_dir / "quant_embedding_tts.pt",
        dest_dir / "hmquant/quant_embedding_tts.pt",
    )
    for i in range(4):
        shutil.move(
            work_dir / hmm_model_dir / f"quant_embedding_tts_code_{i}.pt",
            dest_dir / f"hmquant/quant_embedding_tts_code_{i}.pt",
        )
    ### dvae ###
    hmm_model_dir = "{}-tts_dvae-{}".format(model_name, args.quant_type)
    logger.info(
        msg_output_format(START_MOVE_MSG).format(work_dir / hmm_model_dir, dest_dir)
    )
    shutil.move(work_dir / hmm_model_dir / "hmonnx/tts_dvae", dest_dir / "hmquant/dvae")
    move_models(
        dest_dir, "dvae", "dvae_part1", target_name="hmquant_dvae_part1_with_act.onnx"
    )
    move_models(
        dest_dir, "dvae", "dvae_part2", target_name="hmquant_dvae_part2_with_act.onnx"
    )
    remove_meta_info(Path(dest_dir / "hmquant/dvae/meta_info.json"))
    ### vocos ###
    hmm_model_dir = "{}-tts_vocos-{}".format(model_name, args.quant_type)
    logger.info(
        msg_output_format(START_MOVE_MSG).format(work_dir / hmm_model_dir, dest_dir)
    )
    shutil.move(
        work_dir / hmm_model_dir / "hmonnx/tts_vocos", dest_dir / "hmquant/vocos"
    )
    move_models(dest_dir, "vocos", "vocos", target_name="hmquant_vocos_with_act.onnx")
    remove_meta_info(Path(dest_dir / "hmquant/vocos/meta_info.json"))

    logger.info(msg_output_format("Start remove work_dir: {}".format(work_dir)))
    shutil.rmtree(work_dir, ignore_errors=True)


def get_default_model_dir(model_config: dict) -> str:
    repo_ids = model_config.get("modelscope_repo", [])
    if repo_ids:
        return repo_ids[0].rsplit("/", maxsplit=1)[-1]
    model_name = model_config.get("model_name", "minicpmo")
    model_size = model_config.get("model_size", "7b")
    return f"{model_name}-{model_size}"


def parse_args():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "--config",
        type=str,
        default=DEFAULT_CONFIG_PATH,
        help="path to config.yaml",
    )
    parser.add_argument("--model", type=str, default=None, help="input hf model path")
    parser.add_argument(
        "--model-name",
        type=str,
        default=None,
        help="output hmonnx model name",
    )
    parser.add_argument("--model-size", type=str, default=None, help="model size")
    parser.add_argument("--work-dir", type=str, default="work_dirs/")
    parser.add_argument("--out-dir", type=str, default="output/{}".format(HOUMO_TARGET))
    parser.add_argument("--valid", type=bool, default=False, help="check hmonnx mode")
    parser.add_argument("--debug", action="store_true", help="debug mode")
    parser.add_argument(
        "--context-length", type=str, default=None, help="max sequence length"
    )
    parser.add_argument(
        "--input-sequence-length",
        type=int,
        default=None,
        help="input sequence length",
    )
    parser.add_argument(
        "--quant-type",
        default=None,
        help="quant type, default is w8a8h0_sefp",
    )
    args = parser.parse_args()

    default_model_size, default_model_name, model_configs = get_model_configs(
        args.config
    )
    args.model_name = first_not_none(args.model_name, default_model_name)
    args.model_size = first_not_none(args.model_size, default_model_size)
    model_config = model_configs.get(args.model_name, {}).get(args.model_size, {})
    args.model = first_not_none(args.model, get_default_model_dir(model_config))
    args.context_length = parse_context_length(
        first_not_none(args.context_length, model_config.get("context_length", "4k"))
    )
    args.input_sequence_length = first_not_none(
        args.input_sequence_length, model_config.get("prefill_length", 256)
    )
    args.quant_type = first_not_none(
        args.quant_type, model_config.get("quant_type", "w8a8h0_sefp")
    )
    return args


if __name__ == "__main__":
    assert check_gpu() is True, "Error: Not found GPU device."

    args = parse_args()
    with ProcessMemoryMonitor(interval=2, quiet=True) as monitor:
        for component in ["vision", "audio", "llm", "tts", "tts_dvae", "tts_vocos"]:
            houmo_export_llm(args, component=component)
        move_hmonnx(args)
    print(
        f"\n=== Quantization completed. Peak memory: {monitor.peak_memory_mb:.2f} MB ==="
    )
