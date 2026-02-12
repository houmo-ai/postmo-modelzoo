# Copyright (c) 2025 HOUMO AI
#
# File: test_perf_models.py
# Description:
#   Model Download And Run llm_perf tool Script for Houmo AI LLM and vLLM models.
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
from typing import List, Dict, Union
import sys
HOUMO_EXAMPLES_PATH = os.environ.get("HOUMO_EXAMPLES_PATH", "../..")
sys.path.insert(0, f"{HOUMO_EXAMPLES_PATH}/hmatc")
from loguru import logger
from hmatc.utils.utils import get_file_from_jfrog

import torch
import numpy as np
import shutil
import tcim_lite as tcim

total_device_num = tcim.runtime.get_device_num()
HOUMO_TARGET = os.getenv("HOUMO_TARGET")
assert HOUMO_TARGET in ["xh1", "xh2"], f"Unsupported HOUMO_TARGET: {HOUMO_TARGET}"
DECODE_STOP = 256
ARTIFACTORY_URL = os.getenv("HOUMO_MODELZOO_URL", None)
assert ARTIFACTORY_URL is not None, f"Get ARTIFACTORY_URL Failed!"
VERSION = os.getenv("HOUMO_VERSION", None)
assert VERSION is not None, f"Get VERSION Failed!"
TARGET_PATH = f"{ARTIFACTORY_URL}/models/xh2-v{VERSION}"
os.environ["HDPL_API_TIMEOUT"] = "10000000"

# llm_perf test model lists.
all_test_models = [
    f"{TARGET_PATH}/deepseek/hmm_xh2_deepseek_8b_256_32k_b1_1chip_2cores_v1.0.0.zip",
    f"{TARGET_PATH}/qwen2.5/hmm_xh2_qwen2.5_7b_256_32k_b1_1chip_2cores_v1.0.0.zip",
    f"{TARGET_PATH}/qwen2.5-vl/hmm_xh2_qwen2.5-vl_7b_256_8k_b1_1chip_2cores_v1.0.0.zip",
    f"{TARGET_PATH}/qwen3/hmm_xh2_qwen3_8b_256_16k_b4_1chip_2cores_v1.0.0.zip",
    f"{TARGET_PATH}/qwen3-vl/hmm_xh2_qwen3-vl_8b_256_32k_b1_1chip_2cores_v1.0.0.zip"
]

def parse_hmm_zip_filename(
    filename: Union[str, List[str]]
) -> Union[Dict[str, str], List[Dict[str, str]]]:
    base_fields = [
        "target", "model_name", "model_size", "prefill_len",
        "context_len", "batch", "ndevice", "ncore", "version"
    ]

    def _parse_single(filename: str) -> Dict[str, str]:
        filename = filename.split('/')[-1]
        if filename.lower().endswith(".zip"):
            filename = filename[:-4]
        parts = filename.split("_")
        if parts[0] != "hmm":
            raise ValueError(f"Filename must start with 'hmm_! Filename: {filename}")

        result = {}
        idx = 1
        result["target"] = parts[idx]
        idx += 1
        result["model_name"] = parts[idx]
        idx += 1
        result["model_size"] = parts[idx]
        idx += 1

        while idx < len(parts) - 6:
            if "extension" not in result:
                result["extension"] = parts[idx]
            else:
                result["extension"] += f"_{parts[idx]}"
            idx += 1
        result["prefill_len"] = parts[idx]
        idx += 1
        result["context_len"] = parts[idx]
        idx += 1

        batch_val = parts[idx]
        if not batch_val.startswith("b"):
            raise ValueError(
                f"Invalid batch format! Expected 'b+number' (e.g., b1), got {batch_val}. "
                f"Filename: {filename}"
            )
        result["batch"] = batch_val[1:]
        idx += 1

        result["ndevice"] = parts[idx]
        idx += 1
        result["ncore"] = parts[idx]
        idx += 1
        result["version"] = parts[idx]

        return result

    if isinstance(filename, list):
        return [_parse_single(f) for f in filename]
    elif isinstance(filename, str):
        return _parse_single(filename)
    else:
        raise TypeError("Input must be str (single filename) or list (multiple filenames)")


def find_local(model_path: str, key_string: str) -> str:
    if not os.path.isdir(model_path):
        print(f"Error: Directory not found - {model_path}")
        return None

    match_files = []
    for root, dirs, files in os.walk(model_path):
        for file in files:
            if key_string in file:
                match_files.append(file)

    return match_files[0] if match_files else None

if __name__ == "__main__":
    model_url_lists = []
    relative_model_paths = dict()
    model_infos = dict()

    for idx, model_url in enumerate(all_test_models):
        res = parse_hmm_zip_filename(model_url)
        root_model_dir_name = "customized_" + f"{res['target']}_"  + f"{res['version']}"
        zip_dir = f"customized_tmpzips"
        sub_model_dir_name = ""
        for key, value in res.items():
            if key not in ['target', 'version']:
                if key == 'batch':
                    sub_model_dir_name += 'b'
                sub_model_dir_name = sub_model_dir_name + value
                if key != 'ncore': sub_model_dir_name += '_'
        try:
            context_length = int(res['context_len'].replace('k', '')) * 1024
        except:
            continue
        relative_model_paths[sub_model_dir_name] = {
            "model_path": os.path.join(root_model_dir_name, sub_model_dir_name),
            "download_url": model_url,
            "batch": res["batch"],
            "ndevices": res['ndevice'].replace("chips", "") if "chips" in res['ndevice'] else res['ndevice'].replace("chip", ""),
            "loop" : "1",
            "input": str(DECODE_STOP),
            "stop" : str(DECODE_STOP) if context_length > DECODE_STOP else context_length
        }
        if int(relative_model_paths[sub_model_dir_name]['ndevices']) > total_device_num:
            continue
        logger.info(f"Local Model Path: {relative_model_paths[sub_model_dir_name]['model_path']}")
        get_file_from_jfrog(relative_model_paths[sub_model_dir_name]["download_url"], zip_dir, relative_model_paths[sub_model_dir_name]["model_path"])
        logger.info(f"Local dirs files is {os.listdir(relative_model_paths[sub_model_dir_name]['model_path'])}, prefill file is {find_local(relative_model_paths[sub_model_dir_name]['model_path'], 'prefill')}")
        prefill_path = os.path.join(relative_model_paths[sub_model_dir_name]["model_path"], find_local(relative_model_paths[sub_model_dir_name]["model_path"], "prefill"))
        decode_path = os.path.join(relative_model_paths[sub_model_dir_name]["model_path"], find_local(relative_model_paths[sub_model_dir_name]["model_path"], "decode"))
        visual_path = find_local(relative_model_paths[sub_model_dir_name]["model_path"], "visual")
        if visual_path is not None:
            visual_path = os.path.join(relative_model_paths[sub_model_dir_name]["model_path"], visual_path)
        embedding_path = os.path.join(relative_model_paths[sub_model_dir_name]["model_path"], "hmquant/quant_embedding.pt")
        type = "llm" if "vl" not in sub_model_dir_name else "vllm"
        if os.path.exists(embedding_path) and embedding_path.endswith(".pt"):
            if type == "llm":
                embedding_weight = torch.load(
                    embedding_path, map_location="cpu", weights_only=True
                )
                embedding_weight = embedding_weight['weight']
            if type == "vllm":
                embedding_weight = torch.load(embedding_path, map_location="cpu", weights_only=False)
                if HOUMO_TARGET == "xh2":
                    embedding_weight = embedding_weight.weight
            if embedding_weight.dtype == torch.bfloat16:
                embedding_weight = embedding_weight.float().half()

            embedding_data = embedding_weight.detach().cpu().numpy()
            embedding_data.tofile(embedding_path.replace(".pt", ".bin"))
        perf_config = {}
        perf_config["ModelName"] = sub_model_dir_name
        perf_config["prefill"] = prefill_path
        perf_config["decode"] = decode_path
        if visual_path is not None:
            perf_config["visual"] = visual_path
        perf_config["embedding"] =  os.path.join(relative_model_paths[sub_model_dir_name]["model_path"], "hmquant/quant_embedding.bin")
        perf_config["input"] = relative_model_paths[sub_model_dir_name]["input"]
        perf_config["stop"] = relative_model_paths[sub_model_dir_name]["stop"]
        perf_config["ndevices"] = relative_model_paths[sub_model_dir_name]["ndevices"]
        perf_config["loop"] = "2"
        perf_config["batch"] = relative_model_paths[sub_model_dir_name]["batch"]
        import subprocess
        args = ["llm_perf", "--prefill", perf_config["prefill"], "--decode", perf_config["decode"]]
        if visual_path is not None:
            args.append("--visual")
            args.append(perf_config["visual"])
        args.append("--embedding")
        args.append(perf_config["embedding"])
        args.append("--input")
        args.append(perf_config["input"])
        args.append("--stop")
        args.append(perf_config["stop"])
        args.append("--ndevices")
        args.append(perf_config["ndevices"])
        args.append("--loop")
        args.append(perf_config["loop"])
        args.append("--batch")
        args.append(perf_config["batch"])
        args.append("--LazyMode")
        args.append("--no_warm_up")

        logger.info(f"Running llm_perf for model: {perf_config['ModelName']}")
        process = subprocess.Popen(
            args,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8"
        )
        for line in process.stdout:
            logger.info(line.strip())
        shutil.rmtree(zip_dir)
        shutil.rmtree(relative_model_paths[sub_model_dir_name]["model_path"])