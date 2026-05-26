# Copyright (c) 2025 HOUMO AI
#
# File: build.py
# Description:
#  Qwen3 Speculative  Model Build and Test Tool - Python script for building and testing
# Qwen3 Speculative models.
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
import sys
import numpy as np
import multiprocessing
import argparse
import glob

HOUMO_EXAMPLES_PATH = os.environ.get("HOUMO_EXAMPLES_PATH", "../../..")
sys.path.insert(0, f"{HOUMO_EXAMPLES_PATH}/hmatc")
from hmatc.exec.xh2_exec import Xh2Exec
from hmatc.utils.utils import find_hmonnx_file, get_platform

HOUMO_TARGET = os.getenv("HOUMO_TARGET")
assert HOUMO_TARGET in ["xh2"], f"Unsupported HOUMO_TARGET: {HOUMO_TARGET}"

HOUMO_CORE_NUM = int(os.getenv("HOUMO_CORE_NUM", 2))
GOLDEN_THRESH = 0.98


def sanitize_name(name: str):
    return name.replace(":", "_").replace("/", "_")


def cosine_distance(data1, data2):
    if data1.shape != data2.shape:
        print(f"[error] shape not equal {data1.shape} vs {data2.shape}")
        return -1
    v1_d = data1.flatten().astype("float64")
    v2_d = data2.flatten().astype("float64")
    v1_d[v1_d == np.inf] = np.finfo(np.float16).max
    v2_d[v2_d == np.inf] = np.finfo(np.float16).max
    v1_d[v1_d == -np.inf] = np.finfo(np.float16).min
    v2_d[v2_d == -np.inf] = np.finfo(np.float16).min
    v1_norm = v1_d / np.linalg.norm(v1_d)
    v2_norm = v2_d / np.linalg.norm(v2_d)
    cosine_dist = np.dot(v1_norm, v2_norm)
    if np.isnan(cosine_dist):
        return -1
    return cosine_dist


def get_args() -> argparse.Namespace:
    """Parse commandline."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_dir",
        dest="model_dir",
        type=str,
        default=os.path.join("output", HOUMO_TARGET, "target", "hmquant"),
        help="path to the model dir",
    )
    parser.add_argument(
        "--draft_model_dir",
        dest="draft_model_dir",
        type=str,
        default=os.path.join("output", HOUMO_TARGET, "draft", "hmquant"),
        help="path to the model dir",
    )
    parser.add_argument(
        "--model_name",
        dest="model_name",
        type=str,
        default="qwen3-speculative",
        help="output houmo model name",
    )
    parser.add_argument(
        "--model_size",
        dest="model_size",
        type=str,
        default="14b",
        help="output houmo model size",
    )
    parser.add_argument(
        "--draft_model_size",
        dest="draft_model_size",
        type=str,
        default="0.6b",
        help="output houmo draft model size",
    )
    parser.add_argument(
        "--batch",
        dest="batch",
        type=int,
        default=1,
        help="batch size",
    )
    parser.add_argument(
        "--j",
        dest="j",
        type=int,
        default=multiprocessing.cpu_count(),
        help="build parallel jobs",
    )
    parser.add_argument(
        "--ncore",
        dest="ncore",
        type=int,
        default=HOUMO_CORE_NUM,
        help="core number",
    )
    parser.add_argument(
        "--context_length",
        dest="context_length",
        type=int,
        default=32768,
        help="context_length",
    )
    parser.add_argument(
        "--ndevice",
        dest="ndevice",
        type=int,
        default=1,
        help="device number",
    )
    parser.add_argument(
        "--output_dir",
        dest="output_dir",
        type=str,
        default=os.path.join("output", HOUMO_TARGET),
        help="build output dir",
    )
    parser.add_argument(
        "--verify_length",
        dest="verify_length",
        type=int,
        default=5,
        help="verify_length",
    )
    parser.add_argument(
        "--flash_attention",
        dest="flash_attention",
        type=int,
        default=2,
        choices=[0, 1, 2],
        help="flash attention optimization",
    )

    args = parser.parse_args()
    if args.context_length < 2048:
        args.flash_attention = 0
    return args


def _get_decode_dir(model_dir):
    decode_dirs = sorted(
        path
        for path in glob.glob(os.path.join(model_dir, "*decode*"))
        if os.path.isdir(path)
    )
    if not decode_dirs:
        raise FileNotFoundError(
            f'No subdirectory containing "decode" found under: {model_dir}'
        )
    decode_dir = os.path.abspath(decode_dirs[0])

    return decode_dir


def _copy_embedding_if_exists(source_model_dir, target_model_dir, copied_name):
    import shutil

    source_embedding_path = os.path.join(source_model_dir, "quant_embedding.pt")
    copied_embedding_path = os.path.join(target_model_dir, copied_name)
    if os.path.exists(source_embedding_path):
        if os.path.exists(copied_embedding_path):
            os.remove(copied_embedding_path)
        shutil.copy2(source_embedding_path, copied_embedding_path)


if __name__ == "__main__":
    args = get_args()
    print(args)

    target_model_dir = args.model_dir
    draft_model_dir = args.draft_model_dir
    model_name = args.model_name
    output_dir = args.output_dir
    ncore = args.ncore
    batch = args.batch
    ndevice = args.ndevice
    j = args.j
    context_length = args.context_length

    target_prefill_dir = os.path.join(target_model_dir, "prefill")
    draft_decode_dir = _get_decode_dir(draft_model_dir)
    draft_prefill_dir = os.path.join(draft_model_dir, "prefill")

    assert (
        get_platform() == "x86_64"
    ), "Only supported for compilation on the x86_64 platform."

    Xh2Exec.build_from_hmonnx(
        hmonnx=find_hmonnx_file(draft_prefill_dir),
        hmm_name=f"{model_name}-{args.draft_model_size}_prefill_draft",
        output=output_dir,
        ncore=ncore,
        llm_opt=True,
        flash_attn=args.flash_attention,
        context_length=context_length,
        ndevice=ndevice,
        is_prefill=True,
        parallel_jobs=j,
    )
    Xh2Exec.build_from_hmonnx(
        hmonnx=find_hmonnx_file(draft_decode_dir),
        hmm_name=f"{model_name}-{args.draft_model_size}_decode_draft",
        output=output_dir,
        ncore=ncore,
        llm_opt=True,
        llm_batch=batch,
        flash_attn=args.flash_attention,
        context_length=context_length,
        ndevice=ndevice,
        parallel_jobs=j,
    )
    Xh2Exec.build_from_hmonnx(
        hmonnx=find_hmonnx_file(target_prefill_dir),
        hmm_name=f"{model_name}-{args.model_size}_prefill",
        output=output_dir,
        ncore=ncore,
        llm_opt=True,
        flash_attn=args.flash_attention,
        context_length=context_length,
        ndevice=ndevice,
        is_prefill=True,
        parallel_jobs=j,
    )
    Xh2Exec.build_from_hmonnx(
        hmonnx=find_hmonnx_file(target_prefill_dir),
        hmm_name=f"{model_name}-{args.model_size}_verify",
        output=output_dir,
        ncore=ncore,
        llm_opt=True,
        flash_attn=args.flash_attention,
        context_length=context_length,
        prefill_length=args.verify_length,
        all_logits=True,
        ndevice=ndevice,
        is_prefill=True,
        parallel_jobs=j,
    )
    _copy_embedding_if_exists(
        target_model_dir, target_model_dir, "quant_embedding_target.pt"
    )
    _copy_embedding_if_exists(
        draft_model_dir, target_model_dir, "quant_embedding_draft.pt"
    )

    print("\n=== All builds completed. ===")
