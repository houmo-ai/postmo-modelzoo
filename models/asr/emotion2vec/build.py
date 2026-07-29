# Copyright (c) 2025 HOUMO AI
#
# File: build.py
# Description:
#   emotion2vec HMM model build tool.
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

import psutil

from hmatc.exec.xh2_exec import Xh2Exec
from hmatc.utils.monitor import ProcessMemoryMonitor
from hmatc.utils.utils import (
    find_hmonnx_file,
    first_not_none,
    get_model_configs,
    get_platform,
)

HOUMO_TARGET = os.getenv("HOUMO_TARGET", "xh2")
assert HOUMO_TARGET in ["xh2"], f"Unsupported HOUMO_TARGET: {HOUMO_TARGET}"

HOUMO_CORE_NUM = int(os.getenv("HOUMO_CORE_NUM", 2))
DEFAULT_CONFIG_PATH = os.path.join(os.path.dirname(__file__), "config.yaml")
DEFAULT_MODEL_DIR = os.path.join("output", HOUMO_TARGET, "hmquant")
DEFAULT_OUTPUT_DIR = os.path.join("output", HOUMO_TARGET)


def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build the emotion2vec HMM from its quantized HMONNX model."
    )
    parser.add_argument(
        "--config",
        dest="config_path",
        type=str,
        default=DEFAULT_CONFIG_PATH,
        help="path to config.yaml",
    )
    parser.add_argument(
        "--model_dir",
        dest="model_dir",
        type=str,
        default=DEFAULT_MODEL_DIR,
        help="path to the quantized HMONNX model directory",
    )
    parser.add_argument(
        "--model_name",
        dest="model_name",
        type=str,
        default=None,
        help="output model name",
    )
    parser.add_argument(
        "--model_size",
        dest="model_size",
        type=str,
        default=None,
        help="output model size",
    )
    parser.add_argument(
        "--output_dir",
        dest="output_dir",
        type=str,
        default=DEFAULT_OUTPUT_DIR,
        help="output directory for the built HMM",
    )
    parser.add_argument(
        "--ncore",
        dest="ncore",
        type=int,
        default=None,
        help="core number",
    )
    parser.add_argument(
        "--ndevice",
        dest="ndevice",
        type=int,
        default=None,
        help="device number for multi-device",
    )
    parser.add_argument(
        "--j",
        dest="parallel_jobs",
        type=int,
        default=psutil.cpu_count(logical=False) or 1,
        help="build parallel jobs",
    )
    parser.add_argument(
        "--opt_level",
        dest="opt_level",
        type=int,
        default=2,
        choices=[0, 1, 2, 3],
        help="compiler optimization level",
    )
    args = parser.parse_args()

    default_model_size, default_model_name, model_configs = get_model_configs(
        args.config_path
    )
    args.model_name = first_not_none(args.model_name, default_model_name)
    args.model_size = first_not_none(args.model_size, default_model_size)
    model_config = model_configs.get(args.model_name, {}).get(args.model_size, {})
    if not model_config:
        raise ValueError(
            f"Model configuration not found: {args.model_name}/{args.model_size} "
            f"in {args.config_path}"
        )
    args.ncore = first_not_none(args.ncore, model_config.get("ncore", HOUMO_CORE_NUM))
    args.ndevice = first_not_none(args.ndevice, model_config.get("ndevice", 1))
    return args


def main() -> None:
    args = get_args()
    if get_platform() != "x86_64":
        raise RuntimeError("Only supported for compilation on the x86_64 platform.")

    hmonnx_path = os.path.join(args.model_dir, "model")
    hmonnx = find_hmonnx_file(hmonnx_path)
    if not hmonnx:
        raise FileNotFoundError(f"HMONNX model not found under {hmonnx_path}")

    hmm_name = f"{args.model_name}-{args.model_size}"
    with ProcessMemoryMonitor(interval=2, quiet=True) as monitor:
        hmm_path = Xh2Exec.build_from_hmonnx(
            hmonnx=hmonnx,
            hmm_name=hmm_name,
            output=args.output_dir,
            ncore=args.ncore,
            ndevice=args.ndevice,
            opt_level=args.opt_level,
            parallel_jobs=args.parallel_jobs,
            target=HOUMO_TARGET,
        )

    if not os.path.isfile(hmm_path):
        raise RuntimeError(f"Build did not produce the expected HMM: {hmm_path}")

    classifier_src = os.path.join(args.model_dir, "quant_embedding.pt")
    if os.path.isfile(classifier_src):
        classifier_dir = os.path.join(args.output_dir, "hmquant")
        os.makedirs(classifier_dir, exist_ok=True)
        classifier_dst = os.path.join(classifier_dir, "quant_embedding.pt")
        if not os.path.exists(classifier_dst) or not os.path.samefile(
            classifier_src, classifier_dst
        ):
            shutil.copy2(classifier_src, classifier_dst)

    print(f"Built HMM: {hmm_path}")
    print(f"Peak memory: {monitor.peak_memory_mb:.2f} MB")


if __name__ == "__main__":
    main()
