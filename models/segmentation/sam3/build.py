# Copyright (c) 2026 HOUMO AI
#
# File: build.py
# Description:
#   SAM3 Model Build Tool - Python script for building SAM3 hmonnx models.
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
import multiprocessing
import os

from hmatc.exec.xh2_exec import Xh2Exec
from hmatc.utils.monitor import ProcessMemoryMonitor
from hmatc.utils.utils import (
	find_hmonnx_file,
	first_not_none,
	get_model_configs,
	get_platform,
)

HOUMO_TARGET = os.getenv("HOUMO_TARGET")
assert HOUMO_TARGET in ["xh2"], f"Unsupported HOUMO_TARGET: {HOUMO_TARGET}"

DEFAULT_MODEL_NAME = "sam3"
DEFAULT_MODEL_DIR = os.path.join("output", HOUMO_TARGET, "hmquant")
DEFAULT_OUTPUT_DIR = os.path.join("output", HOUMO_TARGET)
DEFAULT_CONFIG_PATH = os.path.join(os.path.dirname(__file__), "config.yml")


def get_args() -> argparse.Namespace:
	parser = argparse.ArgumentParser()
	parser.add_argument(
		"--config",
		dest="config_path",
		type=str,
		default=DEFAULT_CONFIG_PATH,
		help="path to config.yml",
	)
	parser.add_argument(
		"--model_dir",
		dest="model_dir",
		type=str,
		default=DEFAULT_MODEL_DIR,
		help="path to the hmonnx model directory",
	)
	parser.add_argument(
		"--model_name",
		dest="model_name",
		type=str,
		default=None,
		help="output houmo model name",
	)
	parser.add_argument(
		"--model_size",
		dest="model_size",
		type=str,
		default=None,
		help="model size",
	)
	parser.add_argument(
		"--output_dir",
		dest="output_dir",
		type=str,
		default=DEFAULT_OUTPUT_DIR,
		help="build output directory",
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
		help="device number",
	)
	parser.add_argument(
		"--quant_type",
		dest="quant_type",
		type=str,
		default=None,
		help="quantization type used by the HMONNX model",
	)
	parser.add_argument(
		"--max_size_w", type=int, default=None, help="maximum image width"
	)
	parser.add_argument(
		"--max_size_h", type=int, default=None, help="maximum image height"
	)
	parser.add_argument(
		"--j",
		dest="j",
		type=int,
		default=int(multiprocessing.cpu_count() * 0.7),
		help="build parallel jobs",
	)
	parser.add_argument(
		"--stage",
		dest="stage",
		type=str,
		default="build",
		choices=["build", "all"],
		help="build stage",
	)
	parser.add_argument(
		"--flash_attention",
		dest="flash_attention",
		type=int,
		default=1,
		choices=[0, 1, 2],
		help="FlashAttention optimization switch for vision-style model build",
	)
	args = parser.parse_args()
	default_model_size, default_model_name, model_configs = get_model_configs(
		args.config_path
	)
	args.model_name = first_not_none(
		args.model_name, default_model_name, DEFAULT_MODEL_NAME
	)
	args.model_size = first_not_none(args.model_size, default_model_size)
	model_config = model_configs.get(args.model_name, {}).get(args.model_size, {})
	args.ncore = first_not_none(
		args.ncore,
		model_config.get("ncore", int(os.getenv("HOUMO_CORE_NUM", 2))),
	)
	args.ndevice = first_not_none(args.ndevice, model_config.get("ndevice", 1))
	args.quant_type = first_not_none(
		args.quant_type, model_config.get("quant_type", "w8a8")
	)
	args.max_size_w = first_not_none(
		args.max_size_w, model_config.get("max_size_w", 1008)
	)
	args.max_size_h = first_not_none(
		args.max_size_h, model_config.get("max_size_h", 1008)
	)
	return args


if __name__ == "__main__":
	args = get_args()
	print(args)

	with ProcessMemoryMonitor(interval=2, quiet=True) as monitor:
		if args.stage in ["build", "all"]:
			assert (
				get_platform() == "x86_64"
			), "Only supported for compilation on the x86_64 platform."

			hmm_name = (
				f"{args.model_name}_{args.model_size}_"
				f"{args.max_size_w}x{args.max_size_h}"
			)
			hmonnx_name = f"hmquant_{hmm_name}_{args.quant_type}.onnx"
			Xh2Exec.build_from_hmonnx(
				hmonnx=find_hmonnx_file(
					args.model_dir,
					pattern=hmonnx_name,
				),
				hmm_name=hmm_name,
				output=args.output_dir,
				ncore=args.ncore,
				ndevice=args.ndevice,
				flash_attn=args.flash_attention,
				parallel_jobs=args.j,
				target=HOUMO_TARGET,
			)

	print(
		f"\n=== Build completed. Peak memory: {monitor.peak_memory_mb:.2f} MB ==="
	)
