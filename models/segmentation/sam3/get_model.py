# Copyright (c) 2026 HOUMO AI
#
# File: get_model.py
# Description:
#   Download SAM3 raw or compiled model resources.
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

from hmatc.utils.utils import (
	first_not_none,
	get_model_configs,
	hmatc_get_file,
)

HOUMO_TARGET = os.getenv("HOUMO_TARGET")
assert HOUMO_TARGET in ["xh2"], f"Unsupported HOUMO_TARGET: {HOUMO_TARGET}"

DEFAULT_CONFIG_PATH = os.path.join(os.path.dirname(__file__), "config.yml")
HMM_RESOURCE_VERSION = "v1.5.0"


def get_default_model_dir(model_config: dict) -> str:
	repo_ids = model_config.get("modelscope_repo", [])
	if repo_ids:
		return repo_ids[0].rsplit("/", maxsplit=1)[-1]
	return model_config.get("model_name", "sam3")


def get_args() -> argparse.Namespace:
	"""Parse commandline."""
	parser = argparse.ArgumentParser()
	parser.add_argument(
		"--config",
		dest="config_path",
		type=str,
		default=DEFAULT_CONFIG_PATH,
		help="path to config.yml",
	)
	parser.add_argument(
		"--type",
		dest="file_type",
		type=str,
		default="hmm",
		choices=["raw", "hmm"],
		help="which resource to get, choice in [raw, hmm]",
	)
	parser.add_argument(
		"--download_dir",
		dest="download_dir",
		type=str,
		default=".",
		help="where to save downloaded model",
	)
	parser.add_argument(
		"--extract_dir",
		dest="extract_dir",
		type=str,
		default=None,
		help="where to save extracted files",
	)
	parser.add_argument(
		"--source_type",
		dest="source_type",
		type=str,
		default="jfrog",
		choices=["jfrog", "modelscope"],
		help="download the model from which source",
	)
	parser.add_argument("--model_name", type=str, default=None, help="model name")
	parser.add_argument("--model_size", type=str, default=None, help="model size")
	parser.add_argument("--ncore", type=int, default=None, help="number of cores")
	parser.add_argument("--ndevice", type=int, default=None, help="device number")
	parser.add_argument("--batch", type=int, default=None, help="batch size")
	parser.add_argument(
		"--quant_type", type=str, default=None, help="quantization type"
	)
	parser.add_argument(
		"--max_size_w", type=int, default=None, help="maximum image width"
	)
	parser.add_argument(
		"--max_size_h", type=int, default=None, help="maximum image height"
	)
	return parser.parse_args()


if __name__ == "__main__":
	args = get_args()
	default_model_size, default_model_name, model_configs = get_model_configs(
		args.config_path
	)
	model_name = first_not_none(args.model_name, default_model_name)
	model_size = first_not_none(args.model_size, default_model_size)
	model_config = model_configs.get(model_name, {}).get(model_size, {})
	ncore = first_not_none(
		args.ncore, model_config.get("ncore", int(os.getenv("HOUMO_CORE_NUM", 2)))
	)
	ndevice = first_not_none(args.ndevice, model_config.get("ndevice", 1))
	batch = first_not_none(args.batch, model_config.get("batch", 1))
	quant_type = first_not_none(
		args.quant_type, model_config.get("quant_type", "w8a8")
	)
	max_size_w = first_not_none(
		args.max_size_w, model_config.get("max_size_w", 1008)
	)
	max_size_h = first_not_none(
		args.max_size_h, model_config.get("max_size_h", 1008)
	)
	default_model_dir = get_default_model_dir(model_config)
	version = HMM_RESOURCE_VERSION

	model_cfgs = {
		"target": HOUMO_TARGET,
		"version": version,
		"model_type": "llm",
		"model_name": model_name,
		"model_info": {
			"model_size": model_config.get("model_size", model_size),
			"ncore": ncore,
			"ndevice": ndevice,
			"batch": batch,
			"quant_type": quant_type,
			"max_size_w": max_size_w,
			"max_size_h": max_size_h,
		},
		"modelscope_repo": {
			"repo_ids": model_config.get("modelscope_repo", []),
			"local_dirs": [os.path.join(args.download_dir, default_model_dir)],
			"ignore_patterns": ["*.safetensors", "*.pt"],
		},
	}

	_, ret_dict = hmatc_get_file(
		model_cfgs,
		args.file_type,
		args.download_dir,
		args.extract_dir,
		args.source_type,
	)
	if ret_dict.get("ret", False) is False:
		exit(1)