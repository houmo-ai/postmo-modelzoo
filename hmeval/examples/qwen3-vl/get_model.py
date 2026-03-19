# Copyright 2025 HOUMO AI
#
# SPDX-License-Identifier: Apache-2.0

import os
from hmatc.utils.utils import hmatc_get_file, get_houmo_version

HOUMO_TARGET = os.getenv("HOUMO_TARGET")
assert HOUMO_TARGET in ["xh2"], f"Unsupported HOUMO_TARGET: {HOUMO_TARGET}"

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Download Qwen3-VL-8B hmm model")
    parser.add_argument("--download-dir", default="./models", help="Download directory")
    parser.add_argument("--model-size", default="4b", help="Model size")
    parser.add_argument("--context-length", default="32k", help="Context length")
    args = parser.parse_args()

    model_cfgs = {
        "target": HOUMO_TARGET,
        "version": get_houmo_version(),
        "model_type": "llm",
        "model_name": "qwen3-vl",
        "model_info": {
            "model_size": args.model_size,
            "ncore": 2,
            "ndevice": 1,
            "context_len": args.context_length,
            "prefill_len": 256,
            "batch": 1,
        },
        "modelscope_repo": {
            "repo_ids": [f"Qwen/Qwen3-VL-{args.model_size.upper()}-Instruct"],
            "local_dirs": [f"{args.download_dir}/tokenizers"],
        },
    }

    _, ret_dict = hmatc_get_file(
        model_cfgs,
        file_type="hmm",
        download_dir=args.download_dir,
        extract_dir=None,
        source_type="jfrog",
        extract_to_archive_name=True,
    )
    if not ret_dict.get("ret", False):
        exit(1)
