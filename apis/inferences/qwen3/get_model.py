import os
import sys
import argparse

HOUMO_EXAMPLES_PATH = os.environ.get('HOUMO_EXAMPLES_PATH', '../../..')
sys.path.append(f'{HOUMO_EXAMPLES_PATH}/hmatc')
from hmatc.utils.utils import hmatc_get_file, get_houmo_version

HOUMO_TARGET = os.getenv("HOUMO_TARGET")
assert HOUMO_TARGET in ["xh2"], f"Unsupported HOUMO_TARGET: {HOUMO_TARGET}"


def get_args() -> argparse.Namespace:
    """Parse commandline."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--download_dir',
        dest='download_dir',
        type=str,
        default=os.path.join(HOUMO_EXAMPLES_PATH, "apis/models"),
        help='where to save downloaded model',
    )
    parser.add_argument(
        "--extract_dir",
        dest="extract_dir",
        type=str,
        default=".",
        help='where to save extracted files',
    )
    parser.add_argument(
        "--source_type",
        dest="source_type",
        type=str,
        default="jfrog",
        choices=["jfrog", "modelscope"],
        help='download the model from which source',
    )
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = get_args()
    if "HOUMO_MODELZOO_URL" not in os.environ:
        os.environ["HOUMO_MODELZOO_URL"] = (
            "http://139.224.0.199:8082/artifactory/houmo/release"
        )

    model_cfgs = {
        "target": HOUMO_TARGET,
        "version": get_houmo_version(),
        "model_type": "llm",
        "model_name": "qwen3",
        "model_info": {
            "model_size": "8b",
            "ncore": 2,
            "ndevice": 1,
            "context_len": "8k",
            "prefill_len": 256,
            "batch": 1,
        },
        "hmm_files": {
            "other_files": ["models/qwen3/3rdparty.zip"],
        },
        "modelscope_repo": {"repo_ids": ["qwen/qwen3-8b"]},
    }

    _, ret_dict = hmatc_get_file(
        model_cfgs,
        "hmm",
        args.download_dir,
        args.extract_dir,
        args.source_type,
    )
    if ret_dict.get("ret", False) is False:
        exit(1)

    embedding_path = "hmquant/quant_embedding.pt"
    if os.path.exists(embedding_path):
        print(HOUMO_TARGET)
        import torch
        import numpy as np

        embedding_weight = torch.load(
            embedding_path, map_location="cpu", weights_only=True
        )
        if HOUMO_TARGET == "xh2":
            embedding_weight = embedding_weight['weight']
        if embedding_weight.dtype == torch.bfloat16:
            embedding_weight = embedding_weight.float().half()
        embedding_data = embedding_weight.cpu().numpy()
        embedding_data.tofile(embedding_path.replace(".pt", ".bin"))
