import os
import sys
import argparse

HOUMO_EXAMPLES_PATH = os.environ.get('HOUMO_EXAMPLES_PATH', '../../..')
sys.path.append(f'{HOUMO_EXAMPLES_PATH}/hmatc')
from hmatc.utils.utils import get_file_from_jfrog, get_houmo_version

HOUMO_TARGET = os.getenv("HOUMO_TARGET")
assert HOUMO_TARGET in ["xh1", "xh2"], f"Unsupported HOUMO_TARGET: {HOUMO_TARGET}"


def get_args() -> argparse.Namespace:
    """Parse commandline."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--ncore",
        dest="ncore",
        type=int,
        default=4,
        help="which resource to get, choise in [2, 4]",
    )
    parser.add_argument(
        '--model_dir',
        dest='model_dir',
        type=str,
        default='',
        help='where to save downloaded model',
    )
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = get_args()
    if "HOUMO_MODELZOO_URL" not in os.environ:
        os.environ["HOUMO_MODELZOO_URL"] = (
            "http://139.224.0.199:8082/artifactory/houmo/release"
        )
    model_dir = (
        os.path.join(HOUMO_EXAMPLES_PATH, "apis/models")
        if not args.model_dir
        else args.model_dir
    )

    model_name = "qwen3"
    model_size = "8b"
    ncore = "2cores" if HOUMO_TARGET == "xh2" else f"{args.ncore}cores"
    ndevice = "1chip"
    context_len = "8k"
    prefill_len = 256
    batch = 1
    version = get_houmo_version()
    target = HOUMO_TARGET
    hmm_path = f"models/{target.lower()}-{version}/{model_name}/hmm_{target}_{model_name}_{model_size}_{prefill_len}_{context_len}_b{batch}_{ndevice}_{ncore}_{version}.zip"

    from modelscope import snapshot_download

    snapshot_download(
        "qwen/qwen3-8b", local_dir="qwen3-8b", ignore_patterns=["*.safetensors"]
    )

    get_file_from_jfrog(hmm_path, model_dir, ".")

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

    # 下载编译好的三方库
    thirdparty_path = "models/qwen3/3rdparty.zip"
    get_file_from_jfrog(thirdparty_path, model_dir, "./")
