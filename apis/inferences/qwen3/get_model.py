import os
import argparse
from hmatc.utils.utils import get_file_from_jfrog

HOUMO_EXAMPLES_PATH = os.environ.get("HOUMO_EXAMPLES_PATH", "..")
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

    if HOUMO_TARGET == "xh1":
        if args.ncore == 2:
            hmm_path = "models/qwen3/hmm_qwen3_256_8k_2cores_20250603.zip"
        elif args.ncore == 4:
            hmm_path = "models/qwen3/hmm_qwen3_256_8k_4cores_20250728.zip"
    elif HOUMO_TARGET == "xh2":
        hmm_path = "models/qwen3/hmm_xh2_qwen3_8b_256_8k_2cores_20250808.zip"

    from modelscope import snapshot_download

    snapshot_download(
        "qwen/qwen3-8b", local_dir="qwen3-8b", ignore_patterns=["*.safetensors"]
    )

    get_file_from_jfrog(hmm_path, model_dir, "./")
