import os
import sys
import argparse
from hmatc.utils.utils import get_file_from_jfrog


HOUMO_TARGET = os.getenv("HOUMO_TARGET")
assert HOUMO_TARGET in ["xh1", "xh2"], f"Unsupported HOUMO_TARGET: {HOUMO_TARGET}"


def get_args() -> argparse.Namespace:
    """Parse commandline."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--type',
        dest='model_type',
        type=str,
        default='quant',
        help='which resource to get, choise in [raw, quant, hmm]',
    )
    parser.add_argument(
        '--quant_model_dir',
        dest='quant_model_dir',
        type=str,
        default=os.path.join('output', HOUMO_TARGET, 'hmquant'),
        help='where to save quant_model',
    )
    parser.add_argument(
        "--build_model_dir",
        dest="build_model_dir",
        type=str,
        default=os.path.join("output", HOUMO_TARGET),
    )
    parser.add_argument(
        '--model_dir',
        dest='model_dir',
        type=str,
        default='.',
        help='where to save downloaded model',
    )
    parser.add_argument(
        '--model_size',
        dest='model_size',
        type=str,
        default="7b",
        choices=["3b", "7b"],
        help='model size',
    )
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = get_args()
    quant_model_dir = args.quant_model_dir
    build_model_dir = args.build_model_dir
    model_type = args.model_type
    model_dir = args.model_dir
    HOUMO_DATASETS_PATH = os.getenv('HOUMO_DATASETS_PATH', '.')
    HOUMO_MODEL_PATH = os.getenv('HOUMO_MODEL_PATH', '.')
    if HOUMO_TARGET == "xh1" and args.model_size == "3b":
        quant_path = (
            "models_outdated/qwen2.5-vl/hmquant_xh1_qwen2.5-vl_3b_256_2k_20250903.zip"
        )
        hmm_path = "models/qwen2.5-vl/hmm_xh1_qwen2.5-vl_3b_256_2k_4cores_20250923.zip"
    elif HOUMO_TARGET == "xh1" and args.model_size == "7b":
        quant_path = (
            "models_outdated/qwen2.5-vl/hmquant_xh1_qwen2.5-vl_7b_256_2k_20250903.zip"
        )
        hmm_path = "models/qwen2.5-vl/hmm_xh1_qwen2.5-vl_7b_256_2k_4cores_20250923.zip"
    elif HOUMO_TARGET == "xh2" and args.model_size == "7b":
        quant_path = (
            "models_outdated/qwen2.5-vl/hmquant_xh2_qwen2.5-vl_7b_256_2k_20250908.zip"
        )
        hmm_path = "models/qwen2.5-vl/hmm_xh2_qwen2.5-vl_7b_256_2k_2cores_20250908.zip"

    if model_type in ["raw"]:
        ignore_patterns = []
    else:
        ignore_patterns = ["*.safetensors"]

    from modelscope import snapshot_download

    snapshot_download(
        "Qwen/Qwen2.5-VL-7B-Instruct",
        local_dir=f"{model_dir}/qwen2.5-vl",
        ignore_patterns=ignore_patterns,
    )

    if model_type in ["quant"] and not get_file_from_jfrog(
        quant_path, model_dir, quant_model_dir
    ):
        sys.exit(1)

    if model_type in ["hmm"] and not get_file_from_jfrog(
        hmm_path, model_dir, build_model_dir
    ):
        sys.exit(1)
