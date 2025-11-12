import os
import sys
import argparse
from hmatc.utils.utils import get_file_from_jfrog, get_houmo_version


HOUMO_TARGET = os.getenv("HOUMO_TARGET")
assert HOUMO_TARGET in ["xh1", "xh2"], f"Unsupported HOUMO_TARGET: {HOUMO_TARGET}"


def get_args() -> argparse.Namespace:
    """Parse commandline."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--type',
        dest='model_type',
        type=str,
        default='hmm',
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
        default="8b",
        choices=["2b", "4b", "8b"],
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

    model_name = "qwen3-vl"
    model_size = args.model_size
    ncore = "2cores" if HOUMO_TARGET == "xh2" else "4cores"
    ndevice = "1chip"
    context_len = "8k" if HOUMO_TARGET == "xh2" else "2k"
    prefill_len = 256
    batch = 1
    version = get_houmo_version()
    target = HOUMO_TARGET
    hmm_path = f"models/{target}-{version}/{model_name}/hmm_{target}_{model_name}_{model_size}_{prefill_len}_{context_len}_b{batch}_{ndevice}_{ncore}_{version}.zip"

    date_str = "20251019" if HOUMO_TARGET == "xh2" else "20250903"
    quant_path = f"models/qwen3-vl/hmquant_{target}_{model_name}_{model_size}_256_2k_{date_str}.zip"

    if model_type in ["raw"]:
        ignore_patterns = []
    else:
        ignore_patterns = ["*.safetensors"]

    from modelscope import snapshot_download

    snapshot_download(
        "Qwen/Qwen3-VL-8B-Instruct",
        local_dir=f"{model_dir}/qwen3-vl",
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
