import os
import sys
import argparse
from hmatc.utils.utils import get_file_from_jfrog, get_package_version


HOUMO_TARGET = os.getenv("HOUMO_TARGET")
assert HOUMO_TARGET in ["xh1", "xh2"], f"Unsupported HOUMO_TARGET: {HOUMO_TARGET}"

runtime_version = get_package_version(f"houmo_tcim_runtime_{HOUMO_TARGET}")
runtime_version = runtime_version.split(".dev")[0]


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

    model_name = "qwen2.5-vl"
    model_size = args.model_size
    ncore = "2cores" if HOUMO_TARGET == "xh2" else "4cores"
    ndevice = "1chip"
    context_len = "8k" if HOUMO_TARGET == "xh2" else "2k"
    prefill_len = 256
    batch = 1
    version = f"v{runtime_version}"
    target = HOUMO_TARGET
    hmm_path = f"models/{target}-{version}/{model_name}/hmm_{target}_{model_name}_{model_size}_{prefill_len}_{context_len}_b{batch}_{ndevice}_{ncore}_{version}.zip"

    date_str = "20250908" if HOUMO_TARGET == "xh2" else "20250903"
    quant_path = f"models_outdated/qwen2.5-vl/hmquant_{target}_{model_name}_{model_size}_256_2k_{date_str}.zip"

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
