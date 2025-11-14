import os
import sys
import argparse
from hmatc.utils.utils import get_file_from_jfrog, get_houmo_version
HOUMO_TARGET = os.getenv('HOUMO_TARGET')
assert HOUMO_TARGET in ["xh2"], f"Unsupported HOUMO_TARGET: {HOUMO_TARGET}"
def get_args() -> argparse.Namespace:
    """Parse commandline."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--type',
        dest='model_type',
        type=str,
        default='hmm',
        help='which resource to get, choise in [quant, hmm]',
    )
    parser.add_argument(
        '--model_dir',
        dest='model_dir',
        type=str,
        default='.',
        help='where to save downloaded model',
    )
    parser.add_argument(
        "--build_model_dir",
        dest="build_model_dir",
        type=str,
        default=os.path.join("output", HOUMO_TARGET),
    )
    parser.add_argument(
        "--quant_model_dir",
        dest="quant_model_dir",
        type=str,
        default=os.path.join("output", HOUMO_TARGET, "hmquant"),
    )
    args = parser.parse_args()
    return args
if __name__ == '__main__':
    args = get_args()
    model_type = args.model_type
    model_dir = args.model_dir
    build_model_dir = args.build_model_dir
    quant_model_dir = args.quant_model_dir
    version = get_houmo_version()
    model_name = "minicpmo"
    model_size = "7b"
    ncore = "2cores"
    ndevice = "1chip"
    context_len = "4k"
    prefill_len = 256
    batch = 1
    target = HOUMO_TARGET
    HOUMO_MODEL_PATH = os.getenv('HOUMO_MODEL_PATH', '.')
    config_path = "models/minicpmo/MiniCPM-o-2_6_file_20251114.zip"
    hmm_path = f"models/{target}-{version}/{model_name}/hmm_{target}_{model_name}_{model_size}_{prefill_len}_{context_len}_b{batch}_{ndevice}_{ncore}_{version}.zip"
    quant_path = "models/minicpmo/hmquant_xh2_minicpmo_7b_256_4k_20251114.zip"
    if not get_file_from_jfrog(config_path, model_dir, "./"):
        sys.exit(1)
    if model_type in ["hmm"] and not get_file_from_jfrog(
        hmm_path, model_dir, build_model_dir
    ):
        sys.exit(1)
    if model_type in ["quant"] and not get_file_from_jfrog(
        quant_path, model_dir, quant_model_dir
    ):
        sys.exit(1)