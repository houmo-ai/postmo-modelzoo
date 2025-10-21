import os
import argparse
from hmatc.utils.utils import get_file_from_jfrog


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
        help='which resource to get, choise in [raw, hmm]',
    )
    parser.add_argument(
        '--model_dir',
        dest='model_dir',
        type=str,
        default='.',
        help='where to save downloaded model',
    )
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = get_args()
    model_type = args.model_type
    model_dir = args.model_dir
    HOUMO_MODEL_PATH = os.getenv('HOUMO_MODEL_PATH', '.')
    config_path = "http://10.10.1.53:8082/artifactory/toolchain/release/models/minicpmo/MiniCPM-o-2_6_file.zip"
    if HOUMO_TARGET == "xh2":
        hmm_path = "http://10.10.1.53:8082/artifactory/toolchain/release/models/minicpmo/hmm_xh2_minicpmo_7b_256_4k_2core_20251016.zip"
        quant_path = "http://10.10.1.53:8082/artifactory/toolchain/release/models/minicpmo/hmquant_xh2_minicpmo_7b_256_4k_20251016.zip"
    
    try:
        get_file_from_jfrog(config_path, model_dir, "./")
    except Exception as e:
        print(f"Model doesn't exist, error msg: {e}")
    print("model_type:", model_type)
    if model_type == "hmm":
        try:
            get_file_from_jfrog(hmm_path, model_dir, os.path.join('output', HOUMO_TARGET))
        except Exception as e:
            print(f"Model doesn't exist, error msg: {e}")
    elif model_type == "quant":
        try:
            get_file_from_jfrog(quant_path, model_dir, os.path.join('output', HOUMO_TARGET))
        except Exception as e:
            print(f"Model doesn't exist, error msg: {e}")
