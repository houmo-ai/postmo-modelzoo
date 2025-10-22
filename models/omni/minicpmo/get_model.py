import os
import argparse
from hmatc.utils.utils import get_file_from_jfrog, get_package_version


HOUMO_TARGET = os.getenv('HOUMO_TARGET')
assert HOUMO_TARGET in ["xh2"], f"Unsupported HOUMO_TARGET: {HOUMO_TARGET}"

runtime_version = get_package_version(f"houmo_tcim_runtime_{HOUMO_TARGET}")
runtime_version = runtime_version.split(".dev")[0]

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
    version = f"v{runtime_version}"
    model_name = "minicpmo"
    model_size = "7b"
    ncore = "2cores"
    ndevice = "1chip"
    context_len = "2k"
    prefill_len = 256
    batch = 1
    target = HOUMO_TARGET
    HOUMO_MODEL_PATH = os.getenv('HOUMO_MODEL_PATH', '.')
    config_path = "models/minicpmo/MiniCPM-o-2_6_file.zip"
    if HOUMO_TARGET == "xh2":
        hmm_path = f"models/{target}-{version}/{model_name}/hmm_{target}_{model_name}_{model_size}_{context_len}_b{batch}_{ndevice}_{ncore}_{version}.zip"
        quant_path = "models/minicpmo/hmquant_xh2_minicpmo_7b_256_4k_20251016.zip"
    
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
