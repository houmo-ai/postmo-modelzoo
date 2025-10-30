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
        default='hmm',
        help='which resource to get, choise in [raw, hmm]',
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
        "--context_length",
        dest="context_length",
        type=str,
        default="8k",
        choices=["2k", "8k"],
        help="context length",
    )
    parser.add_argument(
        '--batch',
        dest='batch',
        type=int,
        default=1,
        choices=[1, 2, 4],
        help='batch size',
    )
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = get_args()
    build_model_dir = args.build_model_dir
    model_type = args.model_type
    model_dir = args.model_dir
    HOUMO_DATASETS_PATH = os.getenv('HOUMO_DATASETS_PATH', '.')
    HOUMO_MODEL_PATH = os.getenv('HOUMO_MODEL_PATH', '.')
    wiki_path = "models/datasets/wikitext-2-raw-v1.zip"

    model_name = "qwen3"
    model_size = "8b"
    ncore = "2cores" if HOUMO_TARGET == "xh2" else "4cores"
    ndevice = "1chip"
    context_len = args.context_length
    prefill_len = 256
    batch = args.batch
    version = f"v{runtime_version}"
    target = HOUMO_TARGET
    hmm_path = f"models/{target}-{version}/{model_name}/hmm_{target}_{model_name}_{model_size}_{prefill_len}_{context_len}_b{batch}_{ndevice}_{ncore}_{version}.zip"

    if model_type in ["raw"]:
        ignore_patterns = []
        get_file_from_jfrog(wiki_path, model_dir, HOUMO_DATASETS_PATH)
    else:
        ignore_patterns = ["*.safetensors"]

    from modelscope import snapshot_download

    snapshot_download(
        'qwen/qwen3-8b',
        local_dir=f'{model_dir}/qwen3-8b',
        ignore_patterns=ignore_patterns,
    )

    if model_type in ["hmm"] and not get_file_from_jfrog(
        hmm_path, model_dir, build_model_dir
    ):
        sys.exit(1)
