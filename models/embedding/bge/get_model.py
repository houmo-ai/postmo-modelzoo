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
        help='which resource to get, choise in [raw, quant, hmm]',
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
    model_type = args.model_type
    model_dir = args.model_dir

    version = get_houmo_version()
    model_name = "bge"
    ncore = "2cores"
    model_size = "0.5b"
    context_len = "0.5k"
    ndevice = "1chip"
    batch = 10
    target = HOUMO_TARGET
    onnx_path = "models/bge/onnx_bge_10x512.zip"
    quant_path = f"models_outdated/bge/hmquant_{target}_{model_name}_{model_size}_{context_len}_b{batch}_20251022.zip"
    hmm_path = f"models/{target}-{version}/{model_name}/hmm_{target}_{model_name}_{model_size}_{context_len}_b{batch}_{ndevice}_{ncore}_{version}.zip"

    if model_type in ["raw"]:
        ignore_patterns = []
        get_file_from_jfrog(onnx_path, model_dir, "./")
    else:
        ignore_patterns = ["*.safetensors", "*.bin", "onnx/*"]

    from modelscope import snapshot_download

    snapshot_download(
        'BAAI/bge-reranker-v2-m3',
        local_dir=f'{model_dir}/bge-reranker-v2-m3',
        ignore_patterns=ignore_patterns,
    )
    snapshot_download(
        'BAAI/bge-m3',
        local_dir=f'{model_dir}/bge-m3',
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
