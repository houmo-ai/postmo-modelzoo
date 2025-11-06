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
        help='which resource to get, choise in [raw, hmm]',
    )
    parser.add_argument(
        "--build_model_dir",
        dest="build_model_dir",
        type=str,
        default=os.path.join("output", HOUMO_TARGET),
        help="where to save build_model",
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
    build_model_dir = args.build_model_dir

    model_name = "whisper"
    model_size = "medium"
    ncore = "2cores"
    ndevice = "1chip"
    version = get_houmo_version()
    target = HOUMO_TARGET
    hmm_path = f"models/{target}-{version}/{model_name}/hmm_{target}_{model_name}_{model_size}_{ndevice}_{ncore}_{version}.zip"

    if model_type in ["raw"]:
        ignore_patterns = []
    else:
        ignore_patterns = ["*.bin", "*.h5", "*.msgpack", "*.safetensors"]

    from modelscope import snapshot_download

    snapshot_download(
        'openai-mirror/whisper-medium',
        local_dir=f'{model_dir}/whisper-medium',
        ignore_patterns=ignore_patterns,
    )

    if model_type in ["hmm"] and not get_file_from_jfrog(
        hmm_path, model_dir, build_model_dir
    ):
        sys.exit(1)
