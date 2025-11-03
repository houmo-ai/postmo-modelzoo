import os
import onnx
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
    if HOUMO_TARGET == "xh2":
        hmm_path = "models/whisper/hmm_xh2_whisper_medium_2cores_202501103.zip"

    from modelscope import snapshot_download
    snapshot_download('openai-mirror/whisper-medium',
                      local_dir=f'{model_dir}/whisper-medium',
                      ignore_patterns=["*.bin", "*.h5", "*.msgpack", "*.safetensors"])

    print("model_type:", model_type)
    if model_type == "hmm":
        try:
            get_file_from_jfrog(hmm_path, model_dir, os.path.join('output', HOUMO_TARGET))
        except Exception as e:
            print(f"Model doesn't exist, error msg: {e}")
