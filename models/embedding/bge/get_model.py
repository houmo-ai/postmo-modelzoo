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
    onnx_path = "http://10.10.1.53:8082/artifactory/toolchain/release/models/bge/onnx_bge_10x512.zip"
    if HOUMO_TARGET == "xh2":
        hmm_path = "http://10.10.1.53:8082/artifactory/toolchain/release/models/bge/hmm_xh2_bge_10x512_2cores_20251015.zip"

    from modelscope import snapshot_download
    snapshot_download('BAAI/bge-reranker-v2-m3',
                      local_dir=f'{model_dir}/bge-reranker-v2-m3',
                      ignore_patterns=["*.safetensors"])
    snapshot_download('BAAI/bge-m3',
                      local_dir=f'{model_dir}/bge-m3',
                      ignore_patterns=["*.bin", "onnx/*"])
    
    try:
        get_file_from_jfrog(onnx_path, model_dir, "./")
    except Exception as e:
        print(f"Model doesn't exist, error msg: {e}")
    print("model_type:", model_type)
    if model_type == "hmm":
        try:
            get_file_from_jfrog(hmm_path, model_dir, os.path.join('output', HOUMO_TARGET))
        except Exception as e:
            print(f"Model doesn't exist, error msg: {e}")
