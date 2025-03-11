import os
import sys
import onnx

HOUMO_EXAMPLES_PATH = os.environ.get('HOUMO_EXAMPLES_PATH', '..')
sys.path.append(f'{HOUMO_EXAMPLES_PATH}/common/python')
from utils import get_file_from_jfrog


if __name__ == '__main__':
    if "HOUMO_MODELZOO_URL" not in os.environ:
        os.environ["HOUMO_MODELZOO_URL"] = "http://139.224.0.199:8082/artifactory/houmo/release"
    HOUMO_TARGET = os.environ.get('HOUMO_TARGET', 'houmo')
    model_dir = os.path.join(HOUMO_EXAMPLES_PATH, "models")
    hmm_path = "models/qwen2/hmm_qwen2_256_4096_20250222.zip"
    get_file_from_jfrog(hmm_path, model_dir, "./")
