import os
import sys

HOUMO_EXAMPLES_PATH = os.environ.get('HOUMO_EXAMPLES_PATH', '../..')
sys.path.append(f'{HOUMO_EXAMPLES_PATH}/common/python')
from utils import get_file_from_jfrog


if __name__ == '__main__':
    if "HOUMO_MODELZOO_URL" not in os.environ:
        os.environ["HOUMO_MODELZOO_URL"] = "http://139.224.0.199:8082/artifactory/houmo/release"
    HOUMO_TARGET = os.environ.get('HOUMO_TARGET', 'houmo')
    model_dir = "../models"
    hmm_path = "models/resnet50/hmm_resnet50_20250113.zip"
    get_file_from_jfrog(hmm_path, model_dir, "./")
