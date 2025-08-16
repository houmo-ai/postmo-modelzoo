import os
import sys

HOUMO_EXAMPLES_PATH = os.environ.get('HOUMO_EXAMPLES_PATH', '../..')
sys.path.append(f'{HOUMO_EXAMPLES_PATH}/apis/common/python')
from utils import get_file_from_jfrog


if __name__ == '__main__':
    if "HOUMO_MODELZOO_URL" not in os.environ:
        os.environ["HOUMO_MODELZOO_URL"] = "http://139.224.0.199:8082/artifactory/houmo/release"
    HOUMO_TARGET = os.environ.get('HOUMO_TARGET', 'houmo')

    if HOUMO_TARGET != "xh1":
        print("Error: not support houmo target", HOUMO_TARGET)
        sys.exit(-1)
    model_dir = os.path.join(HOUMO_EXAMPLES_PATH, "models")
    dataset_path = "models/datasets/images_1920x1080.zip"
    get_file_from_jfrog(dataset_path, model_dir, "./")
    hmm_path = "models/yolov5s/hmm_yolov5s_1080x1920_xh1_b4_1core_20250724.zip"
    get_file_from_jfrog(hmm_path, model_dir, "./")
    hmm_path = "models/resnet50/hmm_resnet50_1080x1920_xh1_b1_roi4_1core_20250724.zip"
    get_file_from_jfrog(hmm_path, model_dir, "./")
