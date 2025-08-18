import os
import sys

HOUMO_EXAMPLES_PATH = os.environ.get('HOUMO_EXAMPLES_PATH', '../..')
sys.path.append(f'{HOUMO_EXAMPLES_PATH}/common/python')
from utils import get_file_from_jfrog


if __name__ == '__main__':
    if "HOUMO_MODELZOO_URL" not in os.environ:
        os.environ["HOUMO_MODELZOO_URL"] = "http://139.224.0.199:8082/artifactory/houmo/release"
    HOUMO_TARGET =os.getenv("HOUMO_TARGET", "houmo")

    if HOUMO_TARGET == "xh1":
        resnet50_hmm_path = "models/resnet50/hmm_resnet50_20250113.zip"
        yolov5s_hmm_path = "models/yolov5s/hmm_yolov5s_20250113.zip"
    elif HOUMO_TARGET == "xh2":
        resnet50_hmm_path = "models/resnet50/hmm_resnet50_xh2_b1_1core_20250804.zip"
        yolov5s_hmm_path = "models/yolov5s/hmm_yolov5s_xh2_b1_1core_20250804.zip"

    model_dir = os.path.join(HOUMO_EXAMPLES_PATH, "models")
    get_file_from_jfrog(resnet50_hmm_path, model_dir, "./")
    get_file_from_jfrog(yolov5s_hmm_path, model_dir, "./")
