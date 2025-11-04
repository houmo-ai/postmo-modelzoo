import pytest
import os
import logging
from datetime import datetime

script_dir = os.path.dirname(os.path.abspath(__file__))

# download models from local server
os.environ["HOUMO_MODELZOO_URL"] = (
    "http://10.10.1.53:8082/artifactory/toolchain/release"
)
ori_ld = os.getenv("LD_LIBRARY_PATH", "")
append_ld = f"/opt/venv/houmo/lib/python3.12/site-packages/nvidia/cuda_runtime/lib:/opt/venv/houmo/lib/python3.12/site-packages/torch/lib:{script_dir}/../apis/models/3rdparty/onnxruntime/lib:"
os.environ["LD_LIBRARY_PATH"] = f"{ori_ld}:{append_ld}" if ori_ld else append_ld
os.environ["HOUMO_DATASETS_PATH"] = f"{script_dir}/../data/datasets/"
if os.getenv("HOUMO_VERSION", None) is None:
    os.environ["HOUMO_VERSION"] = "2.4.2"
# os.environ["IMODELZOO_MODELS_PATH"] = f"{script_dir}/../../modelzoo/"
# os.environ["IMODELZOO_MODELS_PATH"] = f"/develop02/modelzoo/"
os.makedirs(f"{script_dir}/models/", exist_ok=True)


def pytest_configure(config):
    shared_markers = ["imodelzoo"]
    for markers in shared_markers:
        config.addinivalue_line("markers", markers)


@pytest.fixture(autouse=True)
def setup_logging(request):
    """
    Create an independent log file for each testcase and return the log path.
    """
    current_date = datetime.now().strftime("%Y%m%d")

    # create log folder
    logs_dir = os.path.join(script_dir + "/", f"test_logs/{current_date}/")
    if not os.path.exists(logs_dir):
        os.makedirs(logs_dir, exist_ok=True)
    # generate a log file name
    test_name = request.node.name
    module_name = request.module.__name__ if request.module else "unknown"
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(logs_dir, f"{module_name}_{test_name}_{timestamp}.log")
    # setup logging
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    file_handler = logging.FileHandler(log_file)
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    file_handler.setFormatter(formatter)
    # clear existing handlers and add a new one.
    logger.handlers.clear()
    logger.addHandler(file_handler)
    # pass the log path to the testcase
    yield log_file
    # remove handler
    logger.removeHandler(file_handler)
