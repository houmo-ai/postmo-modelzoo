import pytest
import os
import logging
from datetime import datetime

script_dir = os.path.dirname(os.path.abspath(__file__))

# download models from local server
os.environ["HOUMO_MODELZOO_URL"] = (
    "http://10.10.1.53:8082/artifactory/toolchain/release"
)
os.environ["HOUMO_DATASETS_PATH"] = f"{script_dir}/../data/datasets/"
os.environ["IMODELZOO_MODELS_PATH"] = f"{script_dir}/models/"
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
