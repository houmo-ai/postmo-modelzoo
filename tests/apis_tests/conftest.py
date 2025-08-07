import pytest
import os

script_dir = os.path.dirname(os.path.abspath(__file__))
os.environ['HOUMO_EXAMPLES_PATH'] = f"{script_dir}/../../apis"


def pytest_configure(config):
    apis_type_markers = [
        "apis",
        "inference",
        "multistreams",
        "pipeline",
        "multibatch",
        "video_detect",
    ]
    md_markers = [
        "qwen3",
        "resnet50",
        "yolov5s",
    ]
    for markers in apis_type_markers:
        config.addinivalue_line("markers", markers)
    for markers in md_markers:
        config.addinivalue_line("markers", markers)


def pytest_collection_modifyitems(session, config, items):
    file_order = [
        "apis_tests/test_inferences_apis.py",
        "apis_tests/test_scenes_apis.py",
    ]

    def get_sort_key(item):
        # get test file path
        filename = item.location[0]
        try:
            return file_order.index(filename)
        except ValueError:
            # files not in the file_order list are placed at the end
            return len(file_order)

    items.sort(key=get_sort_key)
