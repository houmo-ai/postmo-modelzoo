import pytest
import os

script_dir = os.path.dirname(os.path.abspath(__file__))


def pytest_configure(config):
    apis_type_markers = ["hmatc"]
    md_markers = [
        "resnet50",
        "yolov5s",
    ]
    for markers in apis_type_markers:
        config.addinivalue_line("markers", markers)
    for markers in md_markers:
        config.addinivalue_line("markers", markers)
