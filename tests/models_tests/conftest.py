import pytest


def pytest_configure(config):
    test_flow_markers = [
        "get_model",
        "quant",
        "compile",
        "demo",
        "compare",
        "eval",
        "perf",
    ]
    md_markers = [
        "sd3",
        "sdxl",
        "resnet50",
        "mobilenetv2",
        "efficientnet",
        "vit",
        "yolov3",
        "yolov5s",
        "yolov5s_feature",
        "yolov8m",
        "yolo12m",
        "yolov8m_pose",
        "yolov8m_seg",
        "qwen2dot5",
        "qwen3",
        "qwen3_14b",
        "deepseek",
        "deepseek_r1_qwen3_8b",
        "qwen2dot5_vl",
        "yolop",
        "wenet",
    ]
    for markers in test_flow_markers:
        config.addinivalue_line("markers", markers)
    for markers in md_markers:
        config.addinivalue_line("markers", markers)


def pytest_collection_modifyitems(session, config, items):
    file_order = [
        "models_tests/test_get_models.py",
        "models_tests/test_quant_models.py",
        "models_tests/test_compile_models.py",
        "models_tests/test_demo_models.py",
        "models_tests/test_compare_models.py",
        "models_tests/test_eval_models.py",
        "models_tests/test_perf_models.py",
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
