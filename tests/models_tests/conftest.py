import pytest
import os


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

    script_dir = os.path.dirname(os.path.abspath(__file__))
    md_markers = list()
    with open(f"{script_dir}/model_names.txt", "r", encoding="utf-8") as f:
        for line in f:
            model_name = line.strip()
            if model_name:  # 跳过空行
                md_markers.append(model_name)
    print("Supported model names:", md_markers)

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
