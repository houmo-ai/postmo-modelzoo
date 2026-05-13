# Copyright 2025 HOUMO AI
#
# File: update_test_py.py
# Description:
#   Update test Python files with new model configurations.
#   This script automatically generates test functions for new models based on their
#   configuration files.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# SPDX-License-Identifier: Apache-2.0

import os
import json
from glob import glob


def _convert_model_name(model_name: str) -> str:
    """
    Convert model name to a valid Python identifier format.

    Args:
        model_name (str): Original model name that may contain special characters

    Returns:
        str: Converted model name suitable for use as a Python identifier
    """
    # example: deepseek-r1-qwen3-8b-->deepseek_r1_qwen3_8b
    tmp_str = model_name.replace("-", "_")
    res_str = tmp_str.replace(".", "dot")
    return res_str


def _append_model_to_txt(new_model: str) -> bool:
    """
    Append a new model name to the model_names.txt file.

    Args:
        new_model (str): New model name to add to the list

    Returns:
        bool: True if the model was successfully added or already existed, False otherwise
    """
    script_dir = os.path.dirname(os.path.abspath(__file__))
    file_path = f"{script_dir}/model_names.txt"

    if not os.path.exists(file_path):
        print(f"Error: Not found {file_path}")
        return False

    existing_models = []
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            model = line.strip()
            if model:
                existing_models.append(model)

    if new_model in existing_models:
        print(f"✅ Model {new_model} already exists in {file_path}")
        return True
    else:
        with open(file_path, "a+", encoding="utf-8") as f:
            f.seek(0, 2)
            if f.tell() > 0:
                f.write("\n")
            f.write(new_model)
        print(f"✅ Model {new_model} has been successfully appended to {file_path}")
        return True


def _build_dependency_markers(model_info: dict) -> list[str]:
    """
    Build pytest markers from model dependencies field.

    Rules:
    - ndevice: use first element, e.g. [1, 4] -> pytest.mark.ndevice_1
    - dev_mem: use first element, e.g. ["12g", "48g"] -> pytest.mark.dev_mem_12g
    """
    dependency_markers = []
    dependencies = model_info.get("dependencies")
    if not isinstance(dependencies, dict):
        return dependency_markers

    ndevice_values = dependencies.get("ndevice")
    if isinstance(ndevice_values, list) and len(ndevice_values) > 0:
        ndevice_value = str(ndevice_values[0]).strip()
        if ndevice_value:
            dependency_markers.append(f"ndevice_{ndevice_value}")

    dev_mem_values = dependencies.get("dev_mem")
    if isinstance(dev_mem_values, list) and len(dev_mem_values) > 0:
        dev_mem_value = str(dev_mem_values[0]).strip().lower()
        if dev_mem_value:
            dependency_markers.append(f"dev_mem_{dev_mem_value}")

    return dependency_markers


def main():
    """
    Main function to scan model configurations and update test files.
    """
    script_dir = os.path.dirname(os.path.abspath(__file__))
    model_cfg_dir = script_dir + "/model_configs"

    # Python test file paths mapped to test flow types
    py_path = {
        "get_model": script_dir + "/test_get_models.py",
        "quant": script_dir + "/test_quant_models.py",
        "compile": script_dir + "/test_compile_models.py",
        "demo": script_dir + "/test_demo_models.py",
        "compare": script_dir + "/test_compare_models.py",
        "eval": script_dir + "/test_eval_models.py",
        "perf": script_dir + "/test_perf_models.py",
    }

    # Process each model configuration file
    for file_path in glob(model_cfg_dir + "/*.json"):
        if "template" in file_path:
            continue
        model_name = file_path.rsplit("/", 1)[-1][10:-5]

        with open(file_path, "r", encoding="utf-8") as md_file:
            model_info = json.load(md_file)

        # Skip obsolete models
        if model_info.get("obsolete"):
            continue

        # Get supported flows for both xh1 and xh2 backends
        support_flow_xh1 = model_info["support_flow"].get("xh1", list())
        support_flow_xh2 = model_info["support_flow"].get("xh2", list())
        support_flow = list(set(support_flow_xh1 + support_flow_xh2))
        model_type = model_info["model_dir"].split("/")[1]
        model_name_new = _convert_model_name(model_name)
        dependency_markers = _build_dependency_markers(model_info)

        # Generate test functions for each supported flow
        for flow_name in support_flow:
            if flow_name == "demo_multibatch":
                continue

            with open(py_path[flow_name], "r", encoding="utf-8") as file:
                py_content = file.read()

            func_name = "test_" + model_type + "_" + model_name_new + "_" + flow_name
            if func_name in py_content:
                continue

            print(
                f"Detect new model {model_name}-->{model_name_new}, support flow {support_flow}."
            )
            if not _append_model_to_txt(model_name_new):
                print(f"Failed to add {model_name_new} into model_names.txt")
                continue

            print(f"Add {func_name} into {flow_name} python file")
            with open(py_path[flow_name], "a", encoding="utf-8") as file:
                if py_content and not py_content.endswith("\n"):
                    file.write("\n")

                file.write("\n\n")
                file.write(f"@pytest.mark.{model_name_new}\n")
                for dependency_marker in dependency_markers:
                    file.write(f"@pytest.mark.{dependency_marker}\n")
                file.write(f"@pytest.mark.{flow_name}\n")
                if flow_name == "get_model":
                    file.write(f"@pytest.mark.dependency(name='{func_name}')\n")
                elif flow_name == "quant":
                    file.write(
                        f"@pytest.mark.dependency(name='{func_name}', depends_on=['test_get_models.py::test_{model_type}_{model_name_new}_get_model'])\n"
                    )
                elif flow_name == "compile" and "quant" in support_flow:
                    file.write(
                        f"@pytest.mark.dependency(name='{func_name}', depends_on=['test_quant_models.py::test_{model_type}_{model_name_new}_quant'])\n"
                    )
                elif flow_name == "compile" and "get_model" in support_flow:
                    file.write(
                        f"@pytest.mark.dependency(name='{func_name}', depends_on=['test_get_models.py::test_{model_type}_{model_name_new}_get_model'])\n"
                    )
                file.write(f"def {func_name}(setup_logging) -> None:\n")
                file.write(f'    """{func_name}"""\n')
                file.write(f"    model_name = '{model_name}'\n")
                file.write(f"    _{flow_name}_func(model_name, setup_logging)\n")


if __name__ == "__main__":
    main()
