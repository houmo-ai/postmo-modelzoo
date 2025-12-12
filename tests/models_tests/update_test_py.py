import os
import json
from glob import glob


def _convert_model_name(model_name: str) -> str:
    # example: deepseek-r1-qwen3-8b-->deepseek_r1_qwen3_8b
    tmp_str = model_name.replace("-", "_")
    res_str = tmp_str.replace(".", "dot")
    return res_str


def _append_model_to_txt(new_model: str) -> bool:
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
        print(f"✅ 模型 '{new_model}' 已在 {file_path} 中，无需重复添加")
        return True
    else:
        with open(file_path, "a+", encoding="utf-8") as f:
            f.seek(0, 2)
            if f.tell() > 0:
                f.write("\n")
            f.write(new_model)
        print(f"✅ 模型 '{new_model}' 已成功追加到 {file_path}")
        return True


def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    model_cfg_dir = script_dir + "/model_configs"
    # python test file path
    py_path = {
        "get_model": script_dir + "/test_get_models.py",
        "quant": script_dir + "/test_quant_models.py",
        "compile": script_dir + "/test_compile_models.py",
        "demo": script_dir + "/test_demo_models.py",
        "compare": script_dir + "/test_compare_models.py",
        "eval": script_dir + "/test_eval_models.py",
        "perf": script_dir + "/test_perf_models.py",
    }

    for file_path in glob(model_cfg_dir + "/*.json"):
        if "template" in file_path:
            continue
        model_name = file_path.rsplit("/", 1)[-1][10:-5]

        with open(file_path, 'r', encoding='utf-8') as md_file:
            model_info = json.load(md_file)
        support_flow_xh1 = model_info["support_flow"].get("xh1", list())
        support_flow_xh2 = model_info["support_flow"].get("xh2", list())
        support_flow = list(set(support_flow_xh1 + support_flow_xh2))
        model_type = model_info["model_dir"].split("/")[1]
        model_name_new = _convert_model_name(model_name)

        for flow_name in support_flow:
            if flow_name == "demo_multibatch":
                continue

            with open(py_path[flow_name], 'r', encoding='utf-8') as file:
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
            with open(py_path[flow_name], 'a', encoding='utf-8') as file:
                if py_content and not py_content.endswith('\n'):
                    file.write("\n")

                file.write("\n\n")
                file.write(f"@pytest.mark.{model_name_new}\n")
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
                file.write(f"def {func_name}(setup_logging: type(print)) -> None:\n")
                file.write(f'    """{func_name}"""\n')
                file.write(f"    model_name = '{model_name}'\n")
                file.write(f"    _{flow_name}_func(model_name, setup_logging)\n")


if __name__ == "__main__":
    main()
