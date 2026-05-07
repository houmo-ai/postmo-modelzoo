# Copyright (c) 2025 HOUMO AI
#
# File: set_environs.py
# Description:
#   Windows Environment Setup Tool - Python script for setting up
# development environment variables for HOUMO AI Windows platform.
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
import os, sys
import shutil, re
import json
import importlib.util
from env_utils import *
import copy

class WinEnvironsGenerater:
    def __init__(self):
        self.setting_dir_path = os.path.abspath(os.path.join(os.path.abspath(__file__), "../"))
        self.settings = dict()
        self.examples = None
        self.py_examples = None
        self.cpp_examples = None
        self.base_environments = None
        self.all_environments = None
        self.py_example_environments = None
        self.cpp_example_environments = None
        self.all_py_example_environments = None
        self.all_cpp_example_environments = None
        self.env_supported_py_examples = None
        self.env_supported_cpp_examples = None
        self.develop_mode = False
        self.initial_backup_path = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "initial_env_backup.json"))
        self.env_manager = EnvManager(self.initial_backup_path)
        self.initial_env = dict()

    def load_origin_settings(self):
        with open(os.path.join(self.setting_dir_path, "env.json"), "r", encoding="utf-8") as f:
            self.settings = json.load(f)
            f.close()
        self.examples = self.settings["support_demos"]
        self.all_environments = self.settings["all_environments"]
        self.base_environments = self.settings["get_model_environments"]

        self.py_examples = [key for key, value in self.examples.items() if "python" in value]
        self.cpp_examples = [key for key, value in self.examples.items() if "cpp" in value]

        self.py_example_environments = self.settings["py_example_environments"]
        self.cpp_example_environments = self.settings["cpp_example_environments"]

        self.all_py_example_environments = copy.deepcopy(self.py_example_environments)
        self.all_cpp_example_environments = copy.deepcopy(self.cpp_example_environments)


        for key, value in self.py_example_environments.items():
            self.all_py_example_environments[key] = value + self.base_environments

        for key, value in self.cpp_example_environments.items():
            self.all_cpp_example_environments[key] = value + self.py_example_environments[key]

        self.develop_mode = self.settings["develop_url"]

    def nullEnvManualSet(self, key: str, need: bool):
        self.initial_env = self.env_manager.get_initial() if len(self.initial_env.keys()) == 0 else self.initial_env
        var_name = f"{key} [Required] " if need else f"{key} [Optional] "
        if key in self.initial_env.keys():
            self.all_environments[key] = os.getenv(key)
            assert os.path.exists(self.all_environments[key]), f'{var_name} invalid, path not exists!'
        if self.all_environments[key] == "" and key not in self.initial_env.keys():
            self.all_environments[key] = input(f"Manual Set {var_name} abspath:").strip()
            if need:
                assert self.all_environments[key] != "", f'{var_name} not find, please set it!'
                assert os.path.exists(self.all_environments[key]), f'{var_name} invalid, path not exists!'


    def show_current_env_supported_demos(self):
        print_support_lists("imodelzoo win11 support demos", self.examples)

        py_support_demos = dict()
        for key, value in self.all_py_example_environments.items():
            if all(self.all_environments[env] != "" for env in value):
                py_support_demos[key] = True
            else:
                py_support_demos[key] = False
        print_support_lists("current env support python demos", py_support_demos)

        cpp_support_demos = dict()
        for key, value in self.all_cpp_example_environments.items():
            if all(self.all_environments[env] != "" for env in value):
                cpp_support_demos[key] = True
            else:
                cpp_support_demos[key] = False
        print_support_lists("current env support cpp demos", cpp_support_demos)

    def clearEnvirons(self):
        if os.path.exists(self.initial_backup_path):
            self.env_manager.restore_to_initial()
            for key, value in self.all_environments.items():
                self.all_environments[key] = ""
            self.env_manager.refresh_envs()
            with open(os.path.join(self.setting_dir_path, "env.json"), "w", encoding="utf-8") as f:
                json.dump(self.settings, f, ensure_ascii=False, indent=4)
            print("========Clear all envs Finished, please reopen cmd window=========")
        else:
            print("==========================No backup envs==========================")
            print("========Clear all envs Finished, please reopen cmd window==========")

    def autoSetEnvirons(self):
        origin_envs = list(self.all_environments.keys())
        tcim_package_path = find_tcim_path()
        assert tcim_package_path is not None, f'Please install houmo_tcim_runtime_xh2 package first, it is Required!'
        self.nullEnvManualSet("HOUMO_SDK_PATH", need=True)
        self.all_environments["HOUMO_MODELZOO_URL"] = self.env_manager.read_env_var_from_sh_file("env.sh", "HOUMO_MODELZOO_URL")
        self.all_environments["HOUMO_TARGET"] = self.settings["support_target"]
        self.all_environments["TCIM_BACKEND"] = "Xh2HalBackend" if self.settings["support_target"] == "xh2" else "Xh1HalBackend"
        self.all_environments["HOUMO_EXAMPLES_PATH"] = os.path.abspath(os.path.join(os.path.abspath(__file__), "../../../"))
        self.all_environments["PYTHON_DIR"] = os.path.abspath(os.path.join(sys.executable, "../"))
        self.all_environments["TCIM_RUNTIME_PATH"] = tcim_package_path

        if shutil.which("cmake") is not None:
            self.all_environments["CMAKE_PATH"] = os.path.abspath(os.path.join(shutil.which("cmake"), "../"))
        else:
            self.nullEnvManualSet("CMAKE_PATH", need=False)

        self.all_environments["HOUMO_PATH"] = tcim_package_path

        pattern = r'v\d+\.\d+\.\d+'
        self.all_environments["HOUMO_VERSION"] = re.search(pattern, self.all_environments["HOUMO_SDK_PATH"]).group()
        try:
            import tcim_lite
            tcim_verison = tcim_lite.runtime.get_version().split('\n')[0]
            if tcim_verison != self.all_environments["HOUMO_VERSION"]:
                _tcim_verison = re.search(pattern, tcim_verison).group()
                if _tcim_verison != self.all_environments["HOUMO_VERSION"]:
                    print(f'[ERROR]: tcim version : {tcim_verison}, houmo_sdk version : {self.all_environments["HOUMO_VERSION"]}')
                    print("[ERROR]: tcim version and houmo sdk version not equal!, please check!")
                    exit()
                else:
                    print("[WARNING]: tcim version version date may not equal to houmo_sdk verison date.")
                    print(f'[WARNING]: tcim version : {tcim_verison}, houmo_sdk version : {self.all_environments["HOUMO_VERSION"]}')
            else:
                print(f'[INFO]: tcim version : {tcim_verison}, houmo_sdk version : {self.all_environments["HOUMO_VERSION"]}')
        except Exception as e:
            print(f"[ERROR] {e}, Failed to import tcim_lite, please install runtime sdk!")
            exit()

        tcim_dll_path = os.path.join(self.all_environments["TCIM_RUNTIME_PATH"], "bin")
        xh2a_dll_path = os.path.join(self.all_environments["HOUMO_SDK_PATH"], "hal\\lib")
        env_paths = self.env_manager.get_user_path()
        for env in env_paths:
            if env == '':
                continue
            else:
                if env not in self.all_environments["PATH"]:
                    self.all_environments["PATH"] = f'{env};' + self.all_environments["PATH"]

        if self.all_environments["CMAKE_PATH"] not in self.all_environments["PATH"] and self.all_environments["CMAKE_PATH"] != "":
            self.all_environments["PATH"] = f'{self.all_environments["CMAKE_PATH"]};' + self.all_environments["PATH"]

        if xh2a_dll_path not in self.all_environments["PATH"] and xh2a_dll_path != "":
            self.all_environments["PATH"] = f'{xh2a_dll_path};' + self.all_environments["PATH"]

        if tcim_dll_path not in self.all_environments["PATH"] and tcim_dll_path != "":
            self.all_environments["PATH"] = f'{tcim_dll_path};' + self.all_environments["PATH"]
            python_exe_path = os.path.abspath(os.path.join(self.all_environments["TCIM_RUNTIME_PATH"], "../../Scripts"))
            if python_exe_path not in self.all_environments["PATH"] and python_exe_path != "":
                self.all_environments["PATH"] = f'{python_exe_path};' + self.all_environments["PATH"]

        self.nullEnvManualSet("OPENCV_PATH", need=False)

        self.settings["all_environments"] = self.all_environments

        for key, value in self.all_environments.items():
            if key not in origin_envs:
                self.nullEnvManualSet(key, need=False)
            if key != "PATH":
                self.env_manager.set_env(key, value)
        for _, value in enumerate(self.all_environments["PATH"].split(";")):
            self.env_manager.add_to_path(value)

        self.env_manager.refresh_envs()
        with open(os.path.join(self.setting_dir_path, "env.json"), "w", encoding="utf-8") as f:
            json.dump(self.settings, f, ensure_ascii=False, indent=4)

    def ReSetEnvirons(self):
        self.env_manager.reset_env()

if __name__ == "__main__":
    generater = WinEnvironsGenerater()
    generater.load_origin_settings()
    if "--set" in sys.argv and len(sys.argv) == 2:
        generater.show_current_env_supported_demos()
        generater.autoSetEnvirons()
        generater.show_current_env_supported_demos()
        print("========Set all envs Finished, please reopen cmd window=========")
    if "--reset" in sys.argv and len(sys.argv) == 2:
        generater.ReSetEnvirons()
        print("========Reset current envs Finished, please reopen cmd window=========")
    # if "--clear" in sys.argv and len(sys.argv) == 2:
    #     generater.clearEnvirons()