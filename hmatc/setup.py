# Copyright 2025 HOUMO AI
#
# File: setup.py
# Description:
#   Setup script for the hmatc package.
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
import sys
import argparse
import subprocess
import shutil
import pybind11
from datetime import datetime
from setuptools import setup, find_packages, Extension
from setuptools.command.build_ext import build_ext

try:
    import tcim_lite

    TCIM_RUNTIME_PATH = os.path.dirname(tcim_lite.__file__)
except ImportError:
    raise Exception("Please install tcim_lite in current Python environment")

HOUMO_TARGET = os.getenv("HOUMO_TARGET")
if HOUMO_TARGET != "xh2":
    raise Exception("Please set HOUMO_TARGET to xh2")


def get_version():
    """
    Get package version from environment variable.

    Returns:
        str: Version string from HOUMO_VERSION environment variable
    """
    return os.environ.get("HOUMO_VERSION", "1.4.0.dev")


def get_build_time():
    """
    Get current build time in formatted string.

    Returns:
        str: Current timestamp in format "YYYY-MM-DD HH:MM:SS".
    """
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def get_git_commit():
    """
    Get the short git commit hash for the current repository.

    Returns:
        str: Short git commit hash (7 characters) or "unknown" if git command fails.
    """
    try:
        commit = (
            subprocess.check_output(
                ["git", "rev-parse", "HEAD"], cwd=os.path.dirname(__file__)
            )
            .decode("utf-8")
            .strip()
        )
        return commit[:7]
    except Exception:
        return "unknown"


commit = get_git_commit()
with open(os.path.join("hmatc", "_version.py"), "w") as f:
    f.write(f"__version__ = '{get_version()}'\n")
    f.write(f"__commit__ = '{commit}'\n")
    f.write(f"__build_time__ = '{get_build_time()}'\n")

if sys.platform == "win32":
    extra_compile_args = ["/std:c++17", "/O2", "/w"]
    libraries = ["tcim_runtime_lite"]
    extra_link_args = []
else:
    extra_compile_args = ["-std=c++17", "-O2", "-w"]
    libraries = ["tcim_runtime_lite"]
    extra_link_args = []

ext_modules = [
    Extension(
        name="hmatc.python.perf",
        sources=["hmatc/python/tcim_perf.cpp"],
        include_dirs=[
            pybind11.get_include(),
            os.path.join(TCIM_RUNTIME_PATH, "include"),
            "3rdparty/nlohmann/include",
        ],
        library_dirs=[os.path.join(TCIM_RUNTIME_PATH, "lib")],
        libraries=libraries,
        language="c++",
        extra_compile_args=extra_compile_args,
        extra_link_args=extra_link_args,
    )
]

requirements = []
with open("requirements.txt", "r", encoding="utf-8") as f:
    lists = f.readlines()
    for line in lists:
        if sys.platform == "win32" and "onnxsim" in line:
            continue
        requirements.append(line.strip())

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()


class BuildExtWithDLL(build_ext):
    """
    Custom build extension class that handles Windows-specific deployment.

    On Windows platforms, this class copies required DLL files from the
    TCIM runtime directory to the same location as the compiled Python extension
    after the build process completes.
    """

    def run(self):
        """
        Execute the build process and handle Windows DLL copying.

        This method first calls the parent run() method to perform standard
        build operations, then copies DLL files on Windows platforms to ensure
        proper runtime dependencies are available.
        """
        super().run()

        if sys.platform != "win32":
            return

        dll_dir = os.path.join(TCIM_RUNTIME_PATH, "bin")

        for ext in self.extensions:
            pyd_path = self.get_ext_fullpath(ext.name)
            pyd_dir = os.path.dirname(pyd_path)
            print(f"[INFO] Copying DLLs to: {pyd_dir}")

            if not os.path.exists(dll_dir):
                print(f"[WARN] DLL directory not found: {dll_dir}")
                continue

            for file in os.listdir(dll_dir):
                if file.lower().endswith(".dll"):
                    src = os.path.join(dll_dir, file)
                    dst = os.path.join(pyd_dir, file)
                    print(f"[COPY] {src} -> {dst}")
                    shutil.copy(src, dst)


parser = argparse.ArgumentParser(add_help=False)
parser.add_argument(
    "--enable_smi_support", action="store_true", help="enable Python SMI support"
)
args, remaining_argv = parser.parse_known_args()
sys.argv = [sys.argv[0]] + remaining_argv

if sys.platform == "linux" and args.enable_smi_support:
    HOUMO_SDK_PATH = os.environ.get("HOUMO_SDK_PATH")
    smi_libraries = ["hal_xh2a"]
    smi_compile_args = ["-std=c++17", "-O2", "-w", "-D ENABLE_SMI"]
    smi_ext = Extension(
        name="hmatc.python.smi",
        sources=["hmatc/python/device_smi.cpp"],
        include_dirs=[
            pybind11.get_include(),
            os.path.join(HOUMO_SDK_PATH, "hal/export/include"),
            "3rdparty/nlohmann/include",
        ],
        library_dirs=[os.path.join(HOUMO_SDK_PATH, "hal/lib")],
        libraries=smi_libraries,
        language="c++",
        extra_compile_args=smi_compile_args,
        extra_link_args=extra_link_args,
    )

    ext_modules.append(smi_ext)

setup(
    name="hmatc",
    version=get_version(),
    author="Houmo",
    author_email="weiguo.xing@houmo.ai",
    description="Houmo Model Assist Toolkit",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="http://10.10.1.58/weiguo.xing/hmatc",
    packages=find_packages(),
    package_data={
        "hmatc": ["python/*.so"],
    },
    ext_modules=ext_modules,
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    cmdclass={
        "build_ext": BuildExtWithDLL,
    },
    python_requires=">=3.8",
    install_requires=requirements,
    entry_points={
        "console_scripts": [
            "hmatc = hmatc:main",
        ],
    },
)
