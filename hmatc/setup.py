import os
import subprocess
import pybind11
from datetime import datetime
from setuptools import setup, find_packages, Extension


def get_version():
    return os.environ.get("HOUMO_VERSION", "0.0.0.dev")


def get_build_time():
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def get_git_commit():
    try:
        commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], 
            cwd=os.path.dirname(__file__)
        ).decode("utf-8").strip()
        return commit[:7]  # 取短 Commit 号
    except Exception:
        return "unknown"


commit = get_git_commit()
with open(os.path.join("hmatc", "_version.py"), "w") as f:
    f.write(f"__version__ = '{get_version()}'\n")
    f.write(f"__commit__ = '{commit}'\n")
    f.write(f"__build_time__ = '{get_build_time()}'\n")  # 新增时间字段
    

TCIM_RUNTIME_PATH = os.environ.get("TCIM_RUNTIME_PATH")
if not TCIM_RUNTIME_PATH :
    raise Exception("Please set TCIM_RUNTIME_PATH")

ext_modules = [
    Extension(
        name="hmatc.python.perf",
        sources=["hmatc/python/tcim_perf.cpp"],
        include_dirs=[
            pybind11.get_include(),
            os.path.join(TCIM_RUNTIME_PATH, "include"),
            "3rdparty/nlohmann/include"
        ],
        library_dirs=[
            os.path.join(TCIM_RUNTIME_PATH, "lib")
        ],
        libraries=["tcim_runtime_lite"],
        language="c++",
        extra_compile_args=["-std=c++17", "-O2", "-w"]
    )
]

requirements = []
with open("requirements.txt", "r", encoding="utf-8") as f:
    lists = f.readlines()
    for line in lists:
        requirements.append(line.strip())

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

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
    python_requires=">=3.8",
    install_requires=requirements,
    entry_points={
        "console_scripts": [
            "hmatc = hmatc:main",
        ],
    },
)