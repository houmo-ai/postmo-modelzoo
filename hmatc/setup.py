import os
import sys
import subprocess
import pybind11
import shutil  # ← 新增
import pybind11
from datetime import datetime
from setuptools import setup, find_packages, Extension
from setuptools.command.build_ext import build_ext


TCIM_RUNTIME_PATH = os.environ.get("TCIM_RUNTIME_PATH")
if not TCIM_RUNTIME_PATH:
    try:
        import tcim_lite

        TCIM_RUNTIME_PATH = os.path.dirname(tcim_lite.__file__)
    except ImportError:
        raise Exception("Please set TCIM_RUNTIME_PATH or install tcim_lite")

HOUMO_TARGET = os.getenv("HOUMO_TARGET")
if HOUMO_TARGET not in ["xh1", "xh2"]:
    raise Exception("Please set HOUMO_TARGET to xh1 or xh2")


def get_version():
    return os.environ.get("HOUMO_VERSION", "0.0.0.dev")


def get_build_time():
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def get_git_commit():
    try:
        commit = (
            subprocess.check_output(
                ["git", "rev-parse", "HEAD"], cwd=os.path.dirname(__file__)
            )
            .decode("utf-8")
            .strip()
        )
        return commit[:7]  # 取短 Commit 号
    except Exception:
        return "unknown"


commit = get_git_commit()
with open(os.path.join("hmatc", "_version.py"), "w") as f:
    f.write(f"__version__ = '{get_version()}'\n")
    f.write(f"__commit__ = '{commit}'\n")
    f.write(f"__build_time__ = '{get_build_time()}'\n")  # 新增时间字段

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
    在 Windows 平台构建结束后，将 TCIM_RUNTIME_PATH/bin/*.dll 复制到 pyd 同目录
    """

    def run(self):
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
        "build_ext": BuildExtWithDLL,  # ← 注册
    },
    python_requires=">=3.8",
    install_requires=requirements,
    entry_points={
        "console_scripts": [
            "hmatc = hmatc:main",
        ],
    },
)
