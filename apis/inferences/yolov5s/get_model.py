import os
import sys
import platform
import argparse

HOUMO_EXAMPLES_PATH = os.environ.get("HOUMO_EXAMPLES_PATH", "../..")
sys.path.append(f"{HOUMO_EXAMPLES_PATH}/apis/common/python")
from utils import get_file_from_jfrog


def get_args() -> argparse.Namespace:
    """Parse commandline."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_dir",
        dest="model_dir",
        type=str,
        default="",
        help="where to save downloaded model",
    )
    parser.add_argument(
        "--enable_ort",
        action="store_true",
        help="install onnxruntime environment to support post-processing model inference.",
    )
    args = parser.parse_args()
    return args


def execute_cmd(cmd, shell=False):
    import subprocess

    try:
        result = subprocess.run(
            cmd, capture_output=True, text=True, check=True, shell=shell
        )
        return result.stdout
    except Exception as e:
        print(f"Error occured: {e}")
        raise


def install_ort_env(third_party_dir, ort_pkg_name):
    try:
        print("configure the ORT C++ environment...")

        os.rename(f"{third_party_dir}/{ort_pkg_name}", f"{third_party_dir}/onnxruntime")
        ort_lib_dir = third_party_dir + "/onnxruntime/lib"
        cmd = f"echo {ort_lib_dir} | tee /etc/ld.so.conf.d/onnxruntime.conf"
        execute_cmd(cmd, True)
        cmd = "ldconfig"
        execute_cmd(cmd, True)

        print("The ORT C++ environment configuration has been completed.")
    except Exception as e:
        print(f"Failed to configure the ORT C++ environment, error: {e}")


if __name__ == "__main__":
    args = get_args()
    if "HOUMO_MODELZOO_URL" not in os.environ:
        os.environ["HOUMO_MODELZOO_URL"] = (
            "http://139.224.0.199:8082/artifactory/houmo/release"
        )
    HOUMO_TARGET = os.environ.get("HOUMO_TARGET", "houmo")

    model_dir = (
        os.path.join(HOUMO_EXAMPLES_PATH, "apis/models")
        if not args.model_dir
        else args.model_dir
    )
    if HOUMO_TARGET == "xh1":
        hmm_path = "models/yolov5s/hmm_yolov5s_20250113.zip"
    elif HOUMO_TARGET == "xh2":
        hmm_path = "models/yolov5s/hmm_yolov5s_xh2_b1_1core_20250804.zip"
    get_file_from_jfrog(hmm_path, model_dir, "./")

    if args.enable_ort and platform.system() == "Linux":
        # download yolov5s post-processing onnx model
        onnx_path = "models/yolov5s/yolov5s_640x640_postprocess.onnx"
        get_file_from_jfrog(onnx_path, "./")

        platform_name = platform.machine()
        # download onnxruntime env packages
        if platform_name == "x86_64":
            ort_env_str = "x64"
        elif platform_name == "aarch64":
            ort_env_str = "aarch64"
        else:
            print(
                f"Current platform is {platform_name} and does not support onnxruntime c++ env."
            )
            exit(0)

        third_party_dir = os.path.join(model_dir, "3rdparty")
        ort_pkg_name = "onnxruntime-linux-" + ort_env_str + "-1.22.0"
        ort_pkg_path = "models/3rdparty/" + ort_pkg_name + ".tgz"
        get_file_from_jfrog(ort_pkg_path, third_party_dir, third_party_dir)

        if not os.path.exists(third_party_dir + "/onnxruntime"):
            install_ort_env(third_party_dir, ort_pkg_name)
