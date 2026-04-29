# Copyright 2025 HOUMO AI
#
# File: tests_common_utils.py
# Description:
#   Common utilities for model testing using pytest framework.
#   This file provides shared functionality for model tests, including environment variable
#   configuration, resource locking mechanisms, subprocess execution, and hardware checks.
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
import subprocess
import logging
import json
import shutil
import select
import fcntl
from datetime import datetime
from glob import glob
import enum
import time
import threading
import pytest
from enum import Enum, unique

from .tests_pyvenv_utils import get_site_packages, VENV_NAME

script_dir = os.path.dirname(os.path.abspath(__file__))

HOUMO_BACKEND = os.getenv("HOUMO_TARGET", "xh2")
# ON: quant&compile, OFF:inference
SEPARATE_TEST = os.getenv("SKIP_INFER", None)
HDPL_PLATFORM = os.getenv("HDPL_PLATFORM", "")
MODELS_PATH = os.path.abspath(
    os.getenv("IMODELZOO_MODELS_PATH", f"{script_dir}/../models_{HOUMO_BACKEND}/")
)
MODELS_RES_DIR = os.path.abspath(
    os.path.dirname(os.path.abspath(__file__)) + f"/../model_results_{HOUMO_BACKEND}"
)
USE_RELEASED_MODELS = os.getenv("USE_RELEASED_MODELS", "ON")

NDEVICE_MARKER = "ndevice"
DEVICE_MEM_MARKER = "dev_mem"

logger = logging.getLogger(__name__)


@unique
class TCaseType(Enum):
    """Enum representing different types of test cases"""

    DEFAULT = 0
    SEPARATE_NO_INFER = 1
    SEPARATE_INFER = 2


class ModelResourceLock:
    """
    A context manager for handling model resource locks using file-based locking.
    """

    class LockMode(enum.Enum):
        """Lock mode for a `ModelResourceLock`"""

        NO_LOCK = None
        READ_ONLY = fcntl.LOCK_SH
        WRITE = fcntl.LOCK_EX

    # All tests start simultaneously, expect their maximum end time to be 2 hours
    MAX_RESOURCE_ACCESS_TIME_OUT = 7200

    def __init__(self, lock_file: str, lock_mode: LockMode, lock_purpose: str):
        self.lock_file = lock_file
        self.lock_mode = lock_mode
        self.lock_purpose = lock_purpose
        self.lock_fd = None

    def __enter__(self):
        if self.lock_mode == ModelResourceLock.LockMode.NO_LOCK:
            return None
        self.acquire_lock()
        return self.lock_fd

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.lock_mode != ModelResourceLock.LockMode.NO_LOCK:
            self.release_lock()

    def acquire_lock(self):
        """
        Acquire the file lock with timeout
        """
        assert self.lock_mode in (
            ModelResourceLock.LockMode.READ_ONLY,
            ModelResourceLock.LockMode.WRITE,
        )
        # create dir if not exists, and make it accessible for all
        if not os.path.exists(lock_folder := os.path.dirname(self.lock_file)):
            os.makedirs(lock_folder, exist_ok=True)
            os.chmod(lock_folder, 0o777)
        # require a rw path (currently /develop01/models/toolchain is also rw for dev)
        open_mode = os.O_RDWR | os.O_CREAT | os.O_TRUNC
        # temporarily set umask to 0o666 to get a lock file accessible for all
        original_umask = os.umask(0)
        try:
            fd = os.open(self.lock_file, open_mode, 0o666)
        finally:
            os.umask(original_umask)
        pid = os.getpid()
        lock_file_fd = None
        start_time = current_time = time.time()
        print_flag = True
        while (
            current_time < start_time + ModelResourceLock.MAX_RESOURCE_ACCESS_TIME_OUT
        ):
            try:
                fcntl.flock(fd, self.lock_mode.value | fcntl.LOCK_NB)
            except (IOError, OSError):
                pass
            else:
                lock_file_fd = fd
                break
            if print_flag:
                logger.info(
                    "  %d waiting for %s lock: %s at %s",
                    pid,
                    self.lock_mode,
                    self.lock_file,
                    self.lock_purpose,
                )
                print_flag = False

            time.sleep(5.0)
            current_time = time.time()
        assert lock_file_fd is not None, f"Fails to get lock: {self.lock_file}"
        self.lock_fd = lock_file_fd

    def release_lock(self):
        """
        Release the acquired file lock
        """
        assert self.lock_fd is not None
        fcntl.flock(self.lock_fd, fcntl.LOCK_UN)
        os.close(self.lock_fd)


def load_json(json_path: str) -> dict | None:
    """
    Load JSON data from a file

    :param json_path: Path to the JSON file
    :return: Loaded JSON data as dictionary, or None if file doesn't exist
    """
    if not os.path.exists(json_path):
        return None

    with open(json_path, "r", encoding="utf-8") as f:
        json_info = json.load(f)
    logger.info(f"Loaded config file {json_path}")

    return json_info


class SubprocessLogger:
    """
    Logger for subprocess output that can write to both console and file
    """

    def __init__(self, log_file=""):
        """
        Initialize subprocess logger

        :param log_file: Path to the log file
        """
        self.log_file = log_file
        if log_file:
            os.makedirs(os.path.dirname(log_file), exist_ok=True)
        # Create file lock to ensure thread-safe writing
        self.lock = threading.Lock()
        self.write_flag = True if log_file else False

    def write(self, message, stream=sys.stdout):
        """
        Write message to both screen and log file

        :param message: Message to output
        :param stream: Output stream (stdout or stderr)
        """
        if not message or "MB/s" in message:  # or "kB/s" in message:
            return

        try:
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            log_message = f"[{timestamp}] {message}"

            if self.write_flag:
                # Write to log file (with lock to ensure thread safety)
                with self.lock:
                    with open(self.log_file, "a", encoding="utf-8") as f:
                        f.write(log_message)

            # Output to screen
            stream.write(message)
            stream.flush()
        except Exception as e:
            logger.error(f"Failed to write log: {e}", exc_info=True)


def _process_stream(stream, logger, results: list, is_stderr=False):
    """
    Process subprocess output streams

    :param stream: Subprocess output stream (stdout or stderr)
    :param logger: SubprocessLogger instance
    :param results: List to store output results
    :param is_stderr: Whether this is stderr stream
    """
    stream_obj = sys.stderr if is_stderr else sys.stdout
    outputs = []
    try:
        while True:
            # moniter the stream with timeout to avoid blocking if subprocess hangs or descendant process inherits the stream
            ready, _, _ = select.select([stream], [], [], 1.0)
            if not ready:
                # check if stream is closed, if yes, break the loop to avoid infinite loop
                if stream.closed:
                    break
                continue

            line = stream.readline()
            if not line:  # exit if stream is closed
                break
            logger.write(line, stream_obj)
            outputs.append(line.strip())
    except Exception as e:
        logger.error(f"Failed to process stream: {e}", exc_info=True)
    finally:
        try:
            if not stream.closed:
                stream.close()
        except Exception:
            pass

    results.append("\n".join(outputs))


def execute_test_cmd(
    cmd_list: list,
    log_file: str = "",
    assert_flag: bool = False,
    check_flag: bool = True,
    pyvenv_flag: bool = False,
) -> tuple[bool, any]:
    """
    Execute a command and handle its output

    :param cmd_list: Command to execute as a list
    :param log_file: Log file path
    :param assert_flag: Whether to assert success
    :param check_flag: Whether to check for failure keywords in output
    :return: Tuple of (success flag, output string)
    """
    cmd_str = " ".join(cmd_list)
    logger.info("execute command: %s", cmd_str)

    flag = True
    process = None
    stdout_thread = None
    stderr_thread = None
    subprocess_logger = SubprocessLogger(log_file)
    stdout_res = []
    stderr_res = []
    try:
        env = os.environ.copy()
        if pyvenv_flag is True:
            system_site = get_site_packages("python3")
            venv_py_exe = f"{VENV_NAME}/bin/python3"
            venv_site = get_site_packages(venv_py_exe)
            existing_pythonpath = env.get("PYTHONPATH", "")
            if existing_pythonpath:
                env["PYTHONPATH"] = (
                    f"{venv_site}{os.pathsep}{existing_pythonpath}{os.pathsep}{system_site}"
                )
            else:
                env["PYTHONPATH"] = f"{venv_site}{os.pathsep}{system_site}"
            logger.info(f"Set PYTHONPATH for subprocess: {env['PYTHONPATH']}")

        process = subprocess.Popen(
            cmd_list,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            env=env,
            bufsize=1,  # Line buffering
            close_fds=True,
            preexec_fn=os.setpgrp,  # Added: create a new process group for batch termination of child processes
        )

        # Create threads to handle stdout and stderr
        stdout_thread = threading.Thread(
            target=_process_stream,
            args=(process.stdout, subprocess_logger, stdout_res, False),
            daemon=True,
        )
        stderr_thread = threading.Thread(
            target=_process_stream,
            args=(process.stderr, subprocess_logger, stderr_res, True),
            daemon=True,
        )

        # Start threads
        stdout_thread.start()
        stderr_thread.start()
        # Wait for subprocess to complete
        return_code = process.wait()
        # Wait for threads to process remaining output
        stdout_thread.join()
        stderr_thread.join()

        if return_code != 0:
            flag = False
            logger.error(
                f"Failed to execute command: {cmd_str}, error code: {return_code}"
            )
        elif (
            check_flag
            and len(stdout_res) > 0
            and ("fail" in stdout_res[0] or "[error]" in stdout_res[0])
        ):
            flag = False
            logger.error(f"Result verification: FAILED!, command: {cmd_str}.")

    except Exception as e:
        flag = False
        logger.error(f"Failed to execute command: {cmd_str}, unknown error: {e}")
        subprocess_logger.write(sys.stderr)

    # if flag is False:
    #     reset_chips()

    if assert_flag:
        if flag is False:
            logger.warning(f"remove folder: {os.getcwd()}.")
            shutil.rmtree(os.getcwd(), ignore_errors=True)
        assert flag is True, f"Failed to execute command: {cmd_str}."

    if len(stdout_res) == 0:
        stdout_res.append("")

    return flag, stdout_res[0]


def get_platform(support_list: list) -> str | None:
    """
    Get the current platform architecture

    :param support_list: List of supported architectures
    :return: Platform name if supported, otherwise None
    """
    import platform

    system = platform.system()
    machine = platform.machine()
    logger.info(f"Only supports Linux system, current system is {system}.")

    if system == "Linux" and machine in support_list:
        return machine
    return None


def check_gpu() -> dict:
    """
    Check if NVIDIA GPU is available and retrieve GPU information

    :return: Dictionary containing GPU availability and information
    """
    result = {"has_gpu": False, "gpu_info": []}

    try:
        nvidia_smi_output = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader,nounits"],
            stderr=subprocess.STDOUT,
            text=True,
        )
        for line in nvidia_smi_output.strip().split("\n"):
            if line:
                result["has_gpu"] = True
                result["gpu_info"].append(f"NVIDIA (nvidia-smi): {line.strip()}")
    except subprocess.CalledProcessError:
        pass
    except FileNotFoundError:
        # nvidia-smi doesn't exist
        pass

    return result


def check_device_info(support_list: list) -> bool:
    """
    Check if the device has a supported number of cores

    :param support_list: List of supported core numbers
    :return: True if device has supported core count, False otherwise
    """
    if support_list is None or len(support_list) == 0:
        logger.error("No support hmm models.")
        return False

    core_num_str = "Core_Num" if HOUMO_BACKEND == "xh2" else "Core Num"
    exec_flag, opt_str = execute_test_cmd(["hm_smi", "-a"])

    lines = [
        line.split(":", 1)[-1].strip()
        for line in opt_str.split("\n")
        if core_num_str in line
    ]
    if exec_flag and lines and len(set(lines)) == 1:
        device_core_num = int(lines[0])
        if device_core_num in support_list or any(
            device_core_num % core_num == 0 for core_num in support_list
        ):
            logger.info(f"device core num: {device_core_num}")
            return True
        logger.error(
            f"Unsupported device core num {device_core_num}, expected core num: {support_list}"
        )
    else:
        logger.error(f"Unsupported device: {lines}")

    return False


def check_vpu_status() -> bool:
    """
    Check VPU status for xh1 backend

    :return: True if VPU is being used, False otherwise
    """
    if HOUMO_BACKEND != "xh1":
        return False

    exec_flag, opt_str = execute_test_cmd(["hm_smi", "-a"])
    if exec_flag:
        lines = [
            line.split(":", 1)[-1].strip()
            for line in opt_str.split("\n")
            if "Used" in line
        ]
        used_mem = float(lines[0][:-2]) if len(lines) > 0 and "MB" in lines[0] else 0
        if used_mem > 2000:
            logger.info(f"Device 0 is using the vpu driver, mem info: {lines}")
            return True
    logger.info(f"Device 0 isn't using the vpu driver, mem info: {lines}")
    return False


def install_py_env(env_dir: str, log_file: str, flow_type: str = "default") -> dict:
    """
    Install Python environment according to requirements.txt

    :param env_dir: Environment directory path
    :param log_file: Log file path
    :param flow_type: Type of flow (default, quant, etc.)
    :return: Dictionary of changed libraries
    """
    changed_libs = dict()

    rqmt_name = "requirements.txt"
    if flow_type == "quant":
        tmp_rqmt_path = os.path.join(env_dir, "requirements_ptq.txt")
        if os.path.exists(tmp_rqmt_path) and os.path.isfile(tmp_rqmt_path):
            rqmt_name = "requirements_ptq.txt"

    rqmt_path = os.path.join(env_dir, rqmt_name)
    if not os.path.exists(rqmt_path) or not os.path.isfile(rqmt_path):
        return changed_libs

    # get current python env
    pip_res = subprocess.run(
        ["pip3", "list"],
        check=True,
        text=True,
        capture_output=True,
    )
    py_env_dict = dict()
    for line in pip_res.stdout.split("\n"):
        line = line.strip()
        if "Package" in line or "--" in line:
            continue
        split_res = line.split(" ")
        lib_name = split_res[0]
        lib_ver = split_res[-1]
        py_env_dict[lib_name] = lib_ver

    with open(rqmt_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#") or "http" in line:
                continue
            lib_name = line
            if "==" in line:
                lib_name = line.split("==", 1)[0]
            changed_libs[lib_name] = py_env_dict.get(lib_name, None)

    os.chdir(env_dir)
    if os.path.exists("/opt/venv/houmo/lib/python3.12/site-packages"):
        os.system("sudo chmod 777 /opt/venv/houmo/lib/python3.12/site-packages")
        os.system(
            "sudo chmod -R 777 /opt/venv/houmo/lib/python3.12/site-packages/transformers*"
        )
    ret, _ = execute_test_cmd(["pip3", "install", "-r", rqmt_name], log_file)
    logger.info(f"Install python dependencies for the current testcase, ret: {ret}.")

    return changed_libs


def is_release() -> bool:
    """
    Check if released models should be used

    :return: True if released models should be used, False otherwise
    """
    if USE_RELEASED_MODELS and USE_RELEASED_MODELS in ["on", "ON"]:
        return True
    return False


def is_separate() -> bool:
    """
    Check if separate testing is enabled

    :return: True if separate testing is enabled, False otherwise
    """
    if SEPARATE_TEST and SEPARATE_TEST in ["OFF", "ON"]:
        return True
    return False


def get_test_type():
    """
    Determine the type of test case based on environment settings

    :return: Test case type as defined in TCaseType enum
    """
    if is_separate():
        if HDPL_PLATFORM == "ISIM":
            return TCaseType.SEPARATE_NO_INFER
        else:
            return TCaseType.SEPARATE_INFER

    return TCaseType.DEFAULT


def move_models_res(src_path: str, dst_path: str) -> bool:
    """
    Move model results from source to destination

    :param src_path: Source path
    :param dst_path: Destination path
    :return: True if successful, False otherwise
    """
    if not os.path.isdir(src_path):
        logger.error(f"Invalid source folder: {src_path}")
        return False
    if os.path.exists(dst_path):
        logger.info(f"Target folder {dst_path} already exists.")
        return True

    logger.info(f"Move model results from {src_path} -> {dst_path}")
    os.makedirs(dst_path, exist_ok=True)

    lock_file = dst_path + "/lock.lock"
    with ModelResourceLock(
        lock_file, ModelResourceLock.LockMode.WRITE, "saving model results"
    ):
        # copy onnx & hmm
        for file_path in glob(src_path + "/*"):
            if os.path.isdir(file_path):
                folder_name = file_path.rsplit("/", 1)[-1].strip()
                dst_opt_folder = os.path.join(dst_path, folder_name)
                if os.path.exists(dst_opt_folder):
                    logger.warning(f"remove folder: {dst_opt_folder}.")
                    shutil.rmtree(dst_opt_folder, ignore_errors=True)
                shutil.copytree(file_path, dst_opt_folder)
                logger.info(f"Move model folder {file_path} -> {dst_opt_folder}")
                continue

            if not not file_path.endswith(".hmm") and not file_path.endswith(".onnx"):
                continue
            file_name = file_path.rsplit("/", 1)[-1].strip()
            dst_file_path = os.path.join(dst_path, file_name)
            logger.info(f"Move model file {file_path} -> {dst_file_path}")
            shutil.copy2(file_path, dst_file_path)

    return True


def restore_models_res(src_folder: str, dst_folder: str) -> bool:
    """
    Restore model results from source to destination

    :param src_folder: Source folder path
    :param dst_folder: Destination folder path
    :return: True if successful, False otherwise
    """
    if not os.path.isdir(src_folder):
        logger.info(f"Failed to restore result: {src_folder} -> {dst_folder}")
        return False
    logger.info(f"Restore result: {src_folder} -> {dst_folder}")
    os.makedirs(dst_folder, exist_ok=True)

    for item in os.listdir(src_folder):
        src_path = os.path.join(src_folder, item)
        dst_path = os.path.join(dst_folder, item)

        # Skip files with .lock extension
        if os.path.isfile(src_path) and item.endswith(".lock"):
            continue

        if os.path.isdir(src_path):
            logger.info(f"Restore folder: {src_path} -> {dst_path}")
            if os.path.exists(dst_path):
                shutil.rmtree(dst_path)
            shutil.copytree(src_path, dst_path, copy_function=shutil.copy2)
        elif os.path.isfile(src_path):
            logger.info(f"Restore file: {src_path} -> {dst_path}")
            shutil.copy2(src_path, dst_path)
        else:
            logger.warning(f"Skip result file: {src_path}")


def prepare_test_folder(model_dir: str, test_type: str):
    """
    Prepare a test folder with timestamp

    :param model_dir: Base model directory
    :param test_type: Type of test
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    test_folder = f"{model_dir}_{test_type}_{timestamp}"
    logger.info(f"create test_folder: {model_dir} -> {test_folder}")
    if os.path.exists(test_folder):
        logger.warning(f"remove folder: {test_folder}.")
        shutil.rmtree(test_folder)
    shutil.copytree(model_dir, test_folder)
    os.chdir(test_folder)


def reset_chips():
    """
    Reset hardware chips
    """
    logger.warning("Ready to reset chips.")
    cmd = "/usr/local/houmo-sdk/hal/utility/ipu_reset"
    if HOUMO_BACKEND != "xh2" or not os.path.exists(cmd):
        cmd = "/usr/local/houmo-sdk/scripts/reset_aicore.sh"
    os.system(cmd)


def display_to_console(
    log_file: str,
    test_type: str,
    model_name: str,
    res_flag: bool,
    force_print: bool = False,
):
    """
    Display test results to console

    :param log_file: Log file path
    :param test_type: Type of test
    :param model_name: Name of the model
    :param res_flag: Result flag (True for success, False for failure)
    :param force_print: Whether to force printing regardless of result
    """
    if get_test_type() != TCaseType.DEFAULT and (
        res_flag is False or force_print is True
    ):
        _, log_str = execute_test_cmd(["cat", log_file])
        print(f"[execute {test_type} flow: {model_name}]\n {log_str}")


def check_device_markers(pytest_request):
    all_markers = [marker.name for marker in pytest_request.node.own_markers]
    target_prefixes = [f"{NDEVICE_MARKER}_", f"{DEVICE_MEM_MARKER}_"]
    marker_vals = {}
    for marker in all_markers:
        for prefix in target_prefixes:
            if marker.startswith(prefix):
                marker_key = prefix.rstrip("_")
                marker_vals[marker_key] = marker

    if (
        not marker_vals
        or marker_vals.get(NDEVICE_MARKER, None) is None
        or marker_vals.get(DEVICE_MEM_MARKER, None) is None
    ):
        logger.warning("Device markers are not properly set for this test case.")
        pytest.skip("Device markers are not properly set for this test case.")

    return marker_vals
