import os
import subprocess
import logging
import json
import shutil
import fcntl
from datetime import datetime
from glob import glob
import enum
import time
import threading
import sys

HOUMO_BACKEND = os.getenv("HOUMO_TARGET", "xh1")
SKIP_INFER = os.getenv("SKIP_INFER", None)
HDPL_PLATFORM = os.getenv("HDPL_PLATFORM", "")
MODELS_PATH = os.getenv("IMODELZOO_MODELS_PATH", "./")
CI_MODELS_RES_PATH = os.path.dirname(os.path.abspath(__file__)) + "/../model_results"
logger = logging.getLogger(__name__)


class ModelResourceLock:
    """
    Model resource lock.
    """

    class LockMode(enum.Enum):
        """Lock mode for a `ModelResourceLock`"""

        NO_LOCK = None
        READ_ONLY = fcntl.LOCK_SH
        WRITE = fcntl.LOCK_EX

    # all tests start simultaneously, expect their maximum end time to be 2 hours
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
        assert self.lock_fd is not None
        fcntl.flock(self.lock_fd, fcntl.LOCK_UN)
        os.close(self.lock_fd)


def load_json(json_path: str) -> dict:
    if not os.path.exists(json_path):
        return None

    with open(json_path, "r", encoding="utf-8") as f:
        json_info = json.load(f)
    logger.info(f"Loaded config file {json_path}")

    return json_info


class SubprocessLogger:
    def __init__(self, log_file=""):
        """
        初始化输出日志器

        :param log_file: 日志文件路径
        """
        self.log_file = log_file
        if log_file:
            os.makedirs(os.path.dirname(log_file), exist_ok=True)
        # 创建文件锁，确保多线程写入安全
        self.lock = threading.Lock()
        self.write_flag = True if log_file else False

    def write(self, message, stream=sys.stdout):
        """
        同时输出到屏幕和日志文件

        :param message: 要输出的消息
        :param stream: 输出到屏幕的流（stdout或stderr）
        """
        if not message:
            return

        # 添加时间戳
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        log_message = f"[{timestamp}] {message}"

        if self.write_flag:
            # 写入日志文件（加锁确保线程安全）
            with self.lock:
                with open(self.log_file, 'a', encoding='utf-8') as f:
                    f.write(log_message)

        # 输出到屏幕
        stream.write(message)
        stream.flush()


def _process_stream(stream, logger, results: list, is_stderr=False):
    """
    处理子进程的输出流

    :param stream: 子进程的输出流（stdout或stderr）
    :param logger: SubprocessLogger实例
    :param is_stderr: 是否为错误流
    """
    stream_obj = sys.stderr if is_stderr else sys.stdout
    outputs = list()
    try:
        for line in iter(stream.readline, ''):
            logger.write(line, stream_obj)
            outputs.append(line.strip())
    finally:
        stream.close()

    results.append("\n".join(outputs))


def execute_test_cmd(
    cmd_list: list,
    log_file: str = "",
    assert_flag: bool = False,
    check_flag: bool = True,
) -> tuple[bool, any]:
    cmd_str = " ".join(cmd_list)
    logger.info("execute command: %s", cmd_str)

    flag = True
    subprocess_logger = SubprocessLogger(log_file)
    try:
        process = subprocess.Popen(
            cmd_list,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,  # 行缓冲
            universal_newlines=True,
        )

        stdout_res = list()
        # 创建线程处理stdout和stderr
        stdout_thread = threading.Thread(
            target=_process_stream,
            args=(process.stdout, subprocess_logger, stdout_res, False),
            daemon=True,
        )
        stderr_res = list()
        stderr_thread = threading.Thread(
            target=_process_stream,
            args=(process.stderr, subprocess_logger, stderr_res, True),
            daemon=True,
        )
        # 启动线程
        stdout_thread.start()
        stderr_thread.start()

        # 等待子进程完成
        return_code = process.wait()

        # 等待线程处理剩余输出
        stdout_thread.join()
        stderr_thread.join()

        if return_code != 0:
            flag = False
            logger.error(
                f"Failed to execute command: {cmd_str}, error code: {return_code}"
            )
        else:
            if check_flag and len(stdout_res) > 0 and "fail" in stdout_res[0]:
                flag = False
                logger.error(f"Result verification: FAILED!, command: {cmd_str}.")

    except Exception as e:
        flag = False
        logger.error(f"Failed to execute command: {cmd_str}, unknown error: {e}")
        subprocess_logger.write(sys.stderr)

    if assert_flag:
        if flag is False:
            logger.warning(f"remove folder: {os.getcwd()}.")
            shutil.rmtree(os.getcwd())
        assert flag is True, f"Failed to execute command: {cmd_str}."

    if len(stdout_res) == 0:
        stdout_res.append("")

    return flag, stdout_res[0]


def get_platform(support_list: list) -> str:
    import platform

    system = platform.system()
    machine = platform.machine()
    logger.info(f"Only supports Linux system, current system is {system}.")

    if system == "Linux" and machine in support_list:
        return machine
    return None


def check_device_info(support_list: list) -> bool:
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


def install_py_env(env_dir: str, log_file: str) -> dict:
    """Install python env according to requirements.txt."""
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

    changed_libs = dict()
    rqmt_path = os.path.join(env_dir, "requirements.txt")
    if os.path.exists(rqmt_path) and os.path.isfile(rqmt_path):
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
        ret, _ = execute_test_cmd(
            ["pip3", "install", "-r", "requirements.txt"], log_file
        )
        logger.info(
            f"Install python dependencies for the current testcase, ret: {ret}."
        )

    return changed_libs


def is_ci() -> bool:
    if SKIP_INFER and SKIP_INFER in ["OFF", "ON"]:
        return True
    return False


def check_ci_simulator() -> bool:
    if is_ci() and HDPL_PLATFORM == "ISIM":
        return True
    return False


def move_models_res(src_path: str, dst_path: str) -> bool:
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
            if not file_path.endswith(".hmm") and not file_path.endswith(".onnx"):
                continue
            file_name = file_path.rsplit("/", 1)[-1].strip()
            dst_file_path = os.path.join(dst_path, file_name)
            logger.info(f"Move model file {file_path} -> {dst_file_path}")
            shutil.copy2(file_path, dst_file_path)

        if not os.path.isdir(src_path):
            return True

        src_opt_folder = os.path.join(src_path, "output")
        if os.path.exists(src_opt_folder):
            dst_opt_folder = os.path.join(dst_path, "output")
            if os.path.exists(dst_opt_folder):
                logger.warning(f"remove folder: {dst_opt_folder}.")
                shutil.rmtree(dst_opt_folder)
            shutil.copytree(src_opt_folder, dst_opt_folder)
    return True


def restore_models_res(src_folder: str, dst_folder: str) -> bool:
    if not os.path.isdir(src_folder):
        return False
    os.makedirs(dst_folder, exist_ok=True)

    for item in os.listdir(src_folder):
        src_path = os.path.join(src_folder, item)
        dst_path = os.path.join(dst_folder, item)

        if os.path.isdir(src_path):
            restore_models_res(src_path, dst_path)
        else:
            if os.path.exists(dst_path):
                os.remove(dst_path)
            shutil.copy2(src_path, dst_path)
            logger.info(f"Restore file: {src_path} -> {dst_path}")


def prepare_test_folder(model_dir: str, test_type: str):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    test_folder = f"{model_dir}_{test_type}_{timestamp}"
    logger.info(f"create test_folder: {model_dir} -> {test_folder}")
    if os.path.exists(test_folder):
        logger.warning(f"remove folder: {test_folder}.")
        shutil.rmtree(test_folder)
    shutil.copytree(model_dir, test_folder)
    os.chdir(test_folder)


def display_ci_logs(
    log_file: str,
    test_type: str,
    model_name: str,
    res_flag: bool,
    force_print: bool = False,
):
    if check_ci_simulator() and (res_flag is False or force_print is True):
        _, log_str = execute_test_cmd(["cat", log_file])
        print(f"[execute {test_type} flow: {model_name}]\n {log_str}")
