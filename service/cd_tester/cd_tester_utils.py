import os
import logging
import subprocess
from datetime import datetime
import threading
import sys

HOUMO_BACKEND = os.getenv("HOUMO_TARGET", "xh1")
script_dir = os.path.dirname(os.path.abspath(__file__))
IMODELZOO_REPO_DIR = os.path.abspath(f"{script_dir}/../../")


def setup_logging(log_dir: str = None, log_name: str = "cd_tester"):
    log_file = ""
    logger_handlers = [logging.StreamHandler()]  # 输出到控制台
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    if log_dir is not None:
        os.makedirs(log_dir, exist_ok=True)
        log_file = f"{log_dir}/{log_name}_{timestamp}.log"
        logger_handlers.append(logging.FileHandler(log_file))

    # 日志格式：时间 - 级别 - 模块.函数 - 消息
    log_format = '%(asctime)s - %(levelname)s - %(module)s.%(funcName)s - %(message)s'

    # 配置日志级别（DEBUG < INFO < WARNING < ERROR < CRITICAL）
    logging.basicConfig(
        level=logging.INFO,  # 基础日志级别
        format=log_format,  # 日志格式
        handlers=logger_handlers,
    )

    return log_file


def reset_chips():
    cmd = "/usr/local/houmo-sdk/hal/utility/ipu_reset"
    if HOUMO_BACKEND != "xh2" or not os.path.exists(cmd):
        cmd = "/usr/local/houmo-sdk/scripts/reset_aicore.sh"
    os.system(cmd)


def run_tests(cmd_list, log_file):
    ret = execute_cmd(cmd_list, log_file)
    reset_chips()
    if not ret:
        return False
    return True


class SubprocessLogger:
    def __init__(self, log_file):
        """
        初始化输出日志器

        :param log_file: 日志文件路径
        """
        # self.log_file = log_file
        # os.makedirs(os.path.dirname(log_file), exist_ok=True)
        # 创建文件锁，确保多线程写入安全
        self.lock = threading.Lock()

    def write(self, message, stream=sys.stdout):
        """
        同时输出到屏幕和日志文件

        :param message: 要输出的消息
        :param stream: 输出到屏幕的流（stdout或stderr）
        """
        if (
            not message
            or (" eta " in message and " kB" in message)
            or ("speed:" in message and "last:" in message)
        ):
            return

        # 输出到屏幕
        stream.write(message)
        stream.flush()


def _process_stream(stream, logger, is_stderr=False):
    """
    处理子进程的输出流

    :param stream: 子进程的输出流（stdout或stderr）
    :param logger: SubprocessLogger实例
    :param is_stderr: 是否为错误流
    """
    stream_obj = sys.stderr if is_stderr else sys.stdout
    try:
        for line in iter(stream.readline, ''):
            logger.write(line, stream_obj)
    finally:
        stream.close()


def execute_cmd(cmd_list: list, log_file: str = None) -> bool:
    logger = logging.getLogger(__name__)

    cmd_str = " ".join(str(item) for item in cmd_list)
    logger.info("execute command: %s", cmd_str)

    flag = True
    subprocess_logger = SubprocessLogger(log_file)

    try:
        process = subprocess.Popen(
            cmd_list,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,  # 行缓冲，确保实时输出
            universal_newlines=True,
        )

        # 创建线程处理stdout和stderr
        stdout_thread = threading.Thread(
            target=_process_stream,
            args=(process.stdout, subprocess_logger, False),
            daemon=True,
        )
        stderr_thread = threading.Thread(
            target=_process_stream,
            args=(process.stderr, subprocess_logger, True),
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

        if return_code != 0 and return_code != 5:
            flag = False
            logger.error(
                f"Failed to execute command: {cmd_str}, error code: {return_code}"
            )

    except Exception as e:
        flag = False
        logger.error(f"Failed to execute command: {cmd_str}, unknown error: {e}")
        if process.stdout:
            subprocess_logger.write(process.stdout)
        if process.stderr:
            subprocess_logger.write(process.stderr)

    return flag
