import os
import logging
import subprocess
from datetime import datetime
import threading
import sys


def setup_logging(log_dir: str = None, log_name: str = "compile_llms"):
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


class SubprocessLogger:
    def __init__(self, log_file):
        """
        初始化输出日志器

        :param log_file: 日志文件路径
        """
        self.log_file = log_file
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
        # 创建文件锁，确保多线程写入安全
        self.lock = threading.Lock()

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

        # 写入日志文件（加锁确保线程安全）
        with self.lock:
            with open(self.log_file, 'a', encoding='utf-8') as f:
                f.write(log_message)

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
    if log_file:
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

        if log_file:
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

        if log_file:
            # 等待线程处理剩余输出
            stdout_thread.join()
            stderr_thread.join()

        if return_code != 0:
            flag = False
            logger.error(
                f"Failed to execute command: {cmd_str}, error code: {return_code}"
            )

    except Exception as e:
        flag = False
        logger.error(f"Failed to execute command: {cmd_str}, unknown error: {e}")
        if log_file:
            subprocess_logger.write(sys.stderr)

    return flag


def compress_to_zip(folder_path: str, output_path: str, extensions=None) -> bool:
    """
    将指定文件夹压缩为zip格式

    :param folder_path: 要压缩的文件夹路径
    :param output_path: 输出zip文件的路径
    :param extensions: 只压缩指定格式的文件
    :return: 压缩成功返回True, 失败返回False
    """
    import zipfile

    logger = logging.getLogger(__name__)

    if not os.path.isdir(folder_path):
        logger.error(f"Error: {folder_path} not exist.")
        return False
    if output_path is None:
        logger.error(f"Error: please provide output file path.")
        return False

    # 规范化文件夹路径（去除末尾的斜杠）
    folder_path = os.path.normpath(folder_path)

    try:
        # 创建ZIP文件
        with zipfile.ZipFile(output_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
            file_count = 0
            # 遍历文件夹内所有内容
            for root, dirs, files in os.walk(folder_path):
                for file in files:
                    file_ext = os.path.splitext(file)[1].lower()
                    if extensions and file_ext not in extensions:
                        continue

                    file_path = os.path.join(root, file)
                    arcname = os.path.relpath(file_path, os.path.dirname(folder_path))
                    zipf.write(file_path, arcname)
                    file_count += 1

            if file_count == 0:
                logger.warning(
                    f"Warning: Specified type of file not found.({extensions})"
                )
                # 移除空的zip文件
                os.remove(output_path)
                return False

        logger.info(
            f"Compresss Done, save compressed file to {os.path.abspath(output_path)}"
        )
        return True

    except Exception as e:
        logger.error(f"Compresss Failed: {str(e)}")
        # 清理不完整的输出文件
        if os.path.exists(output_path):
            os.remove(output_path)
        return False


def compress_to_tar_gz(
    folder_path, output_path, extensions=None, include_parent=True, compression_level=9
) -> str:
    """
    将指定文件夹压缩为tar.gz格式

    :param folder_path: 要压缩的文件夹路径
    :param output_path: 输出tar.gz文件的路径，默认为文件夹同名的tar.gz
    :param extensions: 只压缩指定格式的文件
    :param include_parent: 是否包含父文件夹本身（False则只压缩文件夹内内容）
    :param compression_level: 压缩级别（1-9，9为最高压缩率）
    :return: 压缩成功返回tar.gz文件路径，失败返回None
    """
    import tarfile

    logger = logging.getLogger(__name__)
    # 验证文件夹是否存在
    if not os.path.isdir(folder_path):
        logger.error(f"Error: {folder_path} not exist.")
        return None

    # 规范化文件夹路径（去除末尾的斜杠）
    folder_path = os.path.normpath(folder_path)

    # 确定输出文件路径
    if output_path is None:
        folder_name = os.path.basename(folder_path)
        output_path = f"{folder_name}.tar.gz"

    # 验证压缩级别
    if not (1 <= compression_level <= 9):
        logger.warning(
            "Warning: compression level must be between 1 and 9. It has been set to 9."
        )
        compression_level = 9

    try:
        # 创建tar.gz文件，使用gzip压缩
        with tarfile.open(
            output_path,
            mode=f"w:gz",  # w:gz表示写入gzip压缩的tar文件
            compresslevel=compression_level,
        ) as tar:
            file_count = 0
            # 遍历文件夹内所有内容
            for root, dirs, files in os.walk(folder_path):
                for file in files:
                    file_ext = os.path.splitext(file)[1].lower()
                    if extensions and file_ext not in extensions:
                        continue

                    file_path = os.path.join(root, file)
                    # 确定在tar中的相对路径
                    if include_parent:
                        # 包含父文件夹（如"parent/child/file.txt"）
                        arcname = os.path.relpath(
                            file_path, os.path.dirname(folder_path)
                        )
                    else:
                        # 不包含父文件夹（如"child/file.txt"）
                        arcname = os.path.relpath(file_path, folder_path)

                    # 添加文件到tar
                    tar.add(file_path, arcname=arcname)
                    file_count += 1

            if file_count == 0:
                logger.warning(
                    f"Warning: Specified type of file not found.({extensions})"
                )
                # 移除空的tar文件
                os.remove(output_path)
                return None

        logger.info(
            f"Compresss Done, save compressed file to {os.path.abspath(output_path)}"
        )
        return output_path

    except Exception as e:
        logger.error(f"Compresss Failed: {str(e)}")
        # 清理不完整的输出文件
        if os.path.exists(output_path):
            os.remove(output_path)
        return None
