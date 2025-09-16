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
            if process.stdout:
                subprocess_logger.write(process.stdout)
            if process.stderr:
                subprocess_logger.write(process.stderr)

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
    cpp_suffix = ".cpp"
    try:
        files_to_zip = list()
        cpp_files = list()
        for root, dirs, files in os.walk(folder_path):
            relative_path = os.path.relpath(root, folder_path)
            for file in files:
                # 获取文件名和扩展名
                file_name = file
                file_ext = os.path.splitext(file_name)[1].lower()

                # 检查是否符合条件
                if extensions and file_ext not in extensions and file_ext != cpp_suffix:
                    continue
                tcim_flag = True if "tcim" in root.split(os.sep) else False
                if (
                    extensions
                    and ".hmm" in extensions
                    and file_ext == ".hmm"
                    and tcim_flag
                ):
                    # skip tcim/*.hmm
                    continue
                if tcim_flag and (
                    not file_name.endswith(cpp_suffix)
                    or file_name.count(cpp_suffix) != 1
                ):
                    continue

                # 源文件完整路径
                file_path = os.path.join(root, file)
                # 计算在zip中的相对路径（不含根目录）
                arcname = os.path.join(relative_path, file)
                if tcim_flag and file_name.endswith(cpp_suffix):
                    cpp_files.append((file_path, arcname))
                else:
                    files_to_zip.append((file_path, arcname))

        if len(files_to_zip) == 0:
            logger.warning(f"Warning: Specified type of file not found.({extensions})")
            return False

        with zipfile.ZipFile(output_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
            for file_path, arcname in files_to_zip:
                zipf.write(file_path, arcname)

        if len(cpp_files) > 0:
            cpp_zip_path = output_path[:-4] + "_cpps.zip"
            with zipfile.ZipFile(cpp_zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
                for file_path, arcname in cpp_files:
                    zipf.write(file_path, arcname)
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


def delete_files(folder_path: str, extensions: list = list()):
    """
    删除指定文件夹下所有后缀在给定列表中的文件

    参数:
        folder_path (str): 要遍历的文件夹路径
        extensions (list): 要删除的文件后缀列表，例如 ['.txt', '.log']
    """
    logger = logging.getLogger(__name__)

    logger.info(f"清理文件夹: {folder_path}, 指定后缀: {extensions}")
    # 检查文件夹是否存在
    if not os.path.exists(folder_path) or not os.path.isdir(folder_path):
        logger.error(f"错误: 文件夹不存在 - {folder_path}")
        return

    import shutil

    if ".DELETE_ALL" in extensions or not extensions:
        # 删除整个文件夹
        shutil.rmtree(folder_path, ignore_errors=True)
        return

    # 确保后缀以点开头，统一格式
    normalized_extensions = [
        ext if ext.startswith('.') else f'.{ext}' for ext in extensions
    ]
    deleted_files_count = 0
    deleted_folders_count = 0
    # 第一遍：删除符合条件的文件
    for root, dirs, files in os.walk(folder_path):
        for file in files:
            if any(file.endswith(ext) for ext in normalized_extensions):
                file_path = os.path.join(root, file)
                try:
                    os.remove(file_path)
                    deleted_files_count += 1
                except Exception as e:
                    logger.error(f"删除文件失败 {file_path}: {e}")

    # 第二遍：删除空文件夹（需要逆序遍历，先删除子文件夹）
    for root, dirs, files in os.walk(folder_path, topdown=False):
        for dir_name in dirs:
            dir_path = os.path.join(root, dir_name)
            try:
                # 检查文件夹是否为空
                if not os.listdir(dir_path):
                    shutil.rmtree(dir_path, ignore_errors=True)
                    deleted_folders_count += 1
            except Exception as e:
                logger.error(f"删除文件夹失败 {dir_path}: {e}")

    logger.info(
        f"\n操作完成，共删除 {deleted_files_count} 个文件，{deleted_folders_count} 个空文件夹"
    )
