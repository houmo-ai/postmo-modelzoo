# Copyright 2025 HOUMO AI
#
# File: compiler_utils.py
# Description:
#   Compiler utilities for LLM model compilation and execution.
#   This module provides utility functions and classes for compiling LLM models,
#   managing Docker containers for execution, logging, and file operations.
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
import logging
import subprocess
from datetime import datetime
from typing import Tuple
import threading
import sys


def setup_logging(
    log_dir: str = None, log_name: str = "compile_llms", log_file: str = ""
):
    """
    Set up logging configuration for the application.

    Args:
        log_dir: Directory to store log files
        log_name: Base name for the log file
        log_file: Specific log file path

    Returns:
        tuple: (log_file_path, timestamp)
    """
    logger_handlers = [logging.StreamHandler()]  # Output to console
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    if log_dir is not None:
        os.makedirs(log_dir, exist_ok=True)
        log_file = f"{log_dir}/{log_name}_{timestamp}.log"
        logger_handlers.append(logging.FileHandler(log_file))
    elif log_file:
        logger_handlers.append(logging.FileHandler(log_file))

    # Log format: time - level - module.function - message
    log_format = "%(asctime)s - %(levelname)s - %(module)s.%(funcName)s - %(message)s"

    # Configure log level (DEBUG < INFO < WARNING < ERROR < CRITICAL)
    logging.basicConfig(
        level=logging.INFO,  # Basic log level
        format=log_format,  # Log format
        handlers=logger_handlers,
    )

    return log_file, timestamp


class SubprocessLogger:
    def __init__(self, log_file=None):
        """
        Initialize the subprocess output logger.

        Args:
            log_file (str): Path to the log file for output
        """
        self.log_file = log_file if log_file else None
        if log_file:
            os.makedirs(os.path.dirname(log_file), exist_ok=True)
        # Create a file lock to ensure thread-safe writing
        self.lock = threading.Lock()

    def write(self, message, stream=sys.stdout) -> None:
        """
        Output message to both screen and log file simultaneously.

        Args:
            message (str): Message to output
            stream: Output stream (stdout or stderr) to write to screen
        """
        if not message:
            return

        if self.log_file:
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            log_message = f"[{timestamp}] {message}"

            # Write to log file (with locking to ensure thread safety)
            with self.lock:
                with open(self.log_file, "a", encoding="utf-8") as f:
                    f.write(log_message)

        # Output to screen
        stream.write(message)
        stream.flush()


def _process_stream(stream, logger, is_stderr=False, outputs=None):
    """
    Process output streams from subprocess.

    Args:
        stream: Subprocess output stream (stdout or stderr)
        logger (SubprocessLogger): Instance of SubprocessLogger for output
        is_stderr (bool): Whether this stream is stderr (True) or stdout (False)
        outputs (list): List to store output lines
    """
    stream_obj = sys.stderr if is_stderr else sys.stdout
    try:
        for line in iter(stream.readline, ""):
            logger.write(line, stream_obj)
            if outputs is not None:
                outputs.append(line)
    finally:
        stream.close()


def execute_cmd(
    cmd_list: list, log_file: str = None, get_outputs=False
) -> Tuple[bool, list]:
    """Execute a command with real-time output logging.

    Args:
        cmd_list (list): List of command arguments to execute
        log_file (str, optional): Path to log file for output. If None, only console output.
        get_outputs (bool, optional): Whether to return the output lines. Defaults to False.

    Returns:
        Tuple[bool, list]: Tuple of (return code, output)
    """
    if log_file:
        setup_logging(log_file=log_file)
    logger = logging.getLogger(__name__)

    # Convert command list to string for logging
    cmd_str = " ".join(str(item) for item in cmd_list)
    logger.info("execute command: %s", cmd_str)

    flag = True
    # Initialize subprocess logger
    subprocess_logger = SubprocessLogger(log_file)

    outputs = list() if get_outputs else None
    try:
        process = subprocess.Popen(
            cmd_list,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,  # Line buffering to ensure real-time output
            universal_newlines=True,
        )

        # Create threads to handle stdout and stderr streams
        stdout_thread = threading.Thread(
            target=_process_stream,
            args=(process.stdout, subprocess_logger, False, outputs),
            daemon=True,
        )
        stderr_thread = threading.Thread(
            target=_process_stream,
            args=(process.stderr, subprocess_logger, True, outputs),
            daemon=True,
        )

        # Start the output processing threads
        stdout_thread.start()
        stderr_thread.start()
        # Wait for subprocess to complete
        return_code = process.wait()
        # Wait for threads to finish processing remaining output
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
        if process.stdout:
            subprocess_logger.write(process.stdout)
        if process.stderr:
            subprocess_logger.write(process.stderr)

    if get_outputs:
        return flag, outputs
    return flag


def compress_to_zip(folder_path: str, output_path: str, extensions=None) -> bool:
    """Compress a folder to a zip file.

    Args:
        folder_path (str): The path to the folder to be compressed.
        output_path (str): The path to the output zip file.
        extensions (list, optional): A list of file extensions to include in the zip file. Defaults to None.

    Returns:
        bool: True if the compression was successful, False otherwise.
    """
    import zipfile

    logger = logging.getLogger(__name__)

    if not os.path.isdir(folder_path):
        logger.error(f"{folder_path} not exist.")
        return False
    if output_path is None:
        logger.error("Please provide output file path.")
        return False

    folder_path = os.path.normpath(folder_path)
    cpp_suffix = ".cpp"
    try:
        files_to_zip = list()
        cpp_files = list()
        for root, dirs, files in os.walk(folder_path):
            relative_path = os.path.relpath(root, folder_path)
            for file in files:
                # Get the file name and extension
                file_name = file
                file_ext = os.path.splitext(file_name)[1].lower()

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

                # Complete Path of the source file
                file_path = os.path.join(root, file)
                # Calculate the relative path in the zip file (excluding the root directory)
                arcname = os.path.join(relative_path, file)
                if tcim_flag and file_name.endswith(cpp_suffix):
                    cpp_files.append((file_path, arcname))
                else:
                    files_to_zip.append((file_path, arcname))

        if len(files_to_zip) == 0:
            logger.warning(f"Warning: Specified type of file not found.({extensions})")
            return False

        with zipfile.ZipFile(output_path, "w", zipfile.ZIP_DEFLATED) as zipf:
            for file_path, arcname in files_to_zip:
                zipf.write(file_path, arcname)

        if len(cpp_files) > 0:
            cpp_zip_path = output_path[:-4] + "_cpps.zip"
            with zipfile.ZipFile(cpp_zip_path, "w", zipfile.ZIP_DEFLATED) as zipf:
                for file_path, arcname in cpp_files:
                    zipf.write(file_path, arcname)
        logger.info(
            f"Compresss Done, save compressed file to {os.path.abspath(output_path)}"
        )
        return True

    except Exception as e:
        logger.error(f"Compresss Failed: {str(e)}")
        # Clean up incomplete output files
        if os.path.exists(output_path):
            os.remove(output_path)
        return False


def delete_files(folder_path: str, extensions: list = list()):
    """Delete all files in the specified folder whose extensions are in the given list.

    Args:
        folder_path (str): The path of the folder to be traversed
        extensions (list, optional): The list of file suffixes to be deleted, such as ['.txt', '.log']
    """
    logger = logging.getLogger(__name__)

    logger.info(f"Clear folder: {folder_path}, specify extensions: {extensions}")
    # 检查文件夹是否存在
    if not os.path.exists(folder_path) or not os.path.isdir(folder_path):
        logger.error(f"Folder does not exist - {folder_path}")
        return

    import shutil

    if ".DELETE_ALL" in extensions or not extensions:
        # Delete the entire folder
        shutil.rmtree(folder_path, ignore_errors=True)
        return

    # Make sure the suffix starts with a dot and maintain a consistent format.
    normalized_extensions = [
        ext if ext.startswith(".") else f".{ext}" for ext in extensions
    ]
    deleted_files_count = 0
    deleted_folders_count = 0
    # First round: Delete files that meet the criteria
    for root, dirs, files in os.walk(folder_path):
        for file in files:
            if any(file.endswith(ext) for ext in normalized_extensions):
                file_path = os.path.join(root, file)
                try:
                    os.remove(file_path)
                    deleted_files_count += 1
                except Exception as e:
                    logger.error(f"删除文件失败 {file_path}: {e}")

    # Second Pass: Delete Empty Folders (Need to traverse in reverse order, delete subfolders first)
    for root, dirs, files in os.walk(folder_path, topdown=False):
        for dir_name in dirs:
            dir_path = os.path.join(root, dir_name)
            try:
                if not os.listdir(dir_path):
                    shutil.rmtree(dir_path, ignore_errors=True)
                    deleted_folders_count += 1
            except Exception as e:
                logger.error(f"Failed to delete the folder {dir_path}: {e}")

    logger.info(
        f"Operation completed. A total of {deleted_files_count} files and {deleted_folders_count} empty folders were deleted."
    )


def update_perf_file(perf_id: str, update_vals: dict) -> None:
    """Update the perf file with the given values.

    Args:
        perf_id (str): The ID of the perf file to update.
        update_vals (dict): The values to update in the perf file.
    """
    import pandas as pd

    logger = logging.getLogger(__name__)

    file_exists = True
    script_dir = os.path.dirname(os.path.abspath(__file__))
    perf_file = f"{script_dir}/perf_results_{perf_id}.csv"
    if not os.path.exists(perf_file):
        file_exists = False
        os.makedirs(os.path.dirname(perf_file), exist_ok=True)
        logger.info(f"create perf result file: {perf_file}")

    df = pd.DataFrame([update_vals])
    df.to_csv(
        perf_file,
        mode="a",
        header=not file_exists,
        index=False,
        encoding="utf-8",
    )


def get_perf_models(perf_id: str):
    """
    Retrieve performance model data from a CSV file based on the given performance ID.

    Args:
        perf_id (str): The performance test identifier used to locate the corresponding
                      CSV file。

    Returns:
        list or None: A list of dictionaries where each dictionary represents a row
                     from the CSV file with column names as keys. Returns None if
                     the performance result file does not exist.
    """
    import pandas as pd

    logger = logging.getLogger(__name__)

    script_dir = os.path.dirname(os.path.abspath(__file__))
    perf_file = f"{script_dir}/perf_results_{perf_id}.csv"
    if not os.path.exists(perf_file):
        logger.error(f"Failed to read perf result file {perf_file}.")
        return None

    df = pd.read_csv(perf_file, encoding="utf-8")
    perf_models = df.to_dict(orient="records")

    return perf_models


def update_perf_values(perf_id: str, perf_vals: dict) -> bool:
    """
    Update performance values in a CSV file by merging with new performance data.

    Args:
        perf_id (str): The performance test identifier used to locate the corresponding
                      CSV file.
        perf_vals (dict): A dictionary containing new performance values to be merged
                         with existing data.

    Returns:
        bool: True if the update operation was successful, False if the performance
              result file does not exist or if any other error occurs during the process.
    """
    import pandas as pd

    logger = logging.getLogger(__name__)

    script_dir = os.path.dirname(os.path.abspath(__file__))
    perf_file = f"{script_dir}/perf_results_{perf_id}.csv"
    if not os.path.exists(perf_file):
        logger.error(f"Failed to read perf result file {perf_file}.")
        return False

    df = pd.read_csv(perf_file, encoding="utf-8")
    perf_df = pd.DataFrame([perf_vals])
    df = pd.merge(df, perf_df, on=["model"], how="inner")
    df.to_csv(
        perf_file,
        mode="w",
        index=False,
        encoding="utf-8",
    )
    return True
