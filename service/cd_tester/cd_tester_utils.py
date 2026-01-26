# Copyright 2025 HOUMO AI
#
# File: cd_tester_utils.py
# Description:
#   Utility functions for continuous deployment testing, including logging setup,
#   chip reset functionality, subprocess execution with real-time output logging,
#   and test execution management.
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
import threading
import sys

HOUMO_BACKEND = os.getenv("HOUMO_TARGET")

script_dir = os.path.dirname(os.path.abspath(__file__))
IMODELZOO_REPO_DIR = os.path.abspath(f"{script_dir}/../../")


def setup_logging(log_dir: str = None, log_name: str = "cd_tester"):
    """Set up logging configuration for the CD tester utility.

    Args:
        log_dir (str, optional): Directory to save log files. If None, only console logging is enabled.
        log_name (str): Base name for the log file (without extension)

    Returns:
        str: Path to the created log file, or empty string if no file logging
    """
    log_file = ""
    logger_handlers = [logging.StreamHandler()]  # Output to console
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    if log_dir is not None:
        os.makedirs(log_dir, exist_ok=True)
        log_file = f"{log_dir}/{log_name}_{timestamp}.log"
        logger_handlers.append(logging.FileHandler(log_file))

    # Define log format: time - level - module.function - message
    log_format = "%(asctime)s - %(levelname)s - %(module)s.%(funcName)s - %(message)s"

    # Configure logging with specified format and handlers
    logging.basicConfig(
        level=logging.INFO,  # Base logging level
        format=log_format,  # Log format
        handlers=logger_handlers,
    )

    return log_file


def reset_chips() -> None:
    """Reset the AI chips based on the target backend platform."""

    cmd = "/usr/local/houmo-sdk/hal/utility/ipu_reset"
    if HOUMO_BACKEND != "xh2" or not os.path.exists(cmd):
        cmd = "/usr/local/houmo-sdk/scripts/reset_aicore.sh"
    os.system(cmd)


def run_tests(cmd_list, log_file) -> bool:
    """Execute a series of tests and perform chip reset afterwards.

    Args:
        cmd_list (list): List of command arguments to execute
        log_file (str): Path to the log file for output

    Returns:
        bool: True if tests executed successfully, False otherwise
    """

    ret = execute_cmd(cmd_list, log_file)
    reset_chips()
    if not ret:
        return False
    return True


class SubprocessLogger:
    def __init__(self, log_file):
        """
        Initialize the subprocess output logger.

        Args:
            log_file (str): Path to the log file for output
        """
        self.log_file = log_file
        # Create a file lock to ensure thread-safe writing
        self.lock = threading.Lock()

    def write(self, message, stream=sys.stdout) -> None:
        """
        Output message to both screen and log file simultaneously.

        Args:
            message (str): Message to output
            stream: Output stream (stdout or stderr) to write to screen
        """
        if (
            not message
            or (" eta " in message and " kB" in message)
            or ("speed:" in message and "last:" in message)
        ):
            return

        # Output to screen
        stream.write(message)
        stream.flush()


def _process_stream(stream, logger, is_stderr=False) -> None:
    """
    Process output streams from subprocess.

    Args:
        stream: Subprocess output stream (stdout or stderr)
        logger (SubprocessLogger): Instance of SubprocessLogger for output
        is_stderr (bool): Whether this stream is stderr (True) or stdout (False)
    """
    stream_obj = sys.stderr if is_stderr else sys.stdout
    try:
        for line in iter(stream.readline, ""):
            logger.write(line, stream_obj)
    finally:
        # Close the stream when done
        stream.close()


def execute_cmd(cmd_list: list, log_file: str = None) -> bool:
    """Execute a command with real-time output logging.

    Args:
        cmd_list (list): List of command arguments to execute
        log_file (str, optional): Path to log file for output. If None, only console output.

    Returns:
        bool: True if command executed successfully (return code 0 or 5), False otherwise
    """
    logger = logging.getLogger(__name__)

    # Convert command list to string for logging
    cmd_str = " ".join(str(item) for item in cmd_list)
    logger.info("execute command: %s", cmd_str)

    flag = True
    # Initialize subprocess logger
    subprocess_logger = SubprocessLogger(log_file)

    try:
        # Start subprocess with pipe connections for real-time output capture
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
            args=(process.stdout, subprocess_logger, False),
            daemon=True,
        )
        stderr_thread = threading.Thread(
            target=_process_stream,
            args=(process.stderr, subprocess_logger, True),
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
