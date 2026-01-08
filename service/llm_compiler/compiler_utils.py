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
import time
import subprocess
from datetime import datetime
from typing import List, Dict, Optional, Tuple
import threading
import sys
import docker
from docker.errors import NotFound


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


class DockerExecutor:
    def __init__(
        self,
        image: str,
        container_name: str,
        host_log_path: str = "",
        container_workdir: str = "/hmdd",
        keep_container: bool = False,
    ):
        """
        Initialize a DockerExecutor instance.

        Args:
            image (str): The Docker image to use for the container.
            container_name (str): The name of the Docker container.
            host_log_path (str, optional): The path to the host log file. Defaults to "".
            container_workdir (str, optional): The working directory inside the container. Defaults to "/hmdd".
            keep_container (bool, optional): Whether to keep the container after execution. Defaults to False

        Returns:
            DockerExec: The DockerExec object.
        """

        self.image = image
        self.container_name = container_name
        self.container_workdir = container_workdir
        self.keep_container = keep_container

        # Initialize docker client
        try:
            self.client = docker.from_env()
            #  Check docker connection
            self.client.ping()
        except Exception as e:
            raise Exception(f"Cannot connect to docker engine: {str(e)}")

        self.container = None
        self.log_file = (
            host_log_path if host_log_path else f"./{self.container_name}.log"
        )

        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.INFO)

    def pull_image(self) -> None:
        """
        Pull the Docker image from the registry.
        """
        self.logger.info(f"Pulling docker image: {self.image}")
        try:
            self.client.images.pull(self.image)
            self.logger.info(f"Pull image {self.image} successed.")
        except Exception as e:
            raise Exception(f"Failed to pull image, please check image name: {str(e)}")

    def _compare_volumes(self, existing_volumes: list, requested_volumes: dict) -> bool:
        """
        Compare the volumes configuration of the existing container with the requested volumes configuration.

        Args:
            existing_volumes: List of existing volumes in the container
            requested_volumes: Dictionary of requested volumes to be mounted

        Returns:
            bool: True if volumes match, False otherwise
        """
        # Convert requested volumes to the same format as Docker API returns
        requested_mounts = []
        for host_path, container_path in requested_volumes.items():
            if isinstance(container_path, dict):
                # Handle {container_path: {'bind': target_path, 'mode': 'rw'}} format
                target_path = container_path.get("bind")
                mode = container_path.get("mode", "rw")
            else:
                # Handle {host_path: container_path} format
                target_path = container_path
                mode = "rw"

            requested_mounts.append(
                {
                    "Source": (
                        os.path.abspath(host_path)
                        if not host_path.startswith(("/", "."))
                        else host_path
                    ),
                    "Destination": target_path,
                    "Mode": mode,
                }
            )

        # Compare existing volumes with requested volumes
        # Create a dictionary of existing volumes for lookup
        existing_mounts_dict = {
            mount["Destination"]: {"Source": mount["Source"], "Mode": mount["Mode"]}
            for mount in existing_volumes
        }

        # Check if each requested volume exists in existing volumes with the same configuration
        for mount in requested_mounts:
            dest = mount["Destination"]
            if dest not in existing_mounts_dict:
                return False

            existing = existing_mounts_dict[dest]
            if (
                existing["Source"] != mount["Source"]
                or existing["Mode"] != mount["Mode"]
            ):
                return False

        # Check if existing volumes contain additional volumes
        # if len(existing_mounts_dict) != len(requested_mounts):
        #     return False

        return True

    def start_container(
        self,
        volumes: Optional[Dict] = None,
        environment: Optional[Dict] = None,
        network_mode: str = "host",
        task_id=None,
    ) -> None:
        """
        Start the Docker container with the specified configuration.

        Args:
            volumes: Volume mappings for the container
            environment: Environment variables for the container
            network_mode: Network mode for the container
            task_id: Task ID for container reuse logic
        """
        import getpass

        self.logger.info(f"Start container: {self.container_name}")

        # Read-only mode
        default_volumes = {
            "/etc/timezone": {"bind": "/etc/timezone", "mode": "ro"},
            "/etc/localtime": {"bind": "/etc/localtime", "mode": "ro"},
        }
        if volumes:
            default_volumes.update(volumes)

        default_env = {"PS1": "$ "}
        device_requests = list()
        if environment:
            default_env.update(environment)
            device_requests = [
                # -1 means all available GPUs
                docker.types.DeviceRequest(count=-1, capabilities=[["gpu"]])
            ]

        try:
            if task_id:
                # Try to get the container with the same name (whether running or not)
                self.container = self.client.containers.get(self.container_name)
                # Get the volumes configuration of the existing container
                existing_volumes = self.container.attrs.get("Mounts", [])
                # Compare if existing volumes match the requested volumes
                volumes_match = self._compare_volumes(existing_volumes, default_volumes)

                if not volumes_match:
                    # Volumes don't match, need to recreate the container
                    self.logger.info(
                        f"Container {self.container_name} volumes configuration has been updated, recreating container"
                    )
                    self.container.remove(force=True)
                    self.container = None  # Reset container reference
                elif self.container.status != "running":
                    # volumes匹配且容器未运行，则启动容器
                    self.container.start()
                    self.logger.info(f"容器 {self.container_name} 已启动")
                else:
                    # Volumes match and container is not running, start the container
                    self.logger.info(
                        f"Container {self.container_name} has been started"
                    )
            else:
                self.container = None
        except NotFound:
            # Container doesn't exist, continue with creation process
            pass

        # If container is None (doesn't exist or has been deleted), create a new container
        if self.container is None:
            try:
                host_uid = os.getuid()  # 4017
                host_gid = os.getgid()  # 4017
                host_username = getpass.getuser()
                self.logger.info(
                    f"宿主用户信息: username={host_username}, uid={host_uid}, gid={host_gid}"
                )

                self.container = self.client.containers.create(
                    image=self.image,
                    name=self.container_name,
                    privileged=True,
                    volumes=default_volumes,
                    environment=default_env,
                    working_dir=self.container_workdir,
                    tty=True,  # Allocate pseudo-TTY to ensure commands execute properly
                    stdin_open=True,  # Keep stdin open
                    network_mode=network_mode,
                    pid_mode="host",
                    command="/bin/bash",  # Start bash to keep container running
                    shm_size="64g",  # Configure shared memory size
                    # Configure GPU mapping, corresponding to --gpus all parameter
                    device_requests=device_requests,
                    user=f"{host_uid}:{host_gid}",
                    detach=True,
                )
                self.container.start()
                self.logger.info(
                    f"Successfully started container {self.container_name}."
                )

                # Wait for container to start
                time.sleep(2)

                # Execute first command: create container_workdir directory and modify ownership
                exec_cmd1 = f"mkdir -p {self.container_workdir} && chown {host_uid}:{host_gid} {self.container_workdir} && mkdir /.cache && chmod 777 -R /.cache"
                exec_id1 = self.client.api.exec_create(
                    self.container.id,
                    cmd=["bash", "-c", exec_cmd1],
                    user="0:0",  # Execute as root user
                )
                ret = self.client.api.exec_start(exec_id1["Id"])
                self.logger.info(
                    f"The {self.container_workdir} directory has been created and its ownership has been modified."
                )

                # Execute second command: create user and grant sudo privileges
                exec_cmd2 = (
                    f"groupadd -g {host_gid} {host_username} && "
                    f"useradd -m -d {self.container_workdir}/{host_username} -u {host_uid} -g {host_username} {host_username} && "
                    f"usermod -a -G sudo {host_username} && "
                    f'echo "{host_username} ALL=(ALL) NOPASSWD: ALL" >> /etc/sudoers'
                )
                exec_id2 = self.client.api.exec_create(
                    self.container.id,
                    cmd=["bash", "-c", exec_cmd2],
                    user="0:0",  # Execute as root user
                )
                ret = self.client.api.exec_start(exec_id2["Id"])
                self.logger.info(
                    f"The user {host_username} has been created and given sudo privileges with /.cache created in cmd1."
                )

                if environment:
                    # Verify GPU configurations
                    self._verify_configuration()

            except Exception as e:
                self.logger.error(
                    f"The container {self.container_name} failed to start, error msg: {str(e)}"
                )
                if self.container:
                    try:
                        self.container.remove(force=True)
                    except:
                        pass
                raise

    def _verify_configuration(self) -> None:
        """
        Verify whether the shared memory and GPU configuration is effective
        """
        try:
            self.logger.info("Verify GPU configuration...")
            gpu_exec = self.client.api.exec_create(
                self.container.id, "nvidia-smi", tty=True
            )
            gpu_output = "\n".join(
                [
                    line.decode("utf-8").strip()
                    for line in self.client.api.exec_start(gpu_exec["Id"], stream=True)
                ]
            )
            self.logger.info(f"GPU Info:\n{gpu_output}")

            self.logger.info("Verify the shared memory configuration...")
            shm_exec = self.client.api.exec_create(
                self.container.id, "df -h /dev/shm", tty=True
            )
            shm_output = "\n".join(
                [
                    line.decode("utf-8").strip()
                    for line in self.client.api.exec_start(shm_exec["Id"], stream=True)
                ]
            )
            self.logger.info(f"Shared memory information:\n{shm_output}")

        except Exception as e:
            self.logger.error(
                f"An error occurred during configuration verification: {str(e)}"
            )

    def execute_command(self, cmd: str) -> Tuple[int, str]:
        """
        Execute a command in the Docker container and return the result.

        Args:
            cmd (str): Command to execute in the container

        Returns:
            Tuple[int, str]: Exit code and output of the command
        """
        if not self.container:
            raise Exception("The container has not been started.")

        output = []
        try:
            escaped_cmd = cmd.replace("'", "'\\''")
            wrapped_cmd = f"/bin/bash -c '{escaped_cmd}'"

            # 1. Create a docker exec instance
            exec_instance = self.client.api.exec_create(
                self.container.id,
                wrapped_cmd,
                workdir=self.container_workdir,
                tty=True,  # Allocate TTY for command execution
            )
            # 2. Start streaming output
            result = self.client.api.exec_start(
                exec_instance["Id"], stream=True, tty=True  # Match TTY settings
            )

            # 3. Handle the output stream
            for line in result:
                output_line = line.decode("utf-8", errors="replace").strip()
                output.append(output_line)
                print(output_line)  # print to the console

            # 4. After the command is executed, query the exit code
            exit_code = self.client.api.exec_inspect(exec_instance["Id"])["ExitCode"]

            return exit_code, "\n".join(output)

        except Exception as e:
            error_msg = (
                f"An error occurred when executing commands in container: {str(e)}"
            )
            output.append(error_msg)
            self.logger.error(error_msg)
            return -1, "\n".join(output)

    def execute_commands(
        self, commands: List[str], stop_on_error: bool = True
    ) -> List[Dict]:
        """
        Execute a list of commands sequentially in the container.

        Args:
            commands (List[str]): List of commands to execute
            stop_on_error (bool): Whether to stop execution on error

        Returns:
            List[Dict]: Results for each executed command
        """
        self.command_results = []

        with open(self.log_file, "a", encoding="utf-8") as f:
            start_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            header = f"\n{'='*60}\nStart executing the command set - {start_time}\nTotal command number: {len(commands)}\n{'='*60}\n"
            self.logger.info("\n" + header.strip())

            for i, cmd in enumerate(commands, 1):
                cmd_start_time = datetime.now()
                self.logger.info(
                    f"\n{'#'*40}\nExecute command {i}/{len(commands)}: {cmd}\nStart time: {cmd_start_time.strftime('%Y-%m-%d %H:%M:%S')}\n{'#'*40}"
                )

                exit_code, output = self.execute_command(cmd)

                cmd_end_time = datetime.now()
                duration = (cmd_end_time - cmd_start_time).total_seconds()
                result = {
                    "command": cmd,
                    "index": i,
                    "exit_code": exit_code,
                    "start_time": cmd_start_time,
                    "end_time": cmd_end_time,
                    "duration_seconds": duration,
                    "success": exit_code == 0,
                }
                self.command_results.append(result)

                result_msg = f"Command {i} has been completed, exit code: {exit_code}, cost: {duration:.2f} seconds."
                self.logger.info(result_msg)

                if stop_on_error and exit_code != 0:
                    error_msg = "The command execution failed and stop_on_error is set to True. Therefore, no further commands will be executed."
                    self.logger.warning(error_msg)
                    break

            end_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            footer = f"\n{'='*60}\nCommand set execution completed - {end_time}\nSuccess: {sum(1 for r in self.command_results if r['success'])}/{len(self.command_results)}\n{'='*60}\n"
            self.logger.info("\n" + footer.strip())

        self.logger.info(
            f"All commands in Docker have been executed, the logs have been saved to: {self.log_file}"
        )
        return self.command_results

    def stop_and_remove_container(self) -> None:
        """
        Stop and remove the container unless keep_container is True.

        Properly stops and removes the container to clean up resources.
        """
        if self.container and not self.keep_container:
            try:
                self.container.stop()
                self.logger.info(f"Container {self.container_name} has stopped.")
                self.container.remove()
                self.logger.info(f"Container {self.container_name} has been removed.")
            except Exception as e:
                self.logger.error(
                    f"Error occurred when stopping or removing the container: {str(e)}"
                )
        elif self.keep_container:
            self.logger.info(
                f"keep_container=True, container {self.container_name} has been reserved."
            )
        else:
            self.logger.warning("There is no operable container.")

    def __enter__(self):
        """Context manager entry method."""
        self.pull_image()
        self.start_container()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit method."""
        self.stop_and_remove_container()
        if exc_type:
            self.logger.error(f"Error occurred: {exc_val}")
        return False


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
