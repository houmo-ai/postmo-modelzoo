# Copyright 2025 HOUMO AI
#
# File: quant_compile_tests.py
# Description:
#   Execute quantization and compilation tests in Docker containers for continuous deployment.
#   This script manages Docker containers to run model quantization and compilation tests,
#   handles container setup, environment configuration, and test execution orchestration.
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
import docker
import argparse
import logging
import time
from datetime import datetime
from typing import List, Dict, Optional, Tuple
from cd_tester_utils import *

script_dir = os.path.dirname(os.path.abspath(__file__))


def parse_args():
    """Parse command line arguments for the quantization and compilation test execution script."""
    parser = argparse.ArgumentParser(description="CD Tester")
    parser.add_argument(
        "-v",
        "--version",
        required=True,
        type=str,
        help="Houmo Dadao software version, example: 0.7.0, 2.6.0",
    )
    parser.add_argument(
        "-t",
        "--target",
        type=str,
        default="xh2",
        help="Houmo backend, support: xh1, xh2.",
    )
    parser.add_argument(
        "--release",
        type=str,
        default="off",
        help="use release models for testing, support: on, off.",
    )
    parser.add_argument(
        "-k",
        "--key_str",
        type=str,
        default="",
        help="pytest -k value",
    )
    parser.add_argument(
        "-m",
        "--model_str",
        type=str,
        default="",
        help="pytest -m value",
    )

    args = parser.parse_args()
    return args


def _check_args(args: dict) -> bool:
    """Validate the provided command line arguments.

    Args:
        args: Arguments object containing parsed command line options

    Returns:
        bool: True if arguments are valid, False otherwise
    """
    from packaging import version

    # only support xh2 now
    if args.target not in ["xh1", "xh2"]:
        logger.error(f"Invalid houmo target {args.target}.")
        return False

    try:
        # Define minimum supported versions for each target
        xh1_min_ver = version.parse("2.4.2")
        xh2_min_ver = version.parse("0.3.0")
        ver = version.parse(args.version)
        # Check if the version meets minimum requirements for the target
        if (args.target == "xh1" and ver < xh1_min_ver) or (
            args.target == "xh2" and ver < xh2_min_ver
        ):
            logger.error(f"Unsupported version {args.version} on {args.target}.")
            return False
    except version.InvalidVersion as e:
        logger.error(f"Invalid version {args.version}, error msg: {str(e)}")
        return False

    logger.info(
        "\n***** Compilation Configs *****\n"
        "-- DaDao Software version: %s \n"
        "-- Houmo target: %s \n",
        args.version,
        args.target,
    )

    return True


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
        Initialize the Docker Executor to manage containerized test execution.

        Args:
            image (str): Name of the Docker image to use
            container_name (str): Name to assign to the container
            host_log_path (str): Path on the host for log files
            container_workdir (str): Working directory inside the container
            keep_container (bool): Whether to preserve the container after execution
        """
        self.image = image
        self.container_name = container_name
        self.container_workdir = container_workdir
        self.keep_container = keep_container

        # Initialize Docker client
        try:
            self.client = docker.from_env()
            # Verify Docker connection
            self.client.ping()
        except Exception as e:
            raise Exception(f"Cannot connect to docker engine: {str(e)}")

        self.container = None
        self.log_file = (
            host_log_path if host_log_path else f"./{self.container_name}.log"
        )

        # Configure logger for this class
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.INFO)

    def pull_image(self) -> None:
        """
        Pull the Docker image if it doesn't exist locally.
        """
        try:
            self.client.images.get(self.image)
            self.logger.info(f"Use local docker image: {self.image}")
        except docker.errors.ImageNotFound:
            self.logger.info(f"Pulling docker image: {self.image}")
            try:
                self.client.images.pull(self.image)
                self.logger.info(f"Pull image {self.image} successed.")
            except Exception as e:
                raise Exception(
                    f"Failed to pull image, please check image name: {str(e)}"
                )

    def start_container(
        self,
        volumes: Optional[Dict] = None,
        environment: Optional[Dict] = None,
        network_mode: str = "host",
    ) -> None:
        """
        Start a Docker container with the specified configuration.

        Args:
            volumes (Optional[Dict]): Volume mappings from host to container
            environment (Optional[Dict]): Environment variables to set in container
            network_mode (str): Network mode for the container (default: host)
        """
        import getpass

        self.logger.info(f"Start container: {self.container_name}")

        # Define default volumes to mount timezone and locale information (Read-only mode)
        default_volumes = {
            "/etc/timezone": {"bind": "/etc/timezone", "mode": "ro"},
            "/etc/localtime": {"bind": "/etc/localtime", "mode": "ro"},
        }
        if volumes:
            default_volumes.update(volumes)

        # Set up default environment variables
        default_env = {"PS1": "$ "}
        device_requests = list()
        if environment:
            default_env.update(environment)
            device_requests = [
                # -1 means all available GPUs
                docker.types.DeviceRequest(count=-1, capabilities=[["gpu"]])
            ]

        try:
            # Get host user information to match container user
            host_uid = os.getuid()  # 4017
            host_gid = os.getgid()  # 4017
            host_username = getpass.getuser()
            logger.info(
                f"Host user information: username=s{host_username}, uid={host_uid}, gid={host_gid}"
            )

            self.container = self.client.containers.create(
                image=self.image,
                name=self.container_name,
                privileged=True,
                volumes=default_volumes,
                environment=default_env,
                working_dir=self.container_workdir,
                tty=True,  # Allocate pseudo-terminal to ensure commands execute properly
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
            self.logger.info(f"Successfully started container {self.container_name}.")

            # Wait for container to start
            time.sleep(2)

            # Execute first command: create container_workdir directory and modify ownership
            exec_cmd1 = f"mkdir -p {self.container_workdir} && chown {host_uid}:{host_gid} {self.container_workdir}"
            exec_id1 = self.client.api.exec_create(
                self.container.id,
                cmd=["bash", "-c", exec_cmd1],
                user="0:0",  # 以root用户执行
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
                f"The user {host_username} has been created and given sudo privileges."
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
        """Verify whether the shared memory and GPU configuration is effective"""
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
            # Check shared memory size
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
                if " eta " in output_line and " kB" in output_line:
                    continue
                logger.info(output_line)  # print to the console

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
            # Record start time and command count
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

            # Record end time and summary
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


if __name__ == "__main__":
    global logger
    args = parse_args()

    # Set up logging directory
    log_dir = script_dir + "/logs"
    os.makedirs(log_dir, exist_ok=True)
    os.chmod(log_dir, 0o755)
    host_log_file = setup_logging(log_dir)
    logger = logging.getLogger(__name__)

    # Check the validity of parameters
    if _check_args(args) is False:
        exit(1)
    # Load parameters
    target = args.target
    version = args.version
    release = True if args.release in ["on", "ON"] else False

    # Create a result folder for the current container
    container_name = f"tester_cd_{target}_{version}-{int(time.time())}"

    ###### Docker Executor ######
    # Construct Docker image name
    system = "ubuntu20.04" if target == "xh1" else "ubuntu24.04"
    image_name = f"harbor.houmo.ai/toolchain/release:Dadao-{target}-v{version}-{system}-x86.64.latest"
    # Set container configs
    container_home = "/hmdd"
    container_log_file = (
        f"{IMODELZOO_REPO_DIR}/service/cd_tester/logs/"
        + host_log_file.rsplit("/", 1)[-1]
    )
    # Construct the commands to be executed in the container
    commands = ["echo 'Hi CD Tester!'"]
    if target == "xh1":
        commands += [
            "sudo pip3 install -i https://pypi.tuna.tsinghua.edu.cn/simple pytest pytest-dependency"
        ]
    else:
        commands += [
            "pip3 install -i https://pypi.tuna.tsinghua.edu.cn/simple pytest pytest-dependency"
        ]
    # Quantization and compilation tests
    compile_cmd = f"python3 execute_no_infer.py -log {container_log_file}"
    if release:
        compile_cmd += " --release"
    if args.key_str:
        compile_cmd += f" -k '{args.key_str}'"
    if args.model_str:
        compile_cmd += f" -m '{args.model_str}'"
    logger.info(f"compilation cmd: {compile_cmd}")
    commands.append(f"cd {IMODELZOO_REPO_DIR}/service/cd_tester && {compile_cmd}")

    # Create a docker executor
    docker_exec = DockerExecutor(
        image=image_name,
        container_name=container_name,
        host_log_path=host_log_file,
        container_workdir=container_home,
        keep_container=False,
    )
    docker_flag = True
    try:
        docker_exec.pull_image()
        # Map the current directory of the host to the container
        volumes = {
            # Map imodelzoo folder
            IMODELZOO_REPO_DIR: {
                "bind": IMODELZOO_REPO_DIR,
                "mode": "rw",
            },
            # Map modelzoo folder
            "/develop02/imodelzoo": {
                "bind": "/develop02/modelzoo",
                "mode": "rw",
            },
            "/data02/modelzoo": {
                "bind": "/data02/modelzoo",
                "mode": "rw",
            },
            # Map logs folder
            os.path.abspath(log_dir): {
                "bind": f"{container_home}/logs",
                "mode": "rw",
            },
        }
        gpu_env = {
            "NVIDIA_VISIBLE_DEVICES": "all",  # Visible to all GPUs
            "NVIDIA_DRIVER_CAPABILITIES": "compute,utility",  # GPU capabilities
        }
        docker_exec.start_container(volumes=volumes, environment=gpu_env)

        cmds_res = docker_exec.execute_commands(commands, stop_on_error=True)
        for res in cmds_res:
            if res.get("exit_code", -1) != 0:
                docker_flag = False
                break
    except Exception as e:
        docker_flag = False
        logger.error(f"Error occurred: {str(e)}")
    finally:
        docker_exec.stop_and_remove_container()

    if docker_flag is False:
        sys.exit(-1)
    sys.exit(0)
