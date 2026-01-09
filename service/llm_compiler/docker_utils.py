# Copyright 2025 HOUMO AI
#
# File: docker_utils.py
# Description:
#   Docker utilities for LLM model compilation and execution.
#   This module provides utility functions and classes for managing Docker containers,
#   including container lifecycle management, command execution, volume mounting,
#   environment configuration, and logging capabilities for LLM model compilation tasks.
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

import docker
from docker.errors import NotFound
import os
import logging
import time
from datetime import datetime
from typing import List, Dict, Optional, Tuple


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

    def _build_default_volumes(self, volumes: Optional[Dict]) -> Dict:
        """Build default volume mappings with user-provided overrides."""
        default_volumes = {
            "/etc/timezone": {"bind": "/etc/timezone", "mode": "ro"},
            "/etc/localtime": {"bind": "/etc/localtime", "mode": "ro"},
        }
        if volumes:
            default_volumes.update(volumes)
        return default_volumes

    def _build_env_and_device_requests(
        self, environment: Optional[Dict]
    ) -> Tuple[Dict, List]:
        """Build default environment variables and GPU device requests."""
        default_env = {"PS1": "$ "}
        device_requests = []

        if environment:
            default_env.update(environment)
            # Add GPU device request if environment variables are provided
            device_requests = [
                docker.types.DeviceRequest(count=-1, capabilities=[["gpu"]])
            ]

        return default_env, device_requests

    def _get_host_user_info(self) -> Tuple[int, int, str]:
        """Get host user UID, GID and username."""
        import getpass

        host_uid = os.getuid()
        host_gid = os.getgid()
        host_username = getpass.getuser()
        self.logger.info(
            f"Host user information: username={host_username}, uid={host_uid}, gid={host_gid}"
        )
        return host_uid, host_gid, host_username

    def _execute_container_command(
        self, container_id: str, cmd: str, desc: str
    ) -> None:
        """Encapsulate container command execution logic."""
        exec_id = self.client.api.exec_create(
            container_id, cmd=["bash", "-c", cmd], user="0:0"  # Execute as root
        )
        self.client.api.exec_start(exec_id["Id"])
        self.logger.info(desc)

    def _handle_existing_container(self, task_id: str, default_volumes: Dict) -> None:
        """Handle container reuse logic when task_id is provided."""
        try:
            # Get existing container (running or stopped)
            self.container = self.client.containers.get(self.container_name)
            existing_volumes = self.container.attrs.get("Mounts", [])
            volumes_match = self._compare_volumes(existing_volumes, default_volumes)

            if not volumes_match:
                # Recreate container if volumes mismatch
                self.logger.info(
                    f"Container {self.container_name} volumes updated, recreating"
                )
                self.container.remove(force=True)
                self.container = None
            elif self.container.status != "running":
                # Start container if volumes match but container is stopped
                self.container.start()
                self.logger.info(f"Container {self.container_name} started (existing)")
            else:
                self.logger.info(f"Container {self.container_name} is already running")
        except NotFound:
            # Container does not exist - proceed to create new one
            self.container = None

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

    def _create_and_configure_container(
        self,
        default_volumes: Dict,
        default_env: Dict,
        device_requests: List,
        network_mode: str,
    ) -> None:
        """Create new container and execute initialization commands."""
        host_uid, host_gid, host_username = self._get_host_user_info()

        # Create container
        self.container = self.client.containers.create(
            image=self.image,
            name=self.container_name,
            privileged=True,
            volumes=default_volumes,
            environment=default_env,
            working_dir=self.container_workdir,
            tty=True,
            stdin_open=True,
            network_mode=network_mode,
            pid_mode="host",
            command="/bin/bash",
            shm_size="64g",
            device_requests=device_requests,
            user=f"{host_uid}:{host_gid}",
            detach=True,
        )
        self.container.start()
        self.logger.info(f"Successfully started new container {self.container_name}")
        time.sleep(2)  # Wait for container initialization

        # Execute initialization commands
        # 1. Create workdir and cache dir
        workdir_cmd = (
            f"mkdir -p {self.container_workdir} && "
            f"chown {host_uid}:{host_gid} {self.container_workdir} && "
            f"mkdir /.cache && chmod 777 -R /.cache"
        )
        self._execute_container_command(
            self.container.id,
            workdir_cmd,
            f"Created {self.container_workdir} and set ownership; created /.cache with full permissions",
        )

        # 2. Create user and grant sudo access
        user_cmd = (
            f"groupadd -g {host_gid} {host_username} && "
            f"useradd -m -d {self.container_workdir}/{host_username} -u {host_uid} -g {host_username} {host_username} && "
            f"usermod -a -G sudo {host_username} && "
            f'echo "{host_username} ALL=(ALL) NOPASSWD: ALL" >> /etc/sudoers'
        )
        self._execute_container_command(
            self.container.id,
            user_cmd,
            f"Created user {host_username} and granted passwordless sudo privileges",
        )

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
        self.logger.info(f"Start container: {self.container_name}")

        # Build base configurations
        default_volumes = self._build_default_volumes(volumes)
        default_env, device_requests = self._build_env_and_device_requests(environment)

        if task_id:
            self._handle_existing_container(task_id, default_volumes)

        # If container doesn't exist or has been deleted, create a new container
        if self.container is None:
            try:
                self._create_and_configure_container(
                    default_volumes, default_env, device_requests, network_mode
                )
                # Verify GPU config if environment variables are provided
                if environment:
                    self._verify_configuration()
            except Exception as e:
                self.logger.error(
                    f"Failed to start container {self.container_name}: {str(e)}"
                )
                # Clean up failed container
                if self.container:
                    try:
                        self.container.remove(force=True)
                    except Exception:
                        pass
                raise

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
