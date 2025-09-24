import os
import sys
import docker
import getpass
import argparse
import logging
import time
from datetime import datetime
from typing import List, Dict, Optional, Tuple
from cd_tester_utils import setup_logging

script_dir = os.path.dirname(os.path.abspath(__file__))


def parse_args():
    parser = argparse.ArgumentParser(description="CD Tester")
    parser.add_argument(
        "-v",
        "--version",
        required=True,
        type=str,
        help="Houmo Dadao software version, example: 0.3.0, 2.4.2",
    )
    parser.add_argument(
        "-t",
        "--target",
        type=str,
        default="xh2",
        help="Houmo backend, support: xh1, xh2.",
    )

    args = parser.parse_args()
    return args


def _check_args(args: dict):
    from packaging import version

    # only support xh2 now
    if args.target not in ["xh1", "xh2"]:
        logger.error(f"Invalid houmo target {args.target}.")
        return False

    try:
        xh1_min_ver = version.parse("2.4.2")
        xh2_min_ver = version.parse("0.3.0")
        ver = version.parse(args.version)
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
        Init Docker Executor

        :param image: the name of docker image
        :param container_name: the name of container
        :param host_log_path: the path of log file on host
        :param container_workdir: the work directory in container
        :param keep_container: do not remove the container
        """
        self.image = image
        self.container_name = container_name
        self.container_workdir = container_workdir
        self.keep_container = keep_container

        # init docker client
        try:
            self.client = docker.from_env()
            # check docker connection
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
        import getpass

        self.logger.info(f"Start container: {self.container_name}")

        default_volumes = {
            "/etc/timezone": {"bind": "/etc/timezone", "mode": "ro"},  # 只读模式
            "/etc/localtime": {"bind": "/etc/localtime", "mode": "ro"},  # 只读模式
        }
        if volumes:
            default_volumes.update(volumes)

        default_env = {"PS1": "$ "}
        device_requests = list()
        if environment:
            default_env.update(environment)
            device_requests = [
                # -1表示所有可用GPU
                docker.types.DeviceRequest(count=-1, capabilities=[["gpu"]])
            ]

        try:
            host_uid = os.getuid()  # 4017
            host_gid = os.getgid()  # 4017
            host_username = getpass.getuser()  # wanyu.li
            logger.info(
                f"宿主用户信息: username={host_username}, uid={host_uid}, gid={host_gid}"
            )

            self.container = self.client.containers.create(
                image=self.image,
                name=self.container_name,
                privileged=True,
                volumes=default_volumes,
                environment=default_env,
                working_dir=self.container_workdir,
                tty=True,  # 分配伪终端，确保命令正确执行
                stdin_open=True,  # 保持标准输入打开
                network_mode=network_mode,
                pid_mode="host",
                command="/bin/bash",  # 启动bash保持容器运行
                shm_size="64g",  # 配置共享内存大小
                # 配置GPU映射，对应--gpus all参数
                device_requests=device_requests,
                user=f"{host_uid}:{host_gid}",
                detach=True,
            )
            self.container.start()
            self.logger.info(f"Successfully started container {self.container_name}.")

            # 等待容器启动
            time.sleep(2)

            # 执行第一个命令：创建container_workdir目录并修改所有权
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

            # 执行第二个命令：创建用户并赋予sudo权限
            exec_cmd2 = (
                f"groupadd -g {host_gid} {host_username} && "
                f"useradd -m -d {self.container_workdir}/{host_username} -u {host_uid} -g {host_username} {host_username} && "
                f"usermod -a -G sudo {host_username} && "
                f"echo \"{host_username} ALL=(ALL) NOPASSWD: ALL\" >> /etc/sudoers"
            )
            exec_id2 = self.client.api.exec_create(
                self.container.id,
                cmd=["bash", "-c", exec_cmd2],
                user="0:0",  # 以root用户执行
            )
            ret = self.client.api.exec_start(exec_id2["Id"])
            self.logger.info(
                f"The user {host_username} has been created and given sudo privileges."
            )

            if environment:
                # verify gpu configs
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
            # 查看共享内存大小
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
        if not self.container:
            raise Exception("The container has not been started.")

        output = []
        try:
            escaped_cmd = cmd.replace("'", "'\\''")
            wrapped_cmd = f"/bin/bash -c '{escaped_cmd}'"

            # 1. create a docker instance
            exec_instance = self.client.api.exec_create(
                self.container.id,
                wrapped_cmd,
                workdir=self.container_workdir,
                tty=True,  # 为执行命令分配TTY
            )
            # 2. start streaming output
            result = self.client.api.exec_start(
                exec_instance["Id"], stream=True, tty=True  # 匹配TTY设置
            )

            # 3. handle the output stream
            for line in result:
                output_line = line.decode("utf-8", errors="replace").strip()
                output.append(output_line)
                if " eta " in output_line and " kB" in output_line:
                    continue
                logger.info(output_line)  # print to the console

            # 4. after the command is executed, query the exit code
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
                # self.logger.info(output)

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
        self.pull_image()
        self.start_container()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop_and_remove_container()
        if exc_type:
            self.logger.error(f"Error occurred: {exc_val}")
        return False


if __name__ == "__main__":
    global logger
    args = parse_args()

    log_dir = script_dir + "/logs"
    os.makedirs(log_dir, exist_ok=True)
    os.chmod(log_dir, 0o755)
    host_log_file = setup_logging(log_dir)
    logger = logging.getLogger(__name__)

    # check the validity of parameters
    if _check_args(args) is False:
        exit(1)
    # load parameters
    target = args.target
    version = args.version

    # create a result folder for the current container
    container_name = f"tester_cd_{target}_{version}-{int(time.time())}"

    ###### Docker Executor ######
    # construct docker image name
    system = "ubuntu20.04" if target == "xh1" else "ubuntu24.04"
    image_name = f"harbor.houmo.ai/toolchain/release:Dadao-{target}-v{version}-{system}-x86.64.latest"
    # set container configs
    container_home = "/hmdd"
    container_log_file = (
        "/develop02/imodelzoo/service/cd_tester/logs/"
        + host_log_file.rsplit("/", 1)[-1]
    )
    # construct the commands to be executed in the container
    commands = ["echo 'Hi CD Tester!'"]
    if target == "xh1":
        commands += [
            "sudo pip3 install -i https://pypi.tuna.tsinghua.edu.cn/simple pytest pytest-dependency"
        ]
    else:
        commands += [
            "pip3 install -i https://pypi.tuna.tsinghua.edu.cn/simple pytest pytest-dependency"
        ]
    # quant and compile tests
    compile_cmd = f"python3 execute_no_infer.py -log {container_log_file}"
    commands.append(f"cd /develop02/imodelzoo/service/cd_tester && {compile_cmd}")

    # create a docker executor
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
        # map the current directory of the host to the container
        volumes = {
            # map imodelzoo folder
            "/develop02/wanyu.li/imodelzoo_develop": {
                "bind": "/develop02/imodelzoo",
                "mode": "rw",
            },
            # map modelzoo folder
            "/develop02/wanyu.li/modelzoo": {
                "bind": "/develop02/modelzoo",
                "mode": "rw",
            },
            # map logs folder
            os.path.abspath(log_dir): {
                "bind": f"{container_home}/logs",
                "mode": "rw",
            },
        }
        gpu_env = {
            "NVIDIA_VISIBLE_DEVICES": "all",  # 可见所有GPU
            "NVIDIA_DRIVER_CAPABILITIES": "compute,utility",  # GPU能力
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
