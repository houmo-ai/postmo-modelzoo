import os
import logging
import time
import subprocess
from datetime import datetime
from typing import List, Dict, Optional, Tuple
import threading
import sys


def setup_logging(
    log_dir: str = None, log_name: str = "compile_llms", log_file: str = ""
):
    logger_handlers = [logging.StreamHandler()]  # 输出到控制台
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    if log_dir is not None:
        os.makedirs(log_dir, exist_ok=True)
        log_file = f"{log_dir}/{log_name}_{timestamp}.log"
        logger_handlers.append(logging.FileHandler(log_file))
    elif log_file:
        logger_handlers.append(logging.FileHandler(log_file))

    # 日志格式：时间 - 级别 - 模块.函数 - 消息
    log_format = '%(asctime)s - %(levelname)s - %(module)s.%(funcName)s - %(message)s'

    # 配置日志级别（DEBUG < INFO < WARNING < ERROR < CRITICAL）
    logging.basicConfig(
        level=logging.INFO,  # 基础日志级别
        format=log_format,  # 日志格式
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
        Init Docker Executor

        :param image: the name of docker image
        :param container_name: the name of container
        :param host_log_path: the path of log file on host
        :param container_workdir: the work directory in container
        :param keep_container: do not remove the container
        """
        import docker
        from docker.errors import NotFound

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
        self.logger.info(f"Pulling docker image: {self.image}")
        try:
            self.client.images.pull(self.image)
            self.logger.info(f"Pull image {self.image} successed.")
        except Exception as e:
            raise Exception(f"Failed to pull image, please check image name: {str(e)}")

    # 添加一个辅助方法用于比较volumes配置
    def _compare_volumes(self, existing_volumes: list, requested_volumes: dict) -> bool:
        """比较现有容器的volumes配置与请求的volumes配置是否一致"""
        # 将请求的volumes转换为与Docker API返回的格式一致
        requested_mounts = []
        for host_path, container_path in requested_volumes.items():
            if isinstance(container_path, dict):
                # 处理 {container_path: {'bind': target_path, 'mode': 'rw'}} 格式
                target_path = container_path.get('bind')
                mode = container_path.get('mode', 'rw')
            else:
                # 处理 {host_path: container_path} 格式
                target_path = container_path
                mode = 'rw'

            requested_mounts.append(
                {
                    'Source': (
                        os.path.abspath(host_path)
                        if not host_path.startswith(('/', '.'))
                        else host_path
                    ),
                    'Destination': target_path,
                    'Mode': mode,
                }
            )

        # 比较现有的volumes与请求的volumes
        # 注意：这里简化了比较逻辑，实际应用中可能需要更复杂的比较
        # 特别是处理路径规范化和权限模式比较

        # 创建一个现有volumes的字典以便查找
        existing_mounts_dict = {
            mount['Destination']: {'Source': mount['Source'], 'Mode': mount['Mode']}
            for mount in existing_volumes
        }

        # 检查请求的每个volume是否在现有volumes中存在且配置一致
        for mount in requested_mounts:
            dest = mount['Destination']
            if dest not in existing_mounts_dict:
                return False

            existing = existing_mounts_dict[dest]
            if (
                existing['Source'] != mount['Source']
                or existing['Mode'] != mount['Mode']
            ):
                return False

        # 检查现有volumes是否包含额外的volumes
        # 如果严格要求一致，可以启用下面的检查
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
            if task_id:
                # 尝试获取同名容器（无论是否运行中）
                self.container = self.client.containers.get(self.container_name)

                # 获取现有容器的volumes配置
                existing_volumes = self.container.attrs.get('Mounts', [])

                # 比较现有volumes与请求的volumes是否一致
                volumes_match = self._compare_volumes(existing_volumes, default_volumes)

                if not volumes_match:
                    # volumes不匹配，需要重新创建容器
                    self.logger.info(
                        f"容器 {self.container_name} 的volumes配置已更新，正在重新创建容器"
                    )
                    self.container.remove(force=True)
                    self.container = None  # 重置容器引用
                elif self.container.status != "running":
                    # volumes匹配且容器未运行，则启动容器
                    self.container.start()
                    self.logger.info(f"容器 {self.container_name} 已启动")
                else:
                    # volumes匹配且容器已在运行中
                    self.logger.info(
                        f"容器 {self.container_name} 已在运行中且volumes配置匹配"
                    )
            else:
                self.container = None
        except NotFound:
            # 容器不存在，继续创建流程
            pass

        # 如果容器为None（不存在或已被删除），则创建新容器
        if self.container is None:
            try:
                host_uid = os.getuid()  # 4017
                host_gid = os.getgid()  # 4017
                host_username = getpass.getuser()  # wanyu.li
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
                self.logger.info(
                    f"Successfully started container {self.container_name}."
                )

                # 等待容器启动
                time.sleep(2)

                # 执行第一个命令：创建container_workdir目录并修改所有权
                exec_cmd1 = f"mkdir -p {self.container_workdir} && chown {host_uid}:{host_gid} {self.container_workdir} && mkdir /.cache && chmod 777 -R /.cache"
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
                    f"The user {host_username} has been created and given sudo privileges with /.cache created in cmd1."
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
                print(output_line)  # print to the console

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


class SubprocessLogger:
    def __init__(self, log_file=None):
        """
        初始化输出日志器

        :param log_file: 日志文件路径
        """
        self.log_file = log_file if log_file else None
        if log_file:
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

        if self.log_file:
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


def _process_stream(stream, logger, is_stderr=False, outputs=None):
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
            if outputs is not None:
                outputs.append(line)
    finally:
        stream.close()


def execute_cmd(cmd_list: list, log_file: str = None, get_outputs=False) -> bool:
    if log_file:
        setup_logging(log_file=log_file)
    logger = logging.getLogger(__name__)

    cmd_str = " ".join(str(item) for item in cmd_list)
    logger.info("execute command: %s", cmd_str)

    flag = True
    subprocess_logger = SubprocessLogger(log_file)

    outputs = list() if get_outputs else None
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
            args=(process.stdout, subprocess_logger, False, outputs),
            daemon=True,
        )
        stderr_thread = threading.Thread(
            target=_process_stream,
            args=(process.stderr, subprocess_logger, True, outputs),
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


def update_perf_file(perf_id: str, update_vals: dict):
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
        mode='a',
        header=not file_exists,
        index=False,
        encoding='utf-8',
    )


def get_perf_models(perf_id: str):
    import pandas as pd

    logger = logging.getLogger(__name__)

    script_dir = os.path.dirname(os.path.abspath(__file__))
    perf_file = f"{script_dir}/perf_results_{perf_id}.csv"
    if not os.path.exists(perf_file):
        logger.error(f"Failed to read perf result file {perf_file}.")
        return None

    df = pd.read_csv(perf_file, encoding='utf-8')
    perf_models = df.to_dict(orient='records')

    return perf_models


def update_perf_values(perf_id: str, perf_vals: dict):
    import pandas as pd

    logger = logging.getLogger(__name__)

    script_dir = os.path.dirname(os.path.abspath(__file__))
    perf_file = f"{script_dir}/perf_results_{perf_id}.csv"
    if not os.path.exists(perf_file):
        logger.error(f"Failed to read perf result file {perf_file}.")
        return False

    df = pd.read_csv(perf_file, encoding='utf-8')
    perf_df = pd.DataFrame([perf_vals])
    df = pd.merge(df, perf_df, on=['model'], how='inner')
    df.to_csv(
        perf_file,
        mode='w',
        index=False,
        encoding='utf-8',
    )
    return True
