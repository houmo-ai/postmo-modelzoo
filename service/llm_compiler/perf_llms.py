# Copyright 2025 HOUMO AI
#
# File: norm_quant_folder.py
# Description:
#   Execute LLM model performance testing in Docker containers.
#
#   This script orchestrates performance testing of LLM models by running performance evaluation
#   commands inside Docker containers.
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
import argparse
import logging
from datetime import datetime
from docker_utils import DockerExecutor
from compiler_utils import setup_logging

global logger
logger = logging.getLogger(__name__)
script_dir = os.path.dirname(os.path.abspath(__file__))


def parse_args() -> argparse.Namespace:
    """Parse command line arguments for performance testing."""
    parser = argparse.ArgumentParser(description="Perf LLMs")
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
        help="Houmo backend, support: xh2.",
    )
    parser.add_argument(
        "-perf",
        "--perf_cfg",
        type=str,
        default="",
        help="the path of perf result file.",
    )
    parser.add_argument(
        "-log",
        "--log_file",
        type=str,
        default="",
        help="the path of log.",
    )
    parser.add_argument(
        "-no_verify",
        "--no_verify",
        action="store_true",
        help="perf only, skip to verify demo.",
    )

    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()

    setup_logging(log_file=args.log_file)
    logger = logging.getLogger(__name__)
    ts_str = datetime.now().strftime("%Y%m%d_%H%M%S")

    target = args.target
    version = args.version
    system = "ubuntu20.04" if target == "xh1" else "ubuntu24.04"
    image_name = f"harbor.houmo.ai/toolchain/release:Dadao-{target}-v{version}-{system}-x86.64.latest"
    container_name = f"compiler_perf_{target}_{version}_{ts_str}"
    container_home = "/hmdd"

    cmd_list = []
    cmd = f"cd {container_home}/imodelzoo/service/llm_compiler && python3 execute_perf.py --perf_cfg {args.perf_cfg} -log {args.log_file}"
    cmd_list.append(cmd)

    if not args.no_verify:
        logger.info("Will verify the model after running perf")
        verify_cmd = f"cd {container_home}/imodelzoo/service/llm_compiler && python3 execute_demo.py --perf_cfg {args.perf_cfg} -log {args.log_file}"
        cmd_list.append(verify_cmd)

    # Create a docker executor
    docker_exec = DockerExecutor(
        image=image_name,
        container_name=container_name,
        host_log_path=args.log_file,
        container_workdir=container_home,
        keep_container=False,
    )
    docker_flag = True
    try:
        docker_exec.pull_image()
        # Map the current directory of the host to the container
        volumes = {
            # Map imodelzoo folder
            os.path.abspath(f"{script_dir}/../../"): {
                "bind": f"{container_home}/imodelzoo",
                "mode": "rw",
            },
            "/data": {
                "bind": "/data",
                "mode": "rw",
            },
            "/data02": {
                "bind": "/data02",
                "mode": "rw",
            },
        }
        docker_exec.start_container(volumes=volumes)
        cmds_res = docker_exec.execute_commands(cmd_list, stop_on_error=False)
        for res in cmds_res:
            if res.get("exit_code", -1) != 0:
                logger.info("## hm docker_exec commands error occured")
                docker_flag = False
                break
    except Exception as e:
        docker_flag = False
        logger.error(f"Error occurred: {str(e)}")
    finally:
        docker_exec.stop_and_remove_container()
