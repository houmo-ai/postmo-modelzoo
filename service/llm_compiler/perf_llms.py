import os
import argparse
import logging
from datetime import datetime
from compiler_utils import setup_logging, get_perf_models, DockerExecutor

global logger
logger = logging.getLogger(__name__)
script_dir = os.path.dirname(os.path.abspath(__file__))


def parse_args():
    parser = argparse.ArgumentParser(description="Perf LLMs")
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

    args = parser.parse_args()
    return args


def _generate_perf_cmds(container_home, perf_id, log_file):
    cmd_list = list()
    perf_models = get_perf_models(perf_id)
    if perf_models is None:
        return None
    for model_info in perf_models:
        cmd = f"cd {container_home}/imodelzoo/service/llm_compiler && python3 execute_perf.py"
        flag = True
        for key, val in model_info.items():
            logger.info(f"{key}: {val}")
            if key not in ["model", "case_dir"] and not os.path.exists(val):
                flag = False
                logger.error(f"Missing input file {key}: {val}")
                break
            cmd += f" --{key} {val}"
        if flag:
            cmd += f" --perf_id {perf_id} -log {log_file}"
            cmd_list.append(cmd)

    return cmd_list


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

    cmd_list = list()
    cmd = f"cd {container_home}/imodelzoo/service/llm_compiler && python3 execute_perf.py --perf_cfg {args.perf_cfg} -log {args.log_file}"
    cmd_list.append(cmd)

    # create a docker executor
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
        # map the current directory of the host to the container
        volumes = {
            # map imodelzoo folder
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
                logger.info(f"##hm docker_exec commands error occured")
                docker_flag = False
                break
    except Exception as e:
        docker_flag = False
        logger.error(f"Error occurred: {str(e)}")
    finally:
        docker_exec.stop_and_remove_container()
