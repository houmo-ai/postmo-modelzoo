import os
import re
import argparse
import subprocess


def parse_args():
    parser = argparse.ArgumentParser(description="Device Memory Monitor")
    parser.add_argument(
        "--device_id",
        type=int,
        choices=[0, 1, 2, 3],
        help="device id.",
    )

    args = parser.parse_args()
    return args


def _run_command(command):
    """
    执行系统命令并返回输出结果和退出码
    :param command: 要执行的命令（字符串）
    :return: (output, return_code) 输出结果和退出码
    """
    try:
        # 执行命令，捕获 stdout 和 stderr
        result = subprocess.run(
            command,
            shell=True,
            check=False,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        )
        # 返回输出结果和退出码
        return result.stdout, result.returncode
    except Exception as e:
        return f"Failed to execute command: {str(e)}", -1


def get_device_mem(device_id: int) -> dict:
    script_dir = os.path.dirname(os.path.abspath(__file__))
    cmd = f"{script_dir}/bin/dev_monitor"
    if device_id:
        cmd += f" -d {device_id}"
    opt, ret = _run_command(cmd)
    if ret != 0:
        return None

    result = None
    pattern = r"device_id: (?P<device_id>\d+), time: (?P<time>\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}), mem_total: (?P<mem_total>\d+), mem_used: (?P<mem_used>\d+), mem_avail: (?P<mem_avail>\d+)"
    for line in opt.split("\n"):
        if "device_id:" in line:
            match = re.match(pattern, line.strip())
            if match:
                result = match.groupdict()
    return result


# if __name__ == "__main__":
#     args = parse_args()
#     get_device_mem(args.device_id)
