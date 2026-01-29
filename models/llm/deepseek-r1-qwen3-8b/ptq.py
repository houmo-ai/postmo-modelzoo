# Copyright (c) 2025 HOUMO AI
#
# File: ptq.py
# Description:
#   Post-Training Quantization Tool - Python script for quantizing
# DeepSeek models using post-training quantization techniques.
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
import argparse
import os
import time
import psutil
import threading
from quant_pipline import quant_llm, export_llm, move_llm

HOUMO_DATASETS_PATH = os.getenv("HOUMO_DATASETS_PATH", "")
HOUMO_TARGET = os.getenv("HOUMO_TARGET", "")


class ProcessMemoryMonitor:
    """
    Monitors the memory usage of the current Python process in real-time using psutil.
    """

    def __init__(self, interval=2, log_file=None):
        """
        Initializes the monitor.
        Args:
            interval (int): Time between measurements in seconds.
            log_file (str, optional): Path to a file to log results. If None, prints to console.
        """
        self.process = psutil.Process(os.getpid())
        self.interval = interval
        self.log_file = log_file
        self.is_monitoring = False
        self.peak_memory_mb = 0

    def get_memory_info(self):
        """
        Gets current memory usage information.
        Returns:
            dict: A dictionary containing memory usage data.
        """
        memory_info = self.process.memory_info()
        rss_mb = memory_info.rss / (1024 * 1024)  # Resident Set Size in MB
        percent = self.process.memory_percent()  # Percentage of system memory
        return {"rss_mb": rss_mb, "percent": percent}

    def start(self):
        """Starts the monitoring loop in a separate daemon thread."""
        self.is_monitoring = True
        self.peak_memory_mb = 0
        self.monitor_thread = threading.Thread(target=self._monitor_loop)
        self.monitor_thread.daemon = True  # Thread will exit when main program does
        self.monitor_thread.start()
        print(f"Memory monitoring started (interval: {self.interval}s)")

    def _monitor_loop(self):
        """The internal loop that runs in the thread."""
        while self.is_monitoring:
            mem_info = self.get_memory_info()
            self.peak_memory_mb = max(self.peak_memory_mb, mem_info["rss_mb"])

            timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
            log_message = f"{timestamp} - RSS: {mem_info['rss_mb']:.2f} MB, System%: {mem_info['percent']:.2f}%"

            # Output to console or file
            if self.log_file:
                with open(self.log_file, "a") as f:
                    f.write(log_message + "\n")

            time.sleep(self.interval)

    def stop(self):
        """Stops the monitoring loop and prints peak usage."""
        self.is_monitoring = False
        if hasattr(self, "monitor_thread"):
            self.monitor_thread.join(
                timeout=1
            )  # Wait a moment for the thread to finish
        print(f"[Monitoring stopped. Peak RSS: {self.peak_memory_mb:.2f} MB]")


def check_gpu():
    import subprocess

    try:
        result = subprocess.run(
            "nvidia-smi --query-gpu=count --format=csv,noheader,nounits | wc -l",
            shell=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
            text=True,
        )
        if result.returncode == 0 and int(result.stdout.strip()) > 0:
            return True
        return False
    except Exception as e:
        print(f"Not install GPU driver, error msg: {e}")
        return False


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ("yes", "true", "t", "y", "1", ""):
        return True
    elif v.lower() in ("no", "false", "f", "n", "0"):
        return False
    else:
        raise argparse.ArgumentTypeError("Boolean value expected.")


def parse_args():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("--model", type=str, default="DeepSeek-R1-0528-Qwen3-8B")
    parser.add_argument(
        "--model-name", type=str, default="deepseek", help="output hmonnx model name"
    )
    parser.add_argument("--work-dir", type=str, default="work_dirs/")
    parser.add_argument("--out-dir", type=str, default="output/{}".format(HOUMO_TARGET))
    parser.add_argument("--skip-quarot", action="store_true", help="skip_quarot")
    parser.add_argument("--skip-gptq", action="store_true", help="skip_gptq")
    parser.add_argument("--w-bits", type=int, default=4)
    parser.add_argument("--seed", type=int, default=1024)
    parser.add_argument("--resume", action="store_true", help="resume from the cache")
    parser.add_argument("--debug", action="store_true", help="debug mode")
    parser.add_argument(
        "--context-length", type=int, default=2048, help="max sequence length"
    )
    parser.add_argument(
        "--input-sequence-length", type=int, default=256, help="input sequence length"
    )
    parser.add_argument(
        "--quant-type", default="w4a8h0_ssfp", help="quant type, default is w4a8h0_ssfp"
    )
    parser.add_argument(
        "--calib_data", type=str, default="wikitext2", help="calibration dataset choose"
    )
    parser.add_argument(
        "--gptqmodel", action="store_true", help="use gptqmodel to quant"
    )
    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    quant_llm(args)
    export_llm(args)
    move_llm(args)


if __name__ == "__main__":
    if not check_gpu():
        print("Error: Not found GPU device.")
        exit(-1)
    memory_monitor = ProcessMemoryMonitor(interval=2)
    memory_monitor.start()
    main()
    memory_monitor.stop()
