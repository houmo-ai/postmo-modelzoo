# Copyright (c) 2025 HOUMO AI
#
# File: ptq.py
# Description:
#   Post-Training Quantization Tool - Python script for quantizing
# GLM-OCR models using post-training quantization techniques.
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

import argparse, os
import time
import psutil
import threading
from quant_pipeline import export_llm, export_vision, move_llm

HOUMO_TARGET = os.getenv("HOUMO_TARGET")


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


def parse_arguments():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "--hf_model_dir",
        type=str,
        default="glm-ocr",
        help="HuggingFace model directory",
    )
    parser.add_argument(
        "--work_dir",
        type=str,
        default="work_dirs",
        help="output work directory",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=os.path.join("output", HOUMO_TARGET, "hmquant"),
        help="output directory",
    )
    parser.add_argument(
        "--image_path",
        type=str,
        default="../../../data/pic/ocr.jpeg",
    )
    parser.add_argument("--prompt", type=str, default="Text Recognition:")
    parser.add_argument("--max_new_tokens", type=int, default=256)
    parser.add_argument("--attn_implementation", type=str, default="eager")
    parser.add_argument("--seed", type=int, default=128)
    parser.add_argument("--debug", action="store_true", help="debug mode")
    parser.add_argument("--valid", default=True, help="evaluate the model")
    parser.add_argument(
        "--profile_nodes",
        default=False,
        action="store_true",
        help="profile traced/quanted graph node outputs",
    )
    parser.add_argument(
        "--profile_decode_steps",
        type=int,
        default=3,
        help="number of decode steps to profile when --profile_nodes is set",
    )
    parser.add_argument(
        "--profile_dir",
        type=str,
        default=None,
        help="node profile output dir, default work_dir/node_profile",
    )
    parser.add_argument("--image_size_w", type=int, default=336, help="image width")
    parser.add_argument("--image_size_h", type=int, default=336, help="image height")
    parser.add_argument(
        "--max_sequence_length", type=int, default=2048, help="max sequence length"
    )
    parser.add_argument(
        "--input_sequence_length",
        type=int,
        default=256,
        help="prefill input sequence length",
    )
    parser.add_argument(
        "--target_device", type=str, default="XH2a", help="target device"
    )
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--max_size_t", type=int, default=2, help="max temporal size")
    parser.add_argument("--patch_size", type=int, default=14, help="patch size")
    parser.add_argument(
        "--temporal_patch_size", type=int, default=2, help="temporal patch size"
    )
    args = parser.parse_args()
    return args


def main():
    args = parse_arguments()
    export_llm(args)
    export_vision(args)
    move_llm(args)


if __name__ == "__main__":
    if not check_gpu():
        print("Error: Not found GPU device.")
        exit(-1)
    memory_monitor = ProcessMemoryMonitor(interval=2)
    memory_monitor.start()
    main()
    memory_monitor.stop()
