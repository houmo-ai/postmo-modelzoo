import argparse, os
import time
import psutil
import threading

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
        percent = self.process.memory_percent()   # Percentage of system memory
        return {'rss_mb': rss_mb, 'percent': percent}

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
            self.peak_memory_mb = max(self.peak_memory_mb, mem_info['rss_mb'])

            timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
            log_message = f"{timestamp} - RSS: {mem_info['rss_mb']:.2f} MB, System%: {mem_info['percent']:.2f}%"

            # Output to console or file
            if self.log_file:
                with open(self.log_file, 'a') as f:
                    f.write(log_message + '\n')

            time.sleep(self.interval)

    def stop(self):
        """Stops the monitoring loop and prints peak usage."""
        self.is_monitoring = False
        if hasattr(self, 'monitor_thread'):
            self.monitor_thread.join(timeout=1) # Wait a moment for the thread to finish
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


if HOUMO_TARGET == "xh2":
    from quant_pipeline import quant_llm, export_llm, move_llm

    def parse_args():
        parser = argparse.ArgumentParser(
            formatter_class=argparse.ArgumentDefaultsHelpFormatter
        )
        parser.add_argument("--model", type=str, default="qwen2.5-vl")
        parser.add_argument(
            "--model-name", type=str, default="qwen2.5-vl", help="output hmonnx model name"
        )
        parser.add_argument("--skip-quarot", action="store_true", help="skip_quarot")
        parser.add_argument("--skip-gptq", action="store_true", help="skip_gptq")
        parser.add_argument("--w-bits", type=int, default=4)
        parser.add_argument("--w-head-bits", type=int, default=8)
        parser.add_argument("--seed", type=int, default=1024)
        parser.add_argument(
            "--resume", action="store_true", help="resume from the cache"
        )
        parser.add_argument("--debug", action="store_true", help="debug mode")
        parser.add_argument("--work-dir", type=str, default="work_dirs/")
        parser.add_argument(
            "--out-dir", type=str, default="output/{}".format(HOUMO_TARGET)
        )
        parser.add_argument("--validate", action="store_true", help="validate")
        parser.add_argument("--calib-samples", type=int, default=8)
        parser.add_argument("--calib_dataset", type=str, default="laion/220k-GPT4Vision-captions-from-LIVIS")
        parser.add_argument("--data_files", nargs="+", type=str, default=[], help="List of dataset files")
        parser.add_argument("--batch-size", type=int, default=1, help="batch size")
        parser.add_argument(
            "--context-length", type=int, default=2048, help="max sequence length"
        )
        parser.add_argument(
            "--max_pe_length", type=int, default=32768, help="max pe length"
        )
        parser.add_argument(
            "--quant-type", default="w4a8h1_ssfp", help="quant type, default is w8a8"
        )
        parser.add_argument(
            "--image_max_size_h", type=int, default=448, help="image max size height"
        )
        parser.add_argument(
            "--image_max_size_w", type=int, default=448, help="image max size width"
        )
        parser.add_argument(
            "--image_max_size_t",
            type=int,
            default=2,
            help="if image, temporal max size is 2, if video, temporal max size is fps",
        )
        parser.add_argument("--patch_size", type=int, default=14, help="patch size")
        parser.add_argument(
            "--temporal_patch_size", type=int, default=2, help="temporal patch size"
        )
        parser.add_argument(
            "--sample_image_path",
            type=str,
            default="data/images/qwen2_vl_demo.jpeg",
            help="sample image path for generate golden",
        )
        parser.add_argument(
            "--use_gptqmodel", action="store_true", help="use gptqmodel quanted model"
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
