# Copyright (c) 2025 HOUMO AI
#
# File: build.py
# Description:
#   CT-Transformer Model Build and Test Tool - Python script for building and testing
# CT-Transformer models.
#

import os
import numpy as np
import time
import psutil
import threading
import multiprocessing
import argparse
import glob
import logging
import yaml

logging.basicConfig(level="INFO")

HOUMO_TARGET = os.getenv("HOUMO_TARGET", "xh2")
assert HOUMO_TARGET in ["xh2"], f"Unsupported HOUMO_TARGET: {HOUMO_TARGET}"

HOUMO_CORE_NUM = int(os.getenv("HOUMO_CORE_NUM", 2))
GOLDEN_THRESH = 0.98
DEFAULT_CONFIG_PATH = os.path.join(os.path.dirname(__file__), "config.yaml")


def first_not_none(*args):
    """Return the first argument that is not None."""
    for arg in args:
        if arg is not None:
            return arg
    return None


def get_model_configs(config_path: str):
    """Load model configs from yaml file."""
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    default_model_size = config.get("default_model_size", "")
    default_model_name = config.get("default_model_name", "")
    model_configs = config.get("model_configs", {})
    return default_model_size, default_model_name, model_configs


def sanitize_name(name: str):
    return name.replace(":", "_").replace("/", "_")


def cosine_distance(data1, data2):
    if data1.shape != data2.shape:
        print(f"[error] shape not equal {data1.shape} vs {data2.shape}")
        return -1
    v1_d = data1.flatten().astype("float64")
    v2_d = data2.flatten().astype("float64")
    v1_d[v1_d == np.inf] = np.finfo(np.float16).max
    v2_d[v2_d == np.inf] = np.finfo(np.float16).max
    v1_d[v1_d == -np.inf] = np.finfo(np.float16).min
    v2_d[v2_d == -np.inf] = np.finfo(np.float16).min
    v1_norm = v1_d / np.linalg.norm(v1_d)
    v2_norm = v2_d / np.linalg.norm(v2_d)
    cosine_dist = np.dot(v1_norm, v2_norm)
    if np.isnan(cosine_dist):
        return -1
    return cosine_dist


class ProcessMemoryMonitor:
    def __init__(self, interval=2, log_file=None):
        self.process = psutil.Process(os.getpid())
        self.interval = interval
        self.log_file = log_file
        self.is_monitoring = False
        self.peak_memory_mb = 0

    def get_memory_info(self):
        memory_info = self.process.memory_info()
        rss_mb = memory_info.rss / (1024 * 1024)
        percent = self.process.memory_percent()
        return {"rss_mb": rss_mb, "percent": percent}

    def start(self):
        self.is_monitoring = True
        self.peak_memory_mb = 0
        self.monitor_thread = threading.Thread(target=self._monitor_loop)
        self.monitor_thread.daemon = True
        self.monitor_thread.start()
        print(f"Memory monitoring started (interval: {self.interval}s)")

    def _monitor_loop(self):
        while self.is_monitoring:
            mem_info = self.get_memory_info()
            self.peak_memory_mb = max(self.peak_memory_mb, mem_info["rss_mb"])
            if self.log_file:
                timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
                log_message = f"{timestamp} - RSS: {mem_info['rss_mb']:.2f} MB"
                with open(self.log_file, "a") as f:
                    f.write(log_message + "\n")
            time.sleep(self.interval)

    def stop(self):
        self.is_monitoring = False
        if hasattr(self, "monitor_thread"):
            self.monitor_thread.join(timeout=1)
        print(f"[Monitoring stopped. Peak RSS: {self.peak_memory_mb:.2f} MB]")


def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        dest="config_path",
        type=str,
        default=DEFAULT_CONFIG_PATH,
        help="path to config.yaml",
    )
    parser.add_argument(
        "--model_dir",
        dest="model_dir",
        type=str,
        default=os.path.join("work_dirs", "hmonnx"),
        help="path to the hmonnx model dir",
    )
    parser.add_argument(
        "--model_name",
        dest="model_name",
        type=str,
        default=None,
        help="output houmo model name",
    )
    parser.add_argument(
        "--model_size",
        dest="model_size",
        type=str,
        default=None,
        help="model size",
    )
    parser.add_argument(
        "--j",
        dest="j",
        type=int,
        default=multiprocessing.cpu_count(),
        help="build parallel jobs",
    )
    parser.add_argument(
        "--ncore",
        dest="ncore",
        type=int,
        default=None,
        help="core number",
    )
    parser.add_argument(
        "--stage",
        dest="stage",
        type=str,
        default="build",
        choices=["build", "test", "all"],
        help="build stage",
    )
    parser.add_argument(
        "--output_dir",
        dest="output_dir",
        type=str,
        default=os.path.join("output", HOUMO_TARGET),
        help="build output dir",
    )
    args = parser.parse_args()

    default_model_size, default_model_name, model_configs = get_model_configs(
        args.config_path
    )
    args.model_name = first_not_none(args.model_name, default_model_name)
    args.model_size = first_not_none(args.model_size, default_model_size)
    model_config = model_configs.get(args.model_name, {}).get(args.model_size, {})
    args.ncore = first_not_none(args.ncore, model_config.get("ncore", HOUMO_CORE_NUM))

    return args


def build_model(model_name, model_dir, output_dir, ncore, j):
    import tcim

    start = time.time()
    print(f"\n===> {model_name} build start...")
    onnx_files = glob.glob(os.path.join(model_dir, f"*{model_name}*.onnx"))
    
    if not onnx_files:
        print(f"[warning] No ONNX file matching *{model_name}*.onnx found, trying all .onnx files...")
        onnx_files = glob.glob(os.path.join(model_dir, "*.onnx"))
    
    if not onnx_files:
        print(f"[error] Not found ONNX model in {model_dir}")
        exit(-1)
        
    decode_model = os.path.abspath(onnx_files[0])
    print(f"Using HMONNX model for TCIM build: {decode_model}")

    tcim.build_from_hmonnx(
        decode_model,
        output_name=model_name,
        ncore=ncore,
        target=HOUMO_TARGET,
        output_dir=output_dir,
        work_dir=os.path.join(output_dir, "tcim", model_name),
        j=j,
    )
    print(f'{model_name} build completed in {time.time() - start:.3f} s.', flush=True)


def test_model(model_name, model_dir, output_dir, batch=1, prefix=None):
    try:
        import tcim_lite
    except ImportError:
        print("[error] tcim_lite importing failed. Test cancelled.")
        return

    print(f"\n===> {model_name} test start...")
    model_path = os.path.join(output_dir, f"{model_name}.hmm")
    if not os.path.exists(model_path):
        print(f"[error] Cannot find {model_path} for testing. Make sure build is complete.")
        return
        
    start = time.time()
    module = tcim_lite.runtime.load(model_path)
    print(f'{model_name} load completed in {time.time() - start:.3f} s.', flush=True)

    if prefix is None:
        prefix = model_name

    input_num = module.get_num_inputs()
    for id in range(input_num):
        input_name = module.get_input_name(id)
        input_info = module.get_input_info(input_name)
        input_data_path = os.path.join(model_dir, f"{input_name}.npy") # Default xhquant dumped inputs look like this
        
        if not os.path.exists(input_data_path):
            # For backward match
            input_data_path = os.path.join(model_dir, f"hmquant_{prefix}_{sanitize_name(input_name)}_input.npy")

        if os.path.exists(input_data_path):
            input_data = np.load(input_data_path).astype(input_info.dtype)
            input_data = np.concatenate([input_data for _ in range(batch)], axis=0)
            module.set_input(input_name, input_data)
        else:
            print(f"[warning] No input data found for {input_name}, bypassing manual verify. Attempt to run with dummy.")
            dummy_data = np.zeros(input_info.shape, dtype=input_info.dtype)
            module.set_input(input_name, dummy_data)

    start = time.time()
    module.run()
    module.sync()
    print(f'{model_name} infer completed in {(time.time() - start)*1000:.3f} ms.')

    result_check = True
    output_num = module.get_num_outputs()
    for id in range(output_num):
        output_name = module.get_output_name(id)
        output_data = module.get_output(output_name).numpy()
        print(f"output[{output_name}] shape = {output_data.shape}, dtype = {output_data.dtype}")
        
        output_data_path = os.path.join(model_dir, f"{output_name}.npy")
        if not os.path.exists(output_data_path):
            output_data_path = os.path.join(model_dir, f"hmquant_{prefix}_{sanitize_name(output_name)}_output.npy")
            
        if os.path.exists(output_data_path):
            golden_output = np.load(output_data_path)
            golden_output = np.concatenate([golden_output for _ in range(batch)], axis=0)
            if golden_output.shape == output_data.shape:
                cosine_dist = cosine_distance(golden_output, output_data)
                is_match = (golden_output == output_data).all()
                print(f"[compare] golden output [{output_name}] match={is_match}, similarity={cosine_dist:.6f}")
                if not is_match and cosine_dist < GOLDEN_THRESH:
                    result_check = False
            else:
                result_check = False
                print(f"[compare] golden shape mismatch {golden_output.shape} vs {output_data.shape}")
        else:
            print(f"[info] No golden outputs found to compare with {output_name}.")

    if not result_check:
        print("[error] result check failed.")
        exit(-1)
    print(f"<=== {model_name} test success.")


if __name__ == "__main__":
    memory_monitor = ProcessMemoryMonitor(interval=2)
    memory_monitor.start()

    args = get_args()
    print(args)
    
    import platform
    arch = platform.machine()
    
    os.makedirs(args.output_dir, exist_ok=True)

    if args.stage in ["build", "all"]:
        if arch != "x86_64":
            print(f"[error] tcim only supports x86_64 platform, current is: {arch}")
            exit(0)
        build_model(args.model_name, args.model_dir, args.output_dir, args.ncore, args.j)

    if args.stage in ["test", "all"]:
        test_model(args.model_name, args.model_dir, args.output_dir, prefix="ct_transformer")

    memory_monitor.stop()

