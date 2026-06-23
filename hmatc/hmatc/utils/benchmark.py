# Copyright 2025 HOUMO AI
#
# File: benchmark.py
# Description:
#   Benchmark script for HMATC model performance testing on XH2.
#   Outputs two Excel reports: success summary and full results.
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
import time
import pandas as pd
import json
import shutil
import platform
from multiprocessing import get_context, Queue
from . import logger
from .utils import (
    read_yaml_to_dict,
    get_hmquant_xh2_version,
    get_package_version,
    get_houmo_version,
)
from .check import check_cfg
from .onnx_profile import model_profile


def run_single_model(
    location: str,
    cfg_path: str,
    queue: Queue,
    batch_num: int = 1,
    core_num: int = 1,
    thread_num: int = 1,
    device_id: int = 0,
    enable_eval: bool = False,
    enable_static: bool = True,
    warmup: int = 10,
    sample: int = 1000,
):
    """
    Execute a single model test with specified configuration for XH2.

    Returns model info dict via queue containing:
    - Performance metrics (latency, throughput, H2D/D2H latency)
    - Accuracy data (if enable_eval=True and 1c1t config)
    - Success status and error message
    """
    platform_arch = platform.machine().lower()

    result = {
        "input_size": "N/A",
        "gops": 0,
        "platform": platform_arch,
        "batch_num": batch_num,
        "core_num": core_num,
        "thread_num": thread_num,
        "resizer": "Static",
        "dataset": "N/A",
        "dataset_num": "N/A",
        "acc_onnx": "N/A",
        "acc_chip": "N/A",
        "acc_err": "N/A",
        "e2e_avg_cost": "N/A",
        "throughput": "N/A",
        "infer_avg_latency": "N/A",
        "infer_max_latency": "N/A",
        "input_avg_h2d": "N/A",
        "input_max_h2d": "N/A",
        "output_avg_d2h": "N/A",
        "output_max_d2h": "N/A",
        "success": False,
        "msg": "ok",
    }

    # Check model location
    if not os.path.exists(location):
        alt_location = os.path.join(os.environ.get("HOUMO_EXAMPLES_PATH", ""), location)
        if os.path.exists(alt_location):
            location = alt_location
        else:
            result["msg"] = f"Model not found: {location}"
            queue.put(result)
            return

    # Check core limit for XH2 (max 2 cores)
    max_cores = 2
    if core_num > max_cores:
        result["msg"] = f"core_num {core_num} exceeds limit {max_cores}"
        queue.put(result)
        return

    root = os.getcwd()
    os.chdir(location)

    # Download model
    try:
        os.system("python3 get_model.py --type raw")
        logger.info("Model download completed")
    except Exception as e:
        result["msg"] = f"Download failed: {e}"
        os.chdir(root)
        queue.put(result)
        return

    # Load config
    if not os.path.exists(cfg_path):
        result["msg"] = f"Config not found: {cfg_path}"
        os.chdir(root)
        queue.put(result)
        return

    cfg = read_yaml_to_dict(cfg_path)
    if not check_cfg(cfg):
        result["msg"] = f"Invalid config: {cfg_path}"
        os.chdir(root)
        queue.put(result)
        return

    model_name = cfg["model"]["name"]
    inputs_cfg = cfg["model"]["inputs"]
    if len(inputs_cfg) > 1:
        raise ValueError("Only one input is supported")
    ext = "_static" if enable_static else "_dynamic"
    input_name = list(cfg["model"]["inputs"].keys())[0]
    input_cfg = cfg["model"]["inputs"][input_name]
    if "resizer" not in input_cfg:
        ext = ""
    cfg["target"] = "xh2"
    cfg["build"]["ncore"] = core_num
    cfg["build"]["batch"] = batch_num
    if not enable_static and "resizer" in input_cfg:
        if cfg["model"]["inputs"][input_name]["resizer"] is None:
            cfg["model"]["inputs"][input_name]["resizer"] = {"resizer_mode": 1}
        elif isinstance(cfg["model"]["inputs"][input_name]["resizer"], dict):
            cfg["model"]["inputs"][input_name]["resizer"]["resizer_mode"] = 1
        else:
            raise ValueError("Invalid resizer configuration")

    cfg["model"]["save_dir"] = f"benchmark/{get_houmo_version()}/{model_name}{ext}"

    logger.info(f"Config: {json.dumps(cfg, indent=2)}")

    # Create XH2 executor
    try:
        from ..exec.xh2_exec import Xh2Exec

        hm_exec = Xh2Exec(cfg)
    except Exception as e:
        result["msg"] = f"Executor creation failed: {e}"
        os.chdir(root)
        queue.put(result)
        return

    # Get model info
    shape_str = ""
    for idx in range(len(hm_exec.inputs_shape)):
        shape = hm_exec.inputs_shape[idx]
        shape_str = "x".join(map(str, shape))
        if idx == len(hm_exec.inputs_shape) - 1:
            break
        shape_str += "\n"
    result["input_size"] = shape_str
    result["gops"] = model_profile(hm_exec.model_path) * 2 / 1e9

    # Get resizer mode
    result["resizer"] = hm_exec.resizer_mode

    # Quantization (x86_64 only)
    if platform_arch == "x86_64":
        quant_path = hm_exec.quant_onnx_model_path
        if os.path.exists(quant_path):
            logger.info(f"Skip quantization: {quant_path}")
        else:
            try:
                if hm_exec.quantize():
                    logger.info("Quantization completed")
                else:
                    result["msg"] = "Quantization failed"
                    os.chdir(root)
                    queue.put(result)
                    return
            except Exception as e:
                result["msg"] = f"Quantization error: {e}"
                os.chdir(root)
                queue.put(result)
                return

    # Build (x86_64 only)
    if platform_arch == "x86_64":
        if os.path.exists(hm_exec.hmm_path):
            logger.info(f"Skip build: {hm_exec.hmm_path}")
        else:
            try:
                hm_exec.build()
                logger.info("Build completed")
            except Exception as e:
                result["msg"] = f"Build error: {e}"
                os.chdir(root)
                queue.put(result)
                return

    # Check HMM existence (non-x86)
    if platform_arch != "x86_64":
        if not os.path.exists(hm_exec.hmm_path):
            result["msg"] = f"HMM not found: {hm_exec.hmm_path}"
            os.chdir(root)
            queue.put(result)
            return

    # Performance test
    try:
        perf_info = hm_exec.model_perf(
            hm_exec.hmm_path,
            warmup,
            sample,
            loop_num=1,
            thread_num=thread_num,
            devices=[device_id],
        )
        perf = perf_info["perf"]["perf_info"]
        result["e2e_avg_cost"] = f"{perf['avg_cost']:.2f}"
        result["throughput"] = f"{perf['qps'] * batch_num:.2f}"
        result["infer_avg_latency"] = f"{perf['infer_avg_latency']:.2f}"
        result["infer_max_latency"] = f"{perf['infer_max_latency']:.2f}"
        result["input_avg_h2d"] = f"{perf['input_avg_latency']:.2f}"
        result["input_max_h2d"] = f"{perf['input_max_latency']:.2f}"
        result["output_avg_d2h"] = f"{perf['output_avg_latency']:.2f}"
        result["output_max_d2h"] = f"{perf['output_max_latency']:.2f}"
        logger.info("Performance test completed")
    except Exception as e:
        result["msg"] = f"Performance error: {e}"
        os.chdir(root)
        queue.put(result)
        return

    # Accuracy evaluation (only for 1c1t config with enable_eval=True)
    if enable_eval and "eval" in cfg:
        result["dataset"] = cfg["eval"].get("data_dir", "N/A").split("/")[-1]

        # ONNX evaluation
        onnx_info = {}
        onnx_eval_result = {}
        try:
            shutil.rmtree("results_onnx", ignore_errors=True)
            onnx_eval_result = hm_exec.evaluate(backend="onnx")
            onnx_info = (
                onnx_eval_result.get("eval", {}).get("onnx", {}).get("results", {})
            )
        except Exception as e:
            logger.warning(f"ONNX eval failed: {e}")

        # XH2 chip evaluation
        chip_info = {}
        chip_eval_result = {}
        try:
            shutil.rmtree("results_xh2", ignore_errors=True)
            chip_eval_result = hm_exec.evaluate(backend="xh2", device_id=device_id)
            chip_info = (
                chip_eval_result.get("eval", {}).get("xh2", {}).get("results", {})
            )
        except Exception as e:
            logger.warning(f"XH2 eval failed: {e}")

        # Get dataset_num from eval result (prefer onnx, fallback to chip)
        dataset_num = onnx_info.get("num", "0")
        result["dataset_num"] = dataset_num

        # Calculate accuracy error
        if onnx_info:
            if "top1_acc" in onnx_info or "acc" in onnx_info:
                key = "acc" if "acc" in onnx_info else "top1_acc"
                acc_onnx = float(onnx_info.get(key, 0))
                acc_chip = float(chip_info.get(key, 0))
                if acc_onnx > 0:
                    acc_err = (acc_chip / acc_onnx - 1) * 100
                    result["acc_onnx"] = f"{acc_onnx*100:.2f}"
                    result["acc_chip"] = f"{acc_chip*100:.2f}"
                    result["acc_err"] = f"{acc_err:.2f}%"
            elif "map50" in onnx_info:
                map50_onnx = float(onnx_info.get("map50", 0))
                map50_chip = float(chip_info.get("map50", 0))
                map50_95_onnx = float(onnx_info.get("map50_95", 0))
                map50_95_chip = float(chip_info.get("map50_95", 0))
                if map50_onnx > 0:
                    result["acc_onnx"] = f"{map50_95_onnx*100:.2f}/{map50_onnx*100:.2f}"
                    result["acc_chip"] = f"{map50_95_chip*100:.2f}/{map50_chip*100:.2f}"
                    err_95 = (
                        (map50_95_chip / map50_95_onnx - 1) * 100
                        if map50_95_onnx > 0
                        else 0
                    )
                    err_50 = (map50_chip / map50_onnx - 1) * 100
                    result["acc_err"] = f"{err_95:.2f}%/{err_50:.2f}%"
            elif "ap_easy" in onnx_info:
                # WiderFace detection
                onnx_ap_easy = float(onnx_info.get("ap_easy", 0))
                chip_ap_easy = float(chip_info.get("ap_easy", 0))
                onnx_ap_medium = float(onnx_info.get("ap_medium", 0))
                chip_ap_medium = float(chip_info.get("ap_medium", 0))
                onnx_ap_hard = float(onnx_info.get("ap_hard", 0))
                chip_ap_hard = float(chip_info.get("ap_hard", 0))
                if onnx_ap_easy > 0:
                    result["acc_onnx"] = (
                        f"{onnx_ap_easy:.2f}/{onnx_ap_medium:.2f}/{onnx_ap_hard:.2f}"
                    )
                    result["acc_chip"] = (
                        f"{chip_ap_easy:.2f}/{chip_ap_medium:.2f}/{chip_ap_hard:.2f}"
                    )
                    err_easy = (chip_ap_easy / onnx_ap_easy - 1) * 100
                    err_medium = (
                        (chip_ap_medium / onnx_ap_medium - 1) * 100
                        if onnx_ap_medium > 0
                        else 0
                    )
                    err_hard = (
                        (chip_ap_hard / onnx_ap_hard - 1) * 100
                        if onnx_ap_hard > 0
                        else 0
                    )
                    result["acc_err"] = (
                        f"{err_easy:.2f}%/{err_medium:.2f}%/{err_hard:.2f}%"
                    )

        logger.info("Accuracy evaluation completed")

    result["success"] = True
    os.chdir(root)
    queue.put(result)


def run_benchmark(
    config_path: str,
    device_id: int = 0,
):
    """
    Run benchmark for all models on XH2 and generate two Excel reports:
    1. Success summary: Key metrics per model (accuracy + latency1c1t + throughput1c4t)
    2. Full results: All run attempts including failures

    Args:
        config_path: Path to benchmark configuration YAML
        device_id: Device ID for evaluation
    """
    platform_arch = platform.machine().lower()

    # Get versions
    hmquant_version = get_hmquant_xh2_version()
    hmcc_version = get_package_version("houmo_tcim_xh2")
    runtime_version = get_package_version("houmo_tcim_runtime_xh2")

    if runtime_version == "N/A":
        logger.fatal("Runtime not found: houmo_tcim_runtime_xh2")

    houmo_version = get_houmo_version()

    # Load benchmark config
    models_config = read_yaml_to_dict(config_path)
    models = models_config.get("models", {})

    # Prepare output directory
    timestamp = time.strftime("%Y%m%d_%H%M%S", time.localtime())
    os.makedirs("reports", exist_ok=True)
    base_path = os.path.abspath("reports")

    success_file = os.path.join(
        base_path,
        f"benchmark_xh2_{houmo_version}_success_{timestamp}.xlsx",
    )
    full_file = os.path.join(
        base_path,
        f"benchmark_xh2_{houmo_version}_full_{timestamp}.xlsx",
    )

    # Define headers
    headers = [
        "ModelName",
        "Shape",
        "Dataset",
        "DatasetNum",
        "GOPs",
        "Platform",
        "CoreNum",
        "BatchNum",
        "ThreadNum",
        "Resizer",
        "HmquantVersion",
        "CompilerVersion",
        "RuntimeVersion",
        "Accuracy[onnx]",
        "Accuracy[xh2]",
        "AccRelError",
        "End2End_Cost[ms]",
        "Throughput",
        "Infer_Avg[ms]",
        "Infer_Max[ms]",
        "Input_AvgH2D[ms]",
        "Input_MaxH2D[ms]",
        "Output_AvgD2H[ms]",
        "Output_MaxD2H[ms]",
        "Success",
        "ErrorMsg",
    ]

    # Results storage
    all_results = []

    ctx = get_context("spawn")
    queue = ctx.Queue()

    def get_empty_result(
        model_name: str, batch_num: int, core_num: int, thread_num: int, msg: str
    ):
        """Generate empty result dict for failed/timeout cases."""
        return {
            "model_name": model_name,
            "input_size": "N/A",
            "gops": 0,
            "platform": platform_arch,
            "batch_num": batch_num,
            "core_num": core_num,
            "thread_num": thread_num,
            "resizer": "N/A",
            "dataset": "N/A",
            "dataset_num": "N/A",
            "acc_onnx": "N/A",
            "acc_chip": "N/A",
            "acc_err": "N/A",
            "e2e_avg_cost": "N/A",
            "throughput": "N/A",
            "infer_avg_latency": "N/A",
            "infer_max_latency": "N/A",
            "input_avg_h2d": "N/A",
            "input_max_h2d": "N/A",
            "output_avg_d2h": "N/A",
            "output_max_d2h": "N/A",
            "success": False,
            "msg": msg,
        }

    def run_config(
        model_name: str,
        location: str,
        cfg_path: str,
        exec_cfg: dict,
        enable_static=True,
    ):
        """Run single config and return result."""
        batch_num = exec_cfg.get("batch_num", 1)
        core_num = exec_cfg.get("core_num", 1)
        thread_num = exec_cfg.get("thread_num", 1)
        enable_eval = exec_cfg.get("enable_eval", False) if enable_static else False
        warmup = exec_cfg.get("warmup", 10)
        sample = exec_cfg.get("sample", 1000)

        process = ctx.Process(
            target=run_single_model,
            args=(location, cfg_path, queue),
            kwargs={
                "batch_num": batch_num,
                "core_num": core_num,
                "thread_num": thread_num,
                "device_id": device_id,
                "enable_eval": enable_eval,
                "warmup": warmup,
                "sample": sample,
                "enable_static": enable_static,
            },
        )
        process.start()
        process.join(timeout=7200)  # 2 hour timeout

        if process.exitcode is None:
            logger.warning(f"{model_name} timeout, killing process")
            process.kill()
            process.join()
            return get_empty_result(
                model_name, batch_num, core_num, thread_num, "timeout"
            )
        elif process.exitcode != 0:
            logger.error(f"{model_name} failed with exitcode {process.exitcode}")
            return get_empty_result(
                model_name,
                batch_num,
                core_num,
                thread_num,
                f"exitcode={process.exitcode}",
            )
        else:
            if queue.empty():
                return get_empty_result(
                    model_name, batch_num, core_num, thread_num, "no result"
                )
            result = queue.get()
            result["model_name"] = model_name
            return result

    # Run all models
    logger.info(f"Starting XH2 benchmark for {len(models)} models")
    logger.info(f"Success report: {success_file}")
    logger.info(f"Full report: {full_file}")

    for model_name, model_cfg in models.items():
        location = model_cfg["location"]
        cfg_path = model_cfg.get("config", "config.yml")
        exec_cfgs = model_cfg.get("exec_cfgs", [])

        logger.info(f"\n{'='*60}")
        logger.info(f"Model: {model_name}")

        for exec_cfg in exec_cfgs:
            # Run with static resizer (default)
            result_static = run_config(
                model_name, location, cfg_path, exec_cfg, enable_static=True
            )
            # Add version info
            result_static["hmquant_version"] = hmquant_version
            result_static["hmcc_version"] = hmcc_version
            result_static["runtime_version"] = runtime_version

            all_results.append(result_static)

            # Log result
            status = "SUCCESS" if result_static["success"] else "FAILED"
            logger.info(
                f"[{status}] {model_name} (b={result_static['batch_num']}, "
                f"c={result_static['core_num']}, t={result_static['thread_num']}, static) - "
                f"latency={result_static['infer_avg_latency']}ms, throughput={result_static['throughput']}"
            )

            if result_static["resizer"] == "NO RESIZER":
                continue

            # Run with dynamic resizer if model supports it
            result_dynamic = run_config(
                model_name, location, cfg_path, exec_cfg, enable_static=False
            )
            # Add version info
            result_dynamic["hmquant_version"] = hmquant_version
            result_dynamic["hmcc_version"] = hmcc_version
            result_dynamic["runtime_version"] = runtime_version
            all_results.append(result_dynamic)
            # Log result
            status = "SUCCESS" if result_dynamic["success"] else "FAILED"
            logger.info(
                f"[{status}] {model_name} (b={result_dynamic['batch_num']}, "
                f"c={result_dynamic['core_num']}, t={result_dynamic['thread_num']}, dynamic) - "
                f"latency={result_dynamic['infer_avg_latency']}ms, throughput={result_dynamic['throughput']}"
            )

            # Save intermediate results to Excel
            save_results(all_results, full_file, success_file, headers)

    # Final save
    save_results(all_results, full_file, success_file, headers)

    # Summary
    success_count = sum(1 for r in all_results if r["success"])
    logger.info(f"\nBenchmark completed!")
    logger.info(f"Full report: {full_file} ({len(all_results)} entries)")
    logger.info(f"Success report: {success_file} ({success_count} entries)")


def save_results(
    all_results: list,
    full_file: str,
    success_file: str,
    headers: list,
):
    """Save results to Excel files.

    full_file: All records (including failures)
    success_file: Only successful records
    Both use the same headers (26 columns)
    """

    def build_row(r):
        return [
            r["model_name"],
            r["input_size"],
            r["dataset"],
            r["dataset_num"],
            (
                f"{r['gops']:.2f}"
                if isinstance(r["gops"], (int, float)) and r["gops"] > 0
                else "N/A"
            ),
            r["platform"],
            r["core_num"],
            r["batch_num"],
            r["thread_num"],
            r["resizer"],
            r.get("hmquant_version", "N/A"),
            r.get("hmcc_version", "N/A"),
            r.get("runtime_version", "N/A"),
            r["acc_onnx"],
            r["acc_chip"],
            r["acc_err"],
            r["e2e_avg_cost"],
            r["throughput"],
            r["infer_avg_latency"],
            r["infer_max_latency"],
            r["input_avg_h2d"],
            r["input_max_h2d"],
            r["output_avg_d2h"],
            r["output_max_d2h"],
            r["success"],
            r["msg"],
        ]

    # Save full results (all configs including failures)
    full_rows = [build_row(r) for r in all_results]
    df_full = pd.DataFrame(full_rows, columns=headers)
    df_full.to_excel(full_file, index=False, engine="xlsxwriter")

    # Save success results (only successful records)
    success_rows = [build_row(r) for r in all_results if r["success"]]
    df_success = pd.DataFrame(success_rows, columns=headers)
    df_success.to_excel(success_file, index=False, engine="xlsxwriter")
