# Copyright 2025 HOUMO AI
#
# File: main.py
# Description:
#     Main entry point for the HMATC tool.
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
import json
import logging
import os
import time
import shutil
import pandas as pd
import platform
import torch
from multiprocessing import get_context, Queue
from io import StringIO
from prettytable import PrettyTable
from ._version import __build_time__, __commit__, __version__
from .base.base_exec import BaseExec
from .utils import logger
from .utils.check import check_cfg
from .utils.logging_format import LoggingFormatter
from .utils.utils import (
    read_yaml_to_dict,
    save_dict_to_yaml,
    get_hmquant_xh1_version,
    get_hmquant_xh2_version,
    get_package_version,
    get_houmo_version,
    set_random_seed,
)
from .onnx_tool import model_profile


def set_logger(op, log_dir, filename):
    """
    Set up a logger that writes to a timestamped log file in the specified directory.

    Args:
        op (str): Operation name used in the log file name
        log_dir (str): Directory where the log file will be saved
        filename (str): Base name of the log file
    """
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    t = time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime())
    filepath = os.path.join(log_dir, "{}-{}-{}.log".format(filename, op, t))
    file_handler = logging.FileHandler(filepath)
    file_handler.setFormatter(LoggingFormatter())
    logger.addHandler(file_handler)


def run_model(
    location,
    cfg_path,
    target,
    queue,
    batch_num=1,
    core_num=1,
    thread_num=1,
    device_id=0,
    enbale_quantize=True,
    enable_build=True,
    enable_eval=True,
    enable_onnx_eval=True,
    enable_chip_eval=True,
    enable_static=False,
    enable_cuda=False,
    enable_upload=False,
    enable_delete=False,
):
    """
    Execute a model with specified parameters and configurations.

    Args:
        location (str): Path to the model location
        cfg_path (str): Path to the configuration file
        target (str): Target platform (xh1 or xh2)
        queue (Queue): Queue to store the results
        batch_num (int): Number of batches for processing
        core_num (int): Number of cores to use
        thread_num (int): Number of threads to use
        device_id (int): ID of the device to use
        enbale_quantize (bool): Whether to enable quantization
        enable_build (bool): Whether to enable model building
        enable_eval (bool): Whether to enable evaluation
        enable_onnx_eval (bool): Whether to enable ONNX evaluation
        enable_chip_eval (bool): Whether to enable chip evaluation
        enable_static (bool): Whether to enable static mode
        enable_cuda (bool): Whether to enable CUDA for quantization
        enable_upload (bool): Whether to enable upload
        enable_delete (bool): Whether to enable deletion of existing files

    Returns:
        dict: Dictionary containing model information and results
    """
    model_infos = dict(
        input_size="N/A",
        dataset="N/A",
        dataset_num="N/A",
        GOPs=0,
        target=target,
        batch_num=batch_num,
        core_num=core_num,
        thread_num=thread_num,
        resizer="N/A",
        acc_onnx="N/A",
        acc_chip="N/A",
        acc_err="N/A",
        e2e_avg_cost="N/A",
        throughput="N/A",
        infer_avg_latency="N/A",
        infer_max_latency="N/A",
        input_avg_H2D_latency="N/A",
        input_max_H2D_latency="N/A",
        output_avg_D2H_latency="N/A",
        output_max_D2H_latency="N/A",
        enable_static=True,
        msg="ok",
    )

    if not os.path.exists(location):
        # Try to find the model in the HOUMO_MODEL_PATH environment variable
        new_location = os.path.join(os.environ.get("HOUMO_MODEL_PATH", ""), location)
        if os.path.exists(new_location):
            location = new_location
        else:
            msg = f"Not found model: {new_location}"
            logger.error(msg)
            model_infos["msg"] = msg
            queue.put(model_infos)
            return

    if core_num > int(os.getenv("HOUMO_CORE_NUM", 4 if target == "xh1" else 2)):
        msg = f"{target} core_num must less than {os.getenv('HOUMO_CORE_NUM', 4 if target == 'xh1' else 2)}"
        logger.error(msg)
        model_infos["msg"] = msg
        queue.put(model_infos)
        return

    root = os.getcwd()
    os.chdir(location)

    # Download model
    try:
        os.system("python3 get_model.py --type raw")
        logger.info(f"Download done.")
    except Exception as e:
        msg = f"Download failed:\nException: {e}"
        logger.error(msg)
        model_infos["msg"] = msg
        os.chdir(root)
        queue.put(model_infos)
        return

    # Parse cfg
    if not os.path.exists(cfg_path):
        msg = f"{cfg_path} not exists"
        logger.error(msg)
        model_infos["msg"] = msg
        os.chdir(root)
        queue.put(model_infos)
        return

    cfg = read_yaml_to_dict(cfg_path)
    if not check_cfg(cfg):
        msg = f"{cfg_path} is not valid"
        logger.error(msg)
        model_infos["msg"] = msg
        os.chdir(root)
        queue.put(model_infos)
        return

    cfg["target"] = target
    cfg["build"]["ncore"] = core_num
    cfg["build"]["batch"] = batch_num
    inputs_cfg = cfg["model"]["inputs"]
    input_name = list(inputs_cfg.keys())[0]
    if not enable_static:
        if len(inputs_cfg) != 1 or "resizer" not in inputs_cfg[input_name]:
            logger.info("resizer is disabled, dynamic mode skipped")
            os.chdir(root)
            queue.put(model_infos)
            return
        if (
            "enable_static_resizer" in inputs_cfg[input_name]["resizer"]
            and inputs_cfg[input_name]["resizer"]["enable_static_resizer"]
        ):
            logger.info("dynamic_resizer is disabled, skipped")
            os.chdir(root)
            queue.put(model_infos)
            return
        inputs_cfg[input_name]["resizer"]["enable_static_resizer"] = False
        model_infos["enable_static"] = False

    logger.info(f"\n{json.dumps(cfg, indent=2, sort_keys=False)}")

    hm_exec = None
    try:
        if target == "xh1":
            from .exec.xh1_exec import Xh1Exec

            hm_exec = Xh1Exec(cfg)
        elif target == "xh2":
            from .exec.xh2_exec import Xh2Exec

            hm_exec = Xh2Exec(cfg)
    except Exception as e:
        msg = f"Failed to create hm_exec: \n{e}"
        logger.error(msg)
        model_infos["msg"] = msg
        os.chdir(root)
        queue.put(model_infos)
        return
        # return model_infos
    if enable_cuda and target == "xh1" and torch.cuda.is_available():
        hm_exec.device = "cuda"
    hm_exec.enable_upload = enable_upload
    if enable_delete:
        shutil.rmtree(os.path.join(hm_exec.save_dir, target), ignore_errors=True)
    input_size = "x".join(map(str, hm_exec.inputs_shape[0]))
    for idx in range(1, len(hm_exec.inputs_shape)):
        input_size += "\n"
        input_size += "x".join(map(str, hm_exec.inputs_shape[idx]))
    model_infos["input_size"] = input_size
    macs = model_profile(hm_exec.model_path)
    model_infos["GOPs"] = macs * 2 / 1e9

    resizer_mode = hm_exec.resizer_mode
    if resizer_mode == 0:
        resizer_mode = "N/A"
    elif resizer_mode == 1:
        resizer_mode = "Dynamic_v2"
    elif resizer_mode == 2:
        resizer_mode = "Dynamic_v1"
    elif resizer_mode == 3:
        resizer_mode = "Static"
    else:
        resizer_mode = "unknown"
    model_infos["resizer"] = resizer_mode
    platform_arch = platform.machine().lower()

    # Quantization
    if hm_exec is not None and platform_arch == "x86_64" and enbale_quantize:
        try:
            res = hm_exec.quantize()
            if not res:
                msg = "Quantize failed."
                logger.error(msg)
                model_infos["msg"] = msg
                os.chdir(root)
                queue.put(model_infos)
                return
                # return model_infos
            logger.info(f"Quantize done.")
        except Exception as e:
            msg = f"Quantize failed:\nException: {e}"
            logger.error(msg)
            model_infos["msg"] = msg
            os.chdir(root)
            queue.put(model_infos)
            return
            # return model_infos
    # Compilation
    if platform_arch == "x86_64" and enable_build:
        try:
            hm_exec.build()
            logger.info(f"Build done.")
        except Exception as e:
            msg = f"Build failed:\nException: {e}"
            logger.error(msg)
            model_infos["msg"] = msg
            os.chdir(root)
            queue.put(model_infos)
            return
            # return model_infos

    if platform_arch != "x86_64" and not os.path.exists(hm_exec.hmm_path):
        msg = (
            f"[{platform_arch}] HMM model does not exist, and path: {hm_exec.hmm_path}."
        )
        logger.error(msg)
        model_infos["msg"] = msg
        os.chdir(root)
        queue.put(model_infos)
        return

    # Performance testing
    try:
        warmup = 10
        sample = 1000
        loop_num = 1
        perf_info = hm_exec.model_perf(
            hm_exec.hmm_path, warmup, sample, loop_num, thread_num, devices=[device_id]
        )
        perf_info = list(perf_info["perf"].values())[0]["perf_info"]
        model_infos["e2e_avg_cost"] = f"{perf_info['avg_cost']:.2f}"
        model_infos["throughput"] = f"{perf_info['qps'] * batch_num:.2f}"
        model_infos["infer_avg_latency"] = f"{perf_info['infer_avg_latency']:.2f}"
        model_infos["infer_max_latency"] = f"{perf_info['infer_max_latency']:.2f}"
        model_infos["input_avg_H2D_latency"] = f"{perf_info['input_avg_latency']:.2f}"
        model_infos["input_max_H2D_latency"] = f"{perf_info['input_max_latency']:.2f}"
        model_infos["output_avg_D2H_latency"] = f"{perf_info['output_avg_latency']:.2f}"
        model_infos["output_max_D2H_latency"] = f"{perf_info['output_max_latency']:.2f}"
        logger.info(f"Performance done.")
    except Exception as e:
        msg = f"Performance failed:\nException: {e}"
        logger.error(msg)
        model_infos["msg"] = msg
        os.chdir(root)
        queue.put(model_infos)
        return
        # return model_infos

    # ONNX dataset evaluation
    onnx_info = dict()
    if enable_onnx_eval and enable_eval and "eval" in cfg:
        try:
            # Delete cache
            shutil.rmtree("results_onnx", ignore_errors=True)
            onnx_info = hm_exec.evaluate(backend="onnx")
            if not onnx_info:
                msg = f"onnx eval failed"
                model_infos["msg"] = msg
                logger.error(msg)
        except Exception as e:
            msg = f"onnx eval failed:\nException: {e}"
            logger.error(msg)
            model_infos["msg"] = msg
            onnx_info = dict()

    # xh1/xh2 dataset evaluation
    chip_info = dict()
    if enable_chip_eval and enable_eval and "eval" in cfg:
        try:
            shutil.rmtree(f"results_{target}", ignore_errors=True)
            chip_info = hm_exec.evaluate(backend=target, device_id=device_id)
            if not chip_info:
                msg = f"{target} eval failed"
                logger.error(msg)
                model_infos["msg"] = msg
        except Exception as e:
            msg = f"{target} eval failed:\nException: {e}"
            logger.error(msg)
            model_infos["msg"] = msg
            chip_info = dict()

    model_infos["dataset"] = onnx_info.get("dataset", "N/A")
    model_infos["dataset_num"] = onnx_info.get("num", "N/A")
    # Calculate relative error
    if onnx_info and ("top1_acc" in onnx_info or "acc" in onnx_info):
        key = "acc" if "acc" in onnx_info else "top1_acc"
        acc_onnx = float(onnx_info[key])
        acc_chip = float(chip_info[key])
        acc_err = acc_chip / acc_onnx - 1
        model_infos["acc_onnx"] = f"{acc_onnx*100:.2f}"
        model_infos["acc_chip"] = f"{acc_chip*100:.2f}"
        model_infos["acc_err"] = f"{acc_err*100:.2f}%"
    elif onnx_info and "map50" in onnx_info:
        map50_onnx = float(onnx_info["map50"])
        map50_chip = float(chip_info["map50"])
        map50_err = map50_chip / map50_onnx - 1
        map50_95_onnx = float(onnx_info["map50_95"])
        map50_95_chip = float(chip_info["map50_95"])
        map50_95_err = map50_95_chip / map50_95_onnx - 1
        model_infos["acc_onnx"] = f"{map50_95_onnx*100:.2f}/{map50_onnx*100:.2f}"
        model_infos["acc_chip"] = f"{map50_95_chip*100:.2f}/{map50_chip*100:.2f}"
        model_infos["acc_err"] = f"{map50_95_err*100:.2f}%/{map50_err*100:.2f}%"
    elif onnx_info.get("dataset", "N/A") == "widerface":
        onnx_ap_easy = float(onnx_info.get("ap_easy", 0))
        chip_ap_easy = float(chip_info.get("ap_easy", 0))
        onnx_ap_medium = float(onnx_info.get("ap_medium", 0))
        chip_ap_medium = float(chip_info.get("ap_medium", 0))
        onnx_ap_hard = float(onnx_info.get("ap_hard", 0))
        chip_ap_hard = float(chip_info.get("ap_hard", 0))
        ap_err = (
            chip_ap_easy / onnx_ap_easy - 1,
            chip_ap_medium / onnx_ap_medium - 1,
            chip_ap_hard / onnx_ap_hard - 1,
        )
        model_infos["acc_onnx"] = (
            f"{onnx_ap_easy:.2f}/{onnx_ap_medium:.2f}/{onnx_ap_hard:.2f}"
        )
        model_infos["acc_chip"] = (
            f"{chip_ap_easy:.2f}/{chip_ap_medium:.2f}/{chip_ap_hard:.2f}"
        )
        model_infos["acc_err"] = (
            f"{ap_err[0]*100:.2f}%/{ap_err[1]*100:.2f}%/{ap_err[2]*100:.2f}%"
        )

    os.chdir(root)
    queue.put(model_infos)


def run_benchmark(
    config_path: str,
    target: str,
    device_id: int = 0,
    enable_cuda: bool = False,
):
    """
    Run benchmark tests for multiple models with specified configurations.

    Args:
        config_path (str): Path to the configuration file containing model information
        target (str): Target platform (xh1 or xh2)
        device_id (int): ID of the device to use for evaluation
        enable_cuda (bool): Whether to enable CUDA for quantization
    """
    # Get quantization tool version
    hmquant_version = "N/A"
    if target == "xh1":
        hmquant_version = get_hmquant_xh1_version()
    elif target == "xh2":
        hmquant_version = get_hmquant_xh2_version()
    # Get compiler version
    hmcc_version = get_package_version(f"houmo-tcim-{target}")
    runtime_version = get_package_version(f"houmo_tcim_runtime_{target}")
    if runtime_version == "N/A":
        logger.error(f"Not found houmo_tcim_runtime_{target}")
        exit(-1)
    houmo_version = get_houmo_version()
    models = read_yaml_to_dict(config_path)
    t = time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime())
    os.makedirs("reports", exist_ok=True)
    report_file = os.path.abspath(
        os.path.join(
            "reports",
            f"benchmark_{target}_{houmo_version}_{platform.machine().lower()}_{t}.xlsx",
        )
    )

    report_file_pass = os.path.abspath(
        os.path.join(
            "reports",
            f"benchmark_{target}_{houmo_version}_{platform.machine().lower()}_{t}_pass.xlsx",
        )
    )

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
        f"Accuracy[{target}]",
        "AccRelError",
        "End2End_Cost[ms]",
        "Throughput",
        "Infer_Avg[ms]",
        "Infer_Max[ms]",
        "Input_AvgH2D[ms]",
        "Input_MaxH2D[ms]",
        "Output_AvgD2H[ms]",
        "Output_MaxD2H[ms]",
    ]

    ctx = get_context("spawn")
    q = ctx.Queue()

    table = PrettyTable(headers)
    table.title = f"HouMo Model Benchmark Report"
    # Iterate through each model
    models = models["models"]
    for model_name in models:
        model_cfg = models[model_name]
        location = model_cfg["location"]
        cfg_path = model_cfg.get("config", "config.yml")
        exec_cfgs = model_cfg.get("exec_cfgs", list())

        def run_all(enable_static):
            _all_model_infos = list()
            if not enable_static and target == "xh2":
                return _all_model_infos
            for exec_cfg in exec_cfgs:
                batch_num = exec_cfg.get("batch_num", 1)
                core_num = exec_cfg.get("core_num", 1)
                thread_num = exec_cfg.get("thread_num", 1)
                enable_upload = exec_cfg.get("enable_upload", False)
                enable_delete = exec_cfg.get("enable_delete", False)
                enable_eval = (
                    exec_cfg.get("enable_eval", False)
                    if platform.machine().lower() == "x86_64"
                    else False  # TODO need to modify after running eval in non-x86 environment
                )
                p = ctx.Process(
                    target=run_model,
                    args=(location, cfg_path, target, q),
                    kwargs=dict(
                        batch_num=batch_num,
                        core_num=core_num,
                        thread_num=thread_num,
                        enable_eval=enable_eval,
                        enbale_quantize=True,
                        enable_build=True,
                        enable_static=enable_static,
                        device_id=device_id,
                        enable_cuda=enable_cuda,
                        enable_upload=enable_upload and enable_static,
                        enable_delete=enable_delete and enable_static,
                    ),
                )
                p.start()
                p.join(timeout=7200)
                if p.exitcode is None:
                    logger.warning(f"{model_name} run timeout, will kill it")
                    p.kill()
                    p.join()
                if p.exitcode is not None and p.exitcode != 0:
                    logger.error(
                        f"{model_name} run failed, and exitcode is {p.exitcode}"
                    )
                if q.empty():
                    model_infos = dict(
                        input_size="N/A",
                        dataset="N/A",
                        dataset_num="N/A",
                        GOPs=0,
                        target=target,
                        batch_num=batch_num,
                        core_num=core_num,
                        thread_num=thread_num,
                        resizer="N/A",
                        acc_onnx="N/A",
                        acc_chip="N/A",
                        acc_err="N/A",
                        enable_static=False,
                        msg="unknown error",
                    )
                else:
                    model_infos = q.get()
                _all_model_infos.append(model_infos)
            return _all_model_infos

        all_model_infos = run_all(enable_static=False)
        all_model_infos_static = run_all(enable_static=True)

        def add_rows(_all_model_infos, static_mode=False):
            for model_infos in _all_model_infos:
                if static_mode and not model_infos["enable_static"]:
                    continue
                table.add_row(
                    [
                        model_name,
                        model_infos["input_size"],
                        model_infos["dataset"],
                        model_infos["dataset_num"],
                        f"{model_infos['GOPs']:.2f}",
                        platform.machine().lower(),
                        model_infos["core_num"],
                        model_infos["batch_num"],
                        model_infos["thread_num"],
                        model_infos["resizer"],
                        hmquant_version,
                        hmcc_version,
                        runtime_version,
                        model_infos["acc_onnx"],
                        model_infos["acc_chip"],
                        model_infos["acc_err"],
                        model_infos.get("e2e_avg_cost", "N/A"),
                        model_infos.get("throughput", "N/A"),
                        model_infos.get("infer_avg_latency", "N/A"),
                        model_infos.get("infer_max_latency", "N/A"),
                        model_infos.get("input_avg_H2D_latency", "N/A"),
                        model_infos.get("input_max_H2D_latency", "N/A"),
                        model_infos.get("output_avg_D2H_latency", "N/A"),
                        model_infos.get("output_max_D2H_latency", "N/A"),
                    ]
                )

        add_rows(all_model_infos)
        add_rows(all_model_infos_static, static_mode=True)
    logger.info(f"\n{table}")

    def save_to_excel(rows, field_names, report_file):
        df = pd.DataFrame(rows, columns=field_names)
        sheet_name = "Sheet1"
        with pd.ExcelWriter(report_file, engine="xlsxwriter", mode="w") as writer:
            df.to_excel(writer, index=False, sheet_name=sheet_name)
            workbook = writer.book
            worksheet = writer.sheets[sheet_name]
            text_fmt = workbook.add_format({"num_format": "@"})
            number_fmt = workbook.add_format({"num_format": "0.00"})

            def set_column_text_format(column_name):
                col_idx = df.columns.get_loc(column_name)
                excel_idx = chr(ord("A") + col_idx)
                worksheet.set_column(f"{excel_idx}:{excel_idx}", None, text_fmt)

            def set_column_number_format(column_name):
                col_idx = df.columns.get_loc(column_name)
                excel_idx = chr(ord("A") + col_idx)
                worksheet.set_column(f"{excel_idx}:{excel_idx}", None, number_fmt)

            set_column_number_format("GOPs")
            set_column_text_format("Accuracy[onnx]")
            set_column_text_format(f"Accuracy[{target}]")
            set_column_text_format("AccRelError")

    save_to_excel(table.rows, table.field_names, report_file)
    pass_rows = list()
    for row in table.rows:
        if row[16] == "N/A" or row[17] == "N/A":
            continue
        pass_rows.append(row)
    save_to_excel(pass_rows, table.field_names, report_file_pass)
    logger.info("Benchmark done.")


def main():
    """
    Main entry point for the HMATC tool. Parses command-line arguments and executes the appropriate subcommand.
    """
    # Parent parser
    target = os.environ.get("HOUMO_TARGET", "unknown")
    parent_config = argparse.ArgumentParser(add_help=False)
    parent_target = argparse.ArgumentParser(add_help=False)
    parent_onnx = argparse.ArgumentParser(add_help=False)
    parent_hmonnx = argparse.ArgumentParser(add_help=False)
    parent_result_path = argparse.ArgumentParser(add_help=False)
    parent_device_id = argparse.ArgumentParser(add_help=False)
    parent_cuda = argparse.ArgumentParser(add_help=False)
    parent_upload = argparse.ArgumentParser(add_help=False)
    parent_config.add_argument(
        "--config", "-c", type=str, required=True, help="config file path"
    )
    parent_target.add_argument(
        "--target",
        "-t",
        type=str,
        required=target not in ["xh1", "xh2"],
        choices=("xh1", "xh2"),
        default=target,
        help="Specify a chip target",
    )
    parent_onnx.add_argument(
        "--onnx", action="store_true", help="Specify onnx model as the backend"
    )
    parent_hmonnx.add_argument("--hmonnx", action="store_true", help=argparse.SUPPRESS)
    parent_result_path.add_argument(
        "--result_path",
        type=str,
        required=False,
        default="result.yml",
        help="Specify a result path",
    )
    parent_device_id.add_argument(
        "--device_id",
        type=int,
        required=False,
        default=0,
        help="Specify a device, running inference on chip",
    )
    parent_cuda.add_argument(
        "--cuda",
        action="store_true",
        help="Enable cuda quantization",
    )
    parent_upload.add_argument(
        "--upload",
        action="store_true",
        help=argparse.SUPPRESS,
    )
    # Main parser
    parser = argparse.ArgumentParser(description="HouMo Model Assist Tool")
    subparsers = parser.add_subparsers(
        dest="command", required=True, help="qunat build compare perf demo eval"
    )
    # quant
    quant_parser = subparsers.add_parser(
        "quant",
        parents=[parent_config, parent_target, parent_result_path, parent_cuda],
        help="Quantize a model",
    )
    # model config
    model_cfg_parent = argparse.ArgumentParser(add_help=False)
    model_cfg_parent.add_argument(
        "--batch", type=int, required=False, default=1, help="Specify a build batch"
    )
    model_cfg_parent.add_argument(
        "--ncore", type=int, required=False, choices=(1, 2, 4), help="Specify a ncore"
    )
    model_cfg_parent.add_argument(
        "--opt_level",
        type=int,
        required=False,
        choices=(0, 1, 2),
        help="Specify a opt_level",
    )
    model_cfg_parent.add_argument(
        "--roi_num",
        type=int,
        required=False,
        default=1,
        help="Specify a roi_num, only for xh1 yet",
    )
    # build
    build_parser = subparsers.add_parser(
        "build",
        parents=[
            parent_config,
            parent_target,
            parent_result_path,
            model_cfg_parent,
            parent_device_id,
            parent_upload,
        ],
        help="Build a model",
    )
    build_parser.add_argument(
        "--profile",
        action="store_true",
        required=False,
        help="Enable profile",
    )
    # compare
    compare_parser = subparsers.add_parser(
        "compare",
        parents=[
            parent_config,
            parent_target,
            parent_result_path,
            model_cfg_parent,
            parent_device_id,
        ],
        help="Compare onnx/hmquant/chip",
    )
    compare_parser.add_argument(
        "--data_path",
        "-d",
        type=str,
        required=True,
        help="Specify a data path, image or npz",
    )
    # perf
    perf_parser = subparsers.add_parser(
        "perf",
        parents=[parent_target, parent_result_path, model_cfg_parent, parent_device_id],
        help="Test model performance",
    )
    exclusive_group = perf_parser.add_mutually_exclusive_group(required=True)
    exclusive_group.add_argument(
        "--config", "-c", type=str, help="Specify config file path"
    )
    exclusive_group.add_argument("--model", "-m", type=str, help="Specify model path")
    perf_parser.add_argument(
        "--warmup",
        "-wn",
        type=int,
        default=1,
        required=False,
        help="Specify warmup num",
    )
    perf_parser.add_argument(
        "--sample",
        "-sn",
        type=int,
        required=False,
        default=1,
        help="Specify sample num",
    )
    perf_parser.add_argument(
        "--loop_num",
        "-ln",
        type=int,
        required=False,
        default=1,
        help="Specify loop num",
    )
    perf_parser.add_argument(
        "--thread",
        "-tn",
        type=int,
        required=False,
        default=1,
        help="Specify thread num",
    )
    perf_parser.add_argument(
        "--stream",
        type=int,
        required=False,
        default=0,
        help="Specify stream num",
    )
    # demo
    demo_parser = subparsers.add_parser(
        "demo",
        parents=[
            parent_config,
            parent_target,
            parent_onnx,
            parent_hmonnx,
            parent_result_path,
            model_cfg_parent,
            parent_device_id,
        ],
        help="Run model demo",
    )
    # evaluate
    evaluate_parser = subparsers.add_parser(
        "eval",
        parents=[
            parent_config,
            parent_target,
            parent_onnx,
            parent_hmonnx,
            parent_result_path,
            model_cfg_parent,
            parent_device_id,
        ],
        help="Run model evaluate",
    )
    # benchmark
    benchmark_parser = subparsers.add_parser(
        "benchmark",
        parents=[
            parent_config,
            parent_target,
            parent_result_path,
            parent_device_id,
            parent_cuda,
        ],
        help="Run model benchmark",
    )
    # check golden
    check_parser = subparsers.add_parser(
        "check",
        parents=[
            parent_config,
            parent_target,
            parent_device_id,
        ],
        help="Check model golden",
    )
    check_parser.add_argument(
        "--layers",
        action="store_true",
        help="Check model layers output",
    )

    args = parser.parse_args()

    logger.info(
        f"Hmatc version: {__version__}, commit: {__commit__}, build time: {__build_time__}"
    )
    # Set random seed
    set_random_seed(1234)
    # Process batch model benchmark
    current_command = args.command
    if current_command == "benchmark":
        run_benchmark(args.config, args.target, args.device_id, args.cuda)
        exit(0)

    # If result info exists, read back and update before saving
    res_info = dict()
    if (
        hasattr(args, "result_path")
        and args.result_path is not None
        and os.path.exists(args.result_path)
    ):
        res_info = read_yaml_to_dict(args.result_path)

    # Directly specify model for perf can skip config file
    if current_command == "perf" and args.model is not None:
        new_res_info = BaseExec.model_perf(
            args.model,
            args.warmup,
            args.sample,
            args.loop_num,
            args.thread,
            args.stream,
            devices=[args.device_id],
        )
        if "perf" in res_info:
            res_info["perf"].update(new_res_info["perf"])
        else:
            res_info.update(new_res_info)
        exit(0)

    target = args.target
    # set_logger(current_command, "log", "config")
    cfg_path = args.config
    if not os.path.exists(cfg_path):
        logger.error("Config file not found")
        exit(1)
    cfg = read_yaml_to_dict(cfg_path)
    if not check_cfg(cfg):
        logger.error("Config file error")
        exit(-1)

    # Update command line arguments to config file
    cfg["target"] = target
    if current_command in ["build", "perf", "compare", "demo", "eval"]:
        batch = args.batch
        ncore = args.ncore
        opt_level = args.opt_level
        roi_num = args.roi_num
        if batch < 1:
            logger.error("Batch must be greater than 0")
            exit(-1)
        if batch > 1:
            cfg["build"]["batch"] = batch
        if batch > 1 and roi_num != 1 and target == "xh1":
            logger.error("batch > 1, roi_num must be == 1")
            exit(-1)
        if batch == 1 and roi_num < 1 and target == "xh1":
            logger.error("batch == 1, roi_num must be >= 1")
            exit(-1)
        if roi_num > 1 and target == "xh1":
            cfg["build"]["roi_num"] = roi_num
        if opt_level is not None:
            cfg["build"]["opt_level"] = opt_level
        if ncore is not None:
            if ncore == 4 and target == "xh2":
                logger.error("ncore == 4, target must be xh1")
                exit(-1)
            cfg["build"]["ncore"] = ncore
    logger.info(f"\n{json.dumps(cfg, indent=2, sort_keys=False)}")
    hm_exec = None
    if target == "xh1":
        from .exec.xh1_exec import Xh1Exec

        hm_exec = Xh1Exec(cfg)
    elif target == "xh2":
        from .exec.xh2_exec import Xh2Exec

        hm_exec = Xh2Exec(cfg)
    else:
        logger.error(f"Not support target: {target}")
        exit(-1)

    new_res_info = dict()
    backend = target
    # Update onnx backend
    if current_command in ["demo", "eval"]:
        if args.onnx:
            backend = "onnx"
        elif args.hmonnx:
            backend = "hmonnx"
    # Execute the corresponding command
    if current_command == "quant":
        if args.cuda and target == "xh1" and torch.cuda.is_available():
            hm_exec.device = "cuda"
        new_res_info = hm_exec.quantize()
    elif current_command == "build":
        hm_exec.enable_upload = args.upload
        new_res_info = hm_exec.build(enable_profile=args.profile)
        logger.info(f"Build {hm_exec.model_name} done.")
        new_res_info["build"].update(hm_exec.check_golden(args.device_id))
    elif current_command == "check":
        hm_exec.check_golden(args.device_id, args.layers)
    elif current_command == "compare":
        data_path = args.data_path
        if not os.path.exists(data_path):
            # If not exists, look in HOUMO_DATASETS_PATH environment variable
            HOUMO_DATASETS_PATH = os.environ.get(
                "HOUMO_DATASETS_PATH", "/usr/local/src/houmo-modelzoo/data/datasets"
            )
            data_path = os.path.join(HOUMO_DATASETS_PATH, data_path)
            if not os.path.exists(data_path):
                logger.error(f"{data_path} or {args.data_path} not exists.")
                exit(-1)
        new_res_info = hm_exec.compare(data_path, args.device_id)
    elif current_command == "perf":
        new_res_info = hm_exec.model_perf(
            hm_exec.hmm_path,
            args.warmup,
            args.sample,
            args.loop_num,
            args.thread,
            args.stream,
            devices=[args.device_id],
        )
    elif current_command == "demo":
        hm_exec.demo(backend=backend, device_id=args.device_id)
    elif current_command == "eval":
        hm_exec.evaluate(backend=backend, device_id=args.device_id)
    else:
        raise NotImplementedError

    if current_command == "compare" and "compare" in res_info:
        res_info["compare"].update(new_res_info["compare"])
    elif current_command == "perf" and "perf" in res_info:
        res_info["perf"].update(new_res_info["perf"])
    else:
        res_info.update(new_res_info)
    if hasattr(args, "result_path") and args.result_path is not None:
        save_dict_to_yaml(res_info, args.result_path)


if __name__ == "__main__":
    main()
