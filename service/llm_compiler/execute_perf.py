# Copyright 2025 HOUMO AI
#
# File: execute_perf.py
# Description:
#   Execute LLM model performance testing and analysis.
#
#   This script processes performance test results from LLM models and generates Excel reports
#   with detailed performance metrics.
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
import shutil
import glob
import time
import json
import argparse
from datetime import datetime
import logging
from compiler_utils import setup_logging, execute_cmd
import pandas as pd
from typing import Optional, Iterable, Dict, List, Callable

script_dir = os.path.dirname(os.path.abspath(__file__))

JSON_SUFFIX = ".json"

MODELS_FILE_SIZE = "models_file_size"

START_TASK_STR = "Start of Task"
MODEL_NAME_STR = "ModelName:"
END_TASK_STR = "End of Task"

MEM_START_STR = "HM Device Memory Usage"
MEM_USED_STR = "memory used:"
MEM_END_LINE_STR = "************************************"

MEM_USED_COLS = "device_mem_used(MB)"


def parse_args() -> argparse.Namespace:
    """Parse command line arguments for performance testing."""
    parser = argparse.ArgumentParser(description="Perf LLMs")
    parser.add_argument(
        "--perf_cfg",
        type=str,
        help="the path of perf_file.",
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


def _find_first_matched_keyword(line: str, keywords: Iterable) -> Optional[str]:
    """
    Find the first keyword that matches in the given line.

    Args:
        line (str): The input line to search for keywords
        keywords (Iterable): Collection of keywords to search for

    Returns:
        Optional[str]: The first matched keyword, or None if no match is found
    """
    if not keywords or not line:
        return None
    for keyword in keywords:
        if keyword in line:
            return keyword  # Return immediately when first match is found
    return None


def _prepare_test_folder(model_dir: str) -> str:
    """
    Prepare a test folder with a timestamp for performance testing.

    Args:
        model_dir (str): Source directory containing the model

    Returns:
        str: Path to the prepared test folder
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    test_folder = f"{model_dir}_compiler_{timestamp}"
    if os.path.exists(test_folder):
        print(f"remove folder: {test_folder}.")
        shutil.rmtree(test_folder, ignore_errors=True)
    shutil.copytree(model_dir, test_folder)
    os.chdir(test_folder)

    return test_folder


def _write_to_xlsx(
    perf_metric, cfg_path: str, sheet_name: str, new_cols: list = []
) -> None:
    """
    Write performance metrics to an Excel file.
    Creates or updates an Excel file with performance data organized in a specified sheet.

    Args:
        perf_metric: Performance metrics data to write
        cfg_path (str): Configuration file path (used to derive Excel file name)
        sheet_name (str): Name of the sheet in the Excel file
        new_cols (list): Column names for the data
    """
    perf_df = pd.DataFrame(perf_metric)
    perf_df.columns = new_cols
    perf_xlsx_path = cfg_path.replace(JSON_SUFFIX, ".xlsx")
    logger.info(f"perf_xlsx_path: {perf_xlsx_path}")
    excel_mode = "a"
    if not os.path.exists(perf_xlsx_path):
        excel_mode = "w"
    with pd.ExcelWriter(perf_xlsx_path, engine="openpyxl", mode=excel_mode) as writer:
        perf_df.to_excel(writer, sheet_name=sheet_name, index=False)


def _get_value_after_colon(line: str, split_space: bool = False) -> str:
    """
    Extract content after the last colon in a line and strip whitespace.

    Args:
        line (str): Input line to process
        split_space (bool): Whether to split the value by space and take the first part

    Returns:
        str: Extracted value after the last colon
    """
    val = line.strip().rsplit(":", 1)[-1].strip()
    if split_space:
        val = val.split(" ", 1)[0]
    return val


#######################################################################
# -------------------- Condition Check Functions -------------------- #
#######################################################################


def _check_start_task(line: str, state: Dict) -> bool:
    """Check if the line indicates the start of a task."""
    return START_TASK_STR in line


def _check_loop(line: str, state: Dict) -> bool:
    """Check if the line contains loop count information."""
    return "loop :" in line or "  loops:" in line


def _check_end_task(line: str, state: Dict) -> bool:
    """Check if the line indicates the end of a task."""
    return END_TASK_STR in line


def _check_perf_flag(line: str, state: Dict) -> bool:
    """Check if the line contains performance metrics (when perf_flag is True)."""
    return state["perf_flag"] is True


def _check_mem_start(line: str, state: Dict) -> bool:
    """Check if the line indicates the start of memory monitoring."""
    return MEM_START_STR in line


def _check_mem_used(line: str, state: Dict) -> bool:
    """Check if the line contains memory usage information (when mem_flag is True)."""
    return state["mem_flag"] is True and MEM_USED_STR in line


def _check_mem_end(line: str, state: Dict) -> bool:
    """Check if the line indicates the end of memory monitoring (when mem_flag is True)."""
    return MEM_END_LINE_STR in line and state["mem_flag"] is True


def _check_samples(line: str, state: Dict) -> bool:
    """Check if the line contains sample count information."""
    return "  samples:" in line


def _check_warmup(line: str, state: Dict) -> bool:
    """Check if the line contains warmup information."""
    return "  warmup:" in line


def _check_device_num(line: str, state: Dict) -> bool:
    """Check if the line contains device number information."""
    return "  device_num:" in line


def _check_latency(line: str, state: Dict) -> bool:
    """Check if the line contains latency information."""
    return "[latency] " in line


def _check_throughput_qps(line: str, state: Dict) -> bool:
    """Check if the line contains throughput QPS information."""
    return "[Throughput] qps" in line


def _check_total_cost(line: str, state: Dict) -> bool:
    """Check if the line contains total cost information (excluding TTS)."""
    return "Total Cost " in line and "TTS" not in line


#######################################################################
# ------------------ Business Processing Functions ------------------ #
#######################################################################


def _handle_mem_start(
    line: str,
    perf_metric: Dict,
    state: Dict,
    keywords_dict: Dict = {},
    perf_key: str = "mem_flag",
) -> None:
    """(common) Handle memory monitoring start line: set mem_flag."""
    state["mem_flag"] = True


def _handle_mem_end(
    line: str,
    perf_metric: Dict,
    state: Dict,
    keywords_dict: Dict = {},
    perf_key: str = "mem_flag",
) -> None:
    """(common) Handle memory monitoring end line: reset mem_flag."""
    state["mem_flag"] = False


def _handle_mem_used(
    line: str,
    perf_metric: Dict,
    state: Dict = {},
    keywords_dict: Dict = {},
    perf_key: str = "device_mem_used",
) -> None:
    """(common) Handle memory usage line: update device memory usage."""
    mem_used_str = _get_value_after_colon(line)
    if perf_metric[perf_key][-1] == "NA":
        perf_metric[perf_key][-1] = mem_used_str
        return
    perf_metric[perf_key][-1] += f"/{mem_used_str}"


def _handle_start_task(
    line: str, perf_metric: Dict, state: Dict, keywords_dict: Dict, perf_key: str
) -> None:
    """(common) Handle task start line: initialize metrics, extract model name."""
    for key in state:
        state[key] = False
    for key in perf_metric.keys():
        perf_metric[key].append("NA")
    model_name = line.strip().split(MODEL_NAME_STR, 1)[-1].strip()
    perf_metric[perf_key][-1] = model_name


def _handle_end_task(
    line: str,
    perf_metric: Dict,
    state: Dict,
    keywords_dict: Dict = {},
    perf_key: str = "",
) -> None:
    """(common) Handle task end line: reset flags."""
    state["perf_flag"] = False
    state["mem_flag"] = False


def _handle_perf_start_line(
    line: str,
    perf_metric: Dict,
    state: Dict,
    keywords_dict: Dict = {},
    perf_key: str = "",
) -> None:
    """(common) Handle performance start line: set perf_flag."""
    state["perf_flag"] = True


def _parse_latency_line(line: str, perf_metric: dict) -> None:
    """(tcim_perf) Parse latency line, update perf_metric."""

    latency_type_map = {
        " Inference": "inference",
        " Input": "input",
        " Output": "output",
        " End2End": "e2e",
    }

    # Match latency type
    key_str = ""
    for pattern, key in latency_type_map.items():
        if pattern in line:
            key_str = key
            break
    if not key_str:
        return

    # Parse latency values (avg/max/min)
    perf_vals = line.strip().split(",")
    avg_val = perf_vals[0].rsplit(":", 1)[-1].strip()[:-3]
    max_val = perf_vals[1].rsplit(":", 1)[-1].strip()[:-3]
    min_val = perf_vals[2].rsplit(":", 1)[-1].strip()[:-3]

    # Update metrics
    perf_metric[f"{key_str}_avg"][-1] = avg_val
    perf_metric[f"{key_str}_max"][-1] = max_val
    perf_metric[f"{key_str}_min"][-1] = min_val


def _handle_simple_metric(line: str, perf_metric: dict, metric_key: str) -> None:
    """(tcim_perf) Handle simple metrics like samples/loops/warmup/device_num."""
    perf_metric[metric_key][-1] = _get_value_after_colon(line)


def _handle_perf_flag_logic(
    line: str, perf_metric: dict, keywords_dict: dict, keywords_dict_2: dict
) -> None:
    """(minicpmo_perf) Handle all logic when perf_flag=True."""
    if "Input Tokens:" in line:
        val = line.strip().rsplit(",", 1)[0].strip().rsplit(":", 1)[-1].strip()
        perf_metric["input_tokens"][-1] = val

    keyword = _find_first_matched_keyword(line, keywords_dict.keys())
    if keyword is not None:
        perf_metric[keywords_dict[keyword]][-1] = _get_value_after_colon(
            line, split_space=True
        )
        return

    keyword_2 = _find_first_matched_keyword(line, keywords_dict_2.keys())
    if keyword_2 is not None:
        perf_metric[keywords_dict_2[keyword_2]][-1] = _get_value_after_colon(line)


def _add_file_size_to_perf(perf_dict, size_dict):
    """
    Add file_size key-value pairs to perf_dict, matching values from size_dict in order of model_name.
    :param perf_dict: Original performance metrics dictionary
    :param size_dict: File size dictionary, key is model name, value is file size string
    :return: Updated perf_dict with added file_size
    """
    model_names = perf_dict.get("model_name", [])
    if not model_names:
        return perf_dict

    file_size_list = []
    for model in model_names:
        model_size = size_dict.get(model, 0)
        file_size_list.append(model_size)

    perf_dict["file_size"] = file_size_list
    return perf_dict


def _extract_device_mem_from_log(log_lines: list, model_name: str) -> str:
    """
    Extract device memory usage from log lines for a specific model.

    Args:
        log_lines (list): Lines from the log file
        model_name (str): Name of the model to search for

    Returns:
        str: Device memory usage value in format "dev0_mem MB/dev1_mem MB" or "NA" if not found
    """
    device_mems = []
    model_found = False
    mem_usage_section = False

    for line in log_lines:
        if START_TASK_STR in line and model_name in line:
            model_found = True
            continue

        if model_found:
            # Check for HM Device Memory Usage section start
            if MEM_START_STR in line:
                mem_usage_section = True
                continue

            if END_TASK_STR in line or (
                MODEL_NAME_STR in line and model_name not in line
            ):
                break

            # If in HM Device Memory Usage section, look for device memory lines
            if mem_usage_section and MEM_USED_STR in line:
                mem_used_str = _get_value_after_colon(line)
                device_mems.append(mem_used_str)

    # Format the result
    if device_mems:
        return "/".join(device_mems)
    else:
        return "NA"


def _generate_llm_perf_table(cfg_path, outputs, file_size_dict):
    """
    Generate LLM performance table from dump files and configuration.

    Reads performance data from dump_file paths specified in the configuration,
    combines it with configuration parameters and log information to create
    a comprehensive performance table.

    Args:
        cfg_path: Path to the configuration file
        outputs: List of output lines from the log file (for memory extraction)
        file_size_dict: Dictionary containing file sizes
    """
    # Load the configuration file
    try:
        with open(cfg_path, "r", encoding="utf-8") as f:
            cfg_data = json.load(f)
    except Exception as e:
        logger.error(f"Failed to load config file {cfg_path}: {e}")
        return

    # Initialize performance metrics dictionary with required columns
    perf_metric = {
        "file_name": [],  # ModelName from config
        "model": [],  # model_name from config
        "size": [],  # model_size from config
        "p_len(k)": [],  # prefill_length from config
        "ctx_len(k)": [],  # context_length from config
        "batch": [],  # batch from config
        "nchip": [],  # ndevices from config
        "ncore": [],  # ncore from config
        "flash_attention": [],  # Assuming default 0 if not specified
        "input(k)": [],  # input_token from dump file (converted to k)
        "output(k)": [],  # output_token from dump file (converted to k)
        "device_mem(MB)": [],  # Extracted from logs
        "prefill_load_time(ms)": [],  # prefill_load_time from dump file
        "decode_load_time(ms)": [],  # decode_load_time from dump file
        "visual_load_time(ms)": [],  # visual_load_time from dump file
        "prefill_time(ms)": [],  # prefill_time from dump file
        "decode_time(ms)": [],  # decode_time from dump file
        "vision_time(ms)": [],  # vision_time from dump file
        "prefill_speed(tps)": [],  # prefill_speed from dump file
        "decode_speed(tps)": [],  # decode_speed from dump file
        "vision_speed(fps)": [],  # vision_speed from dump file
        "TTFT(ms)": [],  # TTFT from dump file
        "TPOT(ms/token)": [],  # TPOT from dump file
        "e2e_latency(s)": [],  # e2e_latency from dump file
        "e2e_tps(tps)": [],  # e2e_tps from dump file
        "prefill_embedding_time(ms)": [],  # prefill_embedding_time from dump file
        "decode_embedding_time(ms)": [],  # decode_embedding_time from dump file
        "hmm_prefill_size(MB)": [],  # From file_size_dict
        "hmm_decode_size(MB)": [],  # From file_size_dict
        "hmm_visual_size(MB)": [],  # From file_size_dict
    }

    # Process each model in the configuration
    for model_config in cfg_data.get("Streams", []):
        model_name = model_config.get("ModelName", "")
        dump_file_path = model_config.get("dump_file", "")

        # Skip if no dump file specified
        if not dump_file_path:
            logger.warning(f"No dump_file specified for model {model_name}")
            continue

        # Initialize row with NA values
        for key in perf_metric:
            perf_metric[key].append("NA")

        # Fill configuration-based values
        current_index = len(perf_metric["file_name"]) - 1
        perf_metric["file_name"][current_index] = model_name
        perf_metric["model"][current_index] = model_config.get("model_name", "NA")
        perf_metric["size"][current_index] = model_config.get("model_size", "NA")
        prefill_length = model_config.get("prefill_length", "NA")
        perf_metric["p_len(k)"][current_index] = (
            round(prefill_length / 1024.0, 3) if prefill_length != "NA" else "NA"
        )
        perf_metric["ctx_len(k)"][current_index] = model_config.get(
            "context_length", "NA"
        )
        perf_metric["batch"][current_index] = model_config.get("batch", "NA")
        perf_metric["nchip"][current_index] = model_config.get("ndevices", "NA")
        perf_metric["ncore"][current_index] = model_config.get("ncore", "NA")
        perf_metric["flash_attention"][current_index] = model_config.get(
            "flash_attention", "2"
        )

        # Try to read performance data from dump file
        perf_data = {}
        if os.path.exists(dump_file_path):
            try:
                with open(dump_file_path, "r", encoding="utf-8") as f:
                    dump_data = json.load(f)

                # Extract performance metrics from Perf_Metrics array
                perf_metrics_list = dump_data.get("Perf_Metrics", [])
                if perf_metrics_list:
                    perf_data = perf_metrics_list[0]  # Take first entry
            except Exception as e:
                logger.error(f"Failed to read dump file {dump_file_path}: {e}")
        else:
            logger.warning(f"Dump file not found: {dump_file_path}")

        # Fill performance metrics from dump file (convert to appropriate units)
        if perf_data:
            # Convert token counts to k units
            input_tokens = perf_data.get("input_token", 0)
            output_tokens = perf_data.get("output_token", 0)
            perf_metric["input(k)"][current_index] = (
                round(input_tokens / 1024.0, 3) if input_tokens != "NA" else "NA"
            )
            perf_metric["output(k)"][current_index] = (
                round(output_tokens / 1024.0, 3) if output_tokens != "NA" else "NA"
            )

            # Direct mappings from dump file
            perf_metric["prefill_time(ms)"][current_index] = perf_data.get(
                "prefill_time", "NA"
            )
            perf_metric["decode_time(ms)"][current_index] = perf_data.get(
                "decode_time", "NA"
            )
            perf_metric["vision_time(ms)"][current_index] = perf_data.get(
                "vision_time", "NA"
            )
            perf_metric["prefill_speed(tps)"][current_index] = perf_data.get(
                "prefill_speed", "NA"
            )
            perf_metric["decode_speed(tps)"][current_index] = perf_data.get(
                "decode_speed", "NA"
            )
            perf_metric["vision_speed(fps)"][current_index] = perf_data.get(
                "vision_speed", "NA"
            )
            perf_metric["TTFT(ms)"][current_index] = perf_data.get("TTFT", "NA")
            perf_metric["TPOT(ms/token)"][current_index] = perf_data.get("TPOT", "NA")
            perf_metric["e2e_latency(s)"][current_index] = perf_data.get(
                "e2e_latency", "NA"
            )
            perf_metric["e2e_tps(tps)"][current_index] = perf_data.get("e2e_tps", "NA")
            perf_metric["prefill_embedding_time(ms)"][current_index] = perf_data.get(
                "prefill_embedding_time", "NA"
            )
            perf_metric["decode_embedding_time(ms)"][current_index] = perf_data.get(
                "decode_embedding_time", "NA"
            )

            # New load time metrics
            perf_metric["prefill_load_time(ms)"][current_index] = perf_data.get(
                "prefill_load_time", "NA"
            )
            perf_metric["decode_load_time(ms)"][current_index] = perf_data.get(
                "decode_load_time", "NA"
            )
            perf_metric["visual_load_time(ms)"][current_index] = perf_data.get(
                "vision_load_time", "NA"
            )
        # Extract device memory usage from logs
        device_mem = _extract_device_mem_from_log(outputs, model_name)
        perf_metric["device_mem(MB)"][current_index] = device_mem

        # Fill file sizes from file_size_dict
        file_sizes = file_size_dict.get(model_name, "")
        prefill_size = "0"
        decode_size = "0"
        visual_size = "0"

        if file_sizes:
            parts = file_sizes.split(";")
            for part in parts:
                if ":" in part:
                    key, value = part.split(":", 1)
                    size_mb = value.replace(" MB", "").strip()
                    if "prefill" in key:
                        prefill_size = size_mb
                    elif "decode" in key:
                        decode_size = size_mb
                    elif "visual" in key:
                        visual_size = size_mb

        perf_metric["hmm_prefill_size(MB)"][current_index] = prefill_size
        perf_metric["hmm_decode_size(MB)"][current_index] = decode_size
        perf_metric["hmm_visual_size(MB)"][current_index] = visual_size

    # Log the final metrics structure
    for key, value in perf_metric.items():
        logger.info(f"{key}, length: {len(value)}")

    # Define column names for Excel output
    columns = [
        "file_name",
        "model",
        "size",
        "p_len(k)",
        "ctx_len(k)",
        "batch",
        "nchip",
        "ncore",
        "flash_attention",
        "input(k)",
        "output(k)",
        "device_mem(MB)",
        "prefill_load_time(ms)",
        "decode_load_time(ms)",
        "visual_load_time(ms)",
        "prefill_time(ms)",
        "decode_time(ms)",
        "vision_time(ms)",
        "prefill_speed(tps)",
        "decode_speed(tps)",
        "vision_speed(fps)",
        "TTFT(ms)",
        "TPOT(ms/token)",
        "e2e_latency(s)",
        "e2e_tps(tps)",
        "prefill_embedding_time(ms)",
        "decode_embedding_time(ms)",
        "hmm_prefill_size(MB)",
        "hmm_decode_size(MB)",
        "hmm_visual_size(MB)",
    ]

    # Write to Excel file
    _write_to_xlsx(perf_metric, cfg_path, "llm_perf", columns)


def _generate_tcim_perf_table(cfg_path, outputs, file_size_dict):
    """
    Generate TCIM performance table from outputs.

    Parses TCIM performance test outputs and creates an Excel table with metrics
    including inference latency, input/output latency, end-to-end latency, and QPS.

    Args:
        cfg_path: Path to the configuration file
        outputs: List of output lines from the performance test
    """
    perf_metric = {
        "model_name": [],
        "samples": [],
        "loops": [],
        "warmup": [],
        "device_num": [],
        "device_mem_used": [],
        "inference_avg": [],
        "inference_max": [],
        "inference_min": [],
        "input_avg": [],
        "input_max": [],
        "input_min": [],
        "output_avg": [],
        "output_max": [],
        "output_min": [],
        "e2e_avg": [],
        "e2e_max": [],
        "e2e_min": [],
        "qps": [],
    }

    state = {"mem_flag": False}
    processors = [
        # (condition check function, handler function, extra args for handler)
        (_check_start_task, _handle_start_task, (perf_metric, state, {}, "model_name")),
        (_check_samples, _handle_simple_metric, (perf_metric, "samples")),
        (_check_loop, _handle_simple_metric, (perf_metric, "loops")),
        (_check_warmup, _handle_simple_metric, (perf_metric, "warmup")),
        (_check_device_num, _handle_simple_metric, (perf_metric, "device_num")),
        (
            _check_end_task,
            _handle_mem_end,
            (
                perf_metric,
                state,
            ),
        ),
        (
            _check_mem_end,
            _handle_mem_end,
            (
                perf_metric,
                state,
            ),
        ),
        (
            _check_mem_start,
            _handle_mem_start,
            (
                perf_metric,
                state,
            ),
        ),
        (_check_mem_used, _handle_mem_used, (perf_metric,)),
        (_check_latency, _parse_latency_line, (perf_metric,)),
        (_check_throughput_qps, _handle_simple_metric, (perf_metric, "qps")),
    ]
    for line in outputs:
        for check_func, handle_func, args in processors:
            if check_func(line, state):
                handle_func(line, *args)
                break

    perf_metric = _add_file_size_to_perf(perf_metric, file_size_dict)
    for key, value in perf_metric.items():
        print(f"{key}, length: {len(value)}")

    columns = [
        "model_name",
        "samples",
        "loops",
        "warmup",
        "device_num",
        MEM_USED_COLS,
        "inference_avg(ms)",
        "inference_max(ms)",
        "inference_min(ms)",
        "input_avg(ms)",
        "input_max(ms)",
        "input_min(ms)",
        "output_avg(ms)",
        "output_max(ms)",
        "output_min(ms)",
        "e2e_avg(ms)",
        "e2e_max(ms)",
        "e2e_min(ms)",
        "qps",
        "file_size(MB)",
    ]
    _write_to_xlsx(perf_metric, cfg_path, "tcim_perf", columns)


def _generate_demo_perf_table(cfg_path: str, outputs, file_size_sum: float):
    """
    Generate demo performance table from outputs.

    Parses demo performance test outputs and creates an Excel table with metrics
    for complex workflows including LLM, TTS, and multimodal components.

    Args:
        cfg_path: Path to the configuration file
        outputs: List of output lines from the performance test
    """
    perf_metric = {
        "model_name": [],
        "device_mem_used": [],
        "input_tokens": [],
        "output_tokens": [],
        "llm_prefill_speed": [],
        "TTFT": [],
        "TPOT": [],
        "TPS": [],
        "tts_prefill_mean_time": [],
        "tts_decode_mean_time": [],
        "tts_dvae_cost": [],
        "tts_vocos_cost": [],
        "tts_rtf": [],
        "tts_generate_speed": [],
        "e2e_latency": [],
    }

    keywords_dict = {
        "LLM Prefill Speed:": "llm_prefill_speed",
        "TTFT (Time to First Token)": "TTFT",
        "TPOT (Time Per Output Token)": "TPOT",
        "ViT+Whisper+LLM TPS": "TPS",
        "TTS Dvae Cost:": "tts_dvae_cost",
        "TTS Vocos Cost:": "tts_vocos_cost",
        "E2E Latency (End-to-End Latency)": "e2e_latency",
    }
    keywords_dict_2 = {
        "Output tokens:": "output_tokens",
        "TTS Real-Time Factor(RTF)": "tts_rtf",
        "TTS Prefill Mean Time": "tts_prefill_mean_time",
        "TTS Decoder Mean Time": "tts_decode_mean_time",
        "TTS Generate Speed:": "tts_generate_speed",
    }

    state = {"perf_flag": False, "mem_flag": False}
    processors = [
        # (condition check function, handler function, args for handler)
        (_check_start_task, _handle_start_task, (perf_metric, state, {}, "model_name")),
        (
            _check_total_cost,
            _handle_perf_start_line,
            (
                perf_metric,
                state,
            ),
        ),
        (
            _check_perf_flag,
            _handle_perf_flag_logic,
            (
                perf_metric,
                keywords_dict,
                keywords_dict_2,
            ),
        ),
        (
            _check_end_task,
            _handle_end_task,
            (
                perf_metric,
                state,
            ),
        ),
        (
            _check_mem_end,
            _handle_mem_end,
            (
                perf_metric,
                state,
            ),
        ),
        (
            _check_mem_start,
            _handle_mem_start,
            (
                perf_metric,
                state,
            ),
        ),
        (_check_mem_used, _handle_mem_used, (perf_metric,)),
    ]
    for line in outputs:
        for check_func, handle_func, args in processors:
            if check_func(line, state):
                handle_func(line, *args)
                break

    perf_metric["file_size_sum"] = f"{file_size_sum} MB"
    for key, value in perf_metric.items():
        print(f"{key}, length: {len(value)}")

    columns = [
        "model_name",
        MEM_USED_COLS,
        "input_tokens",
        "output_tokens",
        "llm_prefill_speed(tokens/s)",
        "TTFT(ms)",
        "TPOT(tokens/s)",
        "ViT+Whisper+LLM TPS(tokens/s)",
        "tts_prefill_mean_time(ms)",
        "tts_decode_mean_time(ms)",
        "tts_dvae_cost(ms)",
        "tts_vocos_cost(ms)",
        "tts_rtf",
        "tts_generate_speed",
        "e2e_latency(s)",
        "file_size_sum(MB)",
    ]
    _write_to_xlsx(perf_metric, cfg_path, "demo_perf", columns)


def _generate_cfg_paths(
    base_cfg_path: str, suffix_map: Dict[str, str]
) -> Dict[str, str]:
    """
    Generate configuration file paths with suffixes.

    Args:
        base_cfg_path (str): Base configuration file path
        suffix_map (Dict[str, str]): Mapping of names to suffixes

    Returns:
        Dict[str, str]: Dictionary mapping names to full configuration paths
    """
    cfg_paths = {}
    for name, suffix in suffix_map.items():
        cfg_paths[name] = base_cfg_path.replace(JSON_SUFFIX, f"{suffix}{JSON_SUFFIX}")
    return cfg_paths


def _load_config_file(cfg_path: str, default_data: Dict = None) -> Dict:
    """
    Load configuration file with fallback to default data.

    Args:
        cfg_path (str): Path to the configuration file
        default_data (Dict): Default data to return if file doesn't exist or loading fails

    Returns:
        Dict: Loaded configuration data
    """
    default_data = default_data or {"Streams": []}
    if not os.path.exists(cfg_path):
        return default_data
    try:
        with open(cfg_path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        logger.warning(f"Load config {cfg_path} failed: {e}, use default data")
        return default_data


def _copy_file_or_dir(
    src: str,
    dst: str,
    is_dir: bool = False,
    clean_dir: Optional[str] = None,
) -> bool:
    """
    Generic file/directory copy function.

    Args:
        src (str): Source path
        dst (str): Destination path
        is_dir (bool): Whether the source is a directory
        clean_dir (Optional[str]): Directory to clean up if copy fails

    Returns:
        bool: Whether the copy operation was successful
    """
    try:
        if is_dir:
            shutil.copytree(src, dst, dirs_exist_ok=True, ignore_dangling_symlinks=True)
            logger.info(f"Copy folder: {src} -> {dst}")
        else:
            shutil.copy2(src, dst)
            logger.info(f"Copy file: {src} -> {dst}")
        return True
    except Exception as e:
        logger.error(f"Failed to copy {src}: {str(e)}")
        if clean_dir and os.path.exists(clean_dir):
            shutil.rmtree(clean_dir, ignore_errors=True)
        return False


def _get_file_size_mb(file_path: str) -> float:
    """
    Get the size of the specified file in MB.
    :param file_path: str, Absolute or relative path to the file
    :return: float, File size in MB, rounded to 4 decimal places
    """
    # Check if path exists and check if path is a file
    if not os.path.exists(file_path) or not os.path.isfile(file_path):
        return 0

    # Get file size in bytes
    byte_size = os.path.getsize(file_path)
    # Convert to MB
    mb_size = byte_size / (1024 * 1024)

    return round(mb_size, 4)


def _execute_perf_task(
    perf_dir: str,
    cfg_data: Dict,
    cmd_builder: Callable[[Dict, str], Dict[str, List[str]]],
    generate_table_func: Callable[[str, List[str]], None],
    base_cfg_path: str,
    log_file: str,
    task_name: str = "Perf",
) -> None:
    """
    Execute performance task, suitable for llm_perf & tcim_perf.

    Args:
        perf_dir (str): Performance test directory (e.g., llm_perf_dir)
        cfg_data (Dict): Configuration data (e.g., llm_cfg_data)
        cmd_builder (Callable): Command building function (takes perf_md, returns cmds dict)
        generate_table_func (Callable): Function to generate tables
        base_cfg_path (str): Base configuration path (e.g., llm_cfg_path)
        log_file (str): Log file path
        task_name (str): Task name for logging
    """
    cmds = cmd_builder(cfg_data, log_file)
    if len(cmds) == 0:
        logger.info(f"[{task_name}] No commands to execute")
        return

    os.chdir(perf_dir)
    logger.info(f"Current dir: {os.getcwd()}")
    logger.info(f"[{task_name}] cmds: {cmds}")

    outputs_total = []
    for model_name, cmd in cmds.items():
        if model_name == MODELS_FILE_SIZE:
            continue
        outputs_total += [f"****** {START_TASK_STR}, {MODEL_NAME_STR} {model_name}"]
        _, outputs = execute_cmd(cmd, log_file, get_outputs=True)
        outputs_total += outputs
        outputs_total += [f"****** {END_TASK_STR} ******"]
        time.sleep(5)

    generate_table_func(base_cfg_path, outputs_total, cmds[MODELS_FILE_SIZE])


def _build_llm_cmds(cfg_data: Dict, log_file: str) -> Dict[str, List[str]]:
    """
    Build LLM performance test commands from configuration data.

    Args:
        cfg_data (Dict): Configuration data containing performance test parameters
        log_file (str): Path to the log file

    Returns:
        Dict[str, List[str]]: Dictionary mapping model names to command lists
    """
    os.chdir(f"{script_dir}/../../tools/llm_perf")

    cmds = {}
    file_size_dict = {}
    file_keys = ["prefill", "decode", "visual"]
    for perf_md in cfg_data["Streams"]:
        # , "--LazyMode"
        tmp_cmd = ["./llm_perf"]
        # Filter parameters
        for param, param_val in perf_md.items():
            if param in [
                "ModelName",
                "model_name",
                "quant_models",
                "model_size",
                "ncore",
                "prefill_length",
                "context_length",
                "flash_attention",
            ]:
                continue
            tmp_cmd += [f"--{param}", str(param_val)]
        model_name = perf_md["ModelName"]
        cmds[model_name] = tmp_cmd

        # Get file size
        file_size_str = ""
        for file_key in file_keys:
            model_path = perf_md.get(file_key, "")
            model_size = _get_file_size_mb(model_path) if model_path else 0
            file_size_str += f"{file_key}:{model_size:.3f} MB;"
        file_size_dict[model_name] = file_size_str

        # Embed file conversion logic
        embed_bin = perf_md["embedding"]
        embed_pt = embed_bin.replace(".bin", ".pt")
        if os.path.exists(embed_bin):
            continue
        model_type = "llm" if "visual" not in perf_md else "vllm"
        execute_cmd(
            ["python3", "convert_embed.py", "--path", embed_pt, "--type", model_type],
            log_file,
        )
    if cmds:
        cmds[MODELS_FILE_SIZE] = file_size_dict
    return cmds


def _build_tcim_cmds(cfg_data: Dict, log_file: str) -> Dict[str, List[str]]:
    """
    Build TCIM performance test commands from configuration data.

    Args:
        cfg_data (Dict): Configuration data containing performance test parameters
        log_file (str): Path to the log file

    Returns:
        Dict[str, List[str]]: Dictionary mapping model names to command lists
    """
    cmds = {}
    file_size_dict = {}
    for perf_md in cfg_data["Streams"]:
        tmp_cmd = ["./tcim_perf"]
        # Filter parameters
        for param, param_val in perf_md.items():
            if param in ["hmm_list", "ModelName", "model_name", "quant_models"] or (
                isinstance(param_val, int) and param_val <= 0
            ):
                continue
            tmp_cmd += [f"--{param}", str(param_val)]

        # Process hmm_list
        for hmm_path in perf_md["hmm_list"]:
            ori_backup = f"{hmm_path}.ori"
            strip_backup = f"{hmm_path}.strip"
            if os.path.exists(ori_backup) and not os.path.exists(strip_backup):
                shutil.move(hmm_path, strip_backup)
                shutil.move(ori_backup, hmm_path)
                logger.info(f"Restore hmm for tcim_perf, {ori_backup} -> {hmm_path}")

            hmm_name = hmm_path.rsplit("/", 1)[-1]
            tmp_cmd_final = tmp_cmd + ["--model", hmm_path]
            key_name = f"{perf_md['ModelName']}_{hmm_name}"
            cmds[key_name] = tmp_cmd_final

            model_size = _get_file_size_mb(hmm_path)
            file_size_dict[key_name] = f"{model_size:.3f}"

    if cmds:
        cmds[MODELS_FILE_SIZE] = file_size_dict
    return cmds


if __name__ == "__main__":
    args = parse_args()

    setup_logging(log_file=args.log_file)
    logger = logging.getLogger(__name__)
    logger.info("Perf cfg path: %s", args.perf_cfg)

    os.system("pip3 install -i https://pypi.tuna.tsinghua.edu.cn/simple openpyxl")

    cfg_suffix_map = {"llm": "", "tcim": "_tcim", "demo": "_demo"}
    cfg_paths = _generate_cfg_paths(args.perf_cfg, cfg_suffix_map)

    llm_cfg_data = _load_config_file(cfg_paths["llm"])
    tcim_cfg_data = _load_config_file(cfg_paths["tcim"])
    demo_cfg_data = _load_config_file(cfg_paths["demo"])

    cfg_valid = any(
        [
            os.path.exists(cfg_paths["llm"]),
            os.path.exists(cfg_paths["tcim"]),
            os.path.exists(cfg_paths["demo"]),
        ]
    )
    if not cfg_valid:
        logger.error(f"Invalid perf config path: {args.perf_cfg}")
        exit(-1)

    PERF_TASK_CONFIG = {
        "llm": {
            "dir": f"{script_dir}/../../tools/llm_perf",
            "generate_table": _generate_llm_perf_table,
            "task_name": "LLM Perf",
        },
        "tcim": {
            "dir": f"{script_dir}/../../tools/tcim_perf",
            "generate_table": _generate_tcim_perf_table,
            "task_name": "Tcim Perf",
        },
    }
    os.environ["HDPL_PLATFORM"] = "ASIC"
    HOUMO_TARGET = os.getenv("HOUMO_TARGET")

    # ==========  Execute LLM Perf ==========
    _execute_perf_task(
        perf_dir=PERF_TASK_CONFIG["llm"]["dir"],
        cfg_data=llm_cfg_data,
        cmd_builder=_build_llm_cmds,
        generate_table_func=PERF_TASK_CONFIG["llm"]["generate_table"],
        base_cfg_path=cfg_paths["llm"],
        log_file=args.log_file,
        task_name=PERF_TASK_CONFIG["llm"]["task_name"],
    )
    # ==========  Execute Tcim Perf ==========
    _execute_perf_task(
        perf_dir=PERF_TASK_CONFIG["tcim"]["dir"],
        cfg_data=tcim_cfg_data,
        cmd_builder=_build_tcim_cmds,
        generate_table_func=PERF_TASK_CONFIG["tcim"]["generate_table"],
        base_cfg_path=cfg_paths["llm"],
        log_file=args.log_file,
        task_name=PERF_TASK_CONFIG["tcim"]["task_name"],
    )
    # ==========  Execute Demo Perf ==========
    for perf_md in demo_cfg_data["Streams"]:
        hmm_dir = perf_md["hmm_dir"]
        hmm_file_paths = glob.glob(os.path.join(hmm_dir, "*.hmm"))
        source_hmquant = os.path.join(hmm_dir, "hmquant")

        # Check if files exist
        if not hmm_file_paths or not os.path.exists(source_hmquant):
            logger.info(
                f"Warning: Not found hmm files in {hmm_dir} \n or Not found hmquant folder {source_hmquant}."
            )
            continue

        # Prepare test directory
        model_dir = os.path.abspath(f"{script_dir}/../../{perf_md['model_dir']}")
        test_dir = _prepare_test_folder(model_dir)
        target_dir = os.path.join(test_dir, "output", HOUMO_TARGET)
        os.makedirs(target_dir, exist_ok=True)
        logger.info("Current dir: %s", os.getcwd())

        # Copy hmm files
        copy_success = True
        file_size_sum = 0
        for hmm_file in hmm_file_paths:
            file_size_sum += _get_file_size_mb(hmm_file)
            file_name = os.path.basename(hmm_file)
            target_file = os.path.join(target_dir, file_name)
            if not _copy_file_or_dir(hmm_file, target_file, clean_dir=test_dir):
                copy_success = False
                break

        if not copy_success:
            os.chdir(script_dir)
            continue

        # Copy hmquant folder
        target_hmquant = os.path.join(target_dir, "hmquant")
        if not _copy_file_or_dir(
            source_hmquant, target_hmquant, is_dir=True, clean_dir=test_dir
        ):
            os.chdir(script_dir)
            continue

        # Execute demo command
        model_name = perf_md["ModelName"]
        demo_cmd = ["bash", "test.sh", "--step", "demo"]
        logger.info(f"[Demo Perf] execute cmd: {demo_cmd}, folder: {os.getcwd()}")
        ret, outputs = execute_cmd(demo_cmd, args.log_file, get_outputs=True)

        # Generate demo performance table
        if ret:
            outputs_total = [f"****** {START_TASK_STR}, {MODEL_NAME_STR} {model_name}"]
            outputs_total += outputs
            outputs_total += [f"****** {END_TASK_STR} ******"]
            _generate_demo_perf_table(cfg_paths["llm"], outputs_total, file_size_sum)

        # Cleanup directory
        os.chdir(script_dir)
        shutil.rmtree(test_dir, ignore_errors=True)
