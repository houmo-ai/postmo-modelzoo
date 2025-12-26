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

START_TASK_STR = "Start of Task"
MODEL_NAME_STR = "ModelName:"
END_TASK_STR = "End of Task"

MEM_START_STR = "HM Device Memory Usage"
MEM_USED_STR = "memory used:"
MEM_END_LINE_STR = "************************************"

MEM_USED_COLS = "device_mem_used(MB)"


def parse_args():
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
    if not keywords or not line:
        return None
    for keyword in keywords:
        if keyword in line:
            return keyword  # 找到第一个匹配项立即返回
    return None


def _prepare_test_folder(model_dir: str) -> str:
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
    perf_df = pd.DataFrame(perf_metric)
    perf_df.columns = new_cols
    perf_xlsx_path = cfg_path.replace(JSON_SUFFIX, ".xlsx")
    logger.info(f'perf_xlsx_path: {perf_xlsx_path}')
    excel_mode = "a"
    if not os.path.exists(perf_xlsx_path):
        excel_mode = "w"
    with pd.ExcelWriter(perf_xlsx_path, engine="openpyxl", mode=excel_mode) as writer:
        perf_df.to_excel(writer, sheet_name=sheet_name, index=False)


def _get_value_after_colon(line: str, split_space: bool = False) -> str:
    """提取行中最后一个冒号后的内容并去空格"""
    val = line.strip().rsplit(":", 1)[-1].strip()
    if split_space:
        val = val.split(" ", 1)[0]
    return val


######################################################################
# -------------------------- 条件检查函数 -------------------------- #
######################################################################


def _check_start_task(line: str, state: Dict) -> bool:
    """检查是否是任务开始行"""
    return START_TASK_STR in line


def _check_input_token(line: str, state: Dict) -> bool:
    """检查是否是输入token长度行"""
    return "input token len" in line


def _check_stop_token(line: str, state: Dict) -> bool:
    """检查是否是输出token长度行"""
    return "stop token len" in line


def _check_loop(line: str, state: Dict) -> bool:
    """检查是否是循环次数行"""
    return "loop :" in line or "  loops:" in line


def _check_end_task(line: str, state: Dict) -> bool:
    """检查是否是任务结束行"""
    return END_TASK_STR in line


def _check_llm_perf_avg(line: str, state: Dict) -> bool:
    """检查是否是LLM性能平均值开始行"""
    return "LLM Perf Avarage Information" in line


def _check_perf_flag(line: str, state: Dict) -> bool:
    """检查是否是性能指标行(且perf_flag为True)"""
    return state["perf_flag"] is True


def _check_mem_start(line: str, state: Dict) -> bool:
    """检查是否是内存监控开始行"""
    return MEM_START_STR in line


def _check_mem_used(line: str, state: Dict) -> bool:
    """检查是否是内存使用量行(且mem_flag为True)"""
    return state["mem_flag"] is True and MEM_USED_STR in line


def _check_mem_end(line: str, state: Dict) -> bool:
    """检查是否是内存监控结束行(且mem_flag为True)"""
    return MEM_END_LINE_STR in line and state["mem_flag"] is True


def _check_samples(line: str, state: Dict) -> bool:
    """检查是否为samples数量行"""
    return "  samples:" in line


def _check_warmup(line: str, state: Dict) -> bool:
    return "  warmup:" in line


def _check_device_num(line: str, state: Dict) -> bool:
    return "  device_num:" in line


def _check_latency(line: str, state: Dict) -> bool:
    return "[latency] " in line


def _check_throughput_qps(line: str, state: Dict) -> bool:
    return "[Throughput] qps" in line


def _check_total_cost(line: str, state: Dict) -> bool:
    return "Total Cost " in line and "TTS" not in line


######################################################################
# -------------------------- 业务处理函数 -------------------------- #
######################################################################


def _handle_mem_start(
    line: str,
    perf_metric: Dict,
    state: Dict,
    keywords_dict: Dict = {},
    perf_key: str = "mem_flag",
) -> None:
    """(common) 处理内存监控开始行:置位mem_flag"""
    state["mem_flag"] = True


def _handle_mem_end(
    line: str,
    perf_metric: Dict,
    state: Dict,
    keywords_dict: Dict = {},
    perf_key: str = "mem_flag",
) -> None:
    """(common) 处理内存监控结束行:重置mem_flag"""
    state["mem_flag"] = False


def _handle_mem_used(
    line: str,
    perf_metric: Dict,
    state: Dict = {},
    keywords_dict: Dict = {},
    perf_key: str = "device_mem_used",
) -> None:
    """(common) 处理内存使用量行:更新设备内存使用量"""
    mem_used_str = _get_value_after_colon(line)
    if perf_metric[perf_key][-1] == "NA":
        perf_metric[perf_key][-1] = mem_used_str
        return
    perf_metric[perf_key][-1] += f"/{mem_used_str}"


def _handle_start_task(
    line: str, perf_metric: Dict, state: Dict, keywords_dict: Dict, perf_key: str
) -> None:
    """(common) 处理任务开始行:初始化指标、提取模型名"""
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
    """(common) 处理任务结束行:重置flag"""
    state["perf_flag"] = False
    state["mem_flag"] = False


def _handle_perf_start_line(
    line: str,
    perf_metric: Dict,
    state: Dict,
    keywords_dict: Dict = {},
    perf_key: str = "",
) -> None:
    """(common) 处理性能开始行:置位perf_flag"""
    state["perf_flag"] = True


def _handle_token_loop(
    line: str, perf_metric: Dict, state: Dict, keywords_dict: Dict, perf_key: str
) -> None:
    """(llm_perf) 处理输入/输出token长度&循环次数"""
    perf_metric[perf_key][-1] = _get_value_after_colon(line)


def _handle_perf_flag(
    line: str, perf_metric: Dict, state: Dict, keywords_dict: Dict, perf_key: str
) -> None:
    """(llm_perf) 处理性能指标行:提取匹配的指标值"""
    keyword = _find_first_matched_keyword(line, keywords_dict.keys())
    if keyword is not None:
        perf_metric[keywords_dict[keyword]][-1] = (
            line.strip().rsplit(" ", 2)[-2].strip()
        )


def _parse_latency_line(line: str, perf_metric: dict) -> None:
    """(tcim_perf) 解析latency行, 更新perf_metric"""

    latency_type_map = {
        " Inference": "inference",
        " Input": "input",
        " Output": "output",
        " End2End": "e2e",
    }

    # 匹配latency类型
    key_str = ""
    for pattern, key in latency_type_map.items():
        if pattern in line:
            key_str = key
            break
    if not key_str:
        return

    # 解析latency值(avg/max/min)
    perf_vals = line.strip().split(",")
    # 提取值并去除末尾3个字符(如单位)
    avg_val = perf_vals[0].rsplit(":", 1)[-1].strip()[:-3]
    max_val = perf_vals[1].rsplit(":", 1)[-1].strip()[:-3]
    min_val = perf_vals[2].rsplit(":", 1)[-1].strip()[:-3]

    # 更新指标
    perf_metric[f"{key_str}_avg"][-1] = avg_val
    perf_metric[f"{key_str}_max"][-1] = max_val
    perf_metric[f"{key_str}_min"][-1] = min_val


def _handle_simple_metric(line: str, perf_metric: dict, metric_key: str) -> None:
    """(tcim_perf) 处理samples/loops/warmup/device_num等简单指标"""
    perf_metric[metric_key][-1] = _get_value_after_colon(line)


def _handle_perf_flag_logic(
    line: str, perf_metric: dict, keywords_dict: dict, keywords_dict_2: dict
) -> None:
    """(minicpmo_perf) 处理perf_flag=True时的所有逻辑"""
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


def _generate_llm_perf_table(cfg_path, outputs):
    perf_metric = {
        "model_name": [],
        "input_token": [],
        "output_token": [],
        "loop": [],
        "device_mem_used": [],
        "prefill_time": [],
        "decode_time": [],
        "vision_time": [],
        "prefill_speed": [],
        "decode_speed": [],
        "TTFT": [],
        "TPOT": [],
        "e2e_latency": [],
        "e2e_tps": [],
        "embedding_time": [],
    }

    keywords_dict = {
        "Prefill Time": "prefill_time",
        "Decode Time": "decode_time",
        "Vision Time": "vision_time",
        "Prefill Speed": "prefill_speed",
        "Decode Speed": "decode_speed",
        "TTFT": "TTFT",
        "TPOT": "TPOT",
        "E2E Latency": "e2e_latency",
        "E2E TPS": "e2e_tps",
        "Embedding Time": "embedding_time",
    }

    state = {"perf_flag": False, "mem_flag": False}

    processors = [
        (_check_start_task, _handle_start_task, "model_name"),
        (_check_input_token, _handle_token_loop, "input_token"),
        (_check_stop_token, _handle_token_loop, "output_token"),
        (_check_loop, _handle_token_loop, "loop"),
        (_check_end_task, _handle_end_task, ""),
        (_check_mem_end, _handle_mem_end, ""),
        (_check_llm_perf_avg, _handle_perf_start_line, ""),
        (_check_perf_flag, _handle_perf_flag, ""),
        (_check_mem_start, _handle_mem_start, ""),
        (_check_mem_used, _handle_mem_used, "device_mem_used"),
    ]

    for line in outputs:
        for check_func, handle_func, perf_key in processors:
            if check_func(line, state):
                handle_func(line, perf_metric, state, keywords_dict, perf_key)
                break

    for key, value in perf_metric.items():
        logger.info(f"{key}, length: {len(value)}")

    columns = [
        "model_name",
        "input(token)",
        "output(token)",
        "loop",
        MEM_USED_COLS,
        "prefill_time(ms)",
        "decode_time(ms)",
        "vision_time(ms)",
        "prefill_speed(token/s)",
        "decode_speed(token/s)",
        "TTFT(ms)",
        "TPOT(ms/token)",
        "e2e_latency(s)",
        "e2e_tps(tokens/s)",
        "embedding_time(ms)",
    ]
    _write_to_xlsx(perf_metric, cfg_path, "llm_perf", columns)


def _generate_tcim_perf_table(cfg_path, outputs):
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
        # (条件检查函数, 处理函数, 处理函数的额外参数)
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
    ]
    _write_to_xlsx(perf_metric, cfg_path, "tcim_perf", columns)


def _generate_demo_perf_table(cfg_path, outputs):

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
        # (条件检查函数, 处理函数, 处理函数的参数)
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
    ]
    _write_to_xlsx(perf_metric, cfg_path, "demo_perf", columns)


def _generate_cfg_paths(
    base_cfg_path: str, suffix_map: Dict[str, str]
) -> Dict[str, str]:
    """配置路径生成函数"""
    cfg_paths = {}
    for name, suffix in suffix_map.items():
        cfg_paths[name] = base_cfg_path.replace(JSON_SUFFIX, f"{suffix}{JSON_SUFFIX}")
    return cfg_paths


def _load_config_file(cfg_path: str, default_data: Dict = None) -> Dict:
    """配置文件加载函数"""
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
    通用文件/文件夹复制函数
    :param src: 源路径
    :param dst: 目标路径
    :param is_dir: 是否是文件夹
    :param clean_dir: 复制失败时需要清理的目录
    :return: 是否复制成功
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


def _execute_perf_task(
    perf_dir: str,
    cfg_data: Dict,
    cmd_builder: Callable[[Dict], Dict[str, List[str]]],
    generate_table_func: Callable[[str, List[str]], None],
    base_cfg_path: str,
    log_file: str,
    task_name: str = "Perf",
) -> None:
    """
    性能任务执行函数, 适用llm_perf&tcim_perf
    :param perf_dir: 性能测试目录(如llm_perf_dir)
    :param cfg_data: 配置数据(如llm_cfg_data)
    :param cmd_builder: 命令构建函数(输入perf_md, 返回cmds字典)
    :param generate_table_func: 生成表格的函数
    :param base_cfg_path: 基础配置路径(llm_cfg_path)
    :param log_file: 日志文件路径
    :param task_name: 任务名称(用于日志)
    """
    cmds = cmd_builder(cfg_data)
    if len(cmds) == 0:
        logger.info(f"[{task_name}] No commands to execute")
        return

    os.chdir(perf_dir)
    logger.info(f"Current dir: {os.getcwd()}")
    logger.info(f"[{task_name}] cmds: {cmds}")

    outputs_total = []
    for model_name, cmd in cmds.items():
        outputs_total += [f"****** {START_TASK_STR}, {MODEL_NAME_STR} {model_name}"]
        _, outputs = execute_cmd(cmd, log_file, get_outputs=True)
        outputs_total += outputs
        outputs_total += [f"****** {END_TASK_STR} ******"]
        time.sleep(5)

    generate_table_func(base_cfg_path, outputs_total)


def _build_llm_cmds(cfg_data: Dict) -> Dict[str, List[str]]:
    """构建llm_perf命令"""
    os.chdir(f"{script_dir}/../../tools/llm_perf")

    cmds = {}
    for perf_md in cfg_data["Streams"]:
        tmp_cmd = ["./llm_perf"]
        # 过滤参数
        for param, param_val in perf_md.items():
            if param in ["ModelName"]:
                continue
            tmp_cmd += [f"--{param}", str(param_val)]
        cmds[perf_md["ModelName"]] = tmp_cmd

        # 嵌入文件转换逻辑
        embed_bin = perf_md["embedding"]
        embed_pt = embed_bin.replace(".bin", ".pt")
        if os.path.exists(embed_bin):
            continue
        model_type = "llm" if "visual" not in perf_md else "vllm"
        execute_cmd(
            ["python3", "convert_embed.py", "--path", embed_pt, "--type", model_type],
            args.log_file,  # args需在主函数中定义
        )
    return cmds


def _build_tcim_cmds(cfg_data: Dict) -> Dict[str, List[str]]:
    """构建tcim_perf命令"""
    cmds = {}
    for perf_md in cfg_data["Streams"]:
        tmp_cmd = ["./tcim_perf"]
        # 过滤参数
        for param, param_val in perf_md.items():
            if param in ["hmm_list", "ModelName"] or (
                isinstance(param_val, int) and param_val <= 0
            ):
                continue
            tmp_cmd += [f"--{param}", str(param_val)]

        # 处理hmm_list
        for hmm_path in perf_md["hmm_list"]:
            ori_backup = f"{hmm_path}.ori"
            strip_backup = f"{hmm_path}.strip"
            if os.path.exists(ori_backup) and not os.path.exists(strip_backup):
                shutil.move(hmm_path, strip_backup)
                shutil.move(ori_backup, hmm_path)
                logger.info(f"Restore hmm for tcim_perf, {ori_backup} -> {hmm_path}")

            hmm_name = hmm_path.rsplit("/", 1)[-1]
            tmp_cmd_final = tmp_cmd + ["--model", hmm_path]
            key_name = (
                f"{perf_md['ModelName']}_{hmm_name}"  # 修复原代码model_name未定义问题
            )
            cmds[key_name] = tmp_cmd_final
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

    # ==========  执行 LLM Perf ==========
    _execute_perf_task(
        perf_dir=PERF_TASK_CONFIG["llm"]["dir"],
        cfg_data=llm_cfg_data,
        cmd_builder=_build_llm_cmds,
        generate_table_func=PERF_TASK_CONFIG["llm"]["generate_table"],
        base_cfg_path=cfg_paths["llm"],
        log_file=args.log_file,
        task_name=PERF_TASK_CONFIG["llm"]["task_name"],
    )
    # ==========  执行 Tcim Perf ==========
    _execute_perf_task(
        perf_dir=PERF_TASK_CONFIG["tcim"]["dir"],
        cfg_data=tcim_cfg_data,
        cmd_builder=_build_tcim_cmds,
        generate_table_func=PERF_TASK_CONFIG["tcim"]["generate_table"],
        base_cfg_path=cfg_paths["llm"],
        log_file=args.log_file,
        task_name=PERF_TASK_CONFIG["tcim"]["task_name"],
    )
    # ==========  执行 Demo Perf ==========
    for perf_md in demo_cfg_data["Streams"]:
        hmm_dir = perf_md['hmm_dir']
        hmm_file_paths = glob.glob(os.path.join(hmm_dir, "*.hmm"))
        source_hmquant = os.path.join(hmm_dir, "hmquant")

        # 检查文件是否存在
        if not hmm_file_paths or not os.path.exists(source_hmquant):
            logger.info(
                f"Warning: Not found hmm files in {hmm_dir} \n or Not found hmquant folder {source_hmquant}."
            )
            continue

        # 准备测试目录
        model_dir = os.path.abspath(f"{script_dir}/../../{perf_md['model_dir']}")
        test_dir = _prepare_test_folder(model_dir)
        target_dir = os.path.join(test_dir, "output", HOUMO_TARGET)
        os.makedirs(target_dir, exist_ok=True)
        logger.info("Current dir: %s", os.getcwd())

        # 复制hmm文件
        copy_success = True
        for hmm_file in hmm_file_paths:
            file_name = os.path.basename(hmm_file)
            target_file = os.path.join(target_dir, file_name)
            if not _copy_file_or_dir(hmm_file, target_file, clean_dir=test_dir):
                copy_success = False
                break

        if not copy_success:
            os.chdir(script_dir)
            continue

        # 复制hmquant文件夹
        target_hmquant = os.path.join(target_dir, "hmquant")
        if not _copy_file_or_dir(
            source_hmquant, target_hmquant, is_dir=True, clean_dir=test_dir
        ):
            os.chdir(script_dir)
            continue

        # 执行demo命令
        model_name = perf_md["ModelName"]
        demo_cmd = ["bash", "test.sh", "--step", "demo"]
        logger.info(f"[Demo Perf] execute cmd: {demo_cmd}, folder: {os.getcwd()}")
        ret, outputs = execute_cmd(demo_cmd, args.log_file, get_outputs=True)

        # 生成demo性能表格
        if ret:
            outputs_total = [f"****** {START_TASK_STR}, {MODEL_NAME_STR} {model_name}"]
            outputs_total += outputs
            outputs_total += [f"****** {END_TASK_STR} ******"]
            _generate_demo_perf_table(cfg_paths["llm"], outputs_total)

        # 清理目录
        os.chdir(script_dir)
        shutil.rmtree(test_dir, ignore_errors=True)
