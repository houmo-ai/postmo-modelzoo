#!/usr/bin/env python3
# -*- coding:utf-8 -*-
import argparse
import json
import logging
import os
import time
import shutil
import pandas as pd
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
)


def set_logger(op, log_dir, filename):
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    t = time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime())
    filepath = os.path.join(log_dir, "{}-{}-{}.log".format(filename, op, t))
    file_handler = logging.FileHandler(filepath)
    file_handler.setFormatter(LoggingFormatter())
    logger.addHandler(file_handler)


def run_benchmark(
    models, target, hmquant_version, hmcc_version, device_id=0, enbale_static=False
):
    header = [
        "ModelName",
        "Shape",
        "Dataset",
        "DatasetNum",
        "CoreNum",
        "Batch",
        "ThreadNum",
        "Resizer",
        "HmquantVersion",
        "CompilerVersion",
        "Accuracy[onnx]",
        f"Accuracy[{target}]",
        "AccRelError",
        "Latency[ms]",
        "Throughput",
    ]
    table = PrettyTable(header)
    table.title = f"HouMo Model Benchmark Report"
    if target == "xh2" and enbale_static:
        return table
    root = os.getcwd()
    models = models["models"]
    for model_name in models:
        model_cfg = models[model_name]
        location = model_cfg["location"]
        cfg_path = model_cfg.get("config", "config.yml")
        batch = model_cfg.get("batch", 1)
        ncore = model_cfg.get("ncore", 1)
        thread_num = model_cfg.get("thread_num", 8 if target == "xh1" else 4)
        enable_eval = model_cfg.get("eval", True)
        acc_onnx = "N/A"
        acc_chip = "N/A"
        acc_err = "N/A"
        ave_latency = "N/A"
        throughput = "N/A"

        location_ok = os.path.exists(location)
        if not location_ok:
            # 尝试到环境变量HOUMO_MODEL_PATH下寻找
            new_location = os.path.join(
                os.environ.get("HOUMO_MODEL_PATH", ""), location
            )
            if os.path.exists(new_location):
                location = new_location
                location_ok = True
            else:
                logger.error(f"Not found model: {new_location}")

        if location_ok:
            # 切换工作目录至model目录
            os.chdir(location)

        # 下载模型
        download_ok = False
        if location_ok:
            try:
                os.system("python3 get_model.py")
                logger.info(f"Download model: {model_name} done.")
                download_ok = True
            except Exception as e:
                logger.error(f"download model failed: {model_name}\nException: {e}")

        # 检查配置文件是否存在
        cfg_ok = False
        hm_exec = None
        if download_ok:
            if os.path.exists(cfg_path):
                cfg_ok = True
                cfg = read_yaml_to_dict(cfg_path)
                if not check_cfg(cfg):
                    logger.error(f"{cfg_path} is not valid")
                    cfg_ok = False
                if cfg_ok:
                    cfg["target"] = target
                    cfg["build"]["ncore"] = ncore
                    cfg["build"]["batch"] = batch
                    if enbale_static:
                        inputs_cfg = cfg["model"]["inputs"]
                        if len(inputs_cfg) > 1:
                            continue
                        input_name = list(inputs_cfg.keys())[0]
                        input_cfg = inputs_cfg[input_name]
                        resizer_cfg = input_cfg.get("resizer")
                        if resizer_cfg is None:
                            continue
                        if resizer_cfg.get("enable_static_resizer", False):
                            continue
                        cfg["model"]["inputs"][input_name]["resizer"][
                            "enable_static_resizer"
                        ] = True
                    logger.info(f"\n{json.dumps(cfg, indent=2, sort_keys=False)}")
                    if target == "xh1":
                        from .exec.xh1_exec import Xh1Exec

                        hm_exec = Xh1Exec(cfg)
                    elif target == "xh2":
                        from .exec.xh2_exec import Xh2Exec

                        hm_exec = Xh2Exec(cfg)
            else:
                logger.error(f"{cfg_path} not exists")
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
        # 量化
        quantize_ok = False
        if cfg_ok and hm_exec is not None:
            try:
                res = hm_exec.quantize()
                quantize_ok = bool(res)
                logger.info(f"Quantize {model_name} done.")
            except Exception as e:
                logger.error(f"quantize failed: {model_name}\nException: {e}")
        # 编译
        build_ok = False
        if quantize_ok:
            try:
                hm_exec.build()
                build_ok = True
                logger.info(f"Build {model_name} done.")
            except Exception as e:
                logger.error(f"build failed: {model_name}\nException: {e}")
        # 性能测试
        if build_ok:
            try:
                warmup = 10
                sample = 1000
                loop_num = 1
                perf_info = hm_exec.model_perf(
                    hm_exec.hmm_path, warmup, sample, loop_num, device_id, thread_num
                )
                perf_info = list(perf_info["perf"].values())[0]["perf_info"]
                ave_latency = f"{perf_info['avg_cost']:.3f}"
                throughput = f"{perf_info['qps']:.2f}"
                logger.info(f"perf done: {model_name}")
            except Exception as e:
                logger.error(f"perf failed: {model_name}\nException: {e}")

        # onnx 数据集评估
        onnx_info = dict()
        if build_ok and enable_eval:
            try:
                # 删除缓存
                shutil.rmtree("results_onnx", ignore_errors=True)
                onnx_info = hm_exec.evaluate(backend="onnx")
                if not onnx_info:
                    logger.error(f"onnx eval failed: {model_name}")
            except Exception as e:
                logger.error(f"onnx eval failed: {model_name}\nException: {e}")
                onnx_info = dict()
        # xh1/xh2 数据集评估
        chip_info = dict()
        if build_ok and enable_eval:
            try:
                shutil.rmtree(f"results_{target}", ignore_errors=True)
                chip_info = hm_exec.evaluate(backend=target, device_id=device_id)
                if not chip_info:
                    logger.error(f"{target} eval failed: {model_name}")
            except Exception as e:
                logger.error(f"{target} eval failed: {model_name}\nException: {e}")
                chip_info = dict()

        # 计算相对误差"
        if onnx_info and ("top1_acc" in onnx_info or "acc" in onnx_info):
            key = "acc" if "acc" in onnx_info else "top1_acc"
            acc_onnx = float(onnx_info[key])
            acc_chip = float(chip_info[key])
            acc_err = acc_chip / acc_onnx - 1
            acc_onnx = f"{acc_onnx*100:.2f}"
            acc_chip = f"{acc_chip*100:.2f}"
            acc_err = f"{acc_err*100:.2f}%"
        elif onnx_info and "map50" in onnx_info:
            map50_onnx = float(onnx_info["map50"])
            map50_chip = float(chip_info["map50"])
            map50_err = map50_chip / map50_onnx - 1
            map50_95_onnx = float(onnx_info["map50_95"])
            map50_95_chip = float(chip_info["map50_95"])
            map50_95_err = map50_95_chip / map50_95_onnx - 1
            acc_onnx = f"{map50_onnx*100:.2f}/{map50_95_onnx*100:.2f}"
            acc_chip = f"{map50_chip*100:.2f}/{map50_95_chip*100:.2f}"
            acc_err = f"{map50_err*100:.2f}%/{map50_95_err*100:.2f}%"

        input_size = "N/A"
        if hm_exec is not None:
            input_size = "x".join(map(str, hm_exec.inputs_shape[0]))
            for idx in range(1, len(hm_exec.inputs_shape)):
                input_size += "\n"
                input_size += "x".join(map(str, hm_exec.inputs_shape[idx]))

        table.add_row(
            [
                model_name,
                input_size,
                onnx_info.get("dataset", "N/A"),
                onnx_info.get("num", "N/A"),
                ncore,
                batch,
                thread_num,
                resizer_mode,
                hmquant_version,
                hmcc_version,
                acc_onnx,
                acc_chip,
                acc_err,
                ave_latency,
                throughput,
            ]
        )
        # 切回根目录
        os.chdir(root)
    return table


def main():
    # 父解析器
    target = os.environ.get("HOUMO_TARGET", "unknown")
    parent_config = argparse.ArgumentParser(add_help=False)
    parent_target = argparse.ArgumentParser(add_help=False)
    parent_onnx = argparse.ArgumentParser(add_help=False)
    parent_result_path = argparse.ArgumentParser(add_help=False)
    parent_device_id = argparse.ArgumentParser(add_help=False)
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
        help="Specify a device",
    )
    # 主解析器
    parser = argparse.ArgumentParser(description="HouMo Model Assist Tool")
    subparsers = parser.add_subparsers(
        dest="command", required=True, help="qunat build compare perf demo eval"
    )
    # quant
    quant_parser = subparsers.add_parser(
        "quant",
        parents=[parent_config, parent_target, parent_result_path],
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
        ],
        help="Build a model",
    )

    # compare
    compare_parser = subparsers.add_parser(
        "compare",
        parents=[parent_config, parent_target, parent_result_path, model_cfg_parent],
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
        "--warmup", "-wn", type=int, required=True, help="Specify warnup num"
    )
    perf_parser.add_argument(
        "--sample", "-sn", type=int, required=True, help="Specify sample num"
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
    # perf_parser.add_argument(
    #     "--device",
    #     "-dn",
    #     type=int,
    #     required=False,
    #     default=1,
    #     help="Specify device num",
    # )
    # demo
    demo_parser = subparsers.add_parser(
        "demo",
        parents=[
            parent_config,
            parent_target,
            parent_onnx,
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
            parent_result_path,
            model_cfg_parent,
            parent_device_id,
        ],
        help="Run model evaluate",
    )
    # benchmark
    benchmark_parser = subparsers.add_parser(
        "benchmark",
        parents=[parent_config, parent_target, parent_result_path, parent_device_id],
        help="Run model benchmark",
    )

    args = parser.parse_args()

    logger.info(
        f"Hmatc version: {__version__}, commit: {__commit__}, build time: {__build_time__}"
    )

    # 处理批量模型benchmark
    current_command = args.command
    if current_command == "benchmark":
        # 获取量化工具版本
        hmquant_version = "unknown"
        if target == "xh1":
            hmquant_version = get_hmquant_xh1_version()
        elif target == "xh2":
            hmquant_version = get_hmquant_xh2_version()
        if hmquant_version == "unknown":
            logger.error(f"Not found hmquant version for {target}")
            exit(-1)
        # 获取编译器版本
        hmcc_version = get_package_version(f"houmo-tcim-{target}")
        if hmcc_version == "unknown":
            logger.error(f"Not found hmcc version for {target}")
            exit(-1)
        models = read_yaml_to_dict(args.config)
        t = time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime())
        if not os.path.exists("reports"):
            os.makedirs("reports")
        report_file = os.path.abspath(
            os.path.join(
                "reports",
                f"benchmark_{args.target}_v{hmquant_version}_v{hmcc_version}_{t}.csv",
            )
        )
        table = run_benchmark(
            models, args.target, hmquant_version, hmcc_version, args.device_id
        )
        table_static = run_benchmark(
            models,
            args.target,
            hmquant_version,
            hmcc_version,
            args.device_id,
            enbale_static=True,
        )
        combined_table = PrettyTable()
        combined_table.title = table.title
        combined_table.field_names = table.field_names
        table_kv = dict()
        for row in table_static.rows:
            table_kv[row[0]] = row
        for row in table.rows:
            combined_table.add_row(row)
            if row[0] in table_kv:
                combined_table.add_row(table_kv[row[0]])
        logger.info(f"\n{combined_table}")
        with open(report_file, "w", encoding="utf-8", newline="") as f:
            f.write(combined_table.get_csv_string())
        logger.info("Benchmark done.")
        exit(0)

    # 存在结果信息，先读回来更新后再存
    res_info = dict()
    if os.path.exists(args.result_path):
        res_info = read_yaml_to_dict(args.result_path)

    # 直接指定模型的perf可跳过配置文件
    if current_command == "perf" and args.model is not None:
        new_res_info = BaseExec.model_perf(
            args.model,
            args.warmup,
            args.sample,
            args.loop_num,
            args.device_id,
            args.thread,
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

    # 命令行参数更新至配置文件
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
    # 更新onnx backend
    if current_command in ["demo", "eval"] and args.onnx:
        backend = "onnx"
    # 执行对应的命令
    if current_command == "quant":
        new_res_info = hm_exec.quantize()
    elif current_command == "build":
        new_res_info = hm_exec.build()
        logger.info(f"Build {hm_exec.model_name} done.")
        new_res_info["build"].update(hm_exec.check_golden(args.device_id))
    elif current_command == "compare":
        data_path = args.data_path
        if not os.path.exists(data_path):
            # 不存在，到环境变量HOUMO_DATASETS_PATH下查找
            HOUMO_DATASETS_PATH = os.environ.get(
                "HOUMO_DATASETS_PATH", "/usr/local/src/houmo-modelzoo/data/datasets"
            )
            data_path = os.path.join(HOUMO_DATASETS_PATH, data_path)
            if not os.path.exists(data_path):
                logger.error(f"{data_path} or {args.data_path} not exists.")
                exit(-1)
        new_res_info = hm_exec.compare(data_path)
    elif current_command == "perf":
        new_res_info = hm_exec.model_perf(
            hm_exec.hmm_path,
            args.warmup,
            args.sample,
            args.loop_num,
            args.device_id,
            args.thread,
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
    save_dict_to_yaml(res_info, args.result_path)


if __name__ == "__main__":
    main()
