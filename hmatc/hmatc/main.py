#!/usr/bin/env python3
# -*- coding:utf-8 -*-
import os
import logging
import time
import json
import argparse
from .utils import logger
from .utils.utils import read_yaml_to_dict, save_dict_to_yaml
from .utils.logging_format import LoggingFormatter 
from .utils.check import check_cfg
from .base.base_exec import BaseExec
from ._version import __commit__, __version__, __build_time__


def set_logger(op, log_dir, filename):
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    t = time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime())
    filepath = os.path.join(log_dir, "{}-{}-{}.log".format(filename, op, t))
    file_handler = logging.FileHandler(filepath)
    file_handler.setFormatter(LoggingFormatter())
    logger.addHandler(file_handler)


def main():
    # 父解析器
    target = os.environ.get("HOUMO_TARGET", "unknown")
    parent_parser0 = argparse.ArgumentParser(add_help=False)
    parent_parser1 = argparse.ArgumentParser(add_help=False)
    parent_parser2 = argparse.ArgumentParser(add_help=False)
    parent_parser3 = argparse.ArgumentParser(add_help=False)
    parent_parser0.add_argument("--config", "-c", type=str, required=True, help="config file path")
    parent_parser1.add_argument("--target", "-t", type=str, required=target not in ["xh1", "xh2"], choices=("xh1", "xh2"), default=target, help="Specify a chip target")
    parent_parser2.add_argument("--onnx", action="store_true", help="Specify onnx model as the backend")
    parent_parser3.add_argument("--result_path", type=str, required=False, default="result.yml", help="Specify a result path")
    # 主解析器
    parser = argparse.ArgumentParser(description="HouMo Model Assist Tool")
    subparsers = parser.add_subparsers(dest="command", required=True, help="qunat build compare perf demo eval")
    # quant
    quant_parser = subparsers.add_parser("quant", parents=[parent_parser0, parent_parser1, parent_parser3], help="Quantize a model")
    # build
    build_parser = subparsers.add_parser("build", parents=[parent_parser0, parent_parser1, parent_parser3], help="Build a model")
    build_parser.add_argument("--batch", type=int, required=False, default=1, help="Specify a build batch")
    build_parser.add_argument("--ncore", type=int, required=False, choices=(1, 2, 4), help="Specify a ncore, ")
    build_parser.add_argument("--opt_level", type=int, required=False, choices=(0, 1, 2), help="Specify a opt_level")
    build_parser.add_argument("--roi_num", type=int, required=False, default=1, help="Specify a roi_num")
    # compare
    compare_parser = subparsers.add_parser("compare", parents=[parent_parser0, parent_parser1, parent_parser3], help="Compare onnx/hmquant/chip")
    compare_parser.add_argument("--data_path", "-d", type=str, required=True, help="Specify a data path, image or npz")
    # perf
    perf_parser = subparsers.add_parser("perf", parents=[parent_parser1, parent_parser3], help="Test model performance")
    exclusive_group = perf_parser.add_mutually_exclusive_group(required=True)
    exclusive_group.add_argument("--config", "-c", type=str, help="Specify config file path")
    exclusive_group.add_argument("--model",  "-m", type=str, help="Specify model path")
    perf_parser.add_argument("--warmup",   "-wn", type=int, required=True, help="Specify warnup num")
    perf_parser.add_argument("--sample",   "-sn", type=int, required=True, help="Specify sample num")
    perf_parser.add_argument("--loop_num", "-ln", type=int, required=False, default=1, help="Specify loop num")
    perf_parser.add_argument("--thread",   "-tn", type=int, required=False, default=1, help="Specify thread num")
    perf_parser.add_argument("--device",   "-dn", type=int, required=False, default=1, help="Specify device num")
    # demo
    demo_parser = subparsers.add_parser("demo", parents=[parent_parser0, parent_parser1, parent_parser2, parent_parser3], help="Run model demo")
    # evaluate
    evaluate_parser = subparsers.add_parser("eval", parents=[parent_parser0, parent_parser1, parent_parser2, parent_parser3], help="Run model evaluate")
    args = parser.parse_args()
    
    logger.info(f"Hmatc version: {__version__}, commit: {__commit__}, build time: {__build_time__}")

    # 存在结果信息，先读回来更新后再存
    res_info = dict()
    if os.path.exists(args.result_path):
        res_info = read_yaml_to_dict(args.result_path)

    # perf
    current_command = args.command
    if current_command == "perf" and args.model is not None:
        new_res_info = BaseExec.model_perf(args.model, args.warmup, args.sample, args.loop_num, args.device, args.thread)
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
    check_cfg(cfg)
    
    cfg["target"] = target
    if current_command == "build":
        batch = args.batch
        if batch < 1:
            logger.error("Batch must be greater than 0")
            exit(-1)
        if batch > 1:
            cfg["build"]["batch"] = batch
        roi_num = args.roi_num
        if batch > 1 and roi_num != 1:
            logger.error("batch > 1, roi_num must be == 1")
            exit(-1)
        if batch == 1 and roi_num < 1:
            logger.error("batch == 1, roi_num must be >= 1")
            exit(-1)
        cfg["build"]["roi_num"] = roi_num
        ncore = args.ncore
        opt_level = args.opt_level
        if opt_level is not None:
            cfg["build"]["opt_level"] = opt_level
        if ncore is not None:
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
        new_res_info["build"].update(hm_exec.check_golden())
    elif current_command == "compare":
        data_path = args.data_path
        if not os.path.exists(data_path):
            # 不存在，到环境变量HOUMO_DATASETS_PATH下查找
            HOUMO_DATASETS_PATH = os.environ.get(
                "HOUMO_DATASETS_PATH", "/usr/local/src/houmo-modelzoo/data/datasets")
            data_path = os.path.join(HOUMO_DATASETS_PATH, data_path)
            if not os.path.exists(data_path):
                logger.error(f"{data_path} or {args.data_path} not exists.")
                exit(-1)
        new_res_info = hm_exec.compare(data_path)
    elif current_command == "perf":
        new_res_info = hm_exec.model_perf(
            hm_exec.hmm_path, args.warmup, args.sample, 
            args.loop_num, args.device, args.thread)
    elif current_command == "demo":
        hm_exec.demo(backend=backend)
    elif current_command == "eval":
        hm_exec.evaluate(backend=backend)
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
