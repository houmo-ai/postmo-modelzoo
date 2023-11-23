#!/usr/bin/env python

import os
import sys
import traceback
import numpy as np
import argparse
import importlib
import logging
import time
from hmassist.utils import logger
from hmassist.utils.glog_format import GLogFormatter
from hmassist.utils.parser import read_yaml_to_dict
from hmassist.utils.dist_metrics import cosine_distance
from hmassist.utils.check import (
    check_config,
    check_demo_config,
    check_accuracy_config,
    check_file_exist
)
from hmassist.executors.h30_exec import H30Exec
from hmassist.executors.onnx_exec import OnnxExec
from hmassist.models.base_model import BaseModel

def set_logger(op, log_dir, filename):
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    t = time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime())
    filepath = os.path.join(log_dir, "{}-{}-{}.log".format(filename, op, t))
    file_handler = logging.FileHandler(filepath)
    file_handler.setFormatter(GLogFormatter())
    logger.addHandler(file_handler)

def get_executor(cfg):
    target = cfg["build"]["target"]
    if target == ("H30"):
        return H30Exec(cfg)
        # return OnnxExec(cfg)
    else:
        logger.error("Not support target -> {}".format(target))
        exit(-1)

def get_model(cfg, backend=None):
    if "impl_class" in cfg["model"]:
        model_impl_class = cfg["model"]["impl_class"]
    else:
        model_impl_class = None
    hmexec = get_executor(cfg)
    dataset_class = cfg["accuracy"]["dataset_class"]
    data_dir = cfg["accuracy"]["data_dir"]
    try:
        m = importlib.import_module("hm_dataset")
        if hasattr(m, dataset_class):
            # 实例化预处理对象
            dataset = getattr(m, dataset_class)(data_dir)
        else:
            logger.error("hm_dataset.py has no class named {}, please check your config"
                         .format(dataset_class))
            exit(-1)
    except Exception as e:
        logger.warning("can not find hm_dataset.py, use default model will not support accuracy: {}"
                       .format(e))
        dataset = None

    try:
        m = importlib.import_module("hm_model")
        if hasattr(m, model_impl_class):
            # 实例化预处理对象
            model = getattr(m, model_impl_class)(
                executor=hmexec,
                inputs=hmexec.inputs,
                dataset=dataset,
                test_num=0,
                target=hmexec.target,
                # dtype=dtype,   # int8/fp32
                backend=backend
            )
        else:
            logger.error("hm_model.py has no class named {}, please check your config"
                         .format(model_impl_class))
            exit(-1)
    except Exception as e:
        logger.warning("can not find hm_model.py, use default model will not support demo/perf/accuracy: {}"
                       .format(e))
        model = BaseModel(
            executor=hmexec,
            inputs=hmexec.inputs,
            dataset=dataset,
            test_num=0,
            target=hmexec.target,
            # dtype=dtype,   # int8/fp32
            backend=backend
            )
    # del sys.modules["hm_model"]
    # del sys.modules["hm_dataset"]

    return model

def quantize(cfg):
    try:
        logger.info("{}".format(cfg))
        model = get_model(cfg)
        model.executor.quantize(model.get_input_datas)
    except Exception as e:
        logger.error("{}".format(traceback.format_exc()))
        logger.error("HmAssist failed to ptq quantize -> {}".format(e))

def build(cfg):
    try:
        logger.info("{}".format(cfg))
        model = get_model(cfg)       
        model.executor.build(model.build_config())

        # hmexec.model_analysis()
        # hmexec.compress_analysis()
        # hmexec.get_profile_info()
        # hmexec.get_relay_mac()  # print mac/flops/cycles info
        # hmexec.get_device_type()  # print op backend info

        # print span
        # header = ["Phase", "Span/s"]
        # table = PrettyTable(header)
        # table.add_row(["ptq", "{:.3f}".format(hmexec.quantize_span)])
        # table.add_row(["build", "{:.3f}".format(hmexec.build_span)])
        # table.add_row(["infer", "{:.3f}".format(hmexec.iss_simu_span)])
        # logger.info("\n{}".format(table))
    except Exception as e:
        logger.error("{}".format(traceback.format_exc()))
        logger.error("HmAssist failed to build -> {}".format(e))

def test(cfg, backend):
    try:
        if backend not in ["asic", "onnx"]:
            logger.error("infer phase only support asic and onnx")
            exit(-1)
        logger.info("{}".format(cfg))
        hmexec = get_executor(cfg)
        hmexec.backend = backend
        hmexec.load()
        hmexec.print_input_info()
        hmexec.print_output_info()
        inputs = hmexec.get_golden_inputs()
        hmexec.set_fixed_out(True)
        outputs = hmexec.infer(inputs)
        model_name = hmexec.model_name
        # save and compare
        for input_name, input_data in inputs.items():
            input_data.tofile(os.path.join(hmexec.result_dir, "{}_input.bin".format(input_name)))
        for output_name, output_data in outputs.items():
            # 临时添加NCHW
            # output_data = np.transpose(output_data, (0, 2, 3, 1))
            # save fixed result
            logger.info("output[{}] shape = {}, dtype = {}".format(output_name, output_data.shape, output_data.dtype))
            # save origin result
            output_data.tofile(os.path.join(hmexec.result_dir, "{}_output.bin".format(output_name)))
            output_data.tofile(os.path.join(hmexec.result_dir, "{}_output.txt".format(output_name)), sep="\n")
            np.save(os.path.join(hmexec.result_dir, 'tcim_{}_{}_output.npy'.format(model_name, output_name)), output_data)
            logger.info("output[{}] saved in {}".format(output_name, hmexec.result_dir))
            golden_output = hmexec.get_golden_output(output_name)
            if golden_output is not None:
                if golden_output.shape == output_data.shape:
                    cosine_dist = cosine_distance(golden_output, output_data)
                    is_match = (golden_output == output_data).all()
                    logger.info("[compare] {} vs quant output [{}] match={}, similarity={:.6f}"
                                .format(hmexec.target, output_name, is_match, cosine_dist))
                else:
                    logger.error("[compare] {} vs quant output [{}] shape not equal {} vs {}"
                                 .format(hmexec.target, output_name, output_data.shape, golden_output.shape))
        logger.info("success")
    except Exception as e:
        logger.error("{}".format(traceback.format_exc()))
        logger.error("HmAssist failed to infer -> {}".format(e))

def demo(cfg, backend):
    try:
        if backend not in ["asic", "onnx"]:
            logger.error("demo phase only support asic and onnx")
            exit(-1)
        logger.info(cfg)
        if not check_demo_config(cfg):
            exit(-1)
        model = get_model(cfg, backend=backend)
        data_dir = cfg["demo"]["data_dir"]
        test_num = cfg["demo"]["test_num"]

        file_list = []
        if os.path.isfile(data_dir):
            _, ext = os.path.splitext(data_dir)
            if ext in [".jpg", ".JPEG", ".bmp", ".png", ".jpeg", ".BMP"]:
                file_list.append(data_dir)
            else:
                logger.error("file type not support -> {}".format(data_dir))
        elif os.path.isdir(data_dir):
            for filename in os.listdir(data_dir):
                _, ext = os.path.splitext(filename)
                if ext in [".jpg", ".JPEG", ".bmp", ".png", ".jpeg", ".BMP"]:
                    file_list.append(os.path.join(data_dir, filename))
                    if len(file_list) == test_num:
                        break

        if backend == "onnx":
            model.load_json(hmexec.original_json_path)
        elif backend == "asic":
            model.executor.load()
        else:
            logger.error("Not support target({})".format(hmexec.target))
            exit(-1)

        for filepath in file_list:
            model.demo(filepath)
        print("demo success")
        logger.info("[infer] average cost {:.3f}ms".format(model.ave_latency_ms))
        logger.info("[end2end] average cost: {:.3f}ms".format(model.end2end_latency_ms))
        logger.info("success")
    except Exception as e:
        logger.error("{}".format(traceback.format_exc()))
        logger.error("HmAssist failed to demo -> {}".format(e))

def perf(cfg, backend):
    if backend not in ["asic", "onnx"]:
        logger.error("perf phase only support asic and onnx")
        exit(-1)
    logger.info("{}".format(cfg))
    model = get_model(cfg, backend=backend)
    test_num = cfg["perf"]["test_num"]
    model.executor.perf(test_num)
    logger.info("success")

def accuracy(cfg, dtype, backend):
    try:
        # logging.getLogger("").setLevel(logging.WARNING)
        if backend not in ["asic", "onnx"]:
            logger.error("accuracy phase only support asic and onnx")
            exit(-1)
        if not check_accuracy_config(cfg):
            exit(-1)
        model = get_model(cfg, backend=backend)

        data_dir = cfg["accuracy"]["data_dir"]
        test_num = cfg["accuracy"]["test_num"]
        dataset_class = cfg["accuracy"]["dataset_class"]

        m = importlib.import_module("hm_dataset")
        if hasattr(m, dataset_class):
            # 实例化预处理对象
            dataset = getattr(m, dataset_class)(data_dir)
        else:
            logger.error("hm_dataset.py has no class named {}".format(dataset_class))
            exit(-1)

        if backend == "onnx":
            model.load_json(model.executor.original_json_path)
        elif backend == "asic":
            model.load()
        else:
            logger.error("Not support target({})".format(model.executor.target))
            exit(-1)

        res = model.evaluate()
        logger.info("[infer] average cost {:.6f}ms".format(model.ave_latency_ms))
        logger.info("[end2end] average cost: {:.6f}ms".format(model.end2end_latency_ms))
        logger.info("{}".format(res))
        logger.info("success")
        return res
    except Exception as e:
        logger.error("{}".format(traceback.format_exc()))
        logger.error("HmAssist failed to accuracy -> {}".format(e))

def run(config_filepath, phase, dtype, target, backend):
    # 补充自定义预处理文件所在目录，必须与配置文件同目录
    config_abspath = os.path.abspath(config_filepath)
    config_dir = os.path.dirname(config_abspath)
    sys.path.insert(0, config_dir)  # 自定义模块环境变量

    config = read_yaml_to_dict(config_abspath)
    if not check_config(config, phase):
        exit(-1)
    # 更新target，优先使用命令行
    if target is not None:
        config["build"]["target"] = target

    res = dict()
    if phase == "quant":
        quantize(config)
    elif phase == "build":
        build(config)
    elif phase == "test":
        test(config, backend)
    elif phase == "demo":
        demo(config, backend)
    elif phase == "perf":
        perf(config, backend)
    elif phase == "accuracy":
        res = accuracy(config, dtype, backend)
    else:
        logger.error("Not support operation -> {}".format(phase))

    sys.path.remove(config_dir)
    return res


def benchmark(mapping_file, dtype, target, backend):
    import csv
    from prettytable import PrettyTable

    header = ["ModelName", "InputSize", "Dataset", "Num", "Acc./mAP.", "Latency(ms)"]
    table = PrettyTable(header)
    csv_filepath = "benchmark_{}_{}_{}.csv".format(backend, dtype, target)
    f = open(csv_filepath, "w")
    f_csv = csv.writer(f)
    f_csv.writerow(header)

    check_file_exist(mapping_file)
    models_dict = read_yaml_to_dict(mapping_file)["models"]
    root = os.getcwd()
    for model_name in models_dict:
        logger.info("Process {}".format(model_name))
        config_filepath = models_dict[model_name]
        config_abspath = os.path.abspath(config_filepath)
        config_dir = os.path.dirname(config_abspath)

        os.chdir(config_dir)  # 切换至模型目录
        res = run(config_abspath, "accuracy", dtype, target, backend)
        # logger.info("{}".format(res))
        os.chdir(root)  # 切换根目录

        row = list()
        if "top1" in res:
            row = [model_name, res["input_size"], res["dataset"], res["num"], "{}/{}".format(res["top1"], res["top5"]), res["latency"]]
        elif "map" in res:
            row = [model_name, res["input_size"], res["dataset"], res["num"], "{}/{}".format(res["map"], res["map50"]), res["latency"]]
        elif "easy" in res:
            row = [model_name, res["input_size"], res["dataset"], res["num"], "{}/{}/{}".format(res["easy"], res["medium"], res["hard"]), res["latency"]]
        table.add_row(row)
        f_csv.writerow(row)
        logger.info("Finish {}".format(model_name))
    f.close()
    logger.info("\n{}".format(table))
    logger.info("success")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="HmAssist Tool")
    parser.add_argument("type", type=str,
                        choices=("quant", "build", "test", "demo", "perf", "accuracy"),
                        help="Please specify a operator")
    parser.add_argument("--config", "-c", type=str, required=True,
                        help="Please specify a configuration file")
    parser.add_argument("--target", type=str, required=True,
                        choices=("H30"),
                        help="Please specify a chip target")
    parser.add_argument("--dtype", "-t", type=str, default="int8", choices=("int8", "fp32"),
                        help="Please specify one of them, default is int8")
    parser.add_argument("--backend", type=str,
                        default="asic", choices=("asic", "onnx"), 
                        help="Please specify one of them")
    parser.add_argument("--log_dir", type=str, default="./logs",
                        help="Please specify a log dir, default is ./logs")

    args = parser.parse_args()

    print(args)

    # check files
    check_file_exist(args.config)
    basename, _ = os.path.splitext(os.path.basename(args.config))
    # set_logger(args.type, args.log_dir, basename)

    # default config
    if args.backend is None:
        backend = "asic"
    else:
        backend = args.backend

    # TODO: get version
    VERSION = "v0.1.0"
    logger.info("{} with HmAssist version: {}".format(args.type, VERSION))

    if args.type == "benchmark":
        benchmark(args.config, args.dtype, args.target, backend)
    else:
        run(args.config, args.type, args.dtype, args.target, backend)
