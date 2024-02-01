#!/usr/bin/env python

import os
import sys
import traceback
import numpy as np
import argparse
import importlib
import logging
import time
import torch
from hmassist.utils import logger
from hmassist.utils.glog_format import GLogFormatter
from hmassist.utils.parser import read_yaml_to_dict
from hmassist.utils.dist_metrics import cosine_distance
from hmassist.utils.utils import get_random_data
from hmassist.utils.check import (
    check_config,
    check_test_config,
    check_demo_config,
    check_accuracy_config,
    check_args,
    check_datapath
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


def save_data(data, dir, name):
    if not os.path.exists(dir):
        os.makedirs(dir)
    data.tofile(os.path.join(dir, "{}.bin".format(name)))
    data.tofile(os.path.join(dir, "{}.txt".format(name)), sep="\n")
    np.save(os.path.join(dir, '{}.npy'.format(name)), data)


def get_executor(cfg):
    target = cfg["target"]
    if target == "H30":
        return H30Exec(cfg)
    elif target == "onnx":
        return OnnxExec(cfg)
    else:
        logger.error("Not support target -> {}".format(target))
        exit(-1)


def get_model(cfg):
    model_impl_class = cfg["model"].get("impl_class", None)
    executor = get_executor(cfg)

    try:
        m = importlib.import_module("hm_model")
        if hasattr(m, model_impl_class):
            # 实例化预处理对象
            model = getattr(m, model_impl_class)(
                executor=executor,
                dataset=None,
                # dtype=dtype,   # int8/fp32
            )
        else:
            logger.error("hm_model.py has no class named {}, please check your config"
                         .format(model_impl_class))
            exit(-1)
    except Exception as e:
        logger.warning("can not find hm_model.py, use default model will not support demo/perf/accuracy: {}"
                       .format(e))
        model = BaseModel(
            executor=executor,
            dataset=None,
            # dtype=dtype,   # int8/fp32
            )
    # del sys.modules["hm_model"]
    # del sys.modules["hm_dataset"]

    return model


def quantize(cfg):
    logger.info("{}".format(cfg))
    model = get_model(cfg)
    model.executor.quantize(model.get_input_datas)


def build(cfg):
    logger.info("{}".format(cfg))
    model = get_model(cfg)   
    model.executor.build(model.build_options())

    # compare golden data
    model.load()
    model.executor.print_input_info()
    model.executor.print_output_info()
    logger.info("start compare golden data...")
    save_dir = os.path.join(model.executor.result_dir, "tcim")
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    inputs = model.executor.get_golden_inputs()
    if inputs is not None:
        for input_name, input_data in inputs.items():
            input_data.tofile(os.path.join(save_dir, "{}_input.bin".format(input_name)))
        model.executor.set_fixed_out(True)
        outputs = model.executor.infer(inputs)
        # 临时添加NCHW
        # output_data = np.transpose(output_data, (0, 2, 3, 1))
    sum_cos = 0.0
    for output_name, output_data in outputs.items():
        logger.info("{} output[{}] shape = {}, dtype = {}".format(model.target, output_name,
                                                                  output_data.shape, output_data.dtype))
        save_data(output_data, save_dir, output_name)
        logger.info("tcim outputs saved in {}".format(save_dir))
        golden_output = model.executor.get_golden_output(output_name)
        is_match = (output_data == golden_output).all()
        cosine_dist = cosine_distance(output_data, golden_output)
        sum_cos += cosine_dist
        logger.info("[compare] {} vs quant output [{}] match={}, similarity={:.6f}"
                    .format(model.target, output_name, is_match, cosine_dist))
    logger.info("[compare] {} vs quant output average similarity={:.6f}".format(model.target, sum_cos/len(outputs)))
    logger.info("build completed")


def test(cfg):
    logger.info("{}".format(cfg))
    if not check_test_config(cfg):
        exit(-1)
    model = get_model(cfg)
    model.load()
    model.executor.print_input_info()
    model.executor.print_output_info()
    data_path = cfg["test"].get("data_path")
    if data_path:
        dir, file = os.path.split(data_path)
        inputs = model.get_input_datas(dir, file)
    else:
        inputs = {}
        for _input in model.inputs:
            name = _input["name"]
            dtype = _input["dtype"]
            logger.warning("data[{}] will use random data".format(name))
            inputs[name] = get_random_data(name, dtype, model.input_shape)
    inputs = model.executor._preprocess(inputs)
    model.executor.set_fixed_out(False)
    outputs = model.executor.infer(inputs)

    # save datas
    save_dir = os.path.join(model.executor.result_dir, "test")
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    for input_name, input_data in inputs.items():
        save_data(input_data, save_dir, input_name + "_input")
    for output_name, output_data in outputs.items():
        # 临时添加NCHW
        # output_data = np.transpose(output_data, (0, 2, 3, 1))
        logger.info("{} output[{}] shape = {}, dtype = {}".format(model.target, output_name, output_data.shape, output_data.dtype))
        save_data(output_data, save_dir, output_name)
    logger.info("test outputs saved in {}".format(save_dir))

    # compare to framework output
    if model.target in ["H30",]:
        sum_cos = 0.0
        for output_name, output_data in outputs.items():
            output_data_path = os.path.join(model.compare_dir, "test", output_name + '.npy')
            if os.path.exists(output_data_path):
                compare_data = np.load(output_data_path)
                logger.info("{} output[{}] shape = {}, dtype = {}".format(model.framework, output_name,
                                                                          output_data.shape, output_data.dtype))
            else:
                logger.warning("compare canceled while {} output not found -> {}".format(model.framework))
                return None
            is_match = (output_data == compare_data).all()
            cosine_dist = cosine_distance(output_data, compare_data)
            sum_cos += cosine_dist
            logger.info("[compare] {} vs {} output [{}] match={}, similarity={:.6f}"
                        .format(model.target, model.framework, output_name, is_match, cosine_dist))
        logger.info("[compare] {} vs {} output average similarity={:.6f}".format(model.target, model.framework, sum_cos/len(outputs)))
    logger.info("test completed")


def demo(cfg):
    logger.info(cfg)
    if not check_demo_config(cfg):
        exit(-1)
    model = get_model(cfg)
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

    model.load()

    for filepath in file_list:
        model.demo(filepath)
    logger.info("[infer] average cost {:.3f}ms".format(model.ave_latency_ms))
    logger.info("[end2end] average cost: {:.3f}ms".format(model.end2end_latency_ms))
    logger.info("demo completed")


def perf(cfg):
    logger.info("{}".format(cfg))
    model = get_model(cfg)
    test_num = cfg["perf"]["test_num"]
    model.executor.perf(test_num)
    logger.info("perf completed")


def accuracy(cfg):
    # logging.getLogger("").setLevel(logging.WARNING)
    if not check_accuracy_config(cfg):
        exit(-1)
    model = get_model(cfg)

    dataset_class = cfg["accuracy"].get("dataset_class", None)
    if not check_datapath(cfg["accuracy"], "data_dir"):
        return -1
    data_dir = cfg["accuracy"].get("data_dir")
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
        logger.error("can not find hm_dataset.py, use default model will not support accuracy: {}"
                       .format(e))
        exit(-1)

    model.test_num = cfg["accuracy"]["test_num"]
    if os.environ.get("HDPL_PLATFORM") == "ISIM":
        if model.test_num > 20 or model.test_num == 0:
            model.test_num = 20
            logger.warning("test num set to 20 because HDPL_PLATFORM=ISIM may take a lot of time.")
    model.dataset = dataset
    model.load()

    res = model.evaluate()
    logger.info("[infer] average cost {:.6f}ms".format(model.ave_latency_ms))
    logger.info("[end2end] average cost: {:.6f}ms".format(model.end2end_latency_ms))
    logger.info("{}".format(res))
    logger.info("accuracy test completed")
    return res


def run(args):
    # 补充自定义预处理文件所在目录，必须与配置文件同目录
    config_abspath = os.path.abspath(args.config)
    config_dir = os.path.dirname(config_abspath)
    sys.path.insert(0, config_dir)  # 自定义模块环境变量

    config = read_yaml_to_dict(config_abspath)
    if not check_config(config, args.type):
        exit(-1)
    config["target"] = args.target
    config['model']['batch'] = args.batch

    res = dict()
    if args.type == "quant":
        quantize(config)
    elif args.type == "build":
        build(config)
    elif args.type == "test":
        test(config)
    elif args.type == "demo":
        demo(config)
    elif args.type == "perf":
        perf(config)
    elif args.type == "accuracy":
        res = accuracy(config)
    else:
        logger.error("Not support operation -> {}".format(args.type))

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
    logger.info("success")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="HmAssist Tool")
    parser.add_argument("type", type=str,
                        choices=("quant", "build", "test", "demo", "perf", "accuracy"),
                        help="Specify an operation")
    parser.add_argument("--target", type=str, required=True,
                        choices=("H30", "onnx"),
                        help="Specify a chip target")
    parser.add_argument("--config", type=str, default="config.yml",
                        help="Specify a config file, default is config.yml")
    parser.add_argument("--batch", type=int, default=1,
                        help="Specify batch size, default is 1")
    # parser.add_argument("--dtype", type=str, default="int8",
    #                     choices=("int8", "fp32"),
    #                     help="Please specify one of them, default is int8")
    # parser.add_argument("--demo.test_num", type=int, default=-1,
    #                     help="Specify the test number in demo, default is the config in the config file")
    # parser.add_argument("--perf.test_num", type=int, default=-1,
    #                     help="Specify the test number in perf, default is the config in the config file")
    # parser.add_argument("--accuracy.test_num", type=int, default=-1,
    #                     help="Specify the test number in accuracy, default is the config in the config file")

    args = parser.parse_args()
    print(args)
    check_args(args)

    # TODO: get version
    VERSION = "v0.1.0"
    logger.info("{} with HmAssist version: {}".format(args.type, VERSION))

    run(args)
