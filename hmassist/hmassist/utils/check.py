#!/usr/bin/env python3

import os
from . import logger
from enum import Enum


class Phase(Enum):
    BUILD = 0
    COMPARE = 1
    DEMO = 2
    TEST = 3


def check_config(cfg, phase="build"):
    """配置文件参数合法性检查
    :param cfg:
    :param phase:
    :return:
    """
    # 检查是否缺少关键字
    if "model" not in cfg:
        logger.error("The key(model) must be in cfg")
        return False

    if "framework" not in cfg["model"]:
        logger.error("The key(framework) must be in cfg[model]")
        return False

    framework = cfg["model"]["framework"]
    framework_lists = ["onnx"]
    if framework not in framework_lists:
        logger.error("framework({}) must be in {}".format(framework, framework_lists))
        return False

    if "quant" not in cfg:
        logger.error("The key(quant) must be in cfg")
        return False

    if "build" not in cfg:
        logger.error("The key(build) must be in cfg")
        return False

    if "weight" not in cfg["model"]:
        logger.error("The key(weight) must be in cfg[model]")
        return False

    if "calib_dir" not in cfg["quant"]:
        logger.error("The key(calib_dir) must be in cfg[build][quant]")
        return False
    check_datapath(cfg["quant"], "calib_dir")

    if "calib_num" not in cfg["quant"]:
        logger.error("The key(calib_num) must be in cfg[build][quant]")
        return False

    if "debug_level" not in cfg["quant"]:
        logger.error("The key(debug_level) must be in cfg[build][quant]")
        return False

    debug_level = cfg["quant"]["debug_level"]
    debug_level_lists = [-1, 0, 1, 2, 3]
    if debug_level not in debug_level_lists:
        logger.error("debug_level({}) must be in {}".format(debug_level, debug_level_lists))
        return False

    if "calib_method" not in cfg["quant"]:
        logger.error("The key(calib_method) must be in cfg[build][quant]")
        return False

    # calib_method = cfg["quant"]["calib_method"]
    # calib_method_lists = ["kld", "min_max", "l2norm"]
    # if calib_method.startswith("percentile_"):
    #     pass
    # else:
    #     if calib_method not in calib_method_lists:
    #         logger.error("calib_method({}) must be in {}".format(calib_method, calib_method_lists))
    #         return False

    if phase == "quant":
        weight = cfg["model"]["weight"]
        if not os.path.exists(weight):
            logger.error("The model weight not exist -> {}".format(weight))
            return False

    # 多输入且不使用随机数据的情况下必须定义预处理
    input_lists = cfg["model"]["inputs"]
    calib_dir = cfg["quant"]["calib_dir"]

    for _input in input_lists:
        layout = _input["layout"]
        layout_lists = ["NCHW", "NHWC"]
        if layout not in layout_lists:
            logger.error("layout({}) must be in {}".format(layout, layout_lists))
            return False

        if "shape" not in _input:
            logger.error("shape must be in cfg[model][inputs]")
            return False

        if "dtype" in _input:
            dtype = _input["dtype"]
            dype_lists = ["uint8", "float32", "int16", "float16"]
            if dtype not in dype_lists:
                logger.error("dtype({}) must be in {}".format(dtype, dype_lists))
                return False

        if "format" not in _input:
            logger.error("format must be in cfg[model][inputs]")
            return False

        if "layout" not in _input:
            logger.error("layout must be in cfg[model][inputs]")
            return False

        shape = _input["shape"]
        if len(shape) != 4:
            logger.error("input dim must be equal 4")
            return False

        n, c, h, w = shape
        if _input["layout"] == "NHWC":
            n, h, w, c = shape

        if "mean" in _input and "std" in _input:
            mean = _input["mean"]
            std = _input["std"]
            if mean is None:
                mean = [0.0 for _ in range(c)]
            else:
                if len(mean) == 1:
                    mean = [mean[0] for _ in range(c)]
            if std is None:
                std = [1.0 for _ in range(c)]
                if len(std) == 1:
                    std = [std[0] for _ in range(c)]

            if c != len(mean) or c != len(std) or len(mean) != len(std):
                logger.error("input channel must be equal len(mean/std)")
                return False

        format = _input["format"]
        format_lists = ["None", "RGB", "BGR", "GRAY"]
        if format not in format_lists:
            logger.error("format({}) must be in {}".format(format, format_lists))
            return False

    # TODO
    # 检查是否缺少关键字

    return True


def check_test_config(cfg):
    if not check_datapath(cfg["test"], "data_path"):
        return False
    return True


def check_eval_config(cfg):
    if "eval" not in cfg:
        logger.error("Not found key(eval) in config")
        return False

    if "data_dir" not in cfg["eval"]:
        logger.error("Not found key(data_dir) in config[eval]")
        return False

    if not check_datapath(cfg["eval"], "data_dir"):
        return False

    if "test_num" not in cfg["eval"]:
        logger.error("Not found key(test_num) in config[eval]")
        return False

    if "dataset_class" not in cfg["eval"]:
        logger.error("Not found key(dataset_cls) in config[eval]")
        return False

    if "impl_class" not in cfg["model"]:
        logger.error("Not found key(impl_class) in config[model]")
        return False

    data_dir = cfg["eval"]["data_dir"]
    if not os.path.exists(data_dir):
        logger.error("Not found data_dir -> {}".format(data_dir))
        return False

    num = cfg["eval"]["test_num"]
    if not isinstance(num, int):
        logger.error("Not found test_num type not int, -> {}".format(num))
        return False

    if num < 0:
        logger.error("Not found test_num must be >= 0, -> {}".format(num))
        return False

    if not cfg["model"]["impl_class"]:
        logger.error("Not found key(impl_class) in config[model]")
        return False

    return True


def check_demo_config(cfg):
    if "demo" not in cfg:
        logger.error("Not found key(demo) in config[demo]")
        return False

    if "data_dir" not in cfg["demo"]:
        logger.error("Not found key(data_dir) in config[demo]")
        return False
    if not check_datapath(cfg["demo"], "data_dir"):
        return False

    if "test_num" not in cfg["demo"]:
        logger.error("Not found key(test_num) in config[demo]")
        return False

    if "impl_class" not in cfg["model"]:
        logger.error("Not found key(impl_class) in config[model]")
        return False

    test_num = cfg["demo"]["test_num"]
    if not isinstance(test_num, int):
        logger.error("demo test_num type not int, -> {}".format(test_num))
        return False

    if test_num < 0:
        logger.error("demo test_num must be >= 0, -> {}".format(test_num))
        return False

    if not cfg["model"]["impl_class"]:
        logger.error("Not found key(impl_class) in config[model]")
        return False

    return True


def check_args(args):
    if not os.path.exists(args.config):
        logger.error("Not found file -> {}".format(args.config))
        exit(-1)


def check_datapath(cfg, key):
    if key in cfg:
        if cfg[key] is None:
            logger.error("{} can not be None".format(key))
            return False
        if not os.path.exists(cfg[key]):
            comp_path = os.path.join(os.environ.get('DATASETS_PATH'), cfg[key])
            if os.path.exists(comp_path):
                cfg[key] = comp_path
            else:
                logger.error("{} not found -> {}".format(key, cfg[key]))
                return False
    return True
