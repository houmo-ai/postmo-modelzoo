#!/usr/bin/env python3

import os
from . import logger


def check_quant_config(cfg):
    if not check_key(cfg, "model"):
        return False
    if not check_key(cfg["model"], "framework", None, None, ["onnx"]):
        return False
    if not check_key(cfg["model"], "weight"):
        return False
    if not check_path(cfg["model"], "weight", "HOUMO_MODEL_PATH"):
        return False
    if not check_key(cfg["model"], "inputs"):
        return False
    for _input in cfg["model"]["inputs"]:
        if not check_key(_input, "layout", None, None, ["NCHW", "NHWC"]):
            return False
        if not check_key(_input, "shape"):
            return False
        if not check_key(_input, "dtype", None, None, ["uint8", "float32", "int16", "float16"]):
            return False
        if not check_key(_input, "format", None, None, ["RGB", "BGR", "GRAY"]):
            return False
        if not check_key(_input, "layout"):
            return False

        if _input["layout"] == "NHWC":
            c = _input["shape"][-1]
        else:
            c = _input["shape"][1]

        if "mean" in _input and "std" in _input:
            mean = _input["mean"]
            std = _input["std"]
            if mean is None:
                mean = [0.0 for _ in range(c)]
            elif len(mean) == 1:
                mean = [mean[0] for _ in range(c)]
            if std is None:
                std = [1.0 for _ in range(c)]
            elif len(std) == 1:
                std = [std[0] for _ in range(c)]
            if c != len(mean) or c != len(std) or len(mean) != len(std):
                logger.error("input channel must be equal len(mean/std)")
                return False
    if not check_key(cfg, "quant"):
        return False
    if not check_path(cfg["quant"], "ptq_cfg_path", "", "none"):
        return False
    if not check_path(cfg["quant"], "calib_dir", "HOUMO_DATASETS_PATH", "default"):
        return False
    if not check_path(cfg["quant"], "data_path", "HOUMO_DATASETS_PATH", "default"):
        return False
    if not check_key(cfg["quant"], "calib_num", 0):
        return False
    if not check_key(cfg["quant"], "debug_level", 1, None, [0, 1]):
        return False
    if not check_key(cfg["quant"], "mix_search", False, None, [True, False]):
        return False
    if not check_key(cfg["quant"], "calib_method", "minmax"): # ["kl", "minmax", "percent-0.99", "mse", "ema", "aciq"]
        return False
    return True


def check_build_config(cfg):
    if not check_key(cfg, "model"):
        return False
    if not check_key(cfg["model"], "name"):
        return False
    if not check_key(cfg["model"], "save_dir"):
        return False
    if not check_key(cfg, "build"):
        return False
    if not check_key(cfg["build"], "ncore", 1, None, [1, 2, 4]):
        return False
    if not check_key(cfg["build"], "opt_level", 2, None, [0, 1, 2]):
        return False
    return True


def check_test_config(cfg):
    if not check_key(cfg, "test"):
        return False
    if not check_path(cfg["test"], "data_path", "HOUMO_DATASETS_PATH"):
        return False
    return True


def check_demo_config(cfg):
    if not check_key(cfg, "model"):
        return False
    if not check_key(cfg["model"], "impl_class"):
        return False
    if not check_key(cfg, "demo"):
        return False
    if not check_path(cfg["demo"], "data_dir", "HOUMO_DATASETS_PATH"):
        return False
    if not check_key(cfg["demo"], "test_num", 0):
        return False
    return True


def check_perf_config(cfg):
    if not check_key(cfg, "perf"):
        return False
    if not check_key(cfg["perf"], "test_num", 0):
        return False
    if not check_key(cfg["perf"], "infer_only", False):
        return False
    return True


def check_eval_config(cfg):
    if not check_key(cfg, "model"):
        return False
    if not check_key(cfg["model"], "impl_class"):
        return False
    if not check_key(cfg, "eval"):
        return False
    if not check_path(cfg["eval"], "data_dir", "HOUMO_DATASETS_PATH"):
        return False
    if not check_key(cfg["eval"], "test_num", 0):
        return False
    if not check_key(cfg["eval"], "dataset_class"):
        return False
    return True


def check_file(file):
    if not os.path.exists(file):
        logger.error("File not found -> {}".format(file))
        return False
    return True


def check_key(cfg, key, default=None, type=None, value_list=None):
    # 如果default为None则不存在时报错
    if cfg.get(key) is None:
        if default is None:
            logger.error(f"key({key}) not found in {cfg}.")
            return False
        else:
            cfg[key] = default
    if type is not None:
        if isinstance(type, list):
            if type(cfg[key]) not in type:
                logger.error(f"key({key}) type({type(cfg[key])}) must be in {type}.")
                return False
        elif type(cfg[key]) != type:
            logger.error(f"key({key}) type({type(cfg[key])}) must be {type}.")
            return False
    if value_list is not None and cfg[key] not in value_list:
        logger.error(f"key({key}) value({cfg[key]}) must be in {value_list}.")
        return False
    return True


def check_path(cfg, key, env='', default=None):
    # 如果没有配置路径，根据default
    if cfg.get(key) is None:
        if default is None:
            logger.error(f"key({key}) not found in {cfg}.")
            return False
        else:
            cfg[key] = default
            logger.warning(f"key({key}) not found in {cfg}. use default {default}.")
            return True
    # 如果配置了路径但不存在，报错
    if not os.path.exists(cfg[key]):
        comp_path = os.path.join(os.environ.get(env), cfg[key])
        if os.path.exists(comp_path):
            cfg[key] = comp_path
            return True
        logger.error(f"File not found in neither {cfg[key]} nor {comp_path}.")
        return False
    return True
