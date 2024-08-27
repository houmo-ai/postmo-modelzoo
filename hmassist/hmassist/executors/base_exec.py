#!/usr/bin/env python  

import os
import abc
import numpy as np
from ..utils import logger


class BaseExec(object, metaclass=abc.ABCMeta):
    """base hmexec"""

    def __init__(self, cfg: dict):
        """init"""
        # self.cfg = cfg
        self.model = cfg["model"]
        self.quant_cfg = cfg["quant"]
        self.build_cfg = cfg["build"]
        self.test_cfg = cfg["test"]
        self.demo_cfg = cfg["demo"]
        self.perf_cfg = cfg["perf"]
        self.acc_cfg = cfg["eval"]
        self.batch = cfg["batch"]
        if "thread_num" in cfg:
            self.perf_cfg["thread_num"] = cfg["thread_num"]
        else:
            self.perf_cfg["thread_num"] = 1
        # model params
        self.target = cfg["target"]
        self.framework = self.model["framework"]
        self.weight = self.model["weight"]
        if not os.path.exists(self.weight):
            weight = os.path.join(os.getenv("MODEL_PATH", default=""), self.weight)
            if not os.path.exists(weight):
                logger.warning("{} or {} not exist.".format(self.weight, weight))
                # exit(-1)
            self.weight = weight
        self.inputs = self.model["inputs"]
        self.num_inputs = len(self.inputs)
        self.model_name = self.model["name"]
        # default params
        self.build_mode = self.build_cfg.get("mode", "AOT")
        self.opt_level = self.build_cfg.get("opt_level", 2)
        # other params
        self.cur_dir = os.path.abspath("./")
        self.model_dir = os.path.abspath(os.path.join(cfg["model"]["save_dir"], self.target))
        self.result_dir = os.path.join(self.model_dir, "result")
        self.test_dir = os.path.join(self.result_dir, "test")
        self.quant_model_path = os.path.abspath(os.path.join(self.result_dir, 'hmquant_' + self.model_name + '_with_act.onnx'))
        self.golden_data_path = self.result_dir
        if not os.path.exists(self.result_dir):
            os.makedirs(self.result_dir)
        logger.info("model output dir -> {}".format(self.model_dir))

        self.shape_dict = dict()
        self.dtype_dict = dict()

        # self.set_model_name()
        self.set_input_infos()
        # self.set_custom_preprocess()

        self.quantize_span = 0
        self.build_span = 0
        self.layer_compare_span = 0
        self.is_fixed_out = False

    def quantize(self):
        """quantize"""
        logger.error("BaseExec not support quant")
        raise NotImplementedError

    def build(self):
        """build"""
        logger.error("BaseExec not support build")
        raise NotImplementedError

    @abc.abstractmethod
    def load(self):
        """ inference """
        raise NotImplementedError

    @abc.abstractmethod
    def infer(self):
        """ inference """
        raise NotImplementedError

    def set_fixed_out(self, flag):
        self.is_fixed_out = flag

    def get_dataset(self):
        quant_data_dir = self.quant["data_dir"]
        dataset = quant_data_dir
        if not quant_data_dir:  # 未配置量化路径使用随机数据情况
            dataset = self.gen_random_quant_data
        else:
            if self.has_custom_preprocess:  # 配置量化数据目录情况下存在自定义预处理
                dataset = self.preproc_class.get_data
        return dataset

    def set_input_infos(self):
        for idx, _input in enumerate(self.inputs):
            shape = _input["shape"]
            if _input["layout"] == "NCHW":
                n, c, h, w = shape
            elif _input["layout"] == "NHWC":
                n, h, w, c = shape

            if "dtype" not in _input:
                if _input["format"] == "None":
                    _input["dtype"] = "float32"
                else:
                    _input["dtype"] = "uint8"

            self.shape_dict[_input["name"]] = (n, c, h, w)
            self.dtype_dict[_input["name"]] = _input["dtype"]

            # 对mean和std进行广播
            if "mean" in _input and _input["mean"] and len(_input["mean"]) == 1:
                _input["mean"] = [_input["mean"] for _ in range(c)]
            if "std" in _input and _input["std"] and len(_input["std"]) == 1:
                _input["std"] = [_input["std"] for _ in range(c)]

    @staticmethod
    def check_not_exist(filepath):
        if not os.path.exists(filepath):
            logger.error("Not found filepath -> {}".format(filepath))
            exit(-1)

    @staticmethod
    def check_dtype(name, data, target_dtype):
        if data.dtype != target_dtype:
            logger.error("input({}) dtype mismatch {} vs {}".format(name, data.dtype, target_dtype))
            exit(-1)

    def gen_golden(self):
        raise NotImplementedError

    def set_quantize_cfg(self, in_datas):
        """ quantization config
        @param in_datas:
        @return: quantize_config
        """
        in_dtypes, norm = dict(), dict()
        for idx, _input in enumerate(self.inputs):
            name = _input["name"]
            if in_datas[name].dtype == np.uint8:
                data_type = "uint8"
            elif in_datas[name].dtype == np.int16:
                data_type = "int16"
            elif in_datas[name].dtype == np.float16:
                data_type = "float16"
            elif in_datas[name].dtype == np.float32:
                data_type = "float32"
            else:
                logger.error("Not support input dtype -> {}".format(in_datas[name].dtype))
                exit(-1)
            # 与最终量化后的模型输入数据类型相对应
            in_dtypes[name] = data_type
            norm[name] = {"mean": _input["mean"], "std": _input["std"], "axis": 1}
            logger.info("Input({}) dtype -> {}".format(name, in_dtypes[name]))
            logger.info("Input({}) mean/std -> {}".format(name, norm[name]))

        import tvm
        from tvm import relay
        quantize_config = tvm.relay.quantization.get_quantize_config(self.target, in_dtypes)
        quantize_config["calib_method"] = self.quant["calib_method"]

        quantize_config["float_list"] = list()
        skip_layer_idxes = self.quant.get("skip_layer_idxes", list())
        skip_layer_types = self.quant.get("skip_layer_types", list())
        skip_layer_names = self.quant.get("skip_layer_names", list())
        if skip_layer_idxes:
            quantize_config["float_list"].extend(skip_layer_idxes)
        if skip_layer_types:
            quantize_config["float_list"].extend(skip_layer_types)
        if skip_layer_names:
            quantize_config["float_list"].extend(skip_layer_names)
        return quantize_config, norm

    def print_input_info(self):
        input_num = len(self.input_info)
        logger.info("{} input num = {}:".format(self.target, input_num))
        for _input in self.input_info:
            logger.info("{} input[{}] shape = {}, dtype = {}, format = {}".format(self.target, _input["name"],
                                                                                  _input["shape"], _input["dtype"],
                                                                                  _input["format"].name))

    def print_output_info(self):
        output_num = len(self.output_info)
        logger.info("{} output num = {}:".format(self.target, output_num))
        for _output in self.output_info:
            logger.info("{} output[{}] shape = {}, dtype = {}, format = {}".format(self.target, _output["name"],
                                                                                   _output["shape"], _output["dtype"],
                                                                                   _output["format"].name))

    def get_relay_mac(self):
        """get relay func MAC count"""
        raise NotImplementedError

    def get_device_type(self):
        """get op run device"""
        raise NotImplementedError

    def get_version(self):
        """get tytvm version"""
        raise NotImplementedError
