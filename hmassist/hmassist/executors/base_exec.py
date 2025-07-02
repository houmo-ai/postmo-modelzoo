#!/usr/bin/env python  

import os
import abc
import numpy as np
from ..utils import logger


class BaseExec(object, metaclass=abc.ABCMeta):
    """base hmexec"""

    def __init__(self, cfg: dict):
        """init"""
        self.model_cfg = cfg["model"]
        self.quant_cfg = cfg["quant"]
        self.build_cfg = cfg["build"]
        self.test_cfg = cfg["test"]
        self.demo_cfg = cfg.get("demo")
        self.perf_cfg = cfg["perf"]
        self.eval_cfg = cfg.get("eval")
        self.batch = cfg["batch"]
        self.j =  cfg["build"].get('j', None)

        # config from cmd
        self.target = cfg["target"]
        if "thread_num" in cfg:
            self.perf_cfg["thread_num"] = cfg["thread_num"]
        else:
            self.perf_cfg["thread_num"] = 1

        # model params
        self.framework = self.model_cfg["framework"]
        self.weight = self.model_cfg["weight"]
        self.inputs = self.model_cfg["inputs"]
        self.num_inputs = len(self.inputs)
        self.model_name = self.model_cfg["name"]
        # other params
        self.cur_dir = os.path.abspath("./")
        self.model_dir = os.path.abspath(os.path.join(cfg["model"]["save_dir"], self.target))
        self.quant_dir = os.path.join(self.model_dir, "hmquant")
        self.test_dir = os.path.join(self.model_dir, "test")
        self.quant_model_path = os.path.abspath(os.path.join(self.quant_dir, 'hmquant_' + self.model_name + '_with_act.onnx'))
        self.build_dir = os.path.join(self.model_dir, "tcim")
        self.golden_data_path = self.quant_dir
        if not os.path.exists(self.quant_dir):
            os.makedirs(self.quant_dir)
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
            if "dtype" not in _input:
                if _input["format"] == "None":
                    _input["dtype"] = "float32"
                else:
                    _input["dtype"] = "uint8"

            if _input["layout"] == "NCHW":
                n, c, h, w = shape
            elif _input["layout"] == "NHWC":
                n, h, w, c = shape
                shape = (n, c, w, h)
            elif _input["layout"] == "ND":
                pass

            self.shape_dict[_input["name"]] = shape
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

    def gen_golden(self, inputs):
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
        pass

    def print_output_info(self):
        pass

    def get_relay_mac(self):
        """get relay func MAC count"""
        raise NotImplementedError

    def get_device_type(self):
        """get op run device"""
        raise NotImplementedError

    def get_version(self):
        """get tytvm version"""
        raise NotImplementedError
