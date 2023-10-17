#!/usr/bin/env python  

import os
import abc
import cv2
import importlib
import numpy as np
from utils import logger
from utils.enum_type import PaddingMode
from utils.preprocess import calc_padding_size, default_preprocess
from collections import namedtuple, OrderedDict
import torch
import torchvision.transforms as transforms
from torchvision.datasets.folder import pil_loader

# Input = namedtuple("Input", ["idx", "name", "shape", "mean", "std", "layout", "format", "resize_type",
#                              "padding_mode", "padding_value", "enable_aipp", "support"])


class Basehmexec(object, metaclass=abc.ABCMeta):
    """base hmexec"""

    def __init__(self, cfg: dict):
        """init"""
        self.cfg = cfg
        self.model = cfg["model"]
        self.quant = cfg["quant"]
        # model params
        self.framework = cfg["model"]["framework"]
        self.weight = cfg["model"]["weight"]
        self.inputs = cfg["model"]["inputs"]
        self.num_inputs = len(self.inputs)
        self.model_name = cfg["model"]["name"]
        self.quant_model_path = 'quant_' + self.model_name + '.onnx'
        # quant params
        self.preproc_module = self.quant["preproc_module"]
        self.preproc_class = self.quant["preproc_class"]
        # build params
        if "mode" in cfg["build"]:
            self.build_mode = cfg["build"]["mode"]
        else:
            self.build_mode = "AOT"
        self.target = cfg["build"]["target"]
        self.opt_level = cfg["build"].get("opt_level", 2)
        # other params
        self.cur_dir = os.path.abspath("./")
        self.model_dir = os.path.abspath(os.path.join(self.cfg["model"]["save_dir"], self.target))
        self.result_dir = os.path.abspath(os.path.join(self.model_dir, "result"))
        self.golden_data_path = os.path.abspath(os.path.join(self.result_dir, 'hmquant_' + self.model_name + '_with_act'))
        self.backend = "asic"
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
        self.iss_simu_span = 0
        self.layer_compare_span = 0
        self.iss_layerwise_dump_span = 0

    @staticmethod
    def set_env():
        raise NotImplementedError

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
            n, c, h, w = shape
            if _input["layout"] == "NHWC":
                n, h, w, c = shape

            if "dtype" not in _input:
                if _input["format"] == "None":
                    _input["dtype"] = "float32"
                else:
                    _input["dtype"] = "uint8"

            self.shape_dict[_input["name"]] = (n, c, h, w)
            self.dtype_dict[_input["name"]] = _input["dtype"]

            if not _input["mean"]:
                _input["mean"] = [0.0 for _ in range(c)]
            if not _input["std"]:
                _input["std"] = [1.0 for _ in range(c)]

    @property
    def has_custom_preprocess(self):
        return True if self.preproc_class else False

    def set_custom_preprocess(self):
        """检查是否存在自定义预处理
         1.多输入情况需要自定义
         2.默认预处理不能满足的情况
        """
        # 自定义预处理
        if self.preproc_class:
            m = importlib.import_module(self.preproc_module)
            if hasattr(m, self.preproc_class):
                # 实例化预处理对象
                self.preproc_class = getattr(m, self.preproc_class)(
                    self.inputs, self.quant["calib_num"], self.quant["data_dir"])
            else:
                logger.error("{}.py has no class named {}".format(
                    self.preproc_module, self.preproc_class))
                exit(-1)

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

    def get_data(self, name, dtype, shape, filepath=None, transform=None):
        """ 生成数据
        @param name: data name
        @param dtype: data type
        @param shape: data shape
        @param filepath: data file path
        @return: numpy
        """
        import torchvision.transforms as transforms
        from utils.transform import ToTensorNotNormal
        import torch
        
        if filepath:   # 指定输入数据
            logger.info("data[{}] will use file: {}".format(name, filepath))
            data = pil_loader(filepath)
            if transform:
                data = np.array(transform(data, shape))
        else:   # 未指定输入数据，生成随机数
            logger.warning("data[{}] will use random data".format(name))
            n, c, h, w = shape

            if dtype == "float32":
                data = np.random.rand(n, c, h, w).astype(dtype=dtype)   # 数值范围[0, 1)
            elif dtype == "float16":
                data = np.random.rand(n, c, h, w).astype(dtype=dtype)   # 数值范围[0, 1)
            elif dtype == "int16":
                data = np.random.randint(low=-(2**15), high=2**15-1, size=(n, c, h, w), dtype=dtype)
            elif dtype == "uint8":
                data = np.random.randint(low=0, high=255, size=(n, c, h, w), dtype=dtype)
            else:
                logger.error("Not support dtype -> {}".format(dtype))
                exit(-1)
        return data


    def get_datas(self, file_path=None, transform=None, to_tensor=False):
        """ 生成模型输入数据
        @param filepath:  外部指定数据
        @param force_float:  强制输出float数据
        @param force_cr:　是否强制使能CR
        @param force_random:  是否强制使用随机数据，主要用于生成量化数据
        @param to_file:
        @return:
        """
        # in_datas = OrderedDict()  # 保证输入顺序一致
        in_datas = {}
        for idx, _input in enumerate(self.inputs):
            
            name = _input["name"]
            format = _input["format"]
            dtype = _input["dtype"]
            shape = self.shape_dict[name]
            
            in_datas[name] = self.get_data(name, dtype, shape, file_path, transform)

            if to_tensor:
                in_datas[name] = torch.tensor(in_datas[name])

        return in_datas

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

    def save_compare_layer_outputs(self):
        """tvm float vs fixed """
        raise NotImplementedError

    @abc.abstractmethod
    def quantize(self):
        """relay quantize"""
        raise NotImplementedError

    @abc.abstractmethod
    def build(self):
        """relay build"""
        raise NotImplementedError

    def infer(self):
        """ inference on chip/sdk_iss """
        raise NotImplementedError

    def get_relay_mac(self):
        """get relay func MAC count"""
        raise NotImplementedError

    def get_device_type(self):
        """get op run device"""
        raise NotImplementedError

    def get_version(self):
        """get tytvm version"""
        raise NotImplementedError
