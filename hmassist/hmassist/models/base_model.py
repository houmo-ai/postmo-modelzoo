#!/usr/bin/env python3

import abc
import time
import os
import numpy as np
import cv2
from ..utils import logger


class BaseModel(object, metaclass=abc.ABCMeta):
    """模型描述基类，提供2个功能，demo和eval
    """
    def __init__(self, **kwargs):
        """"""
        self.executor = kwargs["executor"]
        self.inputs = kwargs["inputs"]   # multi-input
        self.dataset = kwargs["dataset"]
        self.test_num = kwargs["test_num"]
        self.target = kwargs["target"]
        # self.dtype = kwargs["dtype"]
        self.backend = kwargs["backend"]

        self.total = 0
        self.time_span = 0
        self._infer_latency_ms = 0
        self._end2end_latency_ms = 0
        # self.executor.backend = self.backend

        if self.inputs[0]["layout"] == "NCHW":
            n, c, h, w = self.inputs[0]["shape"]
        elif self.inputs[0]["layout"] == "NHWC":
            n, h, w, c = self.inputs[0]["shape"]
        if "image" in self.inputs[0] and "size" in self.inputs[0]["image"] \
            and self.inputs[0]["image"]["size"]:
            self._input_size = self.inputs[0]["image"]["size"]
        else:
            self._input_size = [h, w]

    def load(self):
        """加载so模型
        :param model_path: 模型目录
        :return:
        """
        self.executor.load()

    @staticmethod
    def build_config():
        logger.warning("can not find hm_model.build_config, use BaseModel.build_config")
        return None

    def get_input_datas(self, filedir, filename):
        # logger.warning("can not find hm_model.get_input_datas, use BaseModel.get_input_datas")
        if len(self.inputs) > 1:
            logger.error(f"default only support 1 input, now is {len(self.inputs)}")
        data = cv2.imread(os.path.join(filedir, filename))
        in_datas = {self.inputs[0]["name"]: data}
        return self._preprocess(in_datas)

    def _preprocess(self, inputs, resize=False):
        """_preprocess
        :param inputs: model inputs dict
        :return: numpy dict, CHW
        """
        logger.warning("can not find hm_model._preprocess, use BaseModel._preprocess")

        datas = {}
        for name, data in inputs.items():
            # from ..utils import utils
            # image = utils.to_pillow(data)
            data = cv2.resize(data, (self._input_size[1], self._input_size[0]))
            datas[name] = np.transpose(data, (2, 0, 1)).astype(np.float32)  # CHW
        return datas

    def _postprocess(self, outputs, image=None):
        """
        :param outputs: model outputs dict
        :param img: origin image
        :return: numpy dict
        """
        logger.warning("can not find hm_model._postprocess, use BaseModel._postprocess")
        return outputs

    def inference(self, inputs):
        inputs = self.executor._preprocess(inputs)
        start = time.time()
        outputs = self.executor.infer(inputs)
        cost = time.time() - start
        self._infer_latency_ms += (cost * 1000)
        self.total += 1
        return outputs

    def evaluate(self):
        """模型指标评估"""
        logger.error("can not find hm_model.evaluate, exit")
        exit(-1)

    def demo(self, inputs):
        """
        模型demo
        :param img_path: 图片路径
        :return:
        """
        logger.error("can not find hm_model.demo, exit")
        exit(-1)

    @property
    def ave_latency_ms(self):
        if self.total == 0:
            return 0
        return self._infer_latency_ms / self.total

    @property
    def end2end_latency_ms(self):
        if self.total == 0:
            return 0
        return self._end2end_latency_ms / self.total
