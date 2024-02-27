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
        self.dataset = kwargs["dataset"]
        self.model = self.executor.model
        self.target = self.executor.target
        self.framework = self.model["framework"]
        self.inputs = self.model["inputs"]
        self.name = self.model["name"]
        self.compare_dir = os.path.abspath(os.path.join(self.model["save_dir"], self.framework, "result"))

        self.total = 0
        self.time_span = 0
        self._infer_latency_ms = 0
        self._end2end_latency_ms = 0

        for input in self.inputs:
            if input["layout"] == "ND":
                self.input_shape = input["shape"]
                break
            if input["layout"] == "NCHW":
                n, c, h, w = input["shape"]
            elif input["layout"] == "NHWC":
                n, h, w, c = input["shape"]
            size = None
            if "image" in input:
                size = input["image"].get("size", [h, w])
            if size:
                self.input_shape = [n, c, size[0], size[1]]
                self._input_size = size
            else:
                self.input_shape = [n, c, h, w]
                self._input_size = [h, w]

    def load(self):
        """加载so模型
        :param model_path: 模型目录
        :return:
        """
        self.executor.load()

    def build_options(self):
        logger.warning("can not find hm_model.build_options, use BaseModel.build_options")
        return None

    def get_input_datas(self, filedir, filename):
        # logger.warning("can not find hm_model.get_input_datas, use BaseModel.get_input_datas")
        if len(self.inputs) > 1:
            logger.error(f"default only support 1 input, now is {len(self.inputs)}")
        data = cv2.imread(os.path.join(filedir, filename))
        in_datas = {self.inputs[0]["name"]: data}
        return self._preprocess(in_datas)

    def _preprocess(self, inputs):
        """_preprocess
        :param inputs: model inputs dict
        :return: numpy dict, CHW
        """
        logger.warning("can not find hm_model._preprocess, use BaseModel._preprocess")

        datas = {}
        for name, data in inputs.items():
            from ..utils import utils
            data = utils.to_opencv(data)
            # convert to 
            data = cv2.cvtColor(data, cv2.COLOR_BGR2RGB)
            data = cv2.resize(data, (self._input_size[1], self._input_size[0]))
            data = np.transpose(data, (2, 0, 1))  # CHW uint8
            datas[name] = np.expand_dims(data, axis=0)  # NCHW uint8
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
