#!/usr/bin/env python3

import abc
import time
from utils import logger

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

        # self.use_norm = True if self.dtype == "fp32" else False

        self.total = 0
        self.time_span = 0
        self._ave_latency_ms = 0
        self._end2end_latency_ms = 0

        # self.executor.backend = self.backend

    def load(self, model_path):
        """加载so模型
        :param model_path: 模型目录
        :return:
        """
        self.executor.load(model_path)

    @staticmethod
    def data_transform(x, shape):
        """data transform
        :param x: pillow
        :param shape: tuple
        :return: torch.Tensor(NCHW, float32)
        """
        import torchvision.transforms as transforms
        from utils.transform import ToTensorNotNormal
        import torch
        logger.warning("can not find hm_model.data_transform, use default")
        def unsqueeze(x):
            return torch.unsqueeze(x, 0)
        _, _, h, w = shape
        transform = transforms.Compose(
            [
                transforms.Resize((h, w)), transforms.CenterCrop((h, w)),
                ToTensorNotNormal(), unsqueeze,
            ],
        )
        return transform(x)

    @staticmethod
    def build_config():
        logger.warning("can not find hm_model.build_config, use default")
        return None

    def _preprocess(self, data):
        """data transform
        :param data:
        :return:
        """
        pass

    def _postprocess(self, outputs, img=None):
        """
        :param outputs: model outputs dict
        :param img: origin image
        :return:
        """
        pass

    def inference(self, input_data):
        start = time.time()
        outputs = self.executor.infer(input_data)
        cost = time.time() - start
        self._ave_latency_ms += (cost * 1000)
        self.total += 1
        return outputs

    def evaluate(self):
        """模型指标评估"""
        pass

    def demo(self, img):
        """
        模型demo
        :param img_path: 图片路径
        :return:
        """
        pass

    @property
    def ave_latency_ms(self):
        if self.total == 0:
            return 0
        return self._ave_latency_ms / self.total

    @property
    def end2end_latency_ms(self):
        if self.total == 0:
            return 0
        return self._end2end_latency_ms / self.total
