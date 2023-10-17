#!/usr/bin/env python3

import abc
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
        self.infer = self.executor
        self.target = kwargs["target"]
        # self.dtype = kwargs["dtype"]
        self.backend = kwargs["backend"]

        # self.use_norm = True if self.dtype == "fp32" else False

        self.total = 0
        self.time_span = 0

        # self.infer.backend = self.backend

    def load(self, model_path):
        """加载so模型
        :param model_path: 模型目录
        :return:
        """
        self.infer.load(model_path)

    @staticmethod
    def data_transform(x, shape):
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

    def _preprocess(self, cv_image):
        """内部预处理调用
        :param cv_image: opencv image
        :return:
        """
        pass

    def _postprocess(self, outputs, img=None):
        """内部后处理调用
        :param outputs: 模型推理输出
        :param cv_image: 原图像
        :return:
        """
        pass

    def inference(self, img):
        """推理接口，目前仅支持batch1
        :param cv_image: opencv image
        :return:
        """
        pass

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
        """模型芯片内部推理时间, 不是严格准确，仅供参考"""
        pass

    @property
    def end2end_latency_ms(self):
        """python推理时间, 包括数据传入传出、预处理、后处理时间"""
        pass
