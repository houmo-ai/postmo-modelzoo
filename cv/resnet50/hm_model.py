#!/usr/bin/env python3

from base.classifier import Classifier
from utils import logger
from utils.postprocess import softmax
import numpy as np

class Resnet50(Classifier):

    @staticmethod
    def data_transform(x, shape):
        import torchvision.transforms as transforms
        from utils.transform import ToTensorNotNormal
        import torch
        def unsqueeze(x):
            return torch.unsqueeze(x, 0)
        transform = transforms.Compose(
            [
                transforms.Resize(256), transforms.CenterCrop(224),
                ToTensorNotNormal(), unsqueeze,
            ],
        )
        return transform(x)
    
    @staticmethod
    def build_config():
        return {
            'tcim.fuse_strategy': 0, #1
            'tcim.gen_intrinsic': 1, #0
            'tcim.codegen_pic': False,
            'tcim.use_convadd': False,
            'tcim.sync_strategy': 1,
            'tcim.for_benchmark': True,
            # 'tcim.multi_stream': 4,
        }

    def _preprocess(self, img):
        import torchvision.transforms as transforms
        from utils.transform import RGB2YUV
        from utils.transform import ToTensorNotNormal
        transform = transforms.Compose(
            [
                transforms.Resize(256), transforms.CenterCrop(224),
                ToTensorNotNormal(), RGB2YUV(), 
            ],
        )

        img = transform(img)
        img = np.expand_dims(img.numpy().astype(np.uint8), 0)
        return img

    def _postprocess(self, outputs, img=None):
        if len(outputs) != 1:
            logger.error("only support signal output, please check")
            exit(-1)
        # outputs = outputs[0]  # [bs, num_cls] or [bs, num_cls, 1, 1]
        for _, name in enumerate(outputs):
            bs = outputs[name].shape[0]
            if bs != 1:
                logger.error("only support bs=1, please check")
                exit(-1)
            out_data = softmax(outputs[name])
        return out_data
