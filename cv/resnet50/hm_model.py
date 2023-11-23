#!/usr/bin/env python3

from hmassist.models.classifier import Classifier
from hmassist.utils.postprocess import softmax
from hmassist.utils.preprocess import centercrop
import numpy as np
import cv2

class Resnet50(Classifier):

    @staticmethod
    def build_config():
        return {
            'tcim.fuse_strategy': 0,
            'tcim.gen_intrinsic': 2,
            'tcim.codegen_pic': False,
            'tcim.sync_strategy': 1,
            'tcim.core_num': 1,
            'tcim.for_benchmark': True,
            # 'tcim.multi_stream': 4,
        }

    def _preprocess(self, inputs):
        assert(len(inputs) == 1)
        datas = {}
        resize_size = 256
        crop_size = 224
        for name, data in inputs.items():
            if data.shape[1] > data.shape[0]:
                h = resize_size
                w = round(data.shape[1] / data.shape[0] * resize_size)
            else:
                h = round(data.shape[0] / data.shape[1] * resize_size)
                w = resize_size
            data = cv2.resize(data, (w, h))  # HWC
            data = centercrop(data, (crop_size, crop_size))
            datas[name] = np.transpose(data, (2, 0, 1)).astype(np.float32)  # CHW
        return datas
