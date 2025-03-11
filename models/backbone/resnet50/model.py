#!/usr/bin/env python3

from hmassist.models.classifier import Classifier
from hmassist.utils.postprocess import softmax
from hmassist.utils.preprocess import centercrop
import numpy as np
import cv2

class Resnet50(Classifier):

    def build_options(self):
        return {}

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
            data = cv2.resize(data, (w, h))  # HWC uint8
            data = cv2.cvtColor(data, cv2.COLOR_BGR2RGB)
            data = centercrop(data, (crop_size, crop_size))
            data = np.transpose(data, (2, 0, 1))  # CHW uint8
            datas[name] = np.expand_dims(data, axis=0)  # NCHW uint8
        return datas
