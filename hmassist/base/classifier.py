#!/usr/bin/env python3

import time
from abc import ABC

import numpy as np
import os
import cv2
import tqdm

from .base_model import BaseModel
from utils import logger
from utils.preprocess import default_preprocess
from torchvision.datasets.folder import pil_loader

class Classifier(BaseModel, ABC):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        n, c, h, w = self.inputs[0]["shape"]
        if self.inputs[0]["layout"] == "NHWC":
            n, h, w, c = self.inputs[0]["shape"]
        self._input_size = (w, h)
        self._infer_latency_ms = 0
        self._end2end_latency_ms = 0

    def _preprocess(self, img):
        return default_preprocess()

    def _postprocess(self, outputs, cv_image=None):
        if len(outputs) != 1:
            logger.error("only support signal output, please check")
            exit(-1)
        outputs = outputs[0]
        bs = outputs.shape[0]
        if bs != 1:
            logger.error("only support bs=1, please check")
            exit(-1)
        return outputs

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

    def load(self):
        self.infer.load()

    def inference(self, input_data):
        infer_start = time.time()
        outputs = self.infer.infer(input_data)
        infer_cost = time.time() - infer_start
        self._infer_latency_ms += (infer_cost * 1000)
        self.total += 1
        return outputs

    def evaluate(self):
        """ top-k
        """
        if not self.dataset:
            logger.error("The dataset is null")
            exit(-1)

        img_paths, labels = self.dataset.get_datas(num=self.test_num)

        k = 5
        top1, top5 = 0, 0
        total_num = len(img_paths)
        for idx, img_path in enumerate(tqdm.tqdm(img_paths)):
            img = pil_loader(img_path)
            if img is None:
                logger.warning("Failed to load image -> {}".format(img_path))
                continue
            input_data = self._preprocess(img)
            output_data = self.inference(input_data)
            output_data = self._postprocess(output_data, img)
            idxes = np.argsort(-output_data, axis=1, kind="quicksort").flatten()[0:k]  # 降序
            logger.info("pred = {}, gt = {}".format(idxes, labels[idx]))
            if labels[idx] == idxes[0]:
                top1 += 1
                top5 += 1
                continue
            if labels[idx] in idxes:
                top5 += 1
        top1, top5 = float(top1)/total_num, float(top5)/total_num
        return {
            "input_size": "{}x{}x{}x{}".format(1, 3, self._input_size[1], self._input_size[0]),
            "dataset": self.dataset.dataset_name,
            "num": total_num,
            "top1": "{:.6f}".format(top1),
            "top5": "{:.6f}".format(top5),
            "latency": "{:.6f}".format(self.ave_latency_ms)
        }

    def demo(self, img_path):
        if not os.path.exists(img_path):
            logger.error("The img path not exist -> {}".format(img_path))
            exit(-1)
        logger.info("process: {}".format(img_path))
        img = pil_loader(img_path)
        if img is None:
            logger.error("Failed to load image -> {}".format(img_path))
            exit(-1)
        end2end_start = time.time()

        input_data = self._preprocess(img)
        output_data = self.inference(input_data)
        output_data = self._postprocess(output_data, img)
        max_idx = np.argmax(output_data, axis=1).flatten()[0]
        max_prob = output_data[0][max_idx].flatten()[0]
        
        end2end_cost = time.time() - end2end_start
        self._end2end_latency_ms += (end2end_cost * 1000)
        logger.info("predict cls = {}, prob = {:.6f}".format(max_idx, max_prob))
