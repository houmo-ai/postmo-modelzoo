#!/usr/bin/env python3

import time
from abc import ABC

import numpy as np
import os
import cv2
import tqdm

from ..models.base_model import BaseModel
from ..utils import logger, utils

class Classifier(BaseModel, ABC):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def get_input_datas(self, filedir, filename):
        if len(self.inputs) > 1:
            logger.error(f"default only support 1 input, now is {len(self.inputs)}")
        data = cv2.imread(os.path.join(filedir, filename))
        inputs = {self.inputs[0]["name"]: data}
        return self._preprocess(inputs)

    def _preprocess(self, inputs):
        datas = {}
        for name, data in inputs.items():
            data = utils.to_opencv(data)
            data = cv2.cvtColor(data, cv2.COLOR_BGR2RGB)
            data = cv2.resize(data, (self._input_size[1], self._input_size[0]))  # HWC uint8
            data = np.transpose(data, (2, 0, 1))  # CHW uint8
            datas[name] = np.expand_dims(data, axis=0)  # NCHW uint8
        return datas

    def _postprocess(self, outputs, image=None):
        if len(outputs) != 1:
            print("only support signal output, please check")
            exit(-1)
        datas = {}
        for name, data in outputs.items():
            bs = data.shape[0]
            if bs != 1:
                print("only support bs=1, please check")
                exit(-1)
            from hmassist.utils.postprocess import softmax
            datas[name] = softmax(data)
        return datas

    def load(self):
        self.executor.load()

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
            img = cv2.imread(img_path)
            if img is None:
                logger.warning("Failed to load image -> {}".format(img_path))
                continue

            end2end_start = time.time()

            inputs = {self.inputs[0]["name"]: img}
            inputs = self._preprocess(inputs)
            outputs = self.inference(inputs)
            outputs = self._postprocess(outputs, img)

            end2end_cost = time.time() - end2end_start
            self._end2end_latency_ms += (end2end_cost * 1000)

            for name, output in outputs.items():
                idxes = np.argsort(-output, axis=1, kind="quicksort").flatten()[0:k]  # 降序
                logger.debug("image:{}, pred = {}, gt = {}".format(img_path, idxes, labels[idx]))
                if labels[idx] == idxes[0]:
                    top1 += 1
                    top5 += 1
                    continue
                if labels[idx] in idxes:
                    top5 += 1
        top1, top5 = float(top1)/total_num, float(top5)/total_num
        return {
            "shape": [self.inputs[0]["shape"]],
            "dataset": self.dataset.dataset_name,
            "test_num": total_num,
            "accuracy": {"top1": top1, "top5": top5},
            "latency": self.ave_latency_ms
        }

    def demo(self, img_path):
        if not os.path.exists(img_path):
            logger.error("The img path not exist -> {}".format(img_path))
            exit(-1)
        logger.info("process: {}".format(img_path))
        img = cv2.imread(img_path)
        end2end_start = time.time()
        inputs = {self.inputs[0]["name"]: img}
        inputs = self._preprocess(inputs)

        # from torchvision.datasets.folder import pil_loader
        # data1 = pil_loader(img_path)
        # import torchvision.transforms as transforms
        # from hmassist.utils.transform import RGB2YUV
        # from hmassist.utils.transform import ToTensorNotNormal
        # transform = transforms.Compose(
        #     [
        #         transforms.Resize(256), transforms.CenterCrop(224),
        #         ToTensorNotNormal(), 
        #         # RGB2YUV(),
        #     ],
        # )
        # a = transform(data1)
        # a = np.expand_dims(a.numpy().astype(np.uint8), 0)
        # datas = {}
        # for input in self.inputs:
        #     datas[input["name"]] = a

        if img is None:
            logger.error("Failed to load image -> {}".format(img_path))
            exit(-1)

        outputs = self.inference(inputs)
        outputs = self._postprocess(outputs, img)
        for name, data in outputs.items():
            max_idx = np.argmax(data, axis=1).flatten()[0]
            max_prob = data.flatten()[max_idx]

        end2end_cost = time.time() - end2end_start
        self._end2end_latency_ms += (end2end_cost * 1000)
        logger.info("predict cls = {}, prob = {:.6f}".format(max_idx, max_prob))
