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

        batch = self.executor.model_input_batch * self.executor.batch
        img_paths, labels = self.dataset.get_datas(num=self.test_num)

        def topk(outputs, k, top1, top5, valid_len=None):
            for _, output in outputs.items():
                if valid_len is None:
                    valid_len = output.shape[0]
                idxes = np.argsort(-output, axis=1, kind="quicksort")[:, 0:k]  # 降序
                for bs in range(valid_len):
                    logger.debug("image:{}, pred = {}, gt = {}".format(img_paths[bs], idxes[bs, 0], labels[bs]))
                    if labels[bs] == idxes[bs, 0]:
                        top1 += 1
                        top5 += 1
                        continue
                    if labels[bs] in idxes[bs, :]:
                        top5 += 1
            return top1, top5
            
        k = 5
        top1, top5 = 0, 0
        total_num = len(img_paths)
        batch_datas = []
        end2end_start = time.time()
        for idx, img_path in enumerate(tqdm.tqdm(img_paths)):
            img = cv2.imread(img_path)
            if img is None:
                logger.warning("Failed to load image -> {}".format(img_path))
                continue
            inputs = {self.inputs[0]["name"]: img}
            inputs = self._preprocess(inputs)
            batch_datas.append(inputs)
            if len(batch_datas) != batch:
                continue
            outputs = self.inference(batch_datas)
            outputs = self._postprocess(outputs)
            batch_datas.clear()
            top1, top5 = topk(outputs, k, top1, top5)
     
        # 不足1batch
        if len(batch_datas) != 0:
            valid_len = len(batch_datas)
            for _ in range(batch - valid_len):
                batch_datas.append(batch_datas[-1])
            outputs = self.inference(batch_datas)
            outputs = self._postprocess(outputs)
            top1, top5 = topk(outputs, k, top1, top5, valid_len)

        end2end_cost = time.time() - end2end_start
        self._end2end_latency_ms += (end2end_cost * 1000)            
        top1, top5 = float(top1)/total_num, float(top5)/total_num
        _, C, H, W = self.inputs[0]["shape"]
        return {
            "shape": [batch, C, H, W],
            "dataset": self.dataset.dataset_name,
            "test_num": total_num,
            "accuracy": {"top1": top1, "top5": top5},
            "latency": self.ave_latency_ms
        }

    def demo(self, file_list: list):
        valid_len = len(file_list)
        batch = self.executor.model_input_batch * self.executor.batch
        assert isinstance(file_list, list)
        assert len(self.inputs) == 1
        input_name = self.inputs[0]["name"]
        batch_datas = []
        end2end_start = time.time()
        for img_path in file_list:
            if not os.path.exists(img_path):
                logger.error("The img path not exist -> {}".format(img_path))
                exit(-1)
            logger.info("process: {}".format(img_path))
            img = cv2.imread(img_path)
            if img is None:
                logger.error("Failed to load image -> {}".format(img_path))
                exit(-1)
            inputs = {input_name: img}
            inputs = self._preprocess(inputs)
            batch_datas.append(inputs)

        # 不足1batch，最后1份数据填充
        if len(batch_datas) != 0 and valid_len < batch:
            for _ in range(batch - valid_len):
                batch_datas.append(batch_datas[-1])
                
        outputs = self.inference(batch_datas)
        outputs = self._postprocess(outputs, img)
        for _, data in outputs.items():
            max_idx = np.argmax(data, axis=1).flatten()  # bs, 1
            for bs, idx in enumerate(max_idx):
                if bs == valid_len:
                    break
                max_prob = data[bs, idx]
                logger.info("predict cls = {}, prob = {:.6f}".format(idx, max_prob))

        end2end_cost = time.time() - end2end_start
        self._end2end_latency_ms += (end2end_cost * 1000)
        
