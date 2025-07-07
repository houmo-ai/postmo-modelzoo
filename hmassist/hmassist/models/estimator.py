#!/usr/bin/env python3

import time
import os
import cv2
import torch
import tqdm
import numpy as np

from ..models.base_model import BaseModel
from ..utils.postprocess import (
    non_max_suppression,
    scale_coords,
)
from ..utils.metrics import (
    coco_eval,
    estimations2txt,
    estimation_txt2json
)
from ..utils import logger

class Estimator(BaseModel):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._iou_threshold = 0.45
        self._conf_threshold = 0.25

    def set_iou_threshold(self, iou_threshold=0.45):
        self._iou_threshold = iou_threshold

    def set_conf_threshold(self, conf_threshold=0.25):
        self._conf_threshold = conf_threshold

    def get_input_datas(self, filedir, filename):
        if len(self.inputs) > 1:
            logger.error(f"default only support 1 input, now is {len(self.inputs)}")
        # from torchvision.datasets.folder import pil_loader
        # data = pil_loader(os.path.join(filedir, filename))
        data = cv2.imread(os.path.join(filedir, filename))
        inputs = {self.inputs[0]["name"]: data}
        return self._preprocess(inputs)

    def _preprocess(self, inputs):
        datas = {}
        for name, data in inputs.items():
            from ..utils import utils
            from ..utils.box_utils import letterbox
            image = utils.to_opencv(data)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image, _, _ = letterbox(image, self._input_size, stride=64, auto=False)  # HWC
            image = np.transpose(image, (2, 0, 1))  # CHW .astype(np.float32)
            image = np.expand_dims(image, axis=0)  # NCHW
            datas[name] = image
        return datas

    def evaluate(self):
        if not self.dataset:
            logger.error("The dataset is null")
            exit(-1)

        self._iou_threshold = 0.25
        self._conf_threshold = 0.45

        batch = self.executor.model_input_batch * self.executor.batch
        img_paths = self.dataset.get_datas(num=self.test_num)

        save_results = os.path.join("output", self.target, "result/eval_results")
        if os.path.exists(save_results):
            import shutil
            shutil.rmtree(save_results)  # 禁用断点续测
        os.makedirs(save_results)

        label_paths = []
        batch_datas = []
        cv_images = []
        end2end_start = time.time()
        for idx, img_path in enumerate(tqdm.tqdm(img_paths)):
            basename = os.path.basename(img_path)
            filename, ext = os.path.splitext(basename)
            label_path = os.path.join(save_results, "{}.txt".format(filename))
            img = cv2.imread(img_path)
            if img is None:
                logger.warning("Failed to decode img by opencv -> {}".format(img_path))
                continue
            inputs = {self.inputs[0]["name"]: img}
            inputs = self._preprocess(inputs)
            batch_datas.append(inputs)
            label_paths.append(label_path)
            cv_images.append(img)
            if len(batch_datas) != batch:
                continue
            outputs = self.inference(batch_datas)  # {output_name: output}
            estimations = self._postprocess(outputs, cv_images)
            if(len(estimations) != 0):
                for bs in range(batch):
                    estimations2txt(estimations[bs], label_paths[bs])
            batch_datas.clear()
            label_paths.clear()
            cv_images.clear()
        # 不足1batch
        if len(batch_datas) != 0:
            valid_len = len(batch_datas)
            for _ in range(batch - valid_len):
                batch_datas.append(batch_datas[-1])
                cv_images.append(cv_images[-1])
            outputs = self.inference(batch_datas)
            detections = self._postprocess(outputs, cv_images)
            if(len(estimations) != 0):
                for idx in range(valid_len):
                    estimations2txt(detections[idx], label_paths[idx])

        end2end_cost = time.time() - end2end_start
        self._end2end_latency_ms += (end2end_cost * 1000)
        pred_json = "pred.json"
        estimation_txt2json(save_results, pred_json)
        _map, map50 = coco_eval(pred_json, self.dataset.annotations_kpt, self.dataset.image_ids, iou_type="keypoints")
        _, C, H, W = self.inputs[0]["shape"]
        return {
            "shape": [batch, C, H, W],
            "dataset": self.dataset.dataset_name,
            "test_num": len(img_paths),
            "accuracy": {"map": float(_map), "map50": float(map50)},
            "latency": self.ave_latency_ms
        }
