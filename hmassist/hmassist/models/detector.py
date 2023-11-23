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
    detection_txt2json,
    detections2txt,
)
from ..utils import logger

class Detector(BaseModel):
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
        from torchvision.datasets.folder import pil_loader
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
            image, _, _ = letterbox(image, self._input_size, stride=64, auto=False)  # HWC
            image = np.transpose(image, (2, 0, 1)).astype(np.float32)  # CHW
            datas[name] = image
        return datas

    def _postprocess(self, outputs, image=None):
        if len(outputs) == 4 or len(outputs) == 1:
            outputs = outputs[0]
        # add yolo process
        outputs = self.yolo_detect(outputs)
        # outputs = torch.from_numpy(outputs)
        outputs = non_max_suppression(outputs, self._conf_threshold, self._iou_threshold)
        outputs = outputs[0]  # bs=1
        outputs[:, :4] = scale_coords(self._input_size, outputs[:, :4], image.shape).round()
        return outputs.numpy()

    def evaluate(self):
        if not self.dataset:
            logger.error("The dataset is null")
            exit(-1)

        self._iou_threshold = 0.65
        self._conf_threshold = 0.01

        img_paths = self.dataset.get_datas(num=self.test_num)

        save_results = "results_{}_{}".format(self.backend, self.dtype)
        if not os.path.exists(save_results):
            os.makedirs(save_results)

        for idx, img_path in enumerate(tqdm.tqdm(img_paths)):
            basename = os.path.basename(img_path)
            filename, ext = os.path.splitext(basename)
            label_path = os.path.join(save_results, "{}.txt".format(filename))
            if os.path.exists(label_path):
                continue
            cv_image = cv2.imread(img_path)
            if cv_image is None:
                logger.warning("Failed to decode img by opencv -> {}".format(img_path))
                continue
            detections = self.inference(cv_image)
            detections2txt(detections, label_path)
        pred_json = "pred.json"
        detection_txt2json(save_results, pred_json)
        _map, map50 = coco_eval(pred_json, self.dataset.annotations_file, self.dataset.image_ids)
        return {
            "input_size": "{}x{}x{}x{}".format(1, 3, self._input_size[1], self._input_size[0]),
            "dataset": self.dataset.dataset_name,
            "num": len(img_paths),
            "map": "{:.6f}".format(_map),
            "map50": "{:.6f}".format(map50),
            "latency": "{:.6f}".format(self.ave_latency_ms)
        }
