#!/usr/bin/env python3

from hmassist.models.detector import Detector
from hmassist.utils.preprocess import letterbox
from hmassist.utils.postprocess import non_max_suppression, scale_coords
# from hmassist.utils.box_utils import non_max_suppression, scale_coords
import numpy as np
import os
import cv2
import torch
import time
import torchvision

class YoloV3(Detector):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._iou_threshold = 0.45
        self._conf_threshold = 0.25

    # @staticmethod
    # def build_options():
    #     return {}

    def _postprocess(self, outputs, cv_image=None):
        assert len(outputs) == 3
        # add yolo process
        outputs = self.yolo_detect(outputs)
        outputs = non_max_suppression(outputs, self._conf_threshold, self._iou_threshold)
        outputs = outputs[0]  # bs=1
        outputs[:, :4] = scale_coords(self._input_size, outputs[:, :4], cv_image.shape).round()
        return outputs.numpy()

    def demo(self, img_path):
        if not os.path.exists(img_path):
            print("[error] The img path not exist -> {}".format(img_path))
            exit(-1)
        filename = os.path.basename(img_path)
        print("process: {}".format(img_path))

        save_results = "demo_results"
        if not os.path.exists(save_results):
            os.makedirs(save_results)

        cv_image = cv2.imread(img_path)
        if cv_image is None:
            print("[error] Failed to decode img by opencv -> {}".format(img_path))
            exit(-1)

        end2end_start = time.time()

        inputs = {self.inputs[0]["name"]: cv_image}
        inputs = self._preprocess(inputs)
        outputs = self.inference(inputs)
        boxes = self._postprocess(outputs, cv_image)

        end2end_cost = time.time() - end2end_start
        self._end2end_latency_ms += (end2end_cost * 1000)

        print("box num = {}".format(len(boxes)))
        for det in boxes:
            (x1, y1, x2, y2), conf, cls = list(map(int, det[0:4])), det[4], int(det[5])
            cv2.rectangle(cv_image, (x1, y1), (x2, y2), (0, 0, 255), 1, 8)
            print("x1:{}, y1:{}, x2:{}, y2:{}, conf:{:.6f}, cls:{}".format(x1, y1, x2, y2, conf, int(cls)))
        cv2.imwrite(os.path.join(save_results, filename), cv_image)

    def yolo_detect(self, feats):
        # in.shape = out.shape: 1x3x80x80x85 1x3x40x40x85 1x3x20x20x85
        output = []

        for i, name in enumerate(feats):
            data = torch.tensor(feats[name])
            assert len(data.shape) == 5

            bs, channel, ny, nx, no = data.shape
            grid, anchor_grid = self._make_grid(nx, ny, i)

            data[..., 0:2] = (data[..., 0:2] * 2 - 0.5 + grid) * self.stride[i]  # xy
            data[..., 2:4] = (data[..., 2:4] * 2) ** 2 * anchor_grid  # wh


            output.append(data.reshape(bs, -1, no))

        return torch.concat(output, dim=1)

    def _make_grid(self, nx=20, ny=20, i=0):
        anchors = torch.tensor([
            [10, 13, 16, 30, 33, 23],
            [30, 61, 62, 45, 59, 119],
            [116, 90, 156, 198, 373, 326],
        ])

        self.stride = torch.tensor([8, 16, 32]).view(-1, 1, 1)
        anchors = anchors.view(3, 3, 2)
        anchors = anchors / self.stride

        yv, xv = torch.meshgrid([torch.arange(ny), torch.arange(nx)])
        grid = torch.stack((xv, yv), 2).expand((1, 3, ny, nx, 2)).float()
        anchor_grid = (anchors[i] * self.stride[i]).view((1, 3, 1, 1, 2)).expand((1, 3, ny, nx, 2)).float()
        return grid, anchor_grid