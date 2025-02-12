#!/usr/bin/env python3

from hmassist.models.detector import Detector
from hmassist.utils.preprocess import letterbox
from hmassist.utils.postprocess import non_max_suppression2, scale_coords
from ultralytics.utils.plotting import Annotator, colors
from hmassist.datasets.coco import coco80_labels

import numpy as np
import os
import cv2
import torch
import time


class YoloV8(Detector):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def _postprocess(self, outputs, cv_image):
        if len(outputs) == 1:
            output = torch.from_numpy(list(outputs.values())[0])
        elif len(outputs) == 2:
            output = self._decode(outputs)
        else:
            print("outputs num", len(outputs), "not supported.")
            assert(0)
        output = non_max_suppression2(output, self._conf_threshold, self._iou_threshold, nc=self._nc)
        output = output[0]
        output[:, :4] = scale_coords(self._input_size, output[:, :4], cv_image.shape).round()

        return output.numpy()

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
            from ultralytics.utils.plotting import Annotator, colors
            from hmassist.datasets.coco import coco80_labels
            label = coco80_labels[cls] + " {:.2f}".format(conf)
            annotator = Annotator(cv_image, line_width=2, example=str(cls))
            annotator.box_label((x1, y1, x2, y2), label, color=colors(cls, True))
            print("x1:{}, y1:{}, x2:{}, y2:{}, conf:{:.6f}, cls:{}".format(x1, y1, x2, y2, conf, cls), flush=True)
        save_path = os.path.join(save_results, filename)
        cv2.imwrite(save_path, cv_image)
        print("demo results saved to", save_path)

    def _decode(self, outputs):
        if "/model.22/Sigmoid_output_0" in outputs:
            output_name_cls = "/model.22/Sigmoid_output_0"
            output_name_box = "/model.22/dfl/Reshape_1_output_0"
        else:
            output_name_cls = "_model.22_Sigmoid_output_0"
            output_name_box = "_model.22_dfl_Reshape_1_output_0"
        cls_data = torch.from_numpy(outputs[output_name_cls])  # 1, 80, 8400
        box_data = torch.from_numpy(outputs[output_name_box])  # 1, 4, 8400

        # decode box
        box_data = box_data.permute(0, 2, 1)  # 1, 8400, 4
        box_data[:, :, 0:2] = self._anchor_points - box_data[:, :, 0:2]
        box_data[:, :, 2:4] = self._anchor_points + box_data[:, :, 2:4]
        box_data_xy = (box_data[:, :, 0:2] + box_data[:, :, 2:4]) * 0.5
        box_data_wh = box_data[:, :, 2:4] - box_data[:, :, 0:2]
        box_data[:, :, 0:2] = box_data_xy
        box_data[:, :, 2:4] = box_data_wh
        box_data *= self._stride_tensor
        box_data = box_data.permute(0, 2, 1).contiguous()  # 1, 4, 8400

        # decode kpt
        return torch.cat([box_data, cls_data], dim=1)   # 1, 84, 8400
