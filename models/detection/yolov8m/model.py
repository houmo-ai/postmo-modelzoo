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

    def _postprocess(self, outputs, cv_images):
        if len(outputs) == 1:
            output = torch.from_numpy(list(outputs.values())[0])
        elif len(outputs) == 2:
            output = self._decode(outputs)
        else:
            print("outputs num", len(outputs), "not supported.")
            assert(0)
        outputs = non_max_suppression2(output, self._conf_threshold, self._iou_threshold, nc=self._nc)
        batch = len(outputs)
        for idx in range(batch):
            outputs[idx][:, :4] = scale_coords(self._input_size, outputs[idx][:, :4], cv_images[idx].shape).round()
            outputs[idx] = outputs[idx].numpy()
        return outputs

    def demo(self, img_paths: list):
        save_results = "demo_results"
        if not os.path.exists(save_results):
            os.makedirs(save_results)
        
        def show_results(cv_images, detections, filenames, valid_len=None):
            if valid_len is None:
                valid_len = len(cv_images)
            for idx, boxes in enumerate(detections):
                if idx == valid_len:
                    break
                print("box num = {}".format(len(boxes)))
                for det in boxes:
                    (x1, y1, x2, y2), conf, cls = list(map(int, det[0:4])), det[4], int(det[5])
                    label = coco80_labels[cls] + " {:.2f}".format(conf)
                    annotator = Annotator(cv_images[idx], line_width=2, example=str(cls))
                    annotator.box_label((x1, y1, x2, y2), label, color=colors(cls, True))
                    print("x1:{}, y1:{}, x2:{}, y2:{}, conf:{:.6f}, cls:{}".format(x1, y1, x2, y2, conf, int(cls)), flush=True)
                save_path = os.path.join(save_results, filenames[idx])
                cv2.imwrite(save_path, cv_images[idx])
                print("demo results saved to", save_path)
                
        batch = self.executor.model_input_batch * self.executor.batch
        batch_datas = []
        cv_images = []
        filenames = []
        
        end2end_start = time.time()
        for img_path in img_paths: 
            if not os.path.exists(img_path):
                print("[error] The img path not exist -> {}".format(img_path))
                exit(-1)
            filename = os.path.basename(img_path)
            print("process: {}".format(img_path))
            cv_image = cv2.imread(img_path)
            if cv_image is None:
                print("[error] Failed to decode img by opencv -> {}".format(img_path))
                exit(-1)

            inputs = {self.inputs[0]["name"]: cv_image}
            inputs = self._preprocess(inputs)
            batch_datas.append(inputs)
            cv_images.append(cv_image)
            filenames.append(filename)
            if len(batch_datas) < batch:
                continue
            outputs = self.inference(batch_datas)
            boxes = self._postprocess(outputs, cv_images)
            show_results(cv_images, boxes, filenames)
            cv_images.clear()
            batch_datas.clear()
            filenames.clear()

        # 不足1batch
        if len(batch_datas) != 0:
            valid_len = len(batch_datas)
            for _ in range(batch - valid_len):
                batch_datas.append(batch_datas[-1])
                cv_images.append(cv_images[-1])
            outputs = self.inference(batch_datas)
            detections = self._postprocess(outputs, cv_images)
            show_results(cv_images, detections, filenames, valid_len)
            
        end2end_cost = time.time() - end2end_start
        self._end2end_latency_ms += (end2end_cost * 1000)

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
