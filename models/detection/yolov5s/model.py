#!/usr/bin/env python3

from hmassist.models.detector import Detector
from hmassist.utils.preprocess import letterbox
from hmassist.utils.postprocess import non_max_suppression, scale_coords
from ultralytics.utils.plotting import Annotator, colors
from hmassist.datasets.coco import coco80_labels

import numpy as np
import os
import cv2
import torch
import time


class YoloV5(Detector):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    # def build_options(self):
    #     return {}

    def _postprocess(self, outputs, cv_images: list):
        assert len(outputs) == 3
        # add yolo process
        outputs = self.yolo_detect(outputs)
        # outputs = torch.from_numpy(outputs)
        outputs = non_max_suppression(outputs, self._conf_threshold, self._iou_threshold)
        batch = len(outputs)
        for idx in range(batch):
            outputs[idx][:, :4] = scale_coords(self._input_size, outputs[idx][:, :4], cv_images[idx].shape).round()
            outputs[idx] = outputs[idx].numpy()
        return outputs

    def demo(self, img_path_list: list):
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
                    # cv2.rectangle(cv_image, (x1, y1), (x2, y2), (0, 0, 255), 1, 8)
                    from ultralytics.utils.plotting import Annotator, colors
                    from hmassist.datasets.coco import coco80_labels
                    label = coco80_labels[cls] + " {:.2f}".format(conf)
                    annotator = Annotator(cv_image, line_width=2, example=str(cls))
                    annotator.box_label((x1, y1, x2, y2), label, color=colors(cls, True))
                    print("x1:{}, y1:{}, x2:{}, y2:{}, conf:{:.6f}, cls:{}".format(x1, y1, x2, y2, conf, int(cls)), flush=True)
                save_path = os.path.join(save_results, filenames[idx])
                cv2.imwrite(save_path, cv_images[idx])
                print("demo results saved to", save_path)
        
        batch = self.executor.model_input_batch * self.executor.batch
        filenames = []
        cv_images = []
        batch_datas = []
        end2end_start = time.time()
        for img_path in img_path_list:
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
            cv_images.append(cv_image)
            batch_datas.append(inputs)
            filenames.append(filename)
            if len(batch_datas) < batch:
                continue
            outputs = self.inference(batch_datas)
            detections = self._postprocess(outputs, cv_images)
            cv_images.clear()
            batch_datas.clear()
            filenames.clear()
            show_results(cv_images, detections, filenames)
            
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

    def yolo_detect(self, feats):
        # in.shape = out.shape: 1x3x80x80x85 1x3x40x40x85 1x3x20x20x85
        output = []

        for i, name in enumerate(feats):
            data = torch.from_numpy(feats[name])
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