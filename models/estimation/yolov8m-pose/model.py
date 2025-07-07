#!/usr/bin/env python3

from hmassist.models.estimator import Estimator
from hmassist.utils.postprocess import non_max_suppression_scale_kpt
from ultralytics.utils.plotting import Annotator, colors

import numpy as np
import os
import cv2
import torch
import time

class YoloV8Pose(Estimator):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.dst_width = 640
        self.dst_height = 640
        self.target_shape  = (self.dst_width, self.dst_height)

    def postprocess_kp(self, result0):
        reshape0 = result0.reshape(1, 17, 3, -1)
        slice0 = reshape0[:, :, :2, :]
        slice1 = reshape0[:, :, 2:3, :]
        mul0 = slice0*2
        add0 = mul0 + np.load("yolov8_pose_data/model_22Add_3.npy")
        mul1 = add0 * np.load("yolov8_pose_data/model_22Mul_4.npy")
        sigmoid0 = 1 / (1 + np.exp(-slice1))
        concat0 = np.concatenate((mul1, sigmoid0), axis=2)
        reshape1 = concat0.reshape(1, 51, -1)
        return reshape1

    def postprocess_bx(self, result1):
        reshape0 = result1.reshape(1, 4, 8400)
        slice0 = reshape0[:, 0:2, :]
        slice1 = reshape0[:, 2:4, :]
        sub0 = np.load("yolov8_pose_data/model_22Sub.npy") - slice0
        add0 = np.load("yolov8_pose_data/model_22Add_1.npy") + slice1
        add1 = sub0 + add0
        sub1 = add0 - sub0
        div0 = add1 * 0.5
        concat0 = np.concatenate((div0, sub1), axis=1)
        mul0 = concat0*np.load("yolov8_pose_data/model_22Mul_2.npy")
        return mul0

    def _decode(self, outputs):
        output_name_kps = "output1"
        output_name_box = "output2"
        output_name_score = "output3"
        kps_data = self.postprocess_kp(outputs[output_name_kps])
        box_data = self.postprocess_bx(outputs[output_name_box])
        kps_data = torch.from_numpy(kps_data)
        box_data = torch.from_numpy(box_data)
        score_data = torch.from_numpy(outputs[output_name_score])
        result = torch.cat((box_data, score_data, kps_data), 1)
        result = result.transpose(-1, -2)  # 1,8400,56
        return result

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
                for box in boxes:
                    if valid_len is None:
                        valid_len = len(cv_images)
                    for idx, boxes in enumerate(detections):
                        if idx == valid_len:
                            break
                        print("box num = {}".format(len(boxes)))
                        for det in boxes:
                            (x1, y1, x2, y2), conf = list(map(int, det[0:4])), det[4]
                            label = "person" + " {:.2f}".format(conf)
                            annotator = Annotator(cv_images[idx], line_width=2, example=str(0))
                            annotator.box_label((x1, y1, x2, y2), label, color=colors(0, True))
                            keypoints = box[5:]
                            keypoints = np.array(keypoints).reshape(-1, 3)
                            annotator.kpts(keypoints, cv_images[idx].shape[:2], radius=5, kpt_line=True, kpt_color=None)
                            print("x1:{}, y1:{}, x2:{}, y2:{}, conf:{:.6f}, cls:{}".format(x1, y1, x2, y2, conf, 0), flush=True)
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

    def _postprocess(self, outputs, cv_images: list):
        if len(outputs) == 1:
            output = torch.from_numpy(list(outputs.values())[0])
        elif len(outputs) == 3:
            output = self._decode(outputs)
        else:
            print("outputs num", len(outputs), "not supported.")
            assert(0)
        outputs = []
        batch = len(output)
        for idx in range(batch):
            image = cv_images[idx]
            pred = output[idx]
            boxes = non_max_suppression_scale_kpt(pred, image, self.target_shape, self._conf_threshold, self._iou_threshold)
            outputs.append(np.array(boxes))
        return outputs