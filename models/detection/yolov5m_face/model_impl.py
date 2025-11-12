import os
import cv2
import torch
import numpy as np
from tqdm import tqdm
from typing import Dict, Any
from hmatc.utils import logger
from hmatc.base.base_model import BaseModel
from hmatc.utils.postprocess import (
    non_max_suppression_face,
    xyxy2xywh,
    scale_coords,
    scale_coords_landmarks,
)
from hmatc.utils.metrics import detections_face2txt
from evaluation import evaluation


class YoloV5MFace(BaseModel):
    def __init__(self, **kwargs):
        super(YoloV5MFace, self).__init__(**kwargs)
        self.input_name = self.inputs_name[0]
        _, C, H, W = self.inputs_cfg[self.input_name]["shape"]
        self.input_size = (H, W)  # HW
        self.conf_threshold = 0.2
        self.iou_threshold = 0.5

    def postprocess(
        self, outs: Dict[str, np.ndarray], in_datas: Dict[str, np.ndarray]
    ) -> Any:
        # 1 (obj_conf) + 4 (c_x, c_y, w, h) + 10 (x1, y1, x2, y2...) + 1 (class_num) = 16
        pred = list(outs.values())[0]  # [bs, 25200, 16]
        pred = torch.from_numpy(pred)
        # 只取batch0，多batch数据是复制来的，不用处理浪费时间
        # pred = pred[:1, ...]  # [1, 25200, 16]
        cv_image = list(in_datas.values())[0]
        output = non_max_suppression_face(
            pred, self.conf_threshold, self.iou_threshold
        )[0]
        gn = torch.tensor(cv_image.shape)[[1, 0, 1, 0]]  # normalization gain whwh
        gn_lks = torch.tensor(cv_image.shape)[
            [1, 0, 1, 0, 1, 0, 1, 0, 1, 0]
        ]  # normalization gain landmarks
        h, w, c = cv_image.shape

        output[:, :4] = scale_coords(
            self.input_size, output[:, :4], cv_image.shape
        ).round()
        output[:, 5:15] = scale_coords_landmarks(
            self.input_size, output[:, 5:15], cv_image.shape
        ).round()
        boxes = []
        landmarks_list = []
        for j in range(output.size()[0]):
            xywh = (xyxy2xywh(output[j, :4].view(1, 4)) / gn).view(-1)
            xywh = xywh.data.cpu().numpy()
            conf = output[j, 4].cpu().numpy()
            landmarks = (output[j, 5:15].view(1, 10) / gn_lks).view(-1).tolist()
            class_num = output[j, 15].cpu().numpy()
            x1 = int(xywh[0] * w - 0.5 * xywh[2] * w)
            y1 = int(xywh[1] * h - 0.5 * xywh[3] * h)
            x2 = int(xywh[0] * w + 0.5 * xywh[2] * w)
            y2 = int(xywh[1] * h + 0.5 * xywh[3] * h)

            boxes.append([x1, y1, (x2 - x1), (y2 - y1), conf])
            landmarks_list.append(landmarks)

        return boxes, landmarks_list

    def demo(self, filepaths: list):
        in_datas = dict()
        save_dir = f"vis_{self.backend}"
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        for idx, filepath in enumerate(filepaths):
            basename, _ = os.path.splitext(os.path.basename(filepath))
            save_path = os.path.join(save_dir, f"{basename}.jpg")
            cv_image = cv2.imread(filepath)
            if cv_image is None:
                logger.warning(f'{filepath} not exists or decode failed')
                continue
            in_datas[self.input_name] = cv_image
            logger.info(f'Image[{idx}] {filepath}')
            outs, landmark_list = self.run(in_datas)
            h, w, c = cv_image.shape
            tl = 1 or round(0.002 * (h + w) / 2) + 1  # line/font thickness
            for idx, detection in enumerate(outs):
                x1 = int(detection[0])
                y1 = int(detection[1])
                x2 = x1 + int(detection[2])
                y2 = y1 + int(detection[3])
                conf = detection[4]
                landmarks = landmark_list[idx]
                logger.info(
                    f'Detection[{idx:2}] x1: {x1:4}, y1: {y1:4}, x2: {x2:4}, y2: {y2:4}, conf: {conf:.3f}'
                )
                cv2.rectangle(
                    cv_image,
                    (x1, y1),
                    (x2, y2),
                    (0, 255, 0),
                    thickness=tl,
                    lineType=cv2.LINE_AA,
                )

                clors = [
                    (255, 0, 0),
                    (0, 255, 0),
                    (0, 0, 255),
                    (255, 255, 0),
                    (0, 255, 255),
                ]
                for i in range(5):
                    point_x = int(landmarks[2 * i] * w)
                    point_y = int(landmarks[2 * i + 1] * h)
                    cv2.circle(cv_image, (point_x, point_y), tl + 1, clors[i], -1)
                tf = max(tl - 1, 1)  # font thickness
                label = str(conf)[:5]
                cv2.putText(
                    cv_image,
                    label,
                    (x1, y1 - 2),
                    0,
                    tl / 3,
                    [225, 255, 255],
                    thickness=tf,
                    lineType=cv2.LINE_AA,
                )
            cv2.imwrite(save_path, cv_image)
            logger.info(f'Save result to {save_path}')

    def evaluate(self, dataset, num=0):
        self.iou_threshold = 0.5
        self.conf_threshold = 0.2
        img_paths = dataset.get_datas(num)
        save_results = f"results_{self.backend}"
        if not os.path.exists(save_results):
            os.makedirs(save_results)
        in_datas = dict()
        for idx, img_path in enumerate(tqdm(img_paths)):
            image_name = os.path.basename(img_path)
            txt_name = os.path.splitext(image_name)[0] + ".txt"
            folder_name = img_path.rsplit("/", 2)[-2]
            out_path = os.path.join(save_results, folder_name, txt_name)
            dirname = os.path.dirname(out_path)
            if not os.path.isdir(dirname):
                os.makedirs(dirname)
            cv_image = cv2.imread(img_path)
            if cv_image is None:
                logger.warning(f'{img_path} not exists or decode failed')
                continue
            in_datas[self.input_name] = cv_image
            logger.debug(f'Image[{idx}] {img_path}')
            detections, _ = self.run(in_datas)
            detections_face2txt(detections, out_path)

        aps = evaluation(save_results, dataset.annotation_path)
        return {
            "input_size": self.inputs_cfg[self.input_name]['shape'],
            "dataset": dataset.dataset_name,
            "num": len(img_paths),
            "ap_easy": f"{aps[0]:.6f}",
            "ap_medium": f"{aps[1]:.6f}",
            "ap_hard": f"{aps[2]:.6f}",
        }
