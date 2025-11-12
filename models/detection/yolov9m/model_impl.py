import os
import cv2
import torch
import numpy as np
from tqdm import tqdm
from typing import Dict, Any
from hmatc.utils import logger
from hmatc.base.base_model import BaseModel
from hmatc.utils.postprocess import non_max_suppression2, scale_coords
from hmatc.utils.metrics import detections2txt, detection_txt2json, coco_eval


class YoloV9(BaseModel):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.input_name = self.inputs_name[0]
        _, C, H, W = self.inputs_cfg[self.input_name]["shape"]
        self.input_size = (H, W)  # HW
        self.conf_threshold = 0.25
        self.iou_threshold = 0.45
        self.to_coco91 = True

    def postprocess(
        self, outs: Dict[str, np.ndarray], in_datas: Dict[str, np.ndarray]
    ) -> Any:
        if len(outs) != 1:
            raise ValueError("YoloV9 model only has one output")
        pred = list(outs.values())[0]  # [bs, 84, 8400]
        pred = torch.from_numpy(pred)
        # 只取batch0，多batch数据是复制来的，不用处理浪费时间
        pred = pred[:1, ...]  # [1, 84, 8400]
        cv_image = list(in_datas.values())[0]
        outputs = non_max_suppression2(pred, self.conf_threshold, self.iou_threshold)
        output = outputs[0]
        output[:, :4] = scale_coords(
            self.input_size, output[:, :4], cv_image.shape
        ).round()
        output = output.detach().cpu().numpy()
        return output

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
                logger.warning(f"{filepath} not exists or decode failed")
                continue
            in_datas[self.input_name] = cv_image
            logger.info(f"Image[{idx}] {filepath}")
            outs = self.run(in_datas)
            for idx, detection in enumerate(outs):
                x1, y1, x2, y2, score, cls_idx = detection
                x1 = int(x1) if x1 > 0 else 0
                y1 = int(y1) if y1 > 0 else 0
                x2 = int(x2) if x2 < cv_image.shape[1] else cv_image.shape[1]
                y2 = int(y2) if y2 < cv_image.shape[0] else cv_image.shape[0]
                cls_idx = int(cls_idx)
                logger.info(
                    f"Detection[{idx:2}] x1: {x1:4}, y1: {y1:4}, x2: {x2:4}, y2: {y2:4}, score: {score:.3f}, cls: {cls_idx:2}"
                )
                cv2.rectangle(cv_image, (x1, y1), (x2, y2), (0, 0, 255), 2)
            cv2.imwrite(save_path, cv_image)
            logger.info(f"Save result to {save_path}")

    def evaluate(self, dataset, num=0):
        self.iou_threshold = 0.65
        self.conf_threshold = 0.01
        img_paths = dataset.get_datas(num)
        save_results = f"results_{self.backend}"
        if not os.path.exists(save_results):
            os.makedirs(save_results)
        in_datas = dict()
        for idx, img_path in enumerate(tqdm(img_paths)):
            basename, _ = os.path.splitext(os.path.basename(img_path))
            image_id = dataset.get_image_id(basename)
            out_path = os.path.join(save_results, f"{image_id}.txt")
            if os.path.exists(out_path):
                continue
            cv_image = cv2.imread(img_path)
            if cv_image is None:
                logger.warning(f"{img_path} not exists or decode failed")
                continue
            in_datas[self.input_name] = cv_image
            logger.debug(f"Image[{idx}] {img_path}")
            detections = self.run(in_datas)
            detections2txt(detections, out_path)
        pred_json = f"pred_{self.backend}.json"
        detection_txt2json(save_results, pred_json, to_coco91=self.to_coco91)
        map50_95, map50 = coco_eval(
            pred_json, dataset.annotations_file, dataset.image_ids
        )
        return {
            "input_size": self.inputs_cfg[self.input_name]["shape"],
            "dataset": dataset.dataset_name,
            "num": len(img_paths),
            "map50_95": f"{map50_95:.6f}",
            "map50": f"{map50:.6f}",
            "latency": f"{self.ave_latency_ms:.6f}",
        }
