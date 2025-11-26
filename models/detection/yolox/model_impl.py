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


def nms(boxes, scores, nms_thr):
    """Single class NMS implemented in Numpy."""
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]

    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    order = scores.argsort()[::-1]

    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])

        w = np.maximum(0.0, xx2 - xx1 + 1)
        h = np.maximum(0.0, yy2 - yy1 + 1)
        inter = w * h
        ovr = inter / (areas[i] + areas[order[1:]] - inter)

        inds = np.where(ovr <= nms_thr)[0]
        order = order[inds + 1]

    return keep

def multiclass_nms_class_agnostic(boxes, scores, nms_thr, score_thr):
    """Multiclass NMS implemented in Numpy. Class-agnostic version."""
    cls_inds = scores.argmax(1)
    cls_scores = scores[np.arange(len(cls_inds)), cls_inds]

    valid_score_mask = cls_scores > score_thr
    if valid_score_mask.sum() == 0:
        dets = []
        return dets
    valid_scores = cls_scores[valid_score_mask]
    valid_boxes = boxes[valid_score_mask]
    valid_cls_inds = cls_inds[valid_score_mask]
    keep = nms(valid_boxes, valid_scores, nms_thr)
    if keep:
        dets = np.concatenate(
            [valid_boxes[keep], valid_scores[keep, None], valid_cls_inds[keep, None]], 1
        )
    return dets

class YoloX(BaseModel):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.input_name = self.inputs_name[0]
        _, C, H, W = self.inputs_cfg[self.input_name]["shape"]
        self.input_size = (H, W)  # HW
        self.conf_threshold = 0.25
        self.iou_threshold = 0.45
        self.feats = [(80, 80), (40, 40), (20, 20)]  # hw
        self.strides = [8., 16., 32.]
        self.to_coco91 = True
        
    def postprocess(self, outs: Dict[str, np.ndarray], in_datas: Dict[str, np.ndarray]) -> Any:
        key = list(outs.keys())[0]
        cv_image = list(in_datas.values())[0]
        outputs = outs[key]
        grids = []
        expanded_strides = []
        strides = self.strides
        img_size = self.input_size
        hsizes = [img_size[0] // stride for stride in strides]
        wsizes = [img_size[1] // stride for stride in strides]
        
        for hsize, wsize, stride in zip(hsizes, wsizes, strides):
            xv, yv = np.meshgrid(np.arange(wsize), np.arange(hsize))
            grid = np.stack((xv, yv), 2).reshape(1, -1, 2)
            grids.append(grid)
            shape = grid.shape[:2]
            expanded_strides.append(np.full((*shape, 1), stride))

        grids = np.concatenate(grids, 1)
        expanded_strides = np.concatenate(expanded_strides, 1)
        outputs[..., :2] = (outputs[..., :2] + grids) * expanded_strides
        outputs[..., 2:4] = np.exp(outputs[..., 2:4]) * expanded_strides
        
        boxes = outputs[:, :, :4]
        scores = outputs[:, :, 4:5] * outputs[:, :, 5:]
        boxes_xyxy = np.ones_like(boxes)
        boxes_xyxy[:, :, 0] = boxes[:, :, 0] - boxes[:, :, 2]/2.
        boxes_xyxy[:, :, 1] = boxes[:, :, 1] - boxes[:, :, 3]/2.
        boxes_xyxy[:, :, 2] = boxes[:, :, 0] + boxes[:, :, 2]/2.
        boxes_xyxy[:, :, 3] = boxes[:, :, 1] + boxes[:, :, 3]/2.
        
        outputs = multiclass_nms_class_agnostic(boxes_xyxy[0, :, :], scores[0, :, :], self.iou_threshold, self.conf_threshold)
        
        return outputs
        
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
            outs = self.run(in_datas)
            for idx, detection in enumerate(outs):
                x1, y1, x2, y2, score, cls_idx = detection
                x1 = int(x1) if x1 > 0 else 0
                y1 = int(y1) if y1 > 0 else 0   
                x2 = int(x2) if x2 < cv_image.shape[1] else cv_image.shape[1]
                y2 = int(y2) if y2 < cv_image.shape[0] else cv_image.shape[0]
                cls_idx = int(cls_idx)
                logger.info(f'Detection[{idx:2}] x1: {x1:4}, y1: {y1:4}, x2: {x2:4}, y2: {y2:4}, score: {score:.3f}, cls: {cls_idx:2}')
                cv2.rectangle(cv_image, (x1, y1), (x2, y2), (0, 0, 255), 2)
            cv2.imwrite(save_path, cv_image)
            logger.info(f'Save result to {save_path}')
    
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
                logger.warning(f'{img_path} not exists or decode failed')
                continue
            in_datas[self.input_name] = cv_image
            logger.debug(f'Image[{idx}] {img_path}')
            detections = self.run(in_datas)
            detections2txt(detections, out_path)
        pred_json = f"pred_{self.backend}.json"
        detection_txt2json(save_results, pred_json, to_coco91=self.to_coco91)
        map50_95, map50 = coco_eval(pred_json, dataset.annotations_file, dataset.image_ids)
        return {
            "input_size": self.inputs_cfg[self.input_name]['shape'],
            "dataset": dataset.dataset_name,
            "num": len(img_paths),
            "map50_95": f"{map50_95:.6f}",
            "map50": f"{map50:.6f}",
            "latency": f"{self.ave_latency_ms:.6f}"
        }
