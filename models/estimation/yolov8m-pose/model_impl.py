import os
import cv2
import torch
import numpy as np
from tqdm import tqdm
from typing import Dict, Any, List
from hmatc.utils import logger
from hmatc.base.base_model import BaseModel
from hmatc.utils.postprocess import (
    non_max_suppression2,
    scale_coords_kpt,
    plot_skeleton_kpts,
)
from hmatc.utils.metrics import detections_kpt2json, merge_json, coco_eval


class YoloV8Pose(BaseModel):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.input_name = self.inputs_name[0]
        _, C, H, W = self.inputs_cfg[self.input_name]["shape"]
        self.input_size = (H, W)  # HW
        self.conf_threshold = 0.25
        self.iou_threshold = 0.45
        self.strides = [8, 16, 32]
        self.proj = torch.arange(16, dtype=torch.float32).view(16, 1)
        self.anchor_points, self.stride_tensor = self.make_anchors(H, W, self.strides)

    @staticmethod
    def make_anchors(H, W, strides, grid_cell_offset=0.5):
        """Generate anchors from features."""
        anchor_points, stride_tensor = [], []
        for i, stride in enumerate(strides):
            h, w = int(H / stride), int(W / stride)
            sx = torch.arange(end=w, dtype=torch.float32) + grid_cell_offset  # shift x
            sy = torch.arange(end=h, dtype=torch.float32) + grid_cell_offset  # shift y
            sy, sx = torch.meshgrid(sy, sx, indexing="ij")
            anchor_points.append(torch.stack((sx, sy), -1).view(-1, 2))
            stride_tensor.append(
                torch.full((h * w, 1), fill_value=stride, dtype=torch.float32)
            )
        return torch.cat(anchor_points), torch.cat(stride_tensor)

    def postprocess(
        self, outs: Dict[str, np.ndarray], in_datas: Dict[str, np.ndarray]
    ) -> Any:
        outs = list(outs.values())
        if len(outs) == 1:
            out = torch.from_numpy(outs[0])
        elif len(outs) == 9:
            out = self._decode(outs)
        # 只取batch0，多batch数据是复制来的，不用处理浪费时间
        pred = out[:1, ...]
        cv_image = list(in_datas.values())[0]
        assert len(pred.shape) == 3, "pred shape error"
        nc = pred.shape[1] - 17 * 3 - 4
        outputs = non_max_suppression2(
            pred, self.conf_threshold, self.iou_threshold, nc=nc
        )
        output = outputs[0]
        output = scale_coords_kpt(self.input_size, output, cv_image.shape)
        return output

    def _decode(self, outputs: List[np.ndarray]):
        bs = outputs[0].shape[0]
        kpt_data = np.concatenate(
            [
                outputs[6].reshape(bs, 51, -1),
                outputs[7].reshape(bs, 51, -1),
                outputs[8].reshape(bs, 51, -1),
            ],
            axis=2,
        )  # bs, 51, 8400

        cls_data = np.concatenate(
            [
                outputs[3].reshape(bs, 1, -1),
                outputs[4].reshape(bs, 1, -1),
                outputs[5].reshape(bs, 1, -1),
            ],
            axis=2,
        )  # bs, 1, 8400

        box_data = np.concatenate(
            [
                outputs[0].reshape(bs, 64, -1),
                outputs[1].reshape(bs, 64, -1),
                outputs[2].reshape(bs, 64, -1),
            ],
            axis=2,
        )  # bs, 64, 8400

        kpt_data = torch.from_numpy(kpt_data)
        cls_data = torch.from_numpy(cls_data)
        box_data = torch.from_numpy(box_data)

        # decode box
        box_data = (
            box_data.view(bs, 4, 16, 8400)
            .permute(0, 3, 1, 2)
            .contiguous()
            .softmax(dim=3)
            .view(-1, 16)
            .matmul(self.proj)
            .view(bs, 8400, 4)
        )
        box_data[:, :, 0:2] = self.anchor_points - box_data[:, :, 0:2]
        box_data[:, :, 2:4] = self.anchor_points + box_data[:, :, 2:4]
        box_data_xy = (box_data[:, :, 0:2] + box_data[:, :, 2:4]) * 0.5
        box_data_wh = box_data[:, :, 2:4] - box_data[:, :, 0:2]
        box_data[:, :, 0:2] = box_data_xy
        box_data[:, :, 2:4] = box_data_wh
        box_data *= self.stride_tensor
        box_data = box_data.permute(0, 2, 1).contiguous()  # bs, 4, 8400

        # decode cls
        cls_data = torch.sigmoid(cls_data)  # bs, 1, 8400

        # decode kpt
        kpt_data[:, 0::3] = (
            kpt_data[:, 0::3] * 2.0 + self.anchor_points[:, 0] - 0.5
        ) * self.stride_tensor.squeeze(1)
        kpt_data[:, 1::3] = (
            kpt_data[:, 1::3] * 2.0 + self.anchor_points[:, 1] - 0.5
        ) * self.stride_tensor.squeeze(1)
        kpt_data[:, 2::3].sigmoid_()
        return torch.cat([box_data, cls_data, kpt_data], dim=1)

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
                plot_skeleton_kpts(cv_image, detection[6:].T, 3)
                (x1, y1, x2, y2, score, cls_idx) = detection[0:6]
                x1 = int(x1) if x1 > 0 else 0
                y1 = int(y1) if y1 > 0 else 0
                x2 = int(x2) if x2 < cv_image.shape[1] else cv_image.shape[1]
                y2 = int(y2) if y2 < cv_image.shape[0] else cv_image.shape[0]
                logger.info(
                    f"Detection[{idx:2}] x1: {x1:4}, y1: {y1:4}, x2: {x2:4}, y2: {y2:4}, score: {score:.3f}, cls: {int(cls_idx)}"
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
            out_path = os.path.join(save_results, f"{image_id}.json")
            if os.path.exists(out_path):
                continue
            cv_image = cv2.imread(img_path)
            if cv_image is None:
                logger.warning(f"{img_path} not exists or decode failed")
                continue
            in_datas[self.input_name] = cv_image
            logger.debug(f"Image[{idx}] {img_path}")
            detections = self.run(in_datas)
            detections_kpt2json(detections, out_path)
        pred_json = f"pred_{self.backend}.json"
        merge_json(save_results, pred_json)
        map50_95, map50 = coco_eval(
            pred_json,
            dataset.annotations_kpt,
            dataset.image_ids,
            iou_type="keypoints",
        )
        return {
            "input_size": self.inputs_cfg[self.input_name]["shape"],
            "dataset": dataset.dataset_name,
            "num": len(img_paths),
            "map50_95": f"{map50_95:.6f}",
            "map50": f"{map50:.6f}",
            "latency": f"{self.ave_latency_ms:.6f}",
        }
