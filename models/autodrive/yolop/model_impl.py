import os
import cv2
import torch
import numpy as np
from tqdm import tqdm
from typing import Dict, Any
from hmtool.utils import logger
from hmtool.base.base_model import BaseModel
from hmtool.utils.preprocess import calc_padding_size
from hmtool.utils.postprocess import non_max_suppression, scale_coords
from hmtool.utils.metrics import detections2txt, detection_txt2json, coco_eval


class YoloP(BaseModel):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.input_name = self.inputs_name[0]
        _, C, H, W = self.inputs_cfg[self.input_name]["shape"]
        self.input_size = (H, W)  # HW
        self.conf_threshold = 0.25
        self.iou_threshold = 0.45
        self.strides = [8., 16., 32.]                                                                                                                                                
        self.anchors = torch.Tensor(
            [3, 9, 5, 11, 4, 20, 7, 18, 6, 39, 12, 31, 19, 50, 38, 81, 68, 157]
        ).float().view(3, 3, 2)
        self.to_coco91 = True
        
    def postprocess(self, outs: Dict[str, np.ndarray], in_datas: Dict[str, np.ndarray]) -> Any:
        bbox_outs = [outs["feat0"], outs["feat1"], outs["feat2"]] 
        da_seg_out = outs["drive_area_seg"]  # bs, 2, 384, 640
        ll_seg_out = outs["lane_line_seg"]  # bs, 2, 384, 640
        # det
        z = list()
        for idx, bbox_out in enumerate(bbox_outs):
            bs, c, ny, nx = bbox_out.shape
            p = torch.from_numpy(bbox_out)
            sx = torch.arange(end=nx, dtype=torch.float32)  # shift x                                                                                                                    
            sy = torch.arange(end=ny, dtype=torch.float32)  # shift y                                                                                                                 
            sy, sx = torch.meshgrid(sy, sx, indexing="ij")                                                                                                                                           
            grid = torch.stack((sx, sy), dim=2).expand(1, 3, ny, nx, 2) - 0.5 
            anchor_grid = self.anchors[idx].view(1, 3, 1, 1, 2).expand(1, 3, ny, nx, 2)  
            p = p.view(bs, 3, c // 3, ny, nx).permute(0, 1, 3, 4, 2).contiguous()  # bs, 3, ny, nx, c//3
            p[..., 0:2] = (p[..., 0:2] * 2.0 + grid) * self.strides[idx]  # xy
            p[..., 2:4] = (p[..., 2:4] * 2) ** 2 * anchor_grid
            z.append(p.view(bs, -1, c // 3))  # bs, -1, c//3
        det_out = torch.cat(z, dim=1)  # bs, -1, c//3
        det_out = det_out[:1, ...]  # [1, -1, c//3]
        cv_image = list(in_datas.values())[0]
        outputs = non_max_suppression(det_out, self.conf_threshold, self.iou_threshold)
        output = outputs[0]
        output[:, :4] = scale_coords(self.input_size, output[:, :4], cv_image.shape).round()
        det_out = output.detach().cpu().numpy()
        
        # da_seg
        # 先插值会原图大小，mask效果比较平滑
        H, W, _ = cv_image.shape
        target_size = (self.input_size[1], self.input_size[0])  # (W, H)
        padding_size, size, scale = calc_padding_size((H, W), target_size, padding_mode=1)
        top, left, bottom, right = padding_size
        nh, nw = size
        # 只取第1个batch
        da_seg_mask = torch.from_numpy(da_seg_out[:, :, top:top + nh, left:left + nw])  # 2, nh, nw
        ll_seg_mask = torch.from_numpy(ll_seg_out[:, :, top:top + nh, left:left + nw])  # 2, nh, nw
        da_seg_mask = torch.nn.functional.interpolate(
            da_seg_mask, size=(H, W), mode='bilinear', align_corners=False).detach().cpu().numpy()
        ll_seg_mask = torch.nn.functional.interpolate(
            ll_seg_mask, size=(H, W), mode='bilinear', align_corners=False).detach().cpu().numpy()
        da_seg_mask = np.argmax(da_seg_mask, axis=1)[0]  # (0|1)
        ll_seg_mask = np.argmax(ll_seg_mask, axis=1)[0]  # (0|1)
        mask = np.zeros((H, W, 3), dtype=np.uint8)
        mask[da_seg_mask == 1] = [0, 255, 0]  # 1: drive area
        mask[ll_seg_mask == 1] = [255, 0, 0]  # 1: lane line
        return det_out, mask
        
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
            det_outs, mask = self.run(in_datas)
            for idx, detection in enumerate(det_outs):
                x1, y1, x2, y2, score, cls_idx = detection
                x1 = int(x1) if x1 > 0 else 0
                y1 = int(y1) if y1 > 0 else 0   
                x2 = int(x2) if x2 < cv_image.shape[1] else cv_image.shape[1]
                y2 = int(y2) if y2 < cv_image.shape[0] else cv_image.shape[0]
                cls_idx = int(cls_idx)
                logger.info(f'Detection[{idx:2}] x1: {x1:4}, y1: {y1:4}, x2: {x2:4}, y2: {y2:4}, score: {score:.3f}, cls: {cls_idx:2}')
                cv2.rectangle(cv_image, (x1, y1), (x2, y2), (0, 0, 255), 2)
            cv_image = np.where(mask == 255, cv_image * 0.5 + mask * 0.5, cv_image).astype(np.uint8)  
            cv2.imwrite(save_path, cv_image)
            logger.info(f'Save result to {save_path}')
    
    def evaluate(self, dataset, num=0):
        raise NotImplementedError("Evaluation is not implemented for YoloP model.")
