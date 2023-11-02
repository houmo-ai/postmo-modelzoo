#!/usr/bin/env python3

from base.detector import Detector
from utils.preprocess import letterbox
import numpy as np
import os
import cv2
import torch
import time
import torchvision

class YoloP(Detector):

    # @staticmethod
    # def data_transform(x, shape):
    #     img = cv2.cvtColor(np.array(x), cv2.COLOR_RGB2BGR)
    #     img, _, _ = letterbox(img, (384, 640), stride=64, auto=False)
    #     img = np.transpose(img, (2, 0, 1)).astype(np.float32)
    #     from utils.transform import ToTensorNotNormal
    #     to_tensor = ToTensorNotNormal()
    #     img = to_tensor(img)
    #     from utils.transform import RGB2YUV
    #     rgb2yuv = RGB2YUV(fmt="422")
    #     img = rgb2yuv(img)
    #     print(img.shape, img.dtype)
    #     return img.unsqueeze(0)

    # @staticmethod
    # def build_config():
    #     return {}

    def _preprocess(self, img):
        img, _, _ = letterbox(img, (384, 640), stride=64, auto=False)
        img = np.transpose(img, (2, 0, 1)).astype(np.float32)
        from utils.transform import BGR2YUV
        rgb2yuv_func = BGR2YUV(fmt="422")
        img = rgb2yuv_func(torch.tensor(img))
        return np.expand_dims(img.numpy().astype(np.uint8), 0)

    def _postprocess(self, outputs, img=None):
        def make_grid(nx=20, ny=20):
            xv, yv = np.meshgrid(np.arange(nx), np.arange(ny))
            return np.stack((xv, yv), 2).reshape(1, 1, ny, nx, 2).astype(np.float32)

        anchors = [
            [3, 9, 5, 11, 4, 20],
            [7, 18, 6, 39, 12, 31],
            [19, 50, 38, 81, 68, 157],
        ]

        stride = [8, 16, 32]
        anchors = np.asarray(anchors, dtype=np.float32).reshape(3, 3, 1, 1, 2)
        anchor_grid = torch.from_numpy(anchors)
        anchors = np.asarray(anchors, dtype=np.float32).reshape(3, 3, 2)
        anchors = anchors / np.array(stride).reshape((3, 1, 1))
        anchors = torch.from_numpy(anchors)
        grid = []
        input_height, input_width = 384, 640

        for i in range(len(stride)):
            h, w = int(input_height / stride[i]), int(input_width / stride[i])
            grid.append(torch.from_numpy(make_grid(w, h)))

        det_out = self._post_process_anchor([outputs["feat0"],
                                             outputs["feat1"],
                                             outputs["feat2"]],
                                             stride, grid, anchor_grid)

        # det_out = torch.from_numpy(det_out).float()
        boxes = self.non_max_suppression(det_out)[0]  # [n,6] [x1,y1,x2,y2,conf,cls]
        boxes = boxes.cpu().numpy().astype(np.float32)

        if boxes.shape[0] == 0:
            print("no bounding boxes detected.")
            return
        print(f"detect {boxes.shape[0]} bounding boxes.")

        da_seg_out = outputs["drive_area_seg"]
        ll_seg_out = outputs["lane_line_seg"]

        return boxes, da_seg_out, ll_seg_out

    def demo(self, img_path):
        if not os.path.exists(img_path):
            print("[error] The img path not exist -> {}".format(img_path))
            exit(-1)
        filename = os.path.basename(img_path)
        print("process: {}".format(img_path))

        save_results = "pictures_{}".format(self.backend)
        if not os.path.exists(save_results):
            os.makedirs(save_results)

        cv_image = cv2.imread(img_path)
        if cv_image is None:
            print("[error] Failed to decode img by opencv -> {}".format(img_path))
            exit(-1)

        t1 = time.time()
        input_data = self._preprocess(cv_image)
        t2 = time.time()
        # self.executor.set_fixed_out(True)
        output_data = self.inference(input_data)
        t3 = time.time()
        boxes, da_seg_out, ll_seg_out = self._postprocess(output_data, cv_image)
        t4 = time.time()

        # resize & normalize
        canvas, r, dw, dh, new_unpad_w, new_unpad_h = self.resize_unscale(cv_image, (384, 640))
        height, width, _ = cv_image.shape

        # scale coords to original size.
        boxes[:, 0] -= dw
        boxes[:, 1] -= dh
        boxes[:, 2] -= dw
        boxes[:, 3] -= dh
        boxes[:, :4] /= r

        # select da & ll segment area.
        da_seg_out = da_seg_out[:, :, dh:dh + new_unpad_h, dw:dw + new_unpad_w]
        ll_seg_out = ll_seg_out[:, :, dh:dh + new_unpad_h, dw:dw + new_unpad_w]

        da_seg_mask = np.argmax(da_seg_out, axis=1)[0]  # (?,?) (0|1)
        ll_seg_mask = np.argmax(ll_seg_out, axis=1)[0]  # (?,?) (0|1)

        color_area = np.zeros((new_unpad_h, new_unpad_w, 3), dtype=np.uint8)
        color_area[da_seg_mask == 1] = [0, 255, 0]
        color_area[ll_seg_mask == 1] = [255, 0, 0]
        color_seg = color_area

        # convert to BGR
        color_seg = color_seg[..., ::-1]
        color_mask = np.mean(color_seg, 2)
        img_merge = canvas[dh:dh + new_unpad_h, dw:dw + new_unpad_w, :]
        img_merge = img_merge[:, :, ::-1]

        # merge: resize to original size
        img_merge[color_mask != 0] = \
            img_merge[color_mask != 0] * 0.5 + color_seg[color_mask != 0] * 0.5
        img_merge = img_merge.astype(np.uint8)
        img_merge = cv2.resize(img_merge, (width, height),
                               interpolation=cv2.INTER_LINEAR)
        for i in range(boxes.shape[0]):
            x1, y1, x2, y2, conf, label = boxes[i]
            x1, y1, x2, y2, label = int(x1), int(y1), int(x2), int(y2), int(label)
            print("x1:{}, y1:{}, x2:{}, y2:{}, conf:{:.6f}, cls:{}".format(x1, y1, x2, y2, conf, int(label)))
            img_merge = cv2.rectangle(img_merge, (x1, y1), (x2, y2), (0, 255, 0), 2, 2)

        t5 = time.time()

        print("preprocess cost {:.3f}ms, infer cost {:.3f}ms, postprocess cost {:.3f}ms, demoprocess cost {:.3f}ms"
              .format((t2-t1)*1000, (t3-t2)*1000, (t4-t3)*1000, (t5-t4)*1000))

        save_merge_path = os.path.join(save_results, "result_" + filename)
        print("result saved in {}".format(save_merge_path))
        cv2.imwrite(save_merge_path, img_merge)

    def _post_process_anchor(self, feat_list, stride, grid, anchor_grid):
        z = list()
        for i, feat in enumerate(feat_list):
            bs, c, ny, nx = feat.shape
            feat = (
                torch.tensor(feat)
                .view(bs, 3, c // 3, ny * nx)
                .permute(0, 1, 3, 2)
                .view(bs, 3, ny, nx, c // 3)
                .contiguous()
            )
            y = feat  # .sigmoid()
            y[..., 0:2] = (y[..., 0:2] * 2.0 - 0.5 + grid[i]) * stride[i]  # xy
            y[..., 2:4] = (y[..., 2:4] * 2) ** 2 * anchor_grid[i]  # wh
            z.append(y.view(bs, -1, c // 3))
        det_out = torch.cat(z, 1)
        return det_out

    def xywh2xyxy(self, x):
        # Convert nx4 boxes from [x, y, w, h] to [x1, y1, x2, y2] where xy1=top-left, xy2=bottom-right
        y = x.clone() if isinstance(x, torch.Tensor) else np.copy(x)
        y[:, 0] = x[:, 0] - x[:, 2] / 2  # top left x
        y[:, 1] = x[:, 1] - x[:, 3] / 2  # top left y
        y[:, 2] = x[:, 0] + x[:, 2] / 2  # bottom right x
        y[:, 3] = x[:, 1] + x[:, 3] / 2  # bottom right y
        return y

    def non_max_suppression(
            self,
            prediction,
            conf_thres=0.25,
            iou_thres=0.45,
            classes=None,
            agnostic=False,
            multi_label=False,
            labels=(),
            max_det=300,
            nm=0,  # number of masks
    ):
        """Non-Maximum Suppression (NMS) on inference results to reject overlapping bounding boxes
        Returns:
            list of detections, on (n,6) tensor per image [xyxy, conf, cls]
        """
        bs = prediction.shape[0]  # batch size
        nc = prediction.shape[2] - nm - 5  # number of classes
        xc = prediction[..., 4] > conf_thres  # candidates

        # Checks
        assert 0 <= conf_thres <= 1, f'Invalid Confidence threshold {conf_thres}, valid values are between 0.0 and 1.0'
        assert 0 <= iou_thres <= 1, f'Invalid IoU {iou_thres}, valid values are between 0.0 and 1.0'

        # Settings
        # min_wh = 2  # (pixels) minimum box width and height
        max_wh = 7680  # (pixels) maximum box width and height
        max_nms = 30000  # maximum number of boxes into torchvision.ops.nms()
        time_limit = 0.5 + 0.05 * bs  # seconds to quit after
        redundant = True  # require redundant detections
        multi_label &= nc > 1  # multiple labels per box (adds 0.5ms/img)
        merge = False  # use merge-NMS

        t = time.time()
        mi = 5 + nc  # mask start index
        output = [torch.zeros((0, 6 + nm), device=prediction.device)] * bs
        for xi, x in enumerate(prediction):  # image index, image inference
            # Apply constraints
            # x[((x[..., 2:4] < min_wh) | (x[..., 2:4] > max_wh)).any(1), 4] = 0  # width-height
            x = x[xc[xi]]  # confidence
            # Cat apriori labels if autolabelling
            if labels and len(labels[xi]):
                lb = labels[xi]
                v = torch.zeros((len(lb), nc + 5), device=x.device)
                v[:, :4] = lb[:, 1:5]  # box
                v[:, 4] = 1.0  # conf
                v[range(len(lb)), lb[:, 0].long() + 5] = 1.0  # cls
                x = torch.cat((x, v), 0)

            # If none remain process next image
            if not x.shape[0]:
                continue

            # Compute conf
            x[:, 5:] *= x[:, 4:5]  # conf = obj_conf * cls_conf

            # Box (center x, center y, width, height) to (x1, y1, x2, y2)
            box = self.xywh2xyxy(x[:, :4])
            mask = x[:, mi:]  # zero columns if no masks

            # Detections matrix nx6 (xyxy, conf, cls)
            if multi_label:
                i, j = (x[:, 5:mi] > conf_thres).nonzero(as_tuple=False).T
                x = torch.cat((box[i], x[i, j + 5, None], j[:, None].float(), mask[i]), dim=1)
            else:  # best class only
                conf, j = x[:, 5:mi].max(1, keepdim=True)
                x = torch.cat((box, conf, j.float(), mask), dim=1)[conf.view(-1) > conf_thres]

            # Filter by class
            if classes is not None:
                x = x[(x[:, 5:6] == torch.tensor(classes, device=x.device)).any(1)]

            # Check shape
            n = x.shape[0]  # number of boxes
            if not n:  # no boxes
                continue

            x = x[x[:, 4].argsort(descending=True)[:max_nms]]  # sort by confidence

            # Batched NMS
            c = x[:, 5:6] * (0 if agnostic else max_wh)  # classes
            boxes, scores = x[:, :4] + c, x[:, 4]  # boxes (offset by class), scores
            i = torchvision.ops.nms(boxes, scores, iou_thres)  # NMS
            i = i[:max_det]
            if merge and (1 < n < 3E3):  # Merge NMS (boxes merged using weighted mean)
                # update boxes as boxes(i,4) = weights(i,n) * boxes(n,4)
                iou = box_iou(boxes[i], boxes) > iou_thres  # iou matrix
                weights = iou * scores[None]  # box weights
                x[i, :4] = torch.mm(weights, x[:, :4]).float() / weights.sum(1, keepdim=True)  # merged boxes
                if redundant:
                    i = i[iou.sum(1) > 1]  # require redundancy

            output[xi] = x[i]
            if (time.time() - t) > time_limit:
                print(f'WARNING: NMS time limit {time_limit:.3f}s exceeded')
                break  # time limit exceeded

        return output


    def resize_unscale(self, img, new_shape=(640, 640), color=114):
        shape = img.shape[:2]  # current shape [height, width]
        if isinstance(new_shape, int):
            new_shape = (new_shape, new_shape)

        canvas = np.zeros((new_shape[0], new_shape[1], 3))
        canvas.fill(color)
        # Scale ratio (new / old) new_shape(h,w)
        r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])

        # Compute padding
        new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))  # w,h
        new_unpad_w = new_unpad[0]
        new_unpad_h = new_unpad[1]
        pad_w, pad_h = new_shape[1] - new_unpad_w, new_shape[0] - new_unpad_h  # wh padding

        dw = pad_w // 2  # divide padding into 2 sides
        dh = pad_h // 2

        if shape[::-1] != new_unpad:  # resize
            img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_AREA)

        canvas[dh:dh + new_unpad_h, dw:dw + new_unpad_w, :] = img

        return canvas, r, dw, dh, new_unpad_w, new_unpad_h  # (dw,dh)