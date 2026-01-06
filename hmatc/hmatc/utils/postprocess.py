# Copyright 2025 HOUMO AI
#
# File: postprocess.py
# Description:
#   Postprocess functions
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# SPDX-License-Identifier: Apache-2.0
import cv2
import numpy as np
import torch
import torch.nn.functional as F
import torchvision


def sigmoid(x):
    """Calculate sigmoid function for input x.

    Args:
        x: Input value or array

    Returns:
        Sigmoid-transformed value(s)
    """
    return 1 / (1 + np.exp(-x))


def softmax(x, axis=1, keepdims=True):
    """Apply softmax function to input array.

    Args:
        x: Input array
        axis: Axis along which to apply softmax
        keepdims: Whether to keep dimensions after operation

    Returns:
        Softmax-transformed array
    """
    e_x = np.exp(x - np.max(x, axis=axis, keepdims=keepdims))
    return e_x / np.sum(e_x, axis=axis, keepdims=keepdims)


def crop_mask(masks, boxes):
    """Crop predicted masks by zeroing out everything not in the predicted bbox.
    Vectorized by Chong (thanks Chong).

    Args:
        - masks should be a size [h, w, n] tensor of masks
        - boxes should be a size [n, 4] tensor of bbox coords in relative point form

    Returns:
        Cropped masks tensor
    """
    n, h, w = masks.shape
    x1, y1, x2, y2 = torch.chunk(boxes[:, :, None], 4, 1)  # x1 shape(1,1,n)
    # rows shape(1,w,1)
    r = torch.arange(w, device=masks.device, dtype=x1.dtype)[None, None, :]
    # cols shape(h,1,1)
    c = torch.arange(h, device=masks.device, dtype=x1.dtype)[None, :, None]

    return masks * ((r >= x1) * (r < x2) * (c >= y1) * (c < y2))


def process_mask(protos, masks_in, bboxes, shape, upsample=False):
    """Crop before upsample.
    proto_out: [mask_dim, mask_h, mask_w]
    out_masks: [n, mask_dim], n is number of masks after nms
    bboxes: [n, 4], n is number of masks after nms
    shape:input_image_size, (h, w)

    return: h, w, n
    """
    c, mh, mw = protos.shape  # CHW
    ih, iw = shape
    masks = (masks_in @ protos.float().view(c, -1)).sigmoid().view(-1, mh, mw)  # CHW

    downsampled_bboxes = bboxes.clone()
    downsampled_bboxes[:, 0] *= mw / iw
    downsampled_bboxes[:, 2] *= mw / iw
    downsampled_bboxes[:, 3] *= mh / ih
    downsampled_bboxes[:, 1] *= mh / ih

    masks = crop_mask(masks, downsampled_bboxes)  # CHW
    if upsample:
        masks = F.interpolate(masks[None], shape, mode="bilinear", align_corners=False)[
            0
        ]  # CHW
    return masks.gt_(0.5)


def clip_coords(boxes, shape):
    """Clip bounding xyxy bounding boxes to image shape (height, width).

    Args:
        boxes: Bounding boxes in xyxy format
        shape: Image shape as (height, width)
    """
    if isinstance(boxes, torch.Tensor):  # faster individually
        boxes[:, 0].clamp_(0, shape[1])  # x1
        boxes[:, 1].clamp_(0, shape[0])  # y1
        boxes[:, 2].clamp_(0, shape[1])  # x2
        boxes[:, 3].clamp_(0, shape[0])  # y2
    else:  # np.array (faster grouped)
        boxes[:, [0, 2]] = boxes[:, [0, 2]].clip(0, shape[1])  # x1, x2
        boxes[:, [1, 3]] = boxes[:, [1, 3]].clip(0, shape[0])  # y1, y2


def scale_coords(img1_shape, coords, img0_shape, need_pad=True):
    """Scale coordinates from img1_shape to img0_shape.

    Args:
        img1_shape: Shape of input image after transformation
        coords: Coordinates to be scaled
        img0_shape: Original image shape
        need_pad: Whether padding was applied

    Returns:
        Scaled coordinates
    """
    gain = min(img1_shape[0] / img0_shape[0], img1_shape[1] / img0_shape[1])
    if need_pad:
        pad = (img1_shape[1] - img0_shape[1] * gain) * 0.5, (
            img1_shape[0] - img0_shape[0] * gain
        ) * 0.5
    else:
        pad = (0, 0)

    coords[:, [0, 2]] -= pad[0]
    coords[:, [1, 3]] -= pad[1]
    coords[:, :4] /= gain
    if isinstance(coords, torch.Tensor):
        coords[:, [0, 2]].clamp_(0, img0_shape[1])
        coords[:, [1, 3]].clamp_(0, img0_shape[0])
    else:
        coords[:, [0, 2]] = coords[:, [0, 2]].clip(0, img0_shape[1])  # x1, x2
        coords[:, [1, 3]] = coords[:, [1, 3]].clip(0, img0_shape[0])  # y1, y2
    return coords


def scale_coords_kpt(img1_shape, coords, img0_shape, ratio_pad=None):
    """Scale coordinates for keypoints from img1_shape to img0_shape.

    Args:
        img1_shape: Shape of input image after transformation
        coords: Keypoint coordinates to be scaled
        img0_shape: Original image shape
        ratio_pad: Optional ratio and padding parameters

    Returns:
        Scaled coordinates as numpy array
    """
    if len(coords) == 0:
        return coords
    if ratio_pad is None:
        # gain  = old / new
        gain = min(img1_shape[0] / img0_shape[0], img1_shape[1] / img0_shape[1])
        pad = (img1_shape[1] - img0_shape[1] * gain) / 2, (
            img1_shape[0] - img0_shape[0] * gain
        ) / 2
    else:
        gain = ratio_pad[0][0]
        pad = ratio_pad[1]

    coords[:, 0] -= pad[0]  # x padding
    coords[:, 1] -= pad[1]  # y padding
    coords[:, 2] -= pad[0]  # x padding
    coords[:, 3] -= pad[1]  # y padding
    coords[:, 6::3] -= pad[0]
    coords[:, 7::3] -= pad[1]
    coords[:, 0:4] /= gain
    coords[:, 6::3] /= gain
    coords[:, 7::3] /= gain
    clip_coords(coords, img0_shape)
    return coords.numpy()


def scale_coords_mask(img1_shape, contours, img0_shape, ratio_pad=None):
    """Scale contour coordinates from img1_shape to img0_shape.

    Args:
        img1_shape: Shape of input image after transformation
        contours: Contour points to be scaled
        img0_shape: Original image shape
        ratio_pad: Optional ratio and padding parameters

    Returns:
        Scaled contours
    """
    if len(contours) == 0:
        return contours
    if ratio_pad is None:  # calculate from img0_shape
        gain = min(
            img1_shape[0] / img0_shape[0], img1_shape[1] / img0_shape[1]
        )  # gain  = old / new
        pad = (img1_shape[1] - img0_shape[1] * gain) / 2, (
            img1_shape[0] - img0_shape[0] * gain
        ) / 2  # wh padding
    else:
        gain = ratio_pad[0][0]
        pad = ratio_pad[1]

    for idx in range(len(contours)):
        contours[idx] = contours[idx].astype("float32")
        contours[idx][:, :, 0] -= pad[0]  # x padding
        contours[idx][:, :, 1] -= pad[1]  # y padding
        contours[idx] /= gain
        contours[idx][:, :, 0] = np.clip(contours[idx][:, :, 0], 0, img0_shape[1] - 1)
        contours[idx][:, :, 1] = np.clip(contours[idx][:, :, 1], 0, img0_shape[0] - 1)
        contours[idx] = contours[idx].round().astype("int32")
    return contours


def scale_coords_landmarks(img1_shape, coords, img0_shape, ratio_pad=None):
    """Scale landmark coordinates from img1_shape to img0_shape.

    Args:
        img1_shape: Shape of input image after transformation
        coords: Landmark coordinates to be scaled
        img0_shape: Original image shape
        ratio_pad: Optional ratio and padding parameters

    Returns:
        Scaled coordinates
    """
    if ratio_pad is None:
        gain = min(img1_shape[0] / img0_shape[0], img1_shape[1] / img0_shape[1])
        pad = (img1_shape[1] - img0_shape[1] * gain) * 0.5, (
            img1_shape[0] - img0_shape[0] * gain
        ) * 0.5
    else:
        gain = ratio_pad[0][0]
        pad = ratio_pad[1]

    coords[:, [0, 2, 4, 6, 8]] -= pad[0]  # x padding
    coords[:, [1, 3, 5, 7, 9]] -= pad[1]  # y padding
    coords[:, :10] /= gain
    # clip
    coords[:, 0].clamp_(0, img0_shape[1])  # x1
    coords[:, 1].clamp_(0, img0_shape[0])  # y1
    coords[:, 2].clamp_(0, img0_shape[1])  # x2
    coords[:, 3].clamp_(0, img0_shape[0])  # y2
    coords[:, 4].clamp_(0, img0_shape[1])  # x3
    coords[:, 5].clamp_(0, img0_shape[0])  # y3
    coords[:, 6].clamp_(0, img0_shape[1])  # x4
    coords[:, 7].clamp_(0, img0_shape[0])  # y4
    coords[:, 8].clamp_(0, img0_shape[1])  # x5
    coords[:, 9].clamp_(0, img0_shape[0])  # y5

    return coords


def xyxy2xywh(x):
    """Convert bounding box format from xyxy to xywh.

    Args:
        x: Bounding boxes in xyxy format [x1, y1, x2, y2]

    Returns:
        Bounding boxes in xywh format [x_center, y_center, width, height]
    """
    y = x.clone() if isinstance(x, torch.Tensor) else np.copy(x)
    y[:, 0] = (x[:, 0] + x[:, 2]) * 0.5
    y[:, 1] = (x[:, 1] + x[:, 3]) * 0.5
    y[:, 2] = x[:, 2] - x[:, 0]
    y[:, 3] = x[:, 3] - x[:, 1]
    return y


def xywh2xyxy(x):
    """Convert bounding box format from xywh to xyxy.

    Args:
        x: Bounding boxes in xywh format [x_center, y_center, width, height]

    Returns:
        Bounding boxes in xyxy format [x1, y1, x2, y2]
    """
    y = x.clone() if isinstance(x, torch.Tensor) else np.copy(x)
    y[:, 0] = x[:, 0] - x[:, 2] * 0.5
    y[:, 1] = x[:, 1] - x[:, 3] * 0.5
    y[:, 2] = x[:, 0] + x[:, 2] * 0.5
    y[:, 3] = x[:, 1] + x[:, 3] * 0.5
    return y


def non_max_suppression(
    preds: torch.Tensor,
    conf_thres=0.25,
    iou_thres=0.45,
    nm=0,
    max_det=300,
    exist_obj_conf=False,
):
    """Perform Non-Maximum Suppression (NMS) on inference results.

    Args:
        preds: Prediction tensor with shape [batch, num_predictions, 4 + nc + nm]
        conf_thres: Confidence threshold for filtering predictions
        iou_thres: IoU threshold for NMS
        nm: Number of mask coefficients
        max_det: Maximum number of detections to keep
        exist_obj_conf: Whether object confidence exists in predictions

    Returns:
        List of detection results for each image in batch
    """
    max_nms = 30000
    max_wh = 7680
    bs = preds.shape[0]
    if preds.shape[1] < preds.shape[2]:
        preds = preds.permute(0, 2, 1)
    if exist_obj_conf:
        box_data = preds[:, :, :4]
        cls_conf = preds[:, :, 4:5] * preds[:, :, 5:]
        preds = torch.cat([box_data, cls_conf], dim=2)
    nc = preds.shape[2] - nm - 4
    mi = 4 + nc
    xc = preds[:, :, 4:mi].amax(dim=2, keepdim=False) > conf_thres
    output = [torch.zeros((0, 6 + nm), device=preds.device)] * bs
    for batch_idx, pred in enumerate(preds):
        pred = pred[xc[batch_idx]]  # confidence
        if not pred.shape[0]:
            continue
        box = xywh2xyxy(pred[:, :4])
        mask = pred[:, mi:]  # zero columns if no masks
        conf, j = pred[:, 4:mi].max(1, keepdim=True)  # (xyxy, conf, cls)
        pred = torch.cat((box, conf, j.float(), mask), dim=1)[
            conf.view(-1) > conf_thres
        ]
        n = pred.shape[0]
        if not n:
            continue
        pred = pred[pred[:, 4].argsort(descending=True)[:max_nms]]
        c = pred[:, 5:6] * max_wh
        boxes, scores = pred[:, :4] + c, pred[:, 4]
        keep_idx = torchvision.ops.nms(boxes, scores, iou_thres)
        keep_idx = keep_idx[:max_det]
        output[batch_idx] = pred[keep_idx]
    return output


def plot_skeleton_kpts(im, kpts, steps, orig_shape=None):
    """Plot skeleton and keypoints for COCO dataset.

    Args:
        im: Image array to draw on
        kpts: Keypoints data
        steps: Number of values per keypoint (2 for x,y or 3 for x,y,conf)
        orig_shape: Original image shape for scaling

    Returns:
        Image with skeleton and keypoints drawn
    """
    palette = np.array(
        [
            [255, 128, 0],
            [255, 153, 51],
            [255, 178, 102],
            [230, 230, 0],
            [255, 153, 255],
            [153, 204, 255],
            [255, 102, 255],
            [255, 51, 255],
            [102, 178, 255],
            [51, 153, 255],
            [255, 153, 153],
            [255, 102, 102],
            [255, 51, 51],
            [153, 255, 153],
            [102, 255, 102],
            [51, 255, 51],
            [0, 255, 0],
            [0, 0, 255],
            [255, 0, 0],
            [255, 255, 255],
        ]
    )

    skeleton = [
        [16, 14],
        [14, 12],
        [17, 15],
        [15, 13],
        [12, 13],
        [6, 12],
        [7, 13],
        [6, 7],
        [6, 8],
        [7, 9],
        [8, 10],
        [9, 11],
        [2, 3],
        [1, 2],
        [1, 3],
        [2, 4],
        [3, 5],
        [4, 6],
        [5, 7],
    ]

    pose_limb_color = palette[
        [9, 9, 9, 9, 7, 7, 7, 0, 0, 0, 0, 0, 16, 16, 16, 16, 16, 16, 16]
    ]
    pose_kpt_color = palette[[16, 16, 16, 16, 16, 0, 0, 0, 0, 0, 0, 9, 9, 9, 9, 9, 9]]
    radius = 5
    num_kpts = len(kpts) // steps

    for kid in range(num_kpts):
        r, g, b = pose_kpt_color[kid]
        x_coord, y_coord = kpts[steps * kid], kpts[steps * kid + 1]
        if not (x_coord % 640 == 0 or y_coord % 640 == 0):
            if steps == 3:
                conf = kpts[steps * kid + 2]
                if conf < 0.5:
                    continue
            cv2.circle(
                im, (int(x_coord), int(y_coord)), radius, (int(r), int(g), int(b)), -1
            )

    for sk_id, sk in enumerate(skeleton):
        r, g, b = pose_limb_color[sk_id]
        pos1 = (int(kpts[(sk[0] - 1) * steps]), int(kpts[(sk[0] - 1) * steps + 1]))
        pos2 = (int(kpts[(sk[1] - 1) * steps]), int(kpts[(sk[1] - 1) * steps + 1]))
        if steps == 3:
            conf1 = kpts[(sk[0] - 1) * steps + 2]
            conf2 = kpts[(sk[1] - 1) * steps + 2]
            if conf1 < 0.5 or conf2 < 0.5:
                continue
        if pos1[0] % 640 == 0 or pos1[1] % 640 == 0 or pos1[0] < 0 or pos1[1] < 0:
            continue
        if pos2[0] % 640 == 0 or pos2[1] % 640 == 0 or pos2[0] < 0 or pos2[1] < 0:
            continue
        cv2.line(im, pos1, pos2, (int(r), int(g), int(b)), thickness=2)
