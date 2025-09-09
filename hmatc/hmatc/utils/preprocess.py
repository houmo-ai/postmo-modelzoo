#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import cv2
import torch
import numpy as np
from enum import Enum
from . import logger
from .transform import BGR2YUV


def calc_padding_size(img_shape, target_size, padding_mode):
    top, bottom, left, right = 0, 0, 0, 0
    tw, th = target_size
    h, w = img_shape
    sw, sh = float(w) / tw, float(h) / th
    if sw > sh:
        s = sw
        nw = tw
        nh = int(h / s)
        nh &= ~1
        if padding_mode == 0:
            bottom = th - nh
        elif padding_mode == 1:
            top = int((th - nh) * 0.5)
            top &= ~1
            bottom = th - nh - top
        else:
            logger.error("Not support padding mode -> {}".format(padding_mode))
            exit(-1)
    else:
        s = sh
        nh = th
        nw = int(w / s)
        nw &= ~1
        if padding_mode == 0:
            right = tw - nw
        elif padding_mode == 1:
            left = int((tw - nw) * 0.5)
            left &= ~1
            right = tw - nw - left
        else:
            logger.error("Not support padding mode -> {}".format(padding_mode))
            exit(-1)

    padding_size = [top, left, bottom, right]
    size = [nh, nw]
    return padding_size, size, float(h) / nh


def resize(
    im,
    size,
    resize_type=1,
    padding_value=114,
    padding_mode=1,
    interpolation=cv2.INTER_LINEAR,
):
    """opencv resize封装，目前仅支持双线性差值
    :param im:
    :param size:
    :param resize_type:  0-长宽分别resize，1-长边等比例resize，2-短边等比例resize，默认为0
    :param padding_value:
    :param padding_mode: 0-LEFT_TOP, 1-CENTER
    :param interpolation:
    :return:
    """
    if resize_type not in [0, 1, 2]:
        logger.error("resize_type must be equal 0 or 1 or 2")
        exit(-1)

    if resize_type == 0:
        return cv2.resize(im, size, interpolation=interpolation)

    if resize_type == 1:
        padding_size, nsize, _ = calc_padding_size(
            (im.shape[0], im.shape[1]), size, padding_mode=padding_mode
        )
        h, w = nsize
        im = cv2.resize(im, (w, h), interpolation=interpolation)
        top, left, bottom, right = padding_size
        return cv2.copyMakeBorder(
            im, top, bottom, left, right, cv2.BORDER_CONSTANT, value=padding_value
        )

    if resize_type == 2:
        logger.error("Not support yet")
        exit(-1)


def convert_bgr_to_yuv(im, fmt="YUV420SP"):
    if fmt == "YUV400":
        assert len(im.shape) == 2
        h, w = im.shape
        yuv_im = im.view(h, w, 1)  # HWC
    else:
        im_chw = torch.squeeze(im, dim=0).type(torch.float32)  # CHW
        yuv_im = BGR2YUV(fmt=fmt)(im_chw)
        yuv_im = yuv_im.type(torch.uint8)  # HWC
    return yuv_im


def default_preprocess(
    im,
    size,
    mean=None,
    std=None,
    use_norm=True,
    use_rgb=False,
    use_resize=True,
    resize_type=0,
    interpolation=cv2.INTER_LINEAR,
    padding_value=114,
    padding_mode=1,
    to_YUV=False,
    fmt="YUV420SP",
):
    """默认预处理函数
    :param im: BGR or GRAY图像
    :param size:
    :param mean:
    :param std:
    :param use_norm:
    :param use_rgb:
    :param use_resize:
    :param interpolation:
    :param resize_type:  0-长宽分别resize，1-长边等比例resize，默认为0
    :param padding_value:
    :param padding_mode:  0-LEFT_TOP, 1-CENTER
    :return:
    """
    if im is None:
        logger.error("Image is None, please check!")
        exit(-1)

    if len(im.shape) not in [2, 3]:
        logger.error("Image must be 2d or 3d")
        exit(-1)

    if use_rgb and len(im.shape) == 3:
        im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)

    if use_resize:
        if (
            not (isinstance(padding_value, list) or isinstance(padding_value, tuple))
            and len(im.shape) == 3
        ):
            padding_value = [padding_value for _ in range(im.shape[2])]
        im = resize(
            im,
            size,
            resize_type=resize_type,
            padding_value=padding_value,
            padding_mode=padding_mode,
            interpolation=interpolation,
        )

    if use_norm:
        if not isinstance(mean, list) and not isinstance(mean, tuple):
            logger.error("mean must be list or tuple")
            exit(-1)
        if not isinstance(std, list) and not isinstance(std, tuple):
            logger.error("mean must be list or tuple")
            exit(-1)
        im = im.astype(dtype=np.float32)
        im -= np.array(mean, dtype=np.float32)
        im /= np.array(std, dtype=np.float32)

    if len(im.shape) == 2:
        im = np.expand_dims(im, 0)
        im = np.expand_dims(im, 3)
    else:
        im = np.expand_dims(im, 0)

    im = np.ascontiguousarray(im.transpose((0, 3, 1, 2)))
    if to_YUV:
        im = torch.from_numpy(im)
        im = convert_bgr_to_yuv(im, fmt=fmt)
    return im


def xh1_preprocess(
    cv_image,
    input_shape,
    max_input_size,
    mean=None,
    std=None,
    use_resize=False,
    use_norm=False,
    use_rgb=True,
    resize_type=1,
    padding_mode=1,
    padding_values=[114, 114, 144],
    is_onnx=False,
    to_YUV=False,
    fmt="YUV420SP",
):
    _, C, H, W = input_shape
    height, width = cv_image.shape[:2]
    # resizer需要全为偶数，一般图片宽高为偶数
    height &= ~1
    width &= ~1
    cv_image = np.ascontiguousarray(cv_image[0:height, 0:width, ...])
    # BGR/GRAY(HWC)->RGB/GRAY(HWC)->NCHW
    im = default_preprocess(
        cv_image,
        (W, H),
        mean,
        std,
        use_norm=use_norm,
        use_rgb=use_rgb,
        use_resize=use_resize,
        resize_type=resize_type,
        padding_mode=padding_mode,
        padding_value=padding_values,
    )
    if is_onnx:
        return torch.from_numpy(im), list()

    _, nc, nh, nw = im.shape
    max_height, max_width = max_input_size
    if nh > max_height or nw > max_width:
        nh, nw = H, W
        if resize_type == 1:
            padding_size, size, _ = calc_padding_size(
                (nh, nw), (max_width, max_height), padding_mode=0
            )
            nh, nw = size
        im = torch.nn.functional.interpolate(
            torch.from_numpy(im), size=(nh, nw), mode="bilinear", align_corners=False
        )
        im = im.detach().cpu().numpy()

    # resizer
    padding_im = np.zeros((1, C, max_height, max_width), dtype=np.uint8)
    padding_im[:, :, 0:nh, 0:nw] = im  # 贴至填充图
    dyn_info = list()
    if resize_type == 1:
        padding_size, size, _ = calc_padding_size((nh, nw), (W, H), padding_mode)
    elif resize_type == 0:
        padding_size = [0, 0, 0, 0]
        size = [H, W]
    resizer_crop = [0, 0, nh, nw]
    resizer_size = size
    resizer_padding = padding_size
    dyn_info.extend(resizer_crop)
    if resize_type == 1:
        dyn_info.extend(resizer_size)
        dyn_info.extend(resizer_padding)
    dyn_info = torch.Tensor(dyn_info).type(torch.int32).view(1, -1)
    padding_im = torch.from_numpy(padding_im)
    if to_YUV:
        padding_im = convert_bgr_to_yuv(padding_im, fmt=fmt)
    return padding_im, dyn_info
