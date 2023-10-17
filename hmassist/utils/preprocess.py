#!/usr/bin/env python3

import cv2
import numpy as np
from utils import logger
from utils.enum_type import PaddingMode


def calc_padding_size(im, target_size, padding_mode):
    top, bottom, left, right = 0, 0, 0, 0
    tw, th = target_size
    h, w = im.shape[0], im.shape[1]
    sw, sh = float(w) / tw, float(h) / th
    if sw > sh:
        s = sw
        nw = tw
        nh = int(h / s)
        if padding_mode == PaddingMode.LEFT_TOP:
            bottom = th - nh
        elif padding_mode == PaddingMode.CENTER:
            top = int((th - nh) * 0.5)
            bottom = th - nh - top
        else:
            logger.error("Not support padding mode -> {}".format(padding_mode))
            exit(-1)
    else:
        s = sh
        nh = th
        nw = int(w / s)
        if padding_mode == PaddingMode.LEFT_TOP:
            right = tw - nw
        elif padding_mode == PaddingMode.CENTER:
            left = int((tw - nw) * 0.5)
            right = tw - nw - left
        else:
            logger.error("Not support padding mode -> {}".format(padding_mode))
            exit(-1)

    padding_size = [top, left, bottom, right]
    size = (nh, nw)
    return padding_size, size


def calc_padding_size2(im, target_size, padding_mode):
    """
    计算padding_size
    :param im:
    :param target_size:
    :param padding_mode:  仅支持左上角(LEFT_TOP)和中心点(CENTER)
    :return: 上(top)/下(bottom)/左(left)/右(right)向外偏移像素，如[0, 10, 10, 20]；留空表示自动计算offset size
    """
    top, bottom, left, right = 0, 0, 0, 0

    tw, th = target_size
    h, w = im.shape[0], im.shape[1]
    if h > w:
        nh = th
        s = float(h) / nh
        nw = int(float(w) / s)
        if padding_mode == PaddingMode.LEFT_TOP:
            right = tw - nw
        elif padding_mode == PaddingMode.CENTER:
            left = int((tw - nw) * 0.5)
            right = tw - nw - left
        else:
            logger.error("Not support padding mode -> {}".format(padding_mode))
            exit(-1)
    else:
        nw = tw
        s = float(w) / nw
        nh = int(float(h) / s)

        if padding_mode == PaddingMode.LEFT_TOP:
            bottom = th - nh
        elif padding_mode == PaddingMode.CENTER:
            top = int((th - nh) * 0.5)
            bottom = th - nh - top
        else:
            logger.error("Not support padding mode -> {}".format(padding_mode))
            exit(-1)

    padding_size = [top, left, bottom, right]
    size = (nh, nw)
    return padding_size, size


def resize(im, size, resize_type=0, padding_value=128, padding_mode=PaddingMode.LEFT_TOP,
           interpolation=cv2.INTER_LINEAR):
    """opencv resize封装，目前仅支持双线性差值
    :param im:
    :param size:
    :param resize_type:  0-长宽分别resize，1-长边等比例resize，2-短边等比例resize，默认为0
    :param padding_value:
    :param padding_mode:
    :param interpolation:
    :return:
    """
    if resize_type not in [0, 1, 2]:
        logger.error("resize_type must be equal 0 or 1 or 2")
        exit(-1)

    if resize_type == 0:
        return cv2.resize(im, size, interpolation=interpolation)

    if resize_type == 1:
        padding_size, nsize = calc_padding_size(im, size, padding_mode=padding_mode)
        h, w = nsize
        im = cv2.resize(im, (w, h), interpolation=interpolation)
        top, left, bottom, right = padding_size

        return cv2.copyMakeBorder(im, top, bottom, left, right, cv2.BORDER_CONSTANT, value=padding_value)

    if resize_type == 2:
        logger.error("Not support yet")
        exit(-1)


def default_preprocess(im, size, mean=None, std=None, use_norm=True, use_rgb=False, use_resize=True, resize_type=0,
                       interpolation=cv2.INTER_LINEAR, padding_value=128, padding_mode=PaddingMode.LEFT_TOP):
    """默认预处理函数
    :param im: BGR or GRAY图像
    :param size:
    :param mean:
    :param std:
    :param use_norm:
    :param use_rgb:
    :param use_resize:
    :param interpolation:
    :param resize_type:  0-长宽分别resize，1-长边等比例resize，2-短边等比例resize，默认为0
    :param padding_value:
    :param padding_mode:  目前仅支持左上角(LEFT_TOP)和中心点(CENTER)
    :return:
    """
    if im is None:
        logger.error("Image is None, please check!")
        exit(-1)

    if use_resize:
        im = resize(im, size, resize_type=resize_type,
                    padding_value=padding_value, padding_mode=padding_mode, interpolation=interpolation)

    if len(im.shape) not in [2, 3]:
        logger.error("Image must be 2d or 3d")
        exit(-1)

    if use_rgb and len(im.shape) == 3:
        im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)

    if use_norm:
        im = im.astype(dtype=np.float32)
        if mean:
            im -= np.array(mean, dtype=np.float32)
        if std:
            im /= np.array(std, dtype=np.float32)

    if len(im.shape) == 2:
        im = np.expand_dims(im, 0)
        im = np.expand_dims(im, 3)
    else:
        im = np.expand_dims(im, 0)

    return np.ascontiguousarray(im.transpose((0, 3, 1, 2)))
