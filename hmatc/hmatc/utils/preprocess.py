# Copyright 2025 HOUMO AI
#
# File: preprocess.py
# Description:
#   preprocess functions
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
import torch
import numpy as np
from enum import Enum
from . import logger
from .transform import BGR2YUV


def calc_padding_size(img_shape, target_size, padding_mode):
    """Calculate padding size for image resizing.

    Args:
        img_shape (tuple): Input image shape as (height, width)
        target_size (tuple): Target size as (width, height)
        padding_mode (int): Padding mode (0: left/top padding, 1: center padding)

    Returns:
        tuple: (padding_size, size, scale_factor) where:
            - padding_size: [top, left, bottom, right] padding values
            - size: new image size [height, width]
            - scale_factor: scaling factor from original to new height
    """
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
            logger.fatal("Not support padding mode -> {}".format(padding_mode))
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
            logger.fatal("Not support padding mode -> {}".format(padding_mode))

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
    """Resize image with different resize strategies.

    Args:
        im: Input image
        size (tuple): Target size as (width, height)
        resize_type (int): Resize type
            - 0: direct resize to target size
            - 1: aspect ratio resize with symmetric padding (keep aspect ratio)
            - 2: fixed height resize with right padding (OCR recognition style)
        padding_value (int or list): Value for padding
        padding_mode (int): Padding mode (0: left/top padding, 1: center padding)
        interpolation: OpenCV interpolation method

    Returns:
        Resized image
    """
    import math

    if resize_type not in [0, 1, 2]:
        logger.fatal("resize_type must be equal 0, 1 or 2")

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
        # Fixed height, aspect-ratio width, right padding
        # Used for OCR recognition models (e.g., PPOCRv3 rec)
        tw, th = size
        h, w = im.shape[:2]
        # Ensure even dimensions
        h &= ~1
        w &= ~1
        im = cv2.resize(im, (w, h), interpolation=interpolation).copy()

        # Calculate aspect-ratio width based on fixed height
        ratio = w / float(h)
        resized_w = int(math.ceil(th * ratio))
        if resized_w > tw:
            resized_w = tw
        else:
            resized_w &= ~1

        resized_h = th
        im = cv2.resize(im, (resized_w, resized_h), interpolation=interpolation)

        # Right padding
        right = tw - resized_w
        if right > 0:
            im = cv2.copyMakeBorder(
                im, 0, 0, 0, right, cv2.BORDER_CONSTANT, value=padding_value
            )
        return im


def convert_bgr_to_yuv(im, fmt="YUV420SP", to_NCHW=False):
    """Convert BGR image to YUV format.

    Args:
        im: Input image tensor in BGR format
        fmt (str): YUV format string

    Returns:
        Converted image in YUV format
    """
    if im.ndim != 4:
        raise ValueError(f"Expected NCHW image tensor, got shape {list(im.shape)}")

    n, c, h, w = im.shape
    if fmt == "YUV400":
        if c != 1:
            raise ValueError(f"YUV400 expects one input channel, got {c}")
        yuv_im = im.permute(0, 2, 3, 1).contiguous()
    else:
        if c != 3:
            raise ValueError(f"{fmt} expects three input channels, got {c}")
        converter = BGR2YUV(fmt=fmt)
        yuv_im = torch.stack(
            [converter(image.type(torch.float32)) for image in im], dim=0
        ).type(torch.uint8)
    if to_NCHW:
        return yuv_im.reshape(n, c, h, w)
    return yuv_im[0] if n == 1 else yuv_im


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
    """Default preprocessing function for images.

    Args:
        im: Input BGR or grayscale image
        size (tuple): Target size as (width, height)
        mean (list or tuple): Mean values for normalization
        std (list or tuple): Standard deviation values for normalization
        use_norm (bool): Whether to apply normalization
        use_rgb (bool): Whether to convert BGR to RGB
        use_resize (bool): Whether to resize the image
        resize_type (int): Resize type (0: direct resize, 1: aspect ratio resize with padding)
        interpolation: OpenCV interpolation method
        padding_value (int or list): Padding value(s)
        padding_mode (int): Padding mode (0: left/top, 1: center)
        to_YUV (bool): Whether to convert to YUV format
        fmt (str): YUV format string

    Returns:
        Preprocessed image tensor
    """
    if im is None:
        logger.fatal("Image is None, please check!")

    if len(im.shape) not in [2, 3]:
        logger.fatal("Image must be 2d or 3d")

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
            logger.fatal("mean must be list or tuple")
        if not isinstance(std, list) and not isinstance(std, tuple):
            logger.fatal("mean must be list or tuple")
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


def clip_resize_scale(src_size, dst_size):
    """Clip resize scale to valid range [1/32, 16].

    Args:
        src_size (tuple): Source size as (height, width)
        dst_size (tuple): Destination size as (height, width)

    Returns:
        tuple: Clipped size (height, width) as even numbers
    """
    nh, nw = src_size
    th, tw = dst_size
    sh = float(th) / nh
    sw = float(tw) / nw
    if sh > 16 or sh < 1.0 / 32:
        nh = int(nh * max(1.0 / 32, min(16, sh))) & ~1
    if sw > 16 or sw < 1.0 / 32:
        nw = int(nw * max(1.0 / 32, min(16, sw))) & ~1
    return nh, nw


def resizer_preprocess_v1(
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
    return_dynamic_v1_format=False,
    crop_size=None,
):
    """Advanced preprocessing function with dynamic resizing and padding.

    Args:
        cv_image: Input OpenCV image
        input_shape (tuple): Input tensor shape (N, C, H, W)
        max_input_size (tuple): Maximum input size (max_height, max_width)
        mean (list or tuple): Mean values for normalization
        std (list or tuple): Standard deviation values for normalization
        use_resize (bool): Whether to resize the image
        use_norm (bool): Whether to apply normalization
        use_rgb (bool): Whether to convert BGR to RGB
        resize_type (int): Resize type (0: direct resize, 1: aspect ratio resize with padding)
        padding_mode (int): Padding mode (0: left/top, 1: center)
        padding_values (list): Padding values for each channel
        is_onnx (bool): Whether to return ONNX-compatible format
        to_YUV (bool): Whether to convert to YUV format
        fmt (str): YUV format string
        return_dynamic_v1_format (bool): Whether to return dynamic v1 format info
        crop_size (list or None): Crop size [start_h, start_w, end_h, end_w]
    Returns:
        tuple: (processed_image, dynamic_info) where:
            - processed_image: Preprocessed image tensor
            - dynamic_info: Dynamic information tensor for variable input shapes
    """
    _, C, H, W = input_shape
    height, width = cv_image.shape[:2]
    height &= ~1
    width &= ~1
    cv_image = cv2.resize(cv_image, (width, height), interpolation=cv2.INTER_LINEAR)
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

    _, _, nh, nw = im.shape
    max_height, max_width = max_input_size
    if nh > max_height or nw > max_width:
        padding_size, size, _ = calc_padding_size(
            (nh, nw),
            (max_width, max_height),
            padding_mode=0,
        )
        nh, nw = size
        im = torch.nn.functional.interpolate(
            torch.from_numpy(im),
            size=(nh, nw),
            mode="bilinear",
            align_corners=False,
        )
        im = im.detach().cpu().numpy()

    if resize_type == 1:
        padding_size, size, _ = calc_padding_size((nh, nw), (W, H), padding_mode)
    elif resize_type == 0:
        padding_size = [0, 0, 0, 0]
        size = [H, W]

    nh, nw = clip_resize_scale((nh, nw), size)
    im = torch.nn.functional.interpolate(
        torch.from_numpy(im),
        size=(nh, nw),
        mode="bilinear",
        align_corners=False,
    )
    im = im.detach().cpu().numpy()
    if resize_type == 1:
        padding_size, size, _ = calc_padding_size((nh, nw), (W, H), padding_mode)

    resizer_crop = [0, 0, nh, nw]
    if crop_size is not None:
        resizer_crop = crop_size
    resizer_size = size
    resizer_padding = padding_size

    # resizer
    padding_im = np.zeros((1, C, max_height, max_width), dtype=np.uint8)
    padding_im[
        :, :, resizer_crop[0] : resizer_crop[2], resizer_crop[1] : resizer_crop[3]
    ] = im
    dyn_info = list()
    dyn_info.extend(resizer_crop)
    if resize_type == 1 or return_dynamic_v1_format:
        dyn_info.extend(resizer_size)
        dyn_info.extend(resizer_padding)
    dyn_info = torch.Tensor(dyn_info).type(torch.int32).view(1, -1)
    padding_im = torch.from_numpy(padding_im)
    if to_YUV:
        padding_im = convert_bgr_to_yuv(padding_im, fmt=fmt)
    return padding_im, dyn_info


def resizer_preprocess(
    cv_image: np.ndarray,
    input_shape: list,
    resizer_input_size: list,
    resizer_crop: list,  # 配置文件的crop参数仅对静态resizer有效
    resizer_mode: int = 3,
    mean: list = None,
    std: list = None,
    use_resize: bool = False,
    use_norm: bool = False,
    use_rgb: bool = False,
    resize_type: int = 1,
    padding_mode: int = 1,
    padding_values: list = [114, 114, 144],
    is_onnx: bool = False,
    to_YUV: bool = False,
    fmt: str = "YUV420SP",
):
    # onnx模型输入
    _, C, H, W = input_shape
    # Ensure even dimensions for input image
    orig_height, orig_width = cv_image.shape[:2]
    orig_height &= ~1
    orig_width &= ~1
    cv_image = cv2.resize(cv_image, (orig_width, orig_height)).copy()

    if is_onnx or resizer_mode == 0:
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
        return torch.from_numpy(im), torch.tensor([], dtype=torch.int32)

    resizer_input_h, resizer_input_w = resizer_input_size
    padded_im = np.zeros((1, C, resizer_input_h, resizer_input_w), dtype=np.uint8)

    if resizer_mode == 3:
        crop_y, crop_x, crop_h, crop_w = resizer_crop
        dyn_tensor = torch.tensor([], dtype=torch.int32)
    elif resizer_mode == 2:
        crop_y, crop_x, crop_h, crop_w = 0, 0, H, W
        dyn_tensor = torch.tensor([[crop_y, crop_x, crop_h, crop_w]], dtype=torch.int32)
    elif resizer_mode == 1:
        crop_y, crop_x, crop_h, crop_w = 0, 0, H, W
        dyn_tensor = torch.tensor(
            [[crop_y, crop_x, crop_h, crop_w, H, W, 0, 0, 0, 0]], dtype=torch.int32
        )
    else:
        logger.fatal(f"Invalid resizer_mode={resizer_mode}")

    im = default_preprocess(
        cv_image,
        (crop_w, crop_h),
        use_norm=False,
        use_resize=True,
        use_rgb=use_rgb,
        resize_type=resize_type,
        padding_mode=padding_mode,
        padding_value=padding_values,
    )
    padded_im[:, :, crop_y : crop_y + crop_h, crop_x : crop_x + crop_w] = im.copy()
    im_tensor = torch.from_numpy(padded_im)
    if to_YUV:
        im_tensor = convert_bgr_to_yuv(im_tensor, fmt=fmt, to_NCHW=True)
    return im_tensor, dyn_tensor
