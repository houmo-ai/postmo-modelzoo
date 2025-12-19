import os
from PIL import Image
from . import logger
from .utils import SUPPORT_IMAGE_FORMATS


def get_image_size(filepath):
    with Image.open(filepath) as im:
        width, height = im.size
        return width, height


def check_cfg(cfg):
    """check config"""
    # model info
    model_cfg = cfg.get("model")
    save_dir = model_cfg.get("save_dir")
    if save_dir is None:
        logger.error("save_dir not found")
        return False
    inputs_cfg = model_cfg.get("inputs", dict())
    if len(inputs_cfg) == 0:
        logger.error("Please set inputs")
        return False
    for input_name in inputs_cfg:
        input_cfg = inputs_cfg[input_name]
        shape = input_cfg.get("shape")
        if shape is None:
            logger.error(f"{input_name} shape error -> {shape}")
            return False
        if not isinstance(shape, list):
            logger.error(f"{input_name} shape must be list")
            return False
        data_format = input_cfg.get("data_format")
        if data_format is None:
            continue
        # 图像
        if data_format not in ["RGB", "BGR", "GRAY"]:
            logger.error(f"Not support data_format: {data_format}")
            return False
        if len(shape) != 4:
            logger.error(f"{input_name} shape must be [N, C, H, W]")
            return False
        channels = shape[1]
        mean = input_cfg.get("mean")
        std = input_cfg.get("std")
        N, C, H, W = shape
        if mean is None:
            logger.error(f"model mean error")
            return False
        if std is None:
            logger.error(f"model std error")
            return False
        if not isinstance(mean, list):
            logger.error(f"model mean must be list")
            return False
        if not isinstance(std, list):
            logger.error(f"model std must be list")
            return False
        resize_type = input_cfg.get("resize_type")
        if resize_type not in [0, 1]:
            logger.error(f"resize_type must be equal 0 or 1")
            return False
        if resize_type == 1:
            padding_mode = input_cfg.get("padding_mode")
            if padding_mode is None:
                logger.error(f"padding_mode error, when resize_type is 1")
                return False
            if padding_mode not in [0, 1]:
                logger.error(
                    f"padding_mode must be equal 0 or 1, when resize_type is 1"
                )
                return False
            padding_values = input_cfg.get("padding_values")
            if padding_values is None:
                logger.error(f"padding_values error, when resize_type is 1")
                return False
            if not isinstance(padding_values, list):
                logger.error(f"padding_values must be list, when resize_type is 1")
                return False
            if len(padding_values) != channels:
                logger.error(
                    f"padding_values length must be equal to channels, when resize_type is 1"
                )
                return False
        # resizer
        resizer_cfg = input_cfg.get("resizer")
        if not isinstance(resizer_cfg, dict) and resizer_cfg is not None:
            logger.error(f"resizer param error, must be dict or None")
            return False
        if resizer_cfg is not None:
            enable_static_resizer = resizer_cfg.get("enable_static_resizer", True)
            if enable_static_resizer not in [False, True]:
                logger.error(f"enable_static_resizer must be equal False or True")
                return False
            max_input_size = resizer_cfg.get("max_input_size", [H, W])
            if not isinstance(max_input_size, list) or len(max_input_size) != 2:
                logger.error(
                    f"max_input_size must be list, and [H, W], when use resizer"
                )
                return False
            # 需保证max_input_size为偶数
            for v in max_input_size:
                if v % 2 != 0:
                    logger.error(f"max_input_size[H, W] must be even number")
                    return False
            # 如果max_input_size比input WH小给出警告
            resizer_input_h, resizer_input_w = max_input_size
            if resizer_input_h < H or resizer_input_w < W:
                logger.warning(f"max_input_size[H, W] should be greater than [H, W]")

            if enable_static_resizer and "crop_size" in resizer_cfg:
                crop_size = resizer_cfg.get("crop_size", [0, 0, H, W])
                y1, x1, crop_height, crop_width = crop_size
                x2, y2 = x1 + crop_width, y1 + crop_height
                # 检查crop_size是否均为偶数
                for v in crop_size:
                    if v % 2 != 0:
                        logger.error(f"crop_size must be even number: {crop_size}")
                        return False
                if x1 < 0 or y1 < 0 or y2 > max_input_size[0] or x2 > max_input_size[1]:
                    logger.error(
                        f"crop_size must be in [0, 0, {max_input_size[0]}, {max_input_size[1]}]"
                    )
                    return False
            toYUV_format = resizer_cfg.get("toYUV_format")
            if toYUV_format not in ["YUV400", "YUV420SP", "YUV422SP", "YUV444SP"]:
                logger.error(
                    "toYUV_format should be in [YUV400, YUV420SP, YUV422SP, YUV444SP], when use resizer"
                )
                return False
            if data_format in ["RGB", "BGR"] and toYUV_format == "YUV400":
                logger.error(
                    f"data_format in [RGB, BGR], toYUV_format must be in [YUV420SP, YUV422SP, YUV444SP]"
                )
                return False
            if data_format == "GRAY" and toYUV_format != "YUV400":
                logger.error(f"data_format = GRAY, toYUV_format must be YUV400")
                return False
    return True
