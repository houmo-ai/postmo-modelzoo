import os
from PIL import Image
from . import logger
from .utils import SUPPORT_IMAGE_FORMATS


def get_image_size(filepath):
    with Image.open(filepath) as im:
        width, height = im.size
        return width, height
    
                 
def check_cfg(cfg):
    """check config
    """    
    # model info
    model_cfg = cfg.get("model")
    save_dir = model_cfg.get("save_dir")
    if save_dir is None:
        logger.error("save_dir not found")
        exit(-1) 
    inputs_cfg = model_cfg.get("inputs")
    for input_name, input_cfg in inputs_cfg.items():
        input_cfg = inputs_cfg[input_name]
        shape = input_cfg.get("shape")
        if shape is None:
            logger.error(f"{input_name} shape error -> {shape}")
            exit(-1)
        if not isinstance(shape, list):
            logger.error(f"{input_name} shape must be list")
            exit(-1)
        data_format = input_cfg.get("data_format")
        if data_format is None:
            continue
        # 图像
        if data_format not in ["RGB", "BGR", "GRAY"]:
            logger.error(f"Not support data_format -> {data_format}")
            exit(-1)
        if len(shape) != 4:
            logger.error(f"{input_name} shape must be [N, C, H, W]")
            exit(-1)
        channels = shape[1]
        mean = input_cfg.get("mean")
        std = input_cfg.get("std")
        if mean is None:
            logger.error(f"model mean error")
            exit(-1)
        if std is None:
            logger.error(f"model std error")
            exit(-1)
        if not isinstance(mean, list):
            logger.error(f"model mean must be list")
            exit(-1)
        if not isinstance(std, list):
            logger.error(f"model std must be list")
            exit(-1)
        resize_type = input_cfg.get("resize_type")
        if resize_type not in [0, 1]:
            logger.error(f"resize_type must be equal 0 or 1")
            exit(-1)
        if resize_type == 1:
            padding_mode = input_cfg.get("padding_mode")
            if padding_mode is None:
                logger.error(f"padding_mode error, when resize_type is 1")
                exit(-1)
            if padding_mode not in [0, 1]:
                logger.error(f"padding_mode must be equal 0 or 1, when resize_type is 1")
                exit(-1)
            padding_values = input_cfg.get("padding_values")
            if padding_values is None:
                logger.error(f"padding_values error, when resize_type is 1")
                exit(-1)
            if not isinstance(padding_values, list):
                logger.error(f"padding_values must be list, when resize_type is 1")
                exit(-1)
            if len(padding_values) != channels:
                logger.error(f"padding_values length must be equal to channels, when resize_type is 1")
                exit(-1)
        # resizer
        resizer_cfg = input_cfg.get("resizer")
        if not isinstance(resizer_cfg, dict) and resizer_cfg is not None:
            logger.error(f"resizer param error, must be dict or None")
            exit(-1)
        if resizer_cfg is not None:
            enable_static_resizer = resizer_cfg.get("enable_static_resizer", False)
            if enable_static_resizer not in [False, True]:
                logger.error(f"enable_static_resizer must be equal False or True")
                exit(-1)
            max_input_size = resizer_cfg.get("max_input_size")
            if not isinstance(max_input_size, list) or len(max_input_size) != 2:
                logger.error(f"max_input_size must be list, and [H, W], when use resizer")
                exit(-1)
            # 需保证max_input_size为偶数
            for v in max_input_size:
                if v % 2 != 0:
                    logger.error(f"max_input_size[H, W] must be even number")
                    exit(-1)
            if enable_static_resizer and resize_type == 1:
                logger.error(f"enable_static_resizer must be = False, when resize_type = 1")
                exit(-1)
            if enable_static_resizer:
                max_height, max_width = max_input_size
                N, C, H, W = shape
                if max_height != H or max_width != W:
                    logger.error(f"max_input_size must be equal to [H, W], when enable_static_resizer = True")
                    exit(-1)
            toYUV_format = resizer_cfg.get("toYUV_format")
            if toYUV_format not in ["YUV400", "YUV420SP", "YUV422SP", "YUV444SP"]:
                logger.error("toYUV_format should be in [YUV400, YUV420SP, YUV422SP, YUV444SP], when use resizer")
                exit(-1)
            if data_format in ["RGB", "BGR"] and toYUV_format == "YUV400":
                logger.error(f"data_format in [RGB, BGR], toYUV_format must be in [YUV420SP, YUV422SP, YUV444SP]")
                exit(-1)
            if data_format == "GRAY" and toYUV_format != "YUV400":
                logger.error(f"data_format = GRAY, toYUV_format must be YUV400")
                exit(-1)
            
            
                
                