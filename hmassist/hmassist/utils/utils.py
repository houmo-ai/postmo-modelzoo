#!/usr/bin/env python3

from ..utils import logger


def to_pillow(data):
    import numpy as np
    import cv2
    from PIL import Image
    if isinstance(data, cv2.UMat) or isinstance(data, np.ndarray):
        img_rgb = cv2.cvtColor(data, cv2.COLOR_BGR2RGB)
        return Image.fromarray(img_rgb)
    elif isinstance(data, Image.Image):
        return data
    else:
        logger.error(f"unsupported type {type(data)}")
        exit(-1)


def to_opencv(data):
    import numpy as np
    import cv2
    from PIL import Image
    if isinstance(data, cv2.UMat) or isinstance(data, np.ndarray):
        return data
    elif isinstance(data, Image.Image):
        img_rgb = np.array(data)
        return cv2.cvtColor(img_rgb, cv2.COLOR_BGR2RGB)
    else:
        logger.error(f"unsupported type {type(data)}")
        exit(-1)


def get_random_data(dtype, shape):
    """ 生成数据
    @param dtype: data type
    @param shape: data shape
    @param filepath: data file path
    @return: numpy
    """
    import numpy as np
    n, c, h, w = shape

    if dtype == "float32":
        data = np.random.rand(n, c, h, w).astype(dtype=dtype)   # 数值范围[0, 1)
    elif dtype == "float16":
        data = np.random.rand(n, c, h, w).astype(dtype=dtype)   # 数值范围[0, 1)
    elif dtype == "int16":
        data = np.random.randint(low=-(2**15), high=2**15-1, size=(n, c, h, w), dtype=dtype)
    elif dtype == "uint8":
        data = np.random.randint(low=0, high=255, size=(n, c, h, w), dtype=dtype)
    else:
        logger.error("Not support dtype -> {}".format(dtype))
        exit(-1)
    return data


def get_host_ip():
    import socket
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(('8.8.8.8', 80))
        ip = s.getsockname()[0]
    finally:
        s.close()
    return ip


def get_md5(data):
    import hashlib
    md5 = hashlib.md5()
    md5.update(data)
    return md5.hexdigest()