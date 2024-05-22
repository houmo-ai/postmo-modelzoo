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


def get_file_md5(file_path):
    import hashlib
    hash_md5 = hashlib.md5()
    with open(file_path, "rb") as file:
        for chunk in iter(lambda: file.read(4096), b""):
            hash_md5.update(chunk)
    return hash_md5.hexdigest()


def get_file_from_jfrog(file_path):
    import requests
    import os
    # response = requests.head(url)
    # url_md5 = response.headers.get('Content-MD5')
    modelzoo_url = os.environ.get("MODELZOO_URL")
    jfrog_base = os.path.dirname(os.path.dirname(modelzoo_url))
    response = requests.get(jfrog_base + '/api/storage/houmo/release/' + file_path)
    if response.status_code == 200:
        url_md5 = response.json()['checksums']['md5']
    else:
        print("failed to retrieve MD5. status code:", response.status_code)
        return
    file_name = os.path.basename(file_path)
    if os.path.exists(file_name):
        from hmassist.utils.utils import get_file_md5
        if (get_file_md5(file_name) == url_md5):
            print(url_md5, file_name, "already exists.")
            return
    if os.path.exists(file_name):
        os.system("rm " + file_name)
    cmd = "wget " + modelzoo_url + "/" + file_path
    os.system(cmd)
    # from tqdm import tqdm
    # desc = "downloading " + file_name
    # progress_bar = tqdm(initial=0, unit='B', unit_divisor=1024, unit_scale=True, desc=desc)
    # url = os.environ.get("MODELZOO_URL") + file_path
    # response = requests.get(url, stream=True)
    # with open(file_name, 'ab') as fp:
    #     for chunk in response.iter_content(chunk_size=10*1024*1024):
    #         if chunk:
    #             fp.write(chunk)
    #             progress_bar.update(len(chunk))
    # progress_bar.close()
    print(file_name, "download success.")


def sanitize_name(name: str):
    return name.replace(":", "_").replace("/", "_")
