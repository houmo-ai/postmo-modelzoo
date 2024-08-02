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


def get_file_from_jfrog(file_path, save_dir=""):
    import requests
    import os
    modelzoo_url = os.environ.get("MODELZOO_URL")
    file_name = os.path.basename(file_path)
    if save_dir == "":
        save_dir = os.getenv("MODEL_PATH", default="")
        if save_dir != "":
            os.system("mkdir -p " + save_dir)
    save_path = os.path.join(save_dir, file_name)
    jfrog_base, jfrog_tail = modelzoo_url.split("artifactory/")
    jfrog_base = jfrog_base + "artifactory/"
    file_info_path = os.path.join(jfrog_base, 'api/storage', jfrog_tail, file_path)
    response = requests.get(file_info_path)
    if response.status_code == 200:
        url_md5 = response.json()['checksums']['md5']
        if os.path.exists(save_path):
            from hmassist.utils.utils import get_file_md5
            if (get_file_md5(save_path) == url_md5):
                print(url_md5, save_path, "already exists.")
                return save_path
    else:
        print("failed to retrieve MD5. status code:", response.status_code)
    if os.path.exists(save_path):
        os.system("rm " + save_path)
    # cmd = "wget -N --ftp-user=ftp001 --ftp-password=3tIx7oMi@R " + modelzoo_url + "/" + file_path
    print("downloading", file_name)
    cmd = "wget -c " + modelzoo_url + "/" + file_path + " -O " + save_path
    os.system(cmd)
    return save_path


def sanitize_name(name: str):
    return name.replace(":", "_").replace("/", "_")
