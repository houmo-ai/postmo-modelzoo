#!/usr/bin/env python3

import os
import sys
import time
import numpy as np
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

    if dtype == "float32":
        data = np.random.rand(*shape).astype(dtype=dtype)   # 数值范围[0, 1)
    elif dtype == "float16":
        data = np.random.rand(*shape).astype(dtype=dtype)   # 数值范围[0, 1)
    elif dtype == "int16":
        data = np.random.randint(low=-(2**15), high=2**15-1, size=shape, dtype=dtype)
    elif dtype == "uint8":
        data = np.random.randint(low=0, high=255, size=shape, dtype=dtype)
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


start_time = time.time()
def download_file(url, save_path):
    def reporthook(block_num, block_size, total_size):
        global start_time
        if total_size is None:
            total_size = total_size
            start_time = time.time()
            print("download begin...")
        elif total_size != -1:
            downloaded = block_num * block_size
            if downloaded < total_size:
                try:
                    speed = downloaded / (time.time() - start_time)
                    remaining_time = (total_size - downloaded) / speed
                    percent = int((downloaded / total_size) * 100)
                    sys.stdout.write("\r[{:<50}] {:>3}% speed: {:.2f}KB/s last: {:.2f}s".format(
                        "=" * percent + " " * (100 - percent), percent, speed / 1024, remaining_time))
                    sys.stdout.flush()
                except ZeroDivisionError:
                    pass
            else:
                print("\ndownload finished.")
    if not os.path.exists(save_path):
        import urllib.request
        print("download from %s to %s" % (url, save_path))
        urllib.request.urlretrieve(url, save_path, reporthook=reporthook)
        return True
    else:
        print("local file %s has already exsit" % save_path)
    return False
    

def get_file_from_jfrog(file_path, save_dir="", extract_dir=None):
    import requests
    import os
    need_download = True
    if "http://" in file_path or "https://" in file_path:
        url, file_path = file_path.split("artifactory/")
        modelzoo_url = url + "artifactory"
    else:
        modelzoo_url = os.environ.get("HOUMO_MODELZOO_URL")
    file_name = os.path.basename(file_path)
    if save_dir == "":
        save_dir = os.getenv("HOUMO_MODEL_PATH", default="./")
    else:
        os.makedirs(save_dir, exist_ok=True)
    save_path = f"{save_dir}/{file_name}"
    jfrog_base, jfrog_tail = modelzoo_url.split("artifactory")
    jfrog_base = jfrog_base + "artifactory"
    file_info_path = f"{jfrog_base}/api/storage/{jfrog_tail}/{file_path}"
    response = requests.get(file_info_path)
    if response.status_code == 200:
        url_md5 = response.json()['checksums']['md5']
        if os.path.exists(save_path):
            if (get_file_md5(save_path) == url_md5):
                print(url_md5, save_path, "already exists.")
                need_download = False
    else:
        print("failed to retrieve MD5. status code:", response.status_code)

    if need_download:
        if os.path.exists(save_path):
            os.remove(save_path)
        url = f"{modelzoo_url}/{file_path}"
        assert(download_file(url, save_path=save_path))

    if extract_dir is not None:
        if save_path.rfind('.zip') > 0:
            import zipfile
            with zipfile.ZipFile(save_path, "r") as zip:
                print("extract to %s" % extract_dir)
                zip.extractall(path=extract_dir)
        elif save_path.rfind('.tar') > 0:
            import tarfile
            with tarfile.open(save_path, "r:gz") as tar:
                print("extract to %s" % extract_dir)
                tar.extractall(path=extract_dir)
    return save_path


def sanitize_name(name: str):
    return name.replace(":", "_").replace("/", "_")


def load_npz(npz_path):
    in_datas = dict()
    with np.load(npz_path) as data:
        keys = data.files
        for key in keys:
            x = data[key]
            in_datas[key] = x.copy()
    return in_datas

