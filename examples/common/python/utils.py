#!/usr/bin/env python3

import os
import sys
import numpy as np
import shutil
import urllib.request
import tarfile
import zipfile
import logging
import time


logger = logging.getLogger()
logger.setLevel(logging.INFO)


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
        print("download from %s to %s" % (url, save_path))
        urllib.request.urlretrieve(url, save_path, reporthook=reporthook)
        return True
    else:
        print("local file %s has already exsit" % save_path)
    return False


def get_file_from_jfrog(file_path, save_dir, extract_dir=None):
    import requests
    import os
    need_download = True
    if "http://" in file_path or "https://" in file_path:
        url, file_path = file_path.split("artifactory/")
        modelzoo_url = url + "artifactory/"
    else:
        modelzoo_url = os.environ.get("HOUMO_MODELZOO_URL")
    file_name = os.path.basename(file_path)
    if save_dir == "":
        save_dir = os.getenv("HOUMO_MODEL_PATH", default="")
    else:
        os.makedirs(save_dir, exist_ok=True)
    save_path = f"{save_dir}/{file_name}"
    jfrog_base, jfrog_tail = modelzoo_url.split("artifactory/")
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
            with zipfile.ZipFile(save_path, "r") as zip:
                print("extract to %s" % extract_dir)
                zip.extractall(path=extract_dir)
        elif save_path.rfind('.tar') > 0:
            with tarfile.open(save_path, "r:gz") as tar:
                print("extract to %s" % extract_dir)
                tar.extractall(path=extract_dir)
    return save_path


def cosine_distance(data1, data2, check_shape=True):
    """calc cosine distance of data1 and data2"""
    if check_shape:
        if data1.shape != data2.shape:
            print("[error] shape not equal {} vs {}".format(data1.shape, data2.shape))
            return -1
    v1_d = data1.flatten().astype("float64")
    v2_d = data2.flatten().astype("float64")
    if len(v1_d) != len(v2_d):
        print("[error] v1 dim {} != v2 dim {}".format(len(v1_d), len(v2_d)))
        return -1
    v1_d[v1_d == np.inf] = np.finfo(np.float16).max
    v2_d[v2_d == np.inf] = np.finfo(np.float16).max
    v1_d[v1_d == -np.inf] = np.finfo(np.float16).min
    v2_d[v2_d == -np.inf] = np.finfo(np.float16).min
    v1_norm = v1_d / np.linalg.norm(v1_d)
    v2_norm = v2_d / np.linalg.norm(v2_d)
    cosine_dist = np.dot(v1_norm, v2_norm)
    if np.isnan(cosine_dist):
        return -1
    return cosine_dist