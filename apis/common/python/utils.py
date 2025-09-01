#!/usr/bin/env python3

import os
import sys
import numpy as np
import logging
from tqdm import tqdm


logger = logging.getLogger()
logger.setLevel(logging.INFO)


def get_file_md5(file_path):
    import hashlib

    hash_md5 = hashlib.md5()
    with open(file_path, "rb") as file:
        for chunk in iter(lambda: file.read(4096), b""):
            hash_md5.update(chunk)
    return hash_md5.hexdigest()


def download_file(url, save_path, file_name, file_size, chunk_size=1024 * 1024):
    if os.path.exists(save_path):
        print("local file %s has already exist" % save_path)
        return False
    if file_size <= 0 or len(file_name) == 0:
        print(f"Invalid file info, file name {file_name}, file size: {file_size}")
        return False

    import requests

    try:
        with requests.get(url, stream=True) as response:
            response.raise_for_status()

            # 创建进度条
            with tqdm(
                total=file_size,
                unit='B',
                unit_scale=True,
                unit_divisor=1024,
                desc=file_name,
            ) as pbar:
                with open(save_path, 'wb') as f:
                    for chunk in response.iter_content(chunk_size=chunk_size):
                        if chunk:  # 过滤掉保持连接的空块
                            f.write(chunk)
                            pbar.update(len(chunk))

            print(f"download {save_path} finished.")
            return True
    except requests.exceptions.RequestException as e:
        print(f"download {save_path} failed, error msg: {str(e)}")
        if os.path.exists(save_path) and os.path.getsize(save_path) != file_size:
            os.remove(save_path)
        return False
    except Exception as e:
        print(f"download {save_path} failed, unknown err: {str(e)}")
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
        save_dir = os.getenv("HOUMO_MODEL_PATH", ".")
    else:
        os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, file_name)
    jfrog_base, jfrog_tail = modelzoo_url.split("artifactory/")
    jfrog_base = jfrog_base + "artifactory"
    file_info_path = f"{jfrog_base}/api/storage/{jfrog_tail}/{file_path}"
    file_size = 0
    response = requests.get(file_info_path)
    if response.status_code == 200:
        url_md5 = response.json()['checksums']['md5']
        file_size = int(response.json()["size"])
        if os.path.exists(save_path) and get_file_md5(save_path) == url_md5:
            print(url_md5, save_path, "already exists.")
            need_download = False
    else:
        print("failed to retrieve MD5. status code:", response.status_code)
        return ""

    if need_download:
        if os.path.exists(save_path):
            os.remove(save_path)
        url = f"{modelzoo_url}/{file_path}"
        if download_file(url, save_path, file_name, file_size) is False:
            return ""

    if extract_dir is not None:
        if save_path.rfind(".zip") > 0:
            import zipfile

            with zipfile.ZipFile(save_path, "r") as zip:
                print("extract to %s" % extract_dir)
                zip.extractall(path=extract_dir)
        elif save_path.rfind(".tar.gz") > 0:
            import tarfile

            with tarfile.open(save_path, "r:gz") as tar:
                print("extract to %s" % extract_dir)
                tar.extractall(path=extract_dir)
        elif save_path.rfind(".tar.xz") > 0:
            import tarfile

            with tarfile.open(save_path, "r:xz") as tar:
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
