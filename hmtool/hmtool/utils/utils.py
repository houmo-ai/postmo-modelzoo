#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os
import sys
import yaml
import time
import json
import hashlib
import torch
import numpy as np
import onnx
from onnx import TensorProto


HOUMO_JFROG_IP = os.getenv("HOUMO_JFROG_IP", "139.224.0.199")
HOUMO_JFROG_PORT = os.getenv("HOUMO_JFROG_PORT", "8082")
HOUMO_MODELZOO_URL = f"http://{HOUMO_JFROG_IP}:{HOUMO_JFROG_PORT}/artifactory/houmo/release"

SUPPORT_IMAGE_FORMATS = [".jpg", ".JPEG", ".bmp", ".png", ".jpeg", ".BMP"]
SUPPORT_BACKEND = ["xh1", "xh2", "onnx"]


def read_yaml_to_dict(file_path: str) -> dict:
    with open(file_path, "r") as f:
        cfg = yaml.load(f, Loader=yaml.FullLoader)
    return cfg


def save_dict_to_yaml(dict_value: dict, yaml_path: str):
    with open(yaml_path, "w") as f:
        f.write(yaml.dump(dict_value, allow_unicode=True, default_flow_style=False))
        

def read_json_to_dict(file_path: str) -> dict:
    with open(file_path, "r") as f:
        cfg = json.load(f)
    return cfg


def get_md5(array: np.ndarray):
    return hashlib.md5(array.tobytes()).hexdigest()


def get_file_md5(file_path):
    hash_md5 = hashlib.md5()
    with open(file_path, "rb") as file:
        for chunk in iter(lambda: file.read(4096), b""):
            hash_md5.update(chunk)
    return hash_md5.hexdigest()


def get_onnx_inputs_info(onnx_path):
    model = onnx.load(onnx_path)
    inputs_info = dict()
    outputs_info = dict()
    for node in model.graph.input:
        name = node.name
        dtype = node.type.tensor_type.elem_type
        dtype_str = TensorProto.DataType.Name(dtype).lower()
        if dtype_str == "float":
            dtype_str = "float32"
        shape = [dim.dim_value for dim in node.type.tensor_type.shape.dim]
        inputs_info[name] = {"dtype": dtype_str, "shape": shape}
    for node in model.graph.output:
        name = node.name
        dtype = node.type.tensor_type.elem_type
        dtype_str = TensorProto.DataType.Name(dtype).lower()
        if dtype_str == "float":
            dtype_str = "float32"
        shape = [dim.dim_value for dim in node.type.tensor_type.shape.dim]
        outputs_info[name] = {"dtype": dtype_str, "shape": shape}
    return inputs_info, outputs_info


def load_npz(npz_path):
    in_datas = dict()
    with np.load(npz_path) as data:
        keys = data.files
        for key in keys:
            x = data[key]
            in_datas[key] = x.copy()
    return in_datas


torch_to_numpy_dtype = {
    torch.float32: "float32",
    torch.float64: "float64",
    torch.float16: "float16",
    torch.bfloat16: "bfloat16",  # numpy doesn"t support bfloat16 directly
    torch.uint8: "uint8",
    torch.int8: "int8",
    torch.int16: "int16",
    torch.int32: "int32",
    torch.int64: "int64",
    torch.bool: "bool",
}

def str_to_torch_dtype(dtype_str):
    if dtype_str == "float32":
        return torch.float32
    elif dtype_str == "float16":
        return torch.float16
    elif dtype_str == "int32":
        return torch.int32
    elif dtype_str == "int64":
        return torch.int64
    elif dtype_str == "int16":
        return torch.int16
    elif dtype_str == "int8":
        return torch.int8
    elif dtype_str == "uint8":
        return torch.uint8
    elif dtype_str == "bool":
        return torch.bool
    else:
        raise f"Not support dtype: {dtype_str}"
    

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
        modelzoo_url = os.environ.get("HOUMO_MODELZOO_URL", HOUMO_MODELZOO_URL)
    file_name = os.path.basename(file_path)
    if save_dir == "":
        save_dir = os.getenv("HOUMO_MODEL_PATH", default="./")
    else:
        os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, file_name)
    jfrog_base, jfrog_tail = modelzoo_url.split("artifactory")
    jfrog_base = jfrog_base + "artifactory"
    file_info_path = f"{jfrog_base}/api/storage/{jfrog_tail}/{file_path}"
    response = requests.get(file_info_path)
    if response.status_code == 200:
        url_md5 = response.json()["checksums"]["md5"]
        if os.path.exists(save_path) and get_file_md5(save_path) == url_md5:
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
        if save_path.rfind(".zip") > 0:
            import zipfile
            with zipfile.ZipFile(save_path, "r") as zip:
                print("extract to %s" % extract_dir)
                zip.extractall(path=extract_dir)
        elif save_path.rfind(".tar") > 0:
            import tarfile
            with tarfile.open(save_path, "r:gz") as tar:
                print("extract to %s" % extract_dir)
                tar.extractall(path=extract_dir)
    return save_path