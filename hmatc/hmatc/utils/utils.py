#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os
import yaml
import time
import json
import hashlib
import lzma
import tarfile
import torch
import numpy as np
import random
import onnx
import fnmatch
import requests
import re
from onnx import TensorProto
from pathlib import Path
from tqdm import tqdm
from importlib.metadata import PackageNotFoundError, version
from requests.auth import HTTPBasicAuth
from urllib.parse import urljoin


HOUMO_JFROG_IP = os.getenv("HOUMO_JFROG_IP", "139.224.0.199")
HOUMO_JFROG_PORT = os.getenv("HOUMO_JFROG_PORT", "8082")
assert HOUMO_JFROG_IP in ["139.224.0.199", "10.10.1.53"]
BASENAME = "houmo" if HOUMO_JFROG_IP == "139.224.0.199" else "toolchain"
HOUMO_MODELZOO_URL = (
    f"http://{HOUMO_JFROG_IP}:{HOUMO_JFROG_PORT}/artifactory/{BASENAME}/release"
)

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


def get_hmquant_xh1_version():
    """获取hmquant版本"""
    try:
        v = version("hmquant-xh1")  # 替换成你想查的包名
        return v
    except PackageNotFoundError:
        return "N/A"


def get_hmquant_xh2_version():
    """获取hmquant版本"""
    try:
        v = version("hmquant_xh2")  # 替换成你想查的包名
        return v
    except PackageNotFoundError:
        return "N/A"


def get_package_version(package_name: str):
    """获取hmquant版本"""
    try:
        v = version(package_name)  # 替换成你想查的包名
        return v
    except PackageNotFoundError:
        return "N/A"


def get_houmo_version():
    """获取houmo版本"""
    v = os.getenv("HOUMO_VERSION")
    if v is None:
        raise ValueError("Please set HOUMO_VERSION env")

    def check_version(version_str):
        pattern = r"^(v)?(\d+)\.(\d+)\.(\d+)(\.dev\d{8})?$"
        return bool(re.match(pattern, version_str))

    if not check_version(v):
        raise ValueError(f"Invalid houmo version: {v}")

    return v if v.startswith("v") else f"v{v}"


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


def set_random_seed(seed=1234):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


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
                unit="B",
                unit_scale=True,
                unit_divisor=1024,
                desc=file_name,
            ) as pbar:
                with open(save_path, "wb") as f:
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


def get_file_from_jfrog(file_path: str, save_dir: str = "", extract_dir=None) -> str:
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
    file_size = 0
    print(f"Get file from jfrog: {file_info_path}")
    response = requests.get(file_info_path)
    if response.status_code == 200:
        url_md5 = response.json()["checksums"]["md5"]
        file_size = int(response.json()["size"])
        if os.path.exists(save_path) and get_file_md5(save_path) == url_md5:
            print(url_md5, save_path, "already exists.")
            need_download = False
    else:
        print("Failed to retrieve MD5. status code:", response.status_code)
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
            try:
                import tarfile

                with tarfile.open(save_path, "r:xz") as tar:
                    print("extract to %s" % extract_dir)
                    tar.extractall(path=extract_dir)
            except Exception as e:
                import subprocess

                subprocess.run(
                    f"tar -xvf {save_path} -C {extract_dir}",
                    check=True,
                    shell=True,
                )
    return save_path


class ProgressFile:
    def __init__(self, fileobj, pbar):
        self.fileobj = fileobj
        self.pbar = pbar

    def read(self, size=-1):
        data = self.fileobj.read(size)
        self.pbar.update(len(data))
        return data


def compress_folder_to_tar_xz_with_progress(
    folder_path: str, output_path: str, exclude=None, preset=9
):
    """
    压缩 folder_path 为 .tar.xz 文件，支持 tar -xvf 解压，
    并用 tqdm 显示压缩进度。

    Args:
        folder_path: 要压缩的文件夹路径
        output_path: 输出文件路径
        exclude: 要排除的文件或目录模式列表，支持通配符
        preset: 压缩级别 (0-9)
    """
    if exclude is None:
        exclude = []

    # 统计所有待压缩文件大小
    file_list = []
    total_size = 0
    folder_abs_path = os.path.abspath(folder_path)

    for root, dirs, files in os.walk(folder_path):
        # 排除目录
        dirs[:] = [
            d
            for d in dirs
            if not any(
                fnmatch.fnmatch(os.path.join(root, d), pattern)
                or fnmatch.fnmatch(d, pattern)
                for pattern in exclude
            )
        ]

        for file in files:
            full_path = os.path.join(root, file)
            rel_path = os.path.relpath(full_path, folder_abs_path)

            # 检查是否应该排除
            should_exclude = any(
                fnmatch.fnmatch(full_path, pattern)
                or fnmatch.fnmatch(rel_path, pattern)
                or fnmatch.fnmatch(file, pattern)
                for pattern in exclude
            )

            if should_exclude:
                continue

            file_list.append(full_path)
            total_size += os.path.getsize(full_path)

    # 确保输出目录存在
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)

    with lzma.open(output_path, "wb", preset=preset | lzma.PRESET_EXTREME) as lzma_file:
        with tarfile.open(fileobj=lzma_file, mode="w|") as tar:
            with tqdm(
                total=total_size, unit="B", unit_scale=True, desc="Compressing"
            ) as pbar:
                for file_path in file_list:
                    # 计算相对路径（相对于文件夹的父目录）
                    arcname = os.path.relpath(
                        file_path, start=os.path.dirname(folder_abs_path)
                    )
                    tarinfo = tar.gettarinfo(file_path, arcname)
                    with open(file_path, "rb") as f:
                        tar.addfile(tarinfo, fileobj=ProgressFile(f, pbar))


def compress_file_to_tar_xz_with_progress(file_path: str, output_path: str, preset=9):
    """
    将单个文件压缩成 .tar.xz，支持 tar -xvf 解压，显示压缩进度。
    """
    total_size = os.path.getsize(file_path)
    with lzma.open(output_path, "wb", preset=preset | lzma.PRESET_EXTREME) as lzma_file:
        with tarfile.open(fileobj=lzma_file, mode="w|") as tar:
            with tqdm(
                total=total_size, unit="B", unit_scale=True, desc="Compressing"
            ) as pbar:
                tarinfo = tar.gettarinfo(file_path, arcname=os.path.basename(file_path))
                with open(file_path, "rb") as f:
                    tar.addfile(tarinfo, fileobj=ProgressFile(f, pbar))


def compress_files_to_tar_xz_with_progress(file_paths, output_path, preset=9):
    """
    将多个文件/目录压缩成 .tar.xz，支持 tar -xvf 解压，显示压缩进度。

    Args:
        file_paths: 要压缩的文件/目录路径列表
        output_path: 输出压缩文件路径
        preset: xz压缩预设 (0-9)
    """
    # 计算总大小
    total_size = 0
    for file_path in file_paths:
        if os.path.isfile(file_path):
            total_size += os.path.getsize(file_path)
        elif os.path.isdir(file_path):
            for root, dirs, files in os.walk(file_path):
                for file in files:
                    total_size += os.path.getsize(os.path.join(root, file))

    with lzma.open(output_path, "wb", preset=preset | lzma.PRESET_EXTREME) as lzma_file:
        with tarfile.open(fileobj=lzma_file, mode="w") as tar:
            with tqdm(
                total=total_size, unit="B", unit_scale=True, desc="Compressing"
            ) as pbar:
                for file_path in file_paths:
                    # 添加文件或目录到tar
                    if os.path.isfile(file_path):
                        arcname = os.path.basename(file_path)
                        tarinfo = tar.gettarinfo(file_path, arcname=arcname)
                        with open(file_path, "rb") as f:
                            tar.addfile(tarinfo, ProgressFile(f, pbar))
                    elif os.path.isdir(file_path):
                        # 添加目录及其内容
                        arcname = os.path.basename(file_path)
                        tar.add(
                            file_path,
                            arcname=arcname,
                            filter=lambda info: (
                                (
                                    pbar.update(os.path.getsize(info.name))
                                    if info.isfile()
                                    else None
                                ),
                                info,
                            )[1],
                        )


def upload_file_to_artifactory(
    file_path, upload_url, username="public", password="Password@123", max_retries=3
):
    """
    上传文件到 Artifactory 服务器，包含校验和验证

    Args:
        file_path (str): 要上传的本地文件路径
        upload_url (str): 目标上传地址（完整URL）
        username (str): 认证用户名（默认：public）
        password (str): 认证密码（默认：Password@123）
        max_retries (int): 失败最大重试次数（默认：3次）

    Returns:
        bool: 上传成功返回True，失败返回False
    """
    BASE_URL = "http://10.10.1.53:8082/artifactory/toolchain/release/"
    upload_url = os.path.join(BASE_URL, upload_url)
    # 检查文件是否存在
    if not os.path.isfile(file_path):
        print(f"Not found file: {file_path}")
        return False

    # 计算文件的 MD5 和 SHA1 校验和
    def calculate_checksums(filepath):
        import hashlib

        md5_hash = hashlib.md5()
        sha1_hash = hashlib.sha1()

        with open(filepath, "rb") as f:
            # 分块读取文件以计算校验和
            for chunk in iter(lambda: f.read(4096), b""):
                md5_hash.update(chunk)
                sha1_hash.update(chunk)

        return md5_hash.hexdigest(), sha1_hash.hexdigest()

    try:
        md5, sha1 = calculate_checksums(file_path)
        print(f"Calaculate checksum file: MD5={md5}, SHA1={sha1}")
    except Exception as e:
        print(f"Failed to calaculate checksum file: {str(e)}")
        return False

    # 准备认证参数
    auth = HTTPBasicAuth(username, password)

    # 读取文件内容（二进制模式）
    with open(file_path, "rb") as file:
        file_data = file.read()

    # 请求头设置，包含校验和信息
    headers = {
        "Content-Type": "application/octet-stream",
        "X-Checksum-Md5": md5,
        "X-Checksum-Sha1": sha1,
    }

    # 重试机制
    for attempt in range(max_retries):
        try:
            response = requests.put(
                url=upload_url,
                data=file_data,
                headers=headers,
                auth=auth,
                timeout=30,  # 设置超时时间（秒）
            )

            # 检查响应状态
            if response.status_code in [200, 201]:
                print(f"Upload: {upload_url}, done.")
                return True
            else:
                print(
                    f"Upload fail (try {attempt + 1}/{max_retries}): HTTP {response.status_code} - {response.text}"
                )

        except requests.exceptions.RequestException as e:
            print(f"Network error (try {attempt + 1}/{max_retries}) : {str(e)}")

        # 如果不是最后一次尝试，则等待后重试
        if attempt < max_retries - 1:
            wait_time = 2**attempt  # 指数退避策略
            print(f"Waiting {wait_time}s retry...")
            time.sleep(wait_time)

    print(f"Upload file failed, retry times: {max_retries}")
    return False
