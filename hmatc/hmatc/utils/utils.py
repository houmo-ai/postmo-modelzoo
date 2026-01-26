# Copyright 2025 HOUMO AI
#
# File: utils.py
# Description:
#   Utils functions
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# SPDX-License-Identifier: Apache-2.0
import os
import yaml
import time
import json
import hashlib
import lzma
import torch
import numpy as np
import random
import onnx
import fnmatch
import requests
import re
import shutil
import subprocess
import tarfile
from onnx import TensorProto
from tqdm import tqdm
from importlib.metadata import PackageNotFoundError, version
from requests.auth import HTTPBasicAuth
from . import logger


JFROG_REPO = "http://artifactory.houmo.ai/artifactory/Dadao"
OSS_REPO = "https://houmo-llm.oss-cn-shanghai.aliyuncs.com/Dadao"

SUPPORT_IMAGE_FORMATS = [".jpg", ".JPEG", ".bmp", ".png", ".jpeg", ".BMP"]
SUPPORT_BACKEND = ["xh1", "xh2", "onnx", "hmonnx"]


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
    """Calculate and return the MD5 hash of a numpy array.

    Args:
        array (np.ndarray): Input numpy array to calculate MD5 hash

    Returns:
        str: MD5 hash string of the input array
    """
    return hashlib.md5(array.tobytes()).hexdigest()


def get_file_md5(file_path):
    """Calculate and return the MD5 hash of a file.

    Args:
        file_path (str): Path to the file to calculate MD5 hash

    Returns:
        str: MD5 hash string of the file content
    """
    hash_md5 = hashlib.md5()
    with open(file_path, "rb") as file:
        for chunk in iter(lambda: file.read(4096), b""):
            hash_md5.update(chunk)
    return hash_md5.hexdigest()


def get_hmquant_xh1_version():
    """Get the version of the hmquant-xh1 package.

    Returns:
        str: Version string of the package, or 'N/A' if not found
    """
    try:
        v = version("hmquant-xh1")
        return v
    except PackageNotFoundError:
        return "N/A"


def get_hmquant_xh2_version():
    """Get the version of the hmquant_xh2 package.

    Returns:
        str: Version string of the package, or 'N/A' if not found
    """
    try:
        v = version("hmquant_xh2")
        return v
    except PackageNotFoundError:
        return "N/A"


def get_package_version(package_name: str):
    """Get the version of a specified package.

    Args:
        package_name (str): Name of the package to get version for

    Returns:
        str: Version string of the package, or 'N/A' if not found
    """
    try:
        v = version(package_name)
        return v
    except PackageNotFoundError:
        return "N/A"


def get_houmo_version():
    """Get the Houmo version from environment variable HOUMO_VERSION.

    Returns:
        str: The Houmo version string with 'v' prefix

    Raises:
        ValueError: If HOUMO_VERSION environment variable is not set or version format is invalid
    """
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
    """Get input and output information from an ONNX model file.

    Args:
        onnx_path (str): Path to the ONNX model file

    Returns:
        tuple: A tuple containing:
            - inputs_info (dict): Dictionary with input names as keys and their dtype/shape as values
            - outputs_info (dict): Dictionary with output names as keys and their dtype/shape as values
    """
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
    """Load data from a .npz file into a dictionary.

    Args:
        npz_path (str): Path to the .npz file to load

    Returns:
        dict: Dictionary with keys from the .npz file and corresponding numpy arrays as values
    """
    in_datas = dict()
    with np.load(npz_path) as data:
        keys = data.files
        for key in keys:
            x = data[key]
            in_datas[key] = x.copy()
    return in_datas


def set_random_seed(seed=1234):
    """Set random seeds for reproducible results across multiple frameworks.

    Args:
        seed (int): Random seed value to set (default: 1234)
    """
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
    """Convert a string representation of a data type to PyTorch data type.

    Args:
        dtype_str (str): String representation of the data type

    Returns:
        torch.dtype: Corresponding PyTorch data type

    Raises:
        ValueError: If the data type string is not supported
    """
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
    """Download a file from URL with progress bar and MD5 verification.

    Args:
        url (str): Download URL
        save_path (str): Local path to save the file
        file_name (str): Name of the file being downloaded (for display)
        file_size (int): Expected file size in bytes
        chunk_size (int): Size of chunks to download at a time (default: 1MB)

    Returns:
        bool: True if download is successful, False otherwise
    """
    if os.path.exists(save_path):
        logger.error("local file %s has already exist" % save_path)
        return False
    if file_size <= 0 or len(file_name) == 0:
        logger.error(
            f"Invalid file info, file name {file_name}, file size: {file_size}"
        )
        return False

    try:
        with requests.get(url, stream=True) as response:
            response.raise_for_status()

            with tqdm(
                total=file_size,
                unit="B",
                unit_scale=True,
                unit_divisor=1024,
                desc=file_name,
            ) as pbar:
                with open(save_path, "wb") as f:
                    for chunk in response.iter_content(chunk_size=chunk_size):
                        if chunk:
                            f.write(chunk)
                            pbar.update(len(chunk))

            logger.info(f"download {save_path} finished.")
            return True
    except requests.exceptions.RequestException as e:
        logger.error(f"download {save_path} failed, error msg: {str(e)}")
        if os.path.exists(save_path) and os.path.getsize(save_path) != file_size:
            os.remove(save_path)
        return False
    except Exception as e:
        logger.error(f"download {save_path} failed, unknown err: {str(e)}")
        return False


def _ping_houmo_domain(timeout: int = 10) -> bool:
    """
    Checks connectivity to the Houmo domain service.
    :param timeout: Sets the timeout for the connection attempt (default: 10)
    :return: Boolean indicating whether the domain is accessible
    """
    try:
        response = requests.get(JFROG_REPO, timeout=timeout)
        response.raise_for_status()
        if response.status_code != 200:
            return False
        return True
    except Exception:
        return False


def _parse_file_url(file_path: str) -> tuple[str, str, str]:
    """
    Parse the file path
    :param file_path: Complete URL / Relative Path
    :return: (modelzoo_url, file_relative_path, repo_type)
             repo_type: "jfrog" / "oss"
    """
    from urllib.parse import unquote

    file_path = unquote(file_path)
    # 1. Handling the situation where the full URL is provided
    if file_path.startswith(("http://", "https://")):
        if "artifactory" in file_path:
            split_key = "artifactory/"
            repo_type = "jfrog"
        elif "oss-cn" in file_path:
            split_key = ".com/"
            repo_type = "oss"
        else:
            raise ValueError(f"Unsupported URL type: {file_path}")
        split_idx = file_path.find(split_key) + len(split_key)
        modelzoo_url = file_path[: split_idx - 1]
        file_relative_path = file_path[split_idx:]
    # 2. Handling the situation of relative paths
    else:
        env_url = os.getenv("HOUMO_MODELZOO_URL")
        if _ping_houmo_domain() is True:
            repo_type = "jfrog"
            modelzoo_url = env_url if env_url else JFROG_REPO
        else:
            repo_type = "oss"
            modelzoo_url = OSS_REPO
        file_relative_path = file_path

    file_relative_path = file_relative_path.strip("/")
    return modelzoo_url, file_relative_path, repo_type


def _get_jfrog_file_md5(
    jfrog_base_url: str, file_relative_path: str
) -> tuple[str, int]:
    """
    Get the MD5 and size of the file from JFrog
    :return: (md5sum, file size)
    """

    try:
        split_parts = jfrog_base_url.rstrip("/").rsplit("artifactory", 1)
        jfrog_base = split_parts[0] + "artifactory"
        jfrog_tail = split_parts[-1].strip("/") if len(split_parts) > 1 else ""
        if jfrog_tail:
            file_info_url = (
                f"{jfrog_base}/api/storage/{jfrog_tail}/{file_relative_path}"
            )
        else:
            file_info_url = f"{jfrog_base}/api/storage/{file_relative_path}"

        logger.info(f"Get file info from Jfrog: {file_info_url}")
        response = requests.get(file_info_url, timeout=10)
        response.raise_for_status()  # Trigger HTTP error (not a 200 status code)

        data = response.json()
        return data["checksums"]["md5"], int(data["size"])
    except Exception as e:
        logger.error(f"Failed to get file information from JFrog: {str(e)}")
        return "", 0


def _get_oss_file_md5(file_relative_path: str) -> tuple[str, int]:
    """
    Get the MD5 and file size from OSS
    :return: (md5sum, file size)
    """
    try:
        file_relative_path = file_relative_path.strip("/")
        file_relative_path = (
            file_relative_path
            if file_relative_path.startswith("Dadao/")
            else f"Dadao/{file_relative_path}"
        )
        oss_api_url = f"https://developer.houmoai.com/api/product/oss_jfrog_file_record/get_file_info/?oss_path={file_relative_path}"
        logger.info(f"Get file info from OSS: {oss_api_url}")

        response = requests.get(oss_api_url, timeout=10)
        response.raise_for_status()  # Trigger HTTP error (not a 200 status code)

        resp_data = response.json()
        if resp_data["code"] == 1:
            return resp_data["data"]["md5"], int(resp_data["data"]["content_length"])

        logger.warning(f'OSS returned an error. Error code: {resp_data["code"]}')
        return "", 0
    except Exception as e:
        logger.warning(f"Failed to get file information from OSS:{str(e)}")
        return "", 0


def _check_file_exists(save_path: str, expected_md5: str) -> bool:
    """Check if the file exists and verify its MD5 checksum matches."""
    if not os.path.exists(save_path) or not expected_md5:
        return False
    try:
        actual_md5 = get_file_md5(save_path)
        if actual_md5 == expected_md5:
            logger.info(
                f"The file already exists and the MD5 checksum matches: {save_path}"
            )
            return True
        else:
            logger.warning(
                f"The file MD5 does not match. It will be re-downloaded: {save_path}"
            )
            return False
    except Exception as e:
        logger.error(f"Failed to calculate the MD5 of the file: {str(e)}")
        return False


def _extract_files(save_path: str, extract_dir: str) -> bool:
    """
    Extract files from compressed file.

    :param save_path: the path of compressed file
    :param extract_dir: the path of extract folder
    :return: return True if decompression is successful, and False if it fails.
    """

    if not save_path.strip() or not os.path.exists(save_path):
        logger.error(f"Invalid path of the compressed file: {save_path}.")
        return False
    if not isinstance(extract_dir, str) or not extract_dir.strip():
        logger.error("The decompression directory cannot be empty.")
        return False

    import zipfile

    extract_dir = os.path.abspath(extract_dir)
    os.makedirs(extract_dir, exist_ok=True)

    extract_mapping = {
        (".zip",): (zipfile.ZipFile, {"mode": "r"}),
        (".tar.gz", ".tgz"): (tarfile.open, {"mode": "r:gz"}),
        (".tar.xz",): (tarfile.open, {"mode": "r:xz"}),
    }
    save_path_lower = save_path.lower()
    extract_func = None
    extract_kwargs = None

    for suffixes, (func, kwargs) in extract_mapping.items():
        if save_path_lower.endswith(suffixes):
            extract_func = func
            extract_kwargs = kwargs
            break

    if not extract_func:
        logger.error(
            f"Unsupported compression format for decompression: {save_path}. Only supported:.zip/.tar.gz/.tgz/.tar.xz"
        )
        return False

    logger.info(f"Start to decompress: {save_path} -> {extract_dir}")
    try:
        with extract_func(save_path, **extract_kwargs) as f:
            f.extractall(path=extract_dir)
        return True
    except Exception as e:
        if save_path_lower.endswith(".tar.xz"):
            logger.warning(
                f"The tarfile extraction failed with error message ({str(e)}). Trying to extract it using the system's tar command."
            )
            try:
                subprocess.run(
                    ["tar", "-xvf", save_path, "-C", extract_dir],
                    check=True,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    encoding="utf-8",
                )
                return True
            except subprocess.CalledProcessError as se:
                logger.error(
                    f"The system tar command failed to decompress successfully: {se.stderr}"
                )
                return False
        else:
            logger.error(f"Decompression failed: {str(e)}")
            return False


def get_file_from_jfrog(file_path: str, save_dir: str = "", extract_dir=None) -> str:
    """Download a file from JFrog repository with MD5 verification and optional extraction.

    Args:
        file_path (str): Complete URL or relative path to the file
        save_dir (str): Directory to save the downloaded file (default: current directory or HOUMO_MODEL_PATH)
        extract_dir (str, optional): Directory to extract compressed files (default: None)

    Returns:
        str: Path to the downloaded file if successful, empty string otherwise
    """
    if not file_path.strip():
        logger.error("The file path cannot be empty.")
        return ""

    try:
        modelzoo_url, file_relative_path, repo_type = _parse_file_url(file_path)
    except ValueError as e:
        logger.error(f"file_path {file_path} parsing failed:{str(e)}")
        return ""

    save_dir = (
        save_dir.strip() if save_dir else os.getenv("HOUMO_MODEL_PATH", default="./")
    )
    save_dir = os.path.abspath(save_dir)
    os.makedirs(save_dir, exist_ok=True)
    file_name = os.path.basename(file_relative_path)
    save_path = os.path.join(save_dir, file_name)

    need_download = True
    file_size = 0
    expected_md5 = ""
    if repo_type in ["jfrog"]:
        expected_md5, file_size = _get_jfrog_file_md5(modelzoo_url, file_relative_path)
    elif repo_type in ["oss"]:
        expected_md5, file_size = _get_oss_file_md5(file_relative_path)
    if not expected_md5 or file_size <= 0:
        return ""

    need_download = not _check_file_exists(save_path, expected_md5)

    if need_download:
        if os.path.exists(save_path):
            os.remove(save_path)
        download_url = f"{modelzoo_url}/{file_relative_path}"
        logger.info(f"Start downloading the file: {download_url}")
        if download_file(download_url, save_path, file_name, file_size) is False:
            logger.error(f"File download failed: {download_url}")
            return ""

    if extract_dir is not None and not _extract_files(save_path, extract_dir):
        return ""

    return save_path


def _generate_extract_dir(file_path, edited_dir):
    generated_dir = None
    if (
        file_path.rfind(".zip")
        or file_path.rfind(".tar.xz")
        or file_path.rfind(".tar.gz")
    ):
        generated_dir = edited_dir

    return generated_dir


def _download_from_modelscope(
    repo_id: str,
    local_dir: str,
    allow_patterns=None,
    ignore_patterns=None,
    revision=None,
) -> bool:
    print(f"Ready to download from modelscope, repo_id: {repo_id}")

    from modelscope import snapshot_download

    local_dir = os.path.abspath(local_dir)
    download_flag = True
    while True:
        try:
            snapshot_download(
                repo_id,
                local_dir=local_dir,
                allow_patterns=allow_patterns,
                ignore_patterns=ignore_patterns,
                revision=revision,
            )
            break
        except Exception as e:
            non_retry_msg = [
                "permission denied",
                "folder not found",
                "invalid token",
                "does not exist",
                "exist",
                "has no revision",
            ]
            print(
                f"Warning:Failed to download models from modelscope, retry, error msg:{e}"
            )
            if any(msg in str(e).lower() for msg in non_retry_msg):
                download_flag = False
                print(
                    f"Error:Failed to download models from modelscope, stop download retry, error msg:{e}"
                )
                break

    return download_flag


def _download_other_files(
    files_key, model_cfgs, download_files, download_dir, extract_dir
):
    if files_key not in download_files:
        download_files[files_key] = dict()
    download_files[files_key]["other_files"] = list()
    for other_file in model_cfgs[files_key]["other_files"]:
        extract_dir_edit = (
            extract_dir
            if extract_dir is not None
            else _generate_extract_dir(other_file, ".")
        )
        tmp_file = get_file_from_jfrog(other_file, download_dir, extract_dir_edit)
        if not tmp_file or not os.path.exists(tmp_file):
            download_files["ret"] = False
        download_files[files_key]["other_files"].append(tmp_file)


def _download_raw_quant(
    file_type: str,
    model_cfgs,
    download_files,
    download_dir,
    extract_dir,
    extract_dir_new,
):
    if file_type not in ["raw", "quant"]:
        return False

    files_key = f"{file_type}_files"
    path_key = f"{file_type}_path"
    if model_cfgs[files_key].get(path_key, None) is not None:
        model_path = model_cfgs[files_key][path_key]
        extract_dir_edit = (
            extract_dir
            if extract_dir is not None
            else _generate_extract_dir(model_path, extract_dir_new)
        )
        download_file = get_file_from_jfrog(model_path, download_dir, extract_dir_edit)
        if not download_file or not os.path.exists(download_file):
            download_files["ret"] = False
        if files_key not in download_files:
            download_files[files_key] = dict()
        download_files[files_key][path_key] = download_file

    if len(model_cfgs[files_key].get("other_files", list())) > 0:
        _download_other_files(
            files_key, model_cfgs, download_files, download_dir, extract_dir
        )

    return True


def _generate_hmm_path(model_cfgs, source_type, model_type, target) -> str:
    # auto generate hmm path
    repo_id = ""
    # required
    version = model_cfgs["version"].lower()
    model_name = model_cfgs["model_name"]
    ncore_val = model_cfgs["model_info"]["ncore"]
    # optional
    batch = model_cfgs["model_info"].get("batch", 0)
    ndevice_val = model_cfgs["model_info"].get("ndevice", 0)
    opt_level = model_cfgs["model_info"].get("opt_level", "NA")
    model_size = model_cfgs["model_info"].get("model_size", "NA")
    prefill_len = model_cfgs["model_info"].get("prefill_len", "")
    context_len = model_cfgs["model_info"].get("context_len", "")
    # convert val to str
    ncore = f"{ncore_val}cores" if ncore_val > 1 else "1core"
    ndevice = f"{ndevice_val}chips" if ndevice_val > 1 else "1chip"

    if source_type == "modelscope" and model_type in ["llm"]:
        repo_id = f"Houmo/{target}_{model_name}_{model_size}"  # repo id
        hmm_path = f"{model_name}_{model_size}"
        if prefill_len:
            hmm_path += f"_{prefill_len}"
        if context_len:
            repo_id += f"_{context_len}"
            hmm_path += f"_{context_len}"
        if batch > 0:
            hmm_path += f"_b{batch}"
        hmm_path += f"_{ndevice}_{ncore}"
    elif model_type in ["llm"]:
        hmm_path = f"models/{target}-{version}/{model_name}/hmm_{target}_{model_name}_{model_size}"
        if prefill_len:
            hmm_path += f"_{prefill_len}"
        if context_len:
            hmm_path += f"_{context_len}"
        if batch > 0:
            hmm_path += f"_b{batch}"
        hmm_path += f"_{ndevice}_{ncore}_{version}.zip"
    else:
        hmm_path = f"models/{target}-{version}/{model_name}/{model_name}_{target}_b{batch}_{ncore_val}core_{opt_level}_{version}.tar.xz"

    return hmm_path, repo_id


def hmatc_get_file(
    model_cfgs: dict,
    file_type: str,
    download_dir: str = "",
    extract_dir=None,
    source_type="jfrog",
) -> str:
    """Compress a folder to .tar.xz format with progress bar and support for excluding files.

    Args:
        folder_path (str): Path to the folder to compress
        output_path (str): Output file path for the compressed archive
        exclude (list, optional): List of patterns to exclude from compression (default: None)
        preset (int): Compression level (0-9, default: 9)
    """
    download_file = ""
    download_files = dict()
    download_files["ret"] = True

    # check model_cfgs
    if (
        not model_cfgs
        or model_cfgs.get("target", None) not in ["xh1", "xh2"]
        or model_cfgs.get("version", None) is None
        or not model_cfgs["version"].startswith("v")
        or model_cfgs.get("model_type", None) not in ["cv", "llm"]
        or model_cfgs.get("model_name", None) is None
        or source_type not in ["jfrog", "modelscope"]
    ):
        print("Error: Missing required fields in model_cfgs.")
        download_files["ret"] = False
        return download_file, download_files

    target = model_cfgs["target"].lower()
    if target != "xh2" and source_type == "modelscope":
        print(
            "Error: Only in the xh2 target, modelscope is supported as the source for downloading hmm models."
        )
        download_files["ret"] = False
        return download_file, download_files

    if download_dir != "":
        download_dir = os.path.abspath(download_dir)
    if extract_dir is not None and extract_dir != "":
        extract_dir = os.path.abspath(extract_dir)
    model_type = model_cfgs["model_type"]
    ignore_patterns = ["*.safetensors"]
    if file_type in ["raw"]:
        ignore_patterns = []
        if model_cfgs.get("raw_files", None) is not None:
            extract_dir_new = os.getenv("HOUMO_DATASETS_PATH", ".")
            _download_raw_quant(
                file_type,
                model_cfgs,
                download_files,
                download_dir,
                extract_dir,
                extract_dir_new,
            )

    elif file_type in ["quant"] and model_cfgs.get("quant_files", None) is not None:
        extract_dir_new = os.path.join("output", target, "hmquant")
        _download_raw_quant(
            file_type,
            model_cfgs,
            download_files,
            download_dir,
            extract_dir,
            extract_dir_new,
        )

    elif file_type in ["hmm"]:
        if (
            source_type == "jfrog"
            and "hmm_files" in model_cfgs
            and model_cfgs["hmm_files"].get("hmm_path", None) is not None
            and model_cfgs["hmm_files"]["hmm_path"].strip()
        ):
            hmm_path = model_cfgs["hmm_files"]["hmm_path"]
        else:
            hmm_path, repo_id = _generate_hmm_path(
                model_cfgs, source_type, model_type, target
            )

        extract_dir_edit = (
            extract_dir
            if extract_dir is not None
            else _generate_extract_dir(
                hmm_path, os.path.abspath(os.path.join("./output", target))
            )
        )

        if source_type == "modelscope" and model_type in ["llm"] and repo_id:
            version = model_cfgs["version"].lower()
            download_dir = os.path.abspath("./") if not download_dir else download_dir
            allow_patterns = [f"{hmm_path}/*"]
            download_flag = _download_from_modelscope(
                repo_id,
                local_dir=download_dir,
                allow_patterns=allow_patterns,
                revision=f"{target}-{version}",
            )

            model_src_dir = f"{download_dir}/{hmm_path}"
            if download_flag and os.path.exists(model_src_dir):
                os.makedirs(extract_dir_edit, exist_ok=True)
                try:
                    for item in os.listdir(model_src_dir):
                        src_item = os.path.join(model_src_dir, item)
                        dst_item = os.path.join(extract_dir_edit, item)
                        shutil.move(src_item, dst_item)
                    shutil.rmtree(model_src_dir, ignore_errors=True)
                    print(
                        f"Rename download model dir: {download_dir}/{hmm_path} -> {extract_dir_edit}"
                    )
                except Exception as e:
                    download_files["ret"] = False
                    print(f"Error: Failed to rename donwnload dir, error msg:{e}")
            elif os.path.exists(f"{download_dir}/{hmm_path}"):
                download_files["ret"] = False
                shutil.rmtree(f"{download_dir}/{hmm_path}", ignore_errors=True)
        else:
            # download from jfrog
            download_file = get_file_from_jfrog(
                hmm_path, download_dir, extract_dir_edit
            )
            if not download_file or not os.path.exists(download_file):
                download_files["ret"] = False
            if "hmm_files" not in download_files:
                download_files["hmm_files"] = dict()
            download_files["hmm_files"]["hmm_path"] = download_file

        if (
            "hmm_files" in model_cfgs
            and model_cfgs["hmm_files"].get("other_files", None) is not None
            and isinstance(model_cfgs["hmm_files"]["other_files"], list)
            and len(model_cfgs["hmm_files"]["other_files"]) > 0
        ):
            _download_other_files(
                "hmm_files", model_cfgs, download_files, download_dir, extract_dir
            )

    # download default files
    if (
        model_cfgs["model_type"] in ["llm"]
        and model_cfgs.get("modelscope_repo", None) is not None
        and len(model_cfgs["modelscope_repo"].get("repo_ids", list())) > 0
    ):
        local_dirs = model_cfgs["modelscope_repo"].get("local_dirs", None)
        if (
            local_dirs is not None
            and isinstance(local_dirs, list)
            and len(local_dirs) > 0
        ):
            local_dirs = model_cfgs["modelscope_repo"]["local_dirs"]
        else:
            local_dirs = None

        cfg_ignore_patterns = model_cfgs["modelscope_repo"].get("ignore_patterns", None)
        if (
            cfg_ignore_patterns is not None
            and isinstance(cfg_ignore_patterns, list)
            and len(cfg_ignore_patterns) > 0
        ):
            ignore_patterns = model_cfgs["modelscope_repo"]["ignore_patterns"]

        for idx, repo_id in enumerate(model_cfgs["modelscope_repo"]["repo_ids"]):
            if local_dirs is not None and len(local_dirs[idx]) > 0:
                local_dir = local_dirs[idx]
            else:
                repo_name = repo_id.strip().rsplit("/", 1)[-1]
                local_dir = f"{download_dir}/{repo_name}"

            download_flag = _download_from_modelscope(
                repo_id, local_dir, ignore_patterns=ignore_patterns
            )
            if not download_flag:
                download_files["ret"] = False
                if os.path.exists(local_dir):
                    shutil.rmtree(local_dir)

    if (
        "default_files" in model_cfgs
        and isinstance(model_cfgs["default_files"], list)
        and len(model_cfgs["default_files"]) > 0
    ):
        download_files["default_files"] = list()
        for default_file in model_cfgs["default_files"]:
            extract_dir_edit = _generate_extract_dir(default_file, ".")
            tmp_file = get_file_from_jfrog(default_file, download_dir, extract_dir_edit)
            if not tmp_file or not os.path.exists(tmp_file):
                download_files["ret"] = False
            download_files["default_files"].append(tmp_file)

    return download_file, download_files


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
    """Compress a single file to .tar.xz format with progress bar.

    Args:
        file_path (str): Path to the file to compress
        output_path (str): Output file path for the compressed archive
        preset (int): Compression level (0-9, default: 9)
    """
    if exclude is None:
        exclude = []

    file_list = []
    total_size = 0
    folder_abs_path = os.path.abspath(folder_path)

    for root, dirs, files in os.walk(folder_path):
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
    """Compress multiple files/directories to .tar.xz format with progress bar.

    Args:
        file_paths: List of file/directory paths to compress
        output_path (str): Output file path for the compressed archive
        preset (int): xz compression preset (0-9, default: 9)
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
    """Compress multiple files/directories to .tar.xz format with progress bar.

    Args:
        file_paths: List of file/directory paths to compress
        output_path (str): Output file path for the compressed archive
        preset (int): xz compression preset (0-9, default: 9)
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
                    if os.path.isfile(file_path):
                        arcname = os.path.basename(file_path)
                        tarinfo = tar.gettarinfo(file_path, arcname=arcname)
                        with open(file_path, "rb") as f:
                            tar.addfile(tarinfo, ProgressFile(f, pbar))
                    elif os.path.isdir(file_path):
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


def upload_file_to_artifactory(file_path, upload_url, max_retries=3):
    """Upload file to Artifactory server with checksum verification and retry mechanism.

    Args:
        file_path (str): Local file path to upload
        upload_url (str): Target upload URL (relative path that will be appended to base URL)
        max_retries (int): Maximum number of retry attempts (default: 3)

    Returns:
        bool: True if upload is successful, False otherwise
    """
    username = os.getenv("JFROG_USERNAME")
    password = os.getenv("JFROG_PASSWORD")
    if not username or not password:
        print("Username and password must be provided")
        return False

    BASE_URL = JFROG_REPO
    upload_url = os.path.join(BASE_URL, upload_url)

    if not os.path.isfile(file_path):
        print(f"Not found file: {file_path}")
        return False

    def calculate_checksums(filepath):
        md5_hash = hashlib.md5()
        sha1_hash = hashlib.sha1()

        with open(filepath, "rb") as f:
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

    auth = HTTPBasicAuth(username, password)

    with open(file_path, "rb") as file:
        file_data = file.read()

    headers = {
        "Content-Type": "application/octet-stream",
        "X-Checksum-Md5": md5,
        "X-Checksum-Sha1": sha1,
    }

    for attempt in range(max_retries):
        try:
            response = requests.put(
                url=upload_url,
                data=file_data,
                headers=headers,
                auth=auth,
                timeout=30,
            )

            if response.status_code in [200, 201]:
                print(f"Upload: {upload_url}, done.")
                return True
            else:
                print(
                    f"Upload fail (try {attempt + 1}/{max_retries}): HTTP {response.status_code} - {response.text}"
                )

        except requests.exceptions.RequestException as e:
            print(f"Network error (try {attempt + 1}/{max_retries}) : {str(e)}")

        if attempt < max_retries - 1:
            wait_time = 2**attempt
            print(f"Waiting {wait_time}s retry...")
            time.sleep(wait_time)

    print(f"Upload file failed, retry times: {max_retries}")
    return False
