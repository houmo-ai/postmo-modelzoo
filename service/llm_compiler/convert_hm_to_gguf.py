#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import ast
import logging
import argparse
import contextlib
import json
import os
import re
import sys
from enum import IntEnum
from pathlib import Path
from hashlib import sha256
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    ContextManager,
    Iterable,
    Iterator,
    Literal,
    Sequence,
    TypeVar,
    cast,
)
from itertools import chain
from transformers import AutoConfig

import math
import numpy as np
import torch
from torch import Tensor
import torch.nn as nn
import re
from compile_llms import _publish_model
from urllib.parse import urlparse
import zipfile
import requests
import subprocess


# if "NO_LOCAL_GGUF" not in os.environ:
#    sys.path.insert(1, str(Path(__file__).parent / "gguf-py"))
import gguf
from datetime import datetime

logger = logging.getLogger("hf-to-gguf")
global model_name_from_arg
global size_label_from_arg
global prefill_len_from_arg
global contxt_len_from_arg
global core_num_from_arg
global device_num_from_arg
global version_from_arg
global target_from_arg
global unzip_file_dir


def run_python_script(script_path, *args):
    """
    执行外部Python脚本
    :param script_path: 脚本路径（相对或绝对路径）
    :param args: 传递给脚本的参数（可选）
    :return: 脚本执行结果（返回码、 stdout、stderr）
    """
    try:
        # 构造命令：python3 + 脚本路径 + 参数
        cmd = ['python3', script_path] + list(args)
        logger.info("##hm run in run_python_script")
        # 执行命令，捕获输出和错误
        result = subprocess.run(
            cmd,
            check=True,  # 命令执行失败时抛出异常
            stdout=subprocess.PIPE,  # 捕获标准输出
            stderr=subprocess.PIPE,  # 捕获标准错误
            text=True  # 输出转为字符串（而非字节）
        )

        return {
            'success': True,
            'returncode': result.returncode,
            'stdout': result.stdout,
            'stderr': result.stderr
        }

    except subprocess.CalledProcessError as e:
        # 命令返回非0状态码（执行失败）
        return {
            'success': False,
            'returncode': e.returncode,
            'stdout': e.stdout,
            'stderr': e.stderr
        }
    except FileNotFoundError:
        # 脚本不存在或python3未找到
        return {
            'success': False,
            'error': f"脚本不存在或Python解释器未找到：{script_path}"
        }
    except Exception as e:
        return {
            'success': False,
            'error': str(e)
        }


def execute_linux_command(command, shell=False):
    """
    执行Linux命令并返回结果字典
    :param command: 命令及参数（列表或字符串）
    :param shell: 是否通过shell执行
    :return: 包含执行状态和信息的字典
    """
    try:
        result = subprocess.run(
            command,
            shell=shell,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        return {
            'success': result.returncode == 0,
            'error': result.stderr.strip()  # 存储错误信息
        }
    except FileNotFoundError:
        cmd_str = command[0] if isinstance(command, list) else command.split()[0]
        return {'success': False, 'error': f"命令不存在: {cmd_str}"}
    except Exception as e:
        return {'success': False, 'error': f"执行错误: {str(e)}"}


###### MODEL DEFINITIONS ######
def download_and_unzip(url, save_dir):
    """
    下载ZIP文件到指定文件夹并解压缩
    :param url: ZIP文件的URL地址
    :param save_dir: 保存和解压缩的目标文件夹
    :return: 字典包含下载和解压缩结果
    """
    try:
        # 1. 创建目标文件夹（如果不存在）
        os.makedirs(save_dir, exist_ok=True)

        # 2. 从URL提取文件名（默认取URL中的文件名，若无则用default.zip）
        parsed_url = urlparse(url)
        filename = os.path.basename(parsed_url.path)
        if not filename.endswith('.zip'):  # 确保文件名以.zip结尾
            filename = f"{os.path.splitext(filename)[0]}.zip"
        if filename == '.zip':  # 处理URL完全没有文件名的情况
            filename = "default.zip"

        # 3. 拼接保存路径
        zip_path = os.path.join(save_dir, filename)

        # 4. 下载ZIP文件
        logger.info(f"##hm wait:downloading file....: {url}")
        response = requests.get(url, stream=True, timeout=60)
        response.raise_for_status()  # 检查HTTP错误（如404、500）

        with open(zip_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        logger.info(f"save zip file success: {zip_path}")

        # 5. 解压缩ZIP文件到目标文件夹
        logger.info(f"starting extract to dir: {save_dir}")
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            # 解压缩所有文件到save_dir
            zip_ref.extractall(save_dir)
        logger.info("decompress done")

        return {
            'success': True,
            'zip_path': zip_path,
            'unzip_dir': save_dir,
            'files': zip_ref.namelist()  # 返回压缩包内的文件列表
        }

    except requests.exceptions.RequestException as e:
        return {'success': False, 'error': f"下载失败: {str(e)}"}
    except zipfile.BadZipFile:
        return {'success': False, 'error': f"文件不是有效的ZIP格式: {zip_path}"}
    except PermissionError:
        return {'success': False, 'error': f"权限不足，无法写入或解压缩到: {save_dir}"}
    except Exception as e:
        return {'success': False, 'error': f"发生错误: {str(e)}"}

def extract_url(input_str):
        # 正则表达式匹配以http://开头的URL（直到空格或字符串结束）
        pattern = r'http://[^ ]+'
        # 搜索匹配的内容
        match = re.search(pattern, input_str)
        if match:
            #return match.group()  # 返回匹配到的URL
            # 提取URL并去除所有换行符（\n和\r）
            url = match.group()
            cleaned_url = url.replace('\n', '').replace('\r', '')
            return cleaned_url
        else:
            return None  # 未找到URL时返回None

def extract_path_and_filename(file_str):
    # 提取文件名（路径中最后一个部分）
    filename = os.path.basename(file_str)

    # 提取路径名（文件名之外的部分）
    pathname = os.path.dirname(file_str)

    return pathname, filename

def capitalize_first(s):
    """
    将字符串的第一个字母改为大写，其余字符保持不变
    :param s: 任意字符串
    :return: 处理后的字符串
    """
    if not s:  # 处理空字符串
        return s
    # 第一个字符转大写 + 剩余字符拼接
    return s[0].upper() + s[1:]

def get_current_file_dir():
    """
    获取当前代码文件所在的目录路径（绝对路径）
    :return: 当前文件所在目录的绝对路径字符串
    """
    # __file__ 是内置变量，存储当前文件的路径（可能是相对路径或绝对路径）
    current_file_path = __file__

    # 转换为绝对路径（处理相对路径的情况）
    absolute_path = os.path.abspath(current_file_path)

    # 提取目录路径（去掉文件名，只保留所在文件夹）
    current_dir = os.path.dirname(absolute_path)

    return current_dir

class SentencePieceTokenTypes(IntEnum):
    NORMAL = 1
    UNKNOWN = 2
    CONTROL = 3
    USER_DEFINED = 4
    UNUSED = 5
    BYTE = 6


class ModelType(IntEnum):
    TEXT = 1
    MMPROJ = 2


AnyModel = TypeVar("AnyModel", bound="type[ModelBase]")


class ModelBase:
    _model_classes: dict[ModelType, dict[str, type[ModelBase]]] = {
        ModelType.TEXT: {},
        ModelType.MMPROJ: {},
    }

    dir_model: Path
    ftype: gguf.LlamaFileType
    fname_out: Path
    is_big_endian: bool
    endianess: gguf.GGUFEndian
    use_temp_file: bool
    lazy: bool
    part_names: list[str]
    is_safetensors: bool
    hparams: dict[str, Any]
    tensor_names: set[str] | None
    gguf_writer: gguf.GGUFWriter
    model_name: str | None
    metadata_override: Path | None
    dir_model_card: Path
    remote_hf_model_id: str | None

    # subclasses should define this!
    model_arch: gguf.MODEL_ARCH
    model_description: str | None = None

    # subclasses should initialize this!
    block_count: int
    tensor_map: gguf.TensorNameMap

    def __init__(
        self,
        dir_model: Path,
        ftype: gguf.LlamaFileType,
        fname_out: Path,
        *,
        is_big_endian: bool = False,
        use_temp_file: bool = False,
        eager: bool = False,
        metadata_override: Path | None = None,
        model_name: str | None = None,
        split_max_tensors: int = 0,
        split_max_size: int = 0,
        dry_run: bool = False,
        small_first_shard: bool = False,
        hparams: dict[str, Any] | None = None,
        remote_hf_model_id: str | None = None,
        model_description: str | None = None,
        hmm_vit_height: int = 0,
        hmm_vit_width: int = 0,
    ):
        if (
            type(self) is ModelBase
            or type(self) is TextModel
            or type(self) is MmprojModel
        ):
            raise TypeError(
                f"{type(self).__name__!r} should not be directly instantiated"
            )

        self.dir_model = dir_model
        self.ftype = ftype
        self.fname_out = fname_out
        self.is_big_endian = is_big_endian
        self.endianess = (
            gguf.GGUFEndian.BIG if is_big_endian else gguf.GGUFEndian.LITTLE
        )
        self.use_temp_file = use_temp_file
        self.lazy = not eager or (remote_hf_model_id is not None)
        self.remote_hf_model_id = remote_hf_model_id
        if remote_hf_model_id is not None:
            self.is_safetensors = True

            def get_remote_tensors() -> Iterator[tuple[str, Tensor]]:
                logger.info(
                    f"Using remote model with HuggingFace id: {remote_hf_model_id}"
                )
                remote_tensors = (
                    gguf.utility.SafetensorRemote.get_list_tensors_hf_model(
                        remote_hf_model_id
                    )
                )
                self.tensor_names = set(name for name in remote_tensors.keys())
                for (
                    name,
                    remote_tensor,
                ) in gguf.utility.SafetensorRemote.get_list_tensors_hf_model(
                    remote_hf_model_id
                ).items():
                    yield (name, LazyTorchTensor.from_remote_tensor(remote_tensor))

            self.get_tensors = get_remote_tensors
        else:
            self.part_names = ModelBase.get_model_part_names(
                self.dir_model, "model", ".safetensors"
            )
            logger.info(f"#hm szj part_names is:{self.part_names}")
            self.is_safetensors = len(self.part_names) > 0
            if not self.is_safetensors:
                self.part_names = ModelBase.get_model_part_names(
                    self.dir_model, "pytorch_model", ".bin"
                )
        self.hparams = (
            ModelBase.load_hparams(self.dir_model) if hparams is None else hparams
        )
        self.tensor_names = None
        self.metadata_override = metadata_override
        self.model_name = model_name
        self.dir_model_card = dir_model  # overridden in convert_lora_to_gguf.py
        self.model_description = model_description
        self.hmm_vit_height = hmm_vit_height
        self.hmm_vit_width = hmm_vit_width

        # Apply heuristics to figure out typical tensor encoding based on first layer tensor encoding type
        if self.ftype == gguf.LlamaFileType.GUESSED:
            # NOTE: can't use field "torch_dtype" in config.json, because some finetunes lie.
            _, first_tensor = next(self.get_tensors())
            if first_tensor.dtype == torch.float16:
                logger.info(
                    f"choosing --outtype f16 from first tensor type ({first_tensor.dtype})"
                )
                self.ftype = gguf.LlamaFileType.MOSTLY_F16
            else:
                logger.info(
                    f"choosing --outtype bf16 from first tensor type ({first_tensor.dtype})"
                )
                self.ftype = gguf.LlamaFileType.MOSTLY_BF16

        # Configure GGUF Writer
        self.gguf_writer = gguf.GGUFWriter(
            path=None,
            arch=gguf.MODEL_ARCH_NAMES[self.model_arch],
            endianess=self.endianess,
            use_temp_file=self.use_temp_file,
            split_max_tensors=split_max_tensors,
            split_max_size=split_max_size,
            dry_run=dry_run,
            small_first_shard=small_first_shard,
        )

    @classmethod
    def add_prefix_to_filename(cls, path: Path, prefix: str) -> Path:
        stem, suffix = path.stem, path.suffix
        new_name = f"{prefix}{stem}{suffix}"
        return path.with_name(new_name)

    def find_hparam(self, keys: Iterable[str], optional: bool = False) -> Any:
        key = next((k for k in keys if k in self.hparams), None)
        if key is not None:
            return self.hparams[key]
        if optional:
            return None
        raise KeyError(f"could not find any of: {keys}")

    def get_tensors(self) -> Iterator[tuple[str, Tensor]]:
        tensor_names_from_parts: set[str] = set()

        index_name = "model.safetensors" if self.is_safetensors else "pytorch_model.bin"
        index_name += ".index.json"
        index_file = self.dir_model / index_name

        if index_file.is_file():
            self.tensor_names = set()
            logger.info(f"gguf: loading model weight map from '{index_name}'")
            with open(index_file, "r", encoding="utf-8") as f:
                index: dict[str, Any] = json.load(f)
                weight_map = index.get("weight_map")
                if weight_map is None or not isinstance(weight_map, dict):
                    raise ValueError(f"Can't load 'weight_map' from {index_name!r}")
                self.tensor_names.update(weight_map.keys())
        else:
            self.tensor_names = tensor_names_from_parts
            weight_map = {}

        for part_name in self.part_names:
            logger.info(f"gguf: loading model part '{part_name}'")
            ctx: ContextManager[Any]
            if self.is_safetensors:
                from safetensors import safe_open

                ctx = cast(
                    ContextManager[Any],
                    safe_open(self.dir_model / part_name, framework="pt", device="cpu"),
                )
            else:
                ctx = contextlib.nullcontext(
                    torch.load(
                        str(self.dir_model / part_name),
                        map_location="cpu",
                        mmap=True,
                        weights_only=True,
                    )
                )

            with ctx as model_part:
                tensor_names_from_parts.update(model_part.keys())

                for name in model_part.keys():
                    if self.is_safetensors:
                        if self.lazy:
                            data = model_part.get_slice(name)
                            data = LazyTorchTensor.from_safetensors_slice(data)
                        else:
                            data = model_part.get_tensor(name)
                    else:
                        data = model_part[name]
                        if self.lazy:
                            data = LazyTorchTensor.from_eager(data)
                    yield name, data

        # verify tensor name presence and identify potentially missing files
        if len(tensor_names_from_parts.symmetric_difference(self.tensor_names)) > 0:
            missing = sorted(self.tensor_names.difference(tensor_names_from_parts))
            extra = sorted(tensor_names_from_parts.difference(self.tensor_names))
            missing_files = sorted(
                set(weight_map[n] for n in missing if n in weight_map)
            )
            if len(extra) == 0 and len(missing_files) > 0:
                raise ValueError(
                    f"Missing or incomplete model files: {missing_files}\n"
                    f"Missing tensors: {missing}"
                )
            else:
                raise ValueError(
                    "Mismatch between weight map and model parts for tensor names:\n"
                    f"Missing tensors: {missing}\n"
                    f"Extra tensors: {extra}"
                )

    def format_tensor_name(
        self, key: gguf.MODEL_TENSOR, bid: int | None = None, suffix: str = ".weight"
    ) -> str:
        if key not in gguf.MODEL_TENSORS[self.model_arch]:
            raise ValueError(
                f"Missing {key!r} for MODEL_TENSORS of {self.model_arch!r}"
            )
        name: str = gguf.TENSOR_NAMES[key]
        if "{bid}" in name:
            assert bid is not None
            name = name.format(bid=bid)
        return name + suffix

    def match_model_tensor_name(
        self,
        name: str,
        key: gguf.MODEL_TENSOR,
        bid: int | None,
        suffix: str = ".weight",
    ) -> bool:
        if key not in gguf.MODEL_TENSORS[self.model_arch]:
            return False
        key_name: str = gguf.TENSOR_NAMES[key]
        if "{bid}" in key_name:
            if bid is None:
                return False
            key_name = key_name.format(bid=bid)
        else:
            if bid is not None:
                return False
        return name == (key_name + suffix)

    def map_tensor_name(
        self, name: str, try_suffixes: Sequence[str] = (".weight", ".bias")
    ) -> str:
        new_name = self.tensor_map.get_name(key=name, try_suffixes=try_suffixes)
        if new_name is None:
            raise ValueError(f"Can not map tensor {name!r}")
        return new_name

    def set_gguf_parameters(self):
        raise NotImplementedError(
            "set_gguf_parameters() must be implemented in subclasses"
        )

    def modify_tensors(
        self, data_torch: Tensor, name: str, bid: int | None
    ) -> Iterable[tuple[str, Tensor]]:
        del bid  # unused

        return [(self.map_tensor_name(name), data_torch)]

    def tensor_force_quant(
        self, name: str, new_name: str, bid: int | None, n_dims: int
    ) -> gguf.GGMLQuantizationType | bool:
        del name, new_name, bid, n_dims  # unused

        return False

    # some models need extra generated tensors (like rope_freqs)
    def generate_extra_tensors(self) -> Iterable[tuple[str, Tensor]]:
        return ()

    def prepare_tensors(self):
        max_name_len = max(len(s) for _, s in self.tensor_map.mapping.values()) + len(
            ".weight,"
        )

        for name, data_torch in chain(
            self.generate_extra_tensors(), self.get_tensors()
        ):
            # we don't need these
            if name.endswith(
                (".attention.masked_bias", ".attention.bias", ".rotary_emb.inv_freq")
            ):
                continue

            old_dtype = data_torch.dtype

            # convert any unsupported data types to float32
            if data_torch.dtype not in (torch.float16, torch.float32):
                data_torch = data_torch.to(torch.float32)

            # use the first number-like part of the tensor name as the block id
            bid = None
            for part in name.split("."):
                if part.isdecimal():
                    bid = int(part)
                    break

            for new_name, data_torch in self.modify_tensors(data_torch, name, bid):
                # TODO: why do we squeeze here?
                # data = data_torch.squeeze().numpy()
                data = data_torch.numpy()

                # if data ends up empty, it means data_torch was a scalar tensor -> restore
                if len(data.shape) == 0:
                    data = data_torch.numpy()

                n_dims = len(data.shape)
                data_qtype: gguf.GGMLQuantizationType | bool = self.tensor_force_quant(
                    name, new_name, bid, n_dims
                )

                # Most of the codebase that takes in 1D tensors or norms only handles F32 tensors
                if n_dims <= 1 or new_name.endswith("_norm.weight"):
                    data_qtype = gguf.GGMLQuantizationType.F32

                # Conditions should closely match those in llama_model_quantize_internal in llama.cpp
                # Some tensor types are always in float32
                if data_qtype is False and (
                    any(
                        self.match_model_tensor_name(new_name, key, bid)
                        for key in (
                            gguf.MODEL_TENSOR.FFN_GATE_INP,
                            gguf.MODEL_TENSOR.POS_EMBD,
                            gguf.MODEL_TENSOR.TOKEN_TYPES,
                            gguf.MODEL_TENSOR.SSM_CONV1D,
                            gguf.MODEL_TENSOR.TIME_MIX_FIRST,
                            gguf.MODEL_TENSOR.TIME_MIX_W1,
                            gguf.MODEL_TENSOR.TIME_MIX_W2,
                            gguf.MODEL_TENSOR.TIME_MIX_DECAY_W1,
                            gguf.MODEL_TENSOR.TIME_MIX_DECAY_W2,
                            gguf.MODEL_TENSOR.TIME_MIX_LERP_FUSED,
                            gguf.MODEL_TENSOR.POSNET_NORM1,
                            gguf.MODEL_TENSOR.POSNET_NORM2,
                            gguf.MODEL_TENSOR.V_ENC_EMBD_POS,
                            gguf.MODEL_TENSOR.A_ENC_EMBD_POS,
                            gguf.MODEL_TENSOR.ALTUP_CORRECT_COEF,
                            gguf.MODEL_TENSOR.ALTUP_PREDICT_COEF,
                        )
                    )
                    or not new_name.endswith(".weight")
                ):
                    data_qtype = gguf.GGMLQuantizationType.F32

                if data_qtype is False and any(
                    self.match_model_tensor_name(new_name, key, bid)
                    for key in (
                        gguf.MODEL_TENSOR.TOKEN_EMBD,
                        gguf.MODEL_TENSOR.PER_LAYER_TOKEN_EMBD,
                        gguf.MODEL_TENSOR.OUTPUT,
                        gguf.MODEL_TENSOR.ALTUP_ROUTER,
                        gguf.MODEL_TENSOR.LAUREL_L,
                        gguf.MODEL_TENSOR.LAUREL_R,
                    )
                ):
                    if self.ftype in (
                        gguf.LlamaFileType.MOSTLY_TQ1_0,
                        gguf.LlamaFileType.MOSTLY_TQ2_0,
                    ):
                        # TODO: use Q4_K and Q6_K
                        data_qtype = gguf.GGMLQuantizationType.F16

                # No override (data_qtype is False), or wants to be quantized (data_qtype is True)
                if isinstance(data_qtype, bool):
                    if self.ftype == gguf.LlamaFileType.ALL_F32:
                        data_qtype = gguf.GGMLQuantizationType.F32
                    elif self.ftype == gguf.LlamaFileType.MOSTLY_F16:
                        data_qtype = gguf.GGMLQuantizationType.F16
                    elif self.ftype == gguf.LlamaFileType.MOSTLY_BF16:
                        data_qtype = gguf.GGMLQuantizationType.BF16
                    elif self.ftype == gguf.LlamaFileType.MOSTLY_Q8_0:
                        data_qtype = gguf.GGMLQuantizationType.Q8_0
                    elif self.ftype == gguf.LlamaFileType.MOSTLY_TQ1_0:
                        data_qtype = gguf.GGMLQuantizationType.TQ1_0
                    elif self.ftype == gguf.LlamaFileType.MOSTLY_TQ2_0:
                        data_qtype = gguf.GGMLQuantizationType.TQ2_0
                    else:
                        raise ValueError(f"Unknown file type: {self.ftype.name}")

                try:
                    data = gguf.quants.quantize(data, data_qtype)
                except gguf.QuantError as e:
                    logger.warning("%s, %s", e, "falling back to F16")
                    data_qtype = gguf.GGMLQuantizationType.F16
                    data = gguf.quants.quantize(data, data_qtype)

                shape = (
                    gguf.quant_shape_from_byte_shape(data.shape, data_qtype)
                    if data.dtype == np.uint8
                    else data.shape
                )

                # reverse shape to make it similar to the internal ggml dimension order
                shape_str = f"{{{', '.join(str(n) for n in reversed(shape))}}}"

                # n_dims is implicit in the shape
                logger.info(
                    f"{f'%-{max_name_len}s' % f'{new_name},'} {old_dtype} --> {data_qtype.name}, shape = {shape_str}"
                )

                self.gguf_writer.add_tensor(new_name, data, raw_dtype=data_qtype)

    def set_type(self):
        self.gguf_writer.add_type(gguf.GGUFType.MODEL)

    def prepare_metadata(self, vocab_only: bool):

        total_params, shared_params, expert_params, expert_count = (
            self.gguf_writer.get_total_parameter_count()
        )

        self.metadata = gguf.Metadata.load(
            self.metadata_override, self.dir_model_card, self.model_name, total_params
        )

        # If we are using HF model id, set the metadata name to the model id
        if self.remote_hf_model_id:
            self.metadata.name = self.remote_hf_model_id

        # Fallback to model directory name if metadata name is still missing
        if self.metadata.name is None:
            self.metadata.name = self.dir_model.name

        # Generate parameter weight class (useful for leader boards) if not yet determined
        if self.metadata.size_label is None and total_params > 0:
            self.metadata.size_label = gguf.size_label(
                total_params, shared_params, expert_params, expert_count
            )

        logger.info(f"#hm calculated self.metadata.size_label is:{self.metadata.size_label}")
        #self.metadata.size_label = size_label_from_arg
        #logger.info(f"#hm from arg parse self.metadata.size_label is:{self.metadata.size_label}")

        self.set_type()

        logger.info("Set meta model")
        self.metadata.set_gguf_meta_model(self.gguf_writer)

        logger.info("Set model parameters")
        self.set_gguf_parameters()

        logger.info("Set model quantization version")
        self.gguf_writer.add_quantization_version(gguf.GGML_QUANT_VERSION)

    def write_vocab(self):
        raise NotImplementedError("write_vocab() must be implemented in subclasses")

    def write(self):
        self.prepare_tensors()
        self.prepare_metadata(vocab_only=False)
        self.gguf_writer.write_header_to_file(path=self.fname_out)
        self.gguf_writer.write_kv_data_to_file()
        self.gguf_writer.write_tensors_to_file(progress=True)
        self.gguf_writer.close()

    @staticmethod
    def get_model_part_names(dir_model: Path, prefix: str, suffix: str) -> list[str]:
        part_names: list[str] = []
        for filename in os.listdir(dir_model):
            if filename.startswith(prefix) and filename.endswith(suffix):
                part_names.append(filename)

        part_names.sort()

        return part_names

    @staticmethod
    def load_hparams(dir_model: Path):
        try:
            # for security reason, we don't allow loading remote code by default
            # if a model need remote code, we will fallback to config.json
            config = AutoConfig.from_pretrained(
                dir_model, trust_remote_code=False
            ).to_dict()
        except Exception as e:
            logger.warning(f"Failed to load model config from {dir_model}: {e}")
            logger.warning("Trying to load config.json instead")
            with open(dir_model / "config.json", "r", encoding="utf-8") as f:
                config = json.load(f)
        if "llm_config" in config:
            # rename for InternVL
            config["text_config"] = config["llm_config"]
        if "thinker_config" in config:
            # rename for Qwen2.5-Omni
            config["text_config"] = config["thinker_config"]["text_config"]
        return config

    @classmethod
    def register(cls, *names: str) -> Callable[[AnyModel], AnyModel]:
        assert names

        def func(modelcls: AnyModel) -> AnyModel:
            model_type = (
                ModelType.MMPROJ
                if modelcls.model_arch == gguf.MODEL_ARCH.MMPROJ
                else ModelType.TEXT
            )
            for name in names:
                cls._model_classes[model_type][name] = modelcls
            return modelcls

        return func

    @classmethod
    def print_registered_models(cls):
        for model_type, model_classes in cls._model_classes.items():
            logger.error(f"{model_type.name} models:")
            for name in sorted(model_classes.keys()):
                logger.error(f"  - {name}")

    @classmethod
    def from_model_architecture(
        cls, arch: str, model_type=ModelType.TEXT
    ) -> type[ModelBase]:
        try:
            return cls._model_classes[model_type][arch]
        except KeyError:
            raise NotImplementedError(f"Architecture {arch!r} not supported!") from None


class TextModel(ModelBase):
    model_type = ModelType.TEXT
    hf_arch: str

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.hf_arch = get_model_architecture(self.hparams, self.model_type)

        if "text_config" in self.hparams:
            # move the text_config to the root level
            self.hparams = {**self.hparams, **self.hparams["text_config"]}

        self.block_count = self.find_hparam(
            ["n_layers", "num_hidden_layers", "n_layer", "num_layers"]
        )
        self.tensor_map = gguf.get_tensor_name_map(self.model_arch, self.block_count)

    @classmethod
    def __init_subclass__(cls):
        # can't use an abstract property, because overriding it without type errors
        # would require using decorated functions instead of simply defining the property
        if "model_arch" not in cls.__dict__:
            raise TypeError(f"Missing property 'model_arch' for {cls.__name__!r}")

    def set_vocab(self):
        self._set_vocab_gpt2()

    def prepare_metadata(self, vocab_only: bool):
        super().prepare_metadata(vocab_only=vocab_only)

        total_params = self.gguf_writer.get_total_parameter_count()[0]
        # Extract the encoding scheme from the file type name. e.g. 'gguf.LlamaFileType.MOSTLY_Q8_0' --> 'Q8_0'
        output_type: str = self.ftype.name.partition("_")[2]

        # Filename Output
        if self.fname_out.is_dir():
            # Generate default filename based on model specification and available metadata
            if not vocab_only:
                logger.info(f"#hm not vocab_only, vocab_only is:{vocab_only}")
                fname_default: str = gguf.naming_convention(
                    self.metadata.name,
                    self.metadata.basename,
                    self.metadata.finetune,
                    self.metadata.version,
                    self.metadata.size_label,
                    output_type,
                    model_type="LoRA" if total_params < 0 else None,
                )
            else:
                logger.info(f"#hm vocab_only, vocab_only is:{vocab_only}")
                fname_default: str = gguf.naming_convention(
                    self.metadata.name,
                    self.metadata.basename,
                    self.metadata.finetune,
                    self.metadata.version,
                    size_label=None,
                    output_type=None,
                    model_type="vocab",
                )

            # Use the default filename
            # change to hm name style
            global model_name_from_arg
            global size_label_from_arg
            global contxt_len_from_arg
            global core_num_from_arg
            global device_num_from_arg
            global version_from_arg
            global target_from_arg
            global prefill_len_from_arg
            now = datetime.now()
            # 格式化时间为：年月日时分秒（例如：20231005143025）
            datetime_str = now.strftime("%Y%m%d%H%M%S")
            model_name_from_arg=capitalize_first(model_name_from_arg)
            fname_default=f"HiModel-{model_name_from_arg}-{size_label_from_arg}-p{prefill_len_from_arg}-{contxt_len_from_arg}K-{core_num_from_arg}core-{device_num_from_arg}d-{target_from_arg}-v{version_from_arg}-{datetime_str}"
            self.fname_out = self.fname_out / f"{fname_default}.gguf"
            logger.info(f"self.fname_out create here1 : {self.fname_out}")
        else:
            # Output path is a custom defined templated filename
            # Note: `not is_dir()` is used because `.is_file()` will not detect
            #       file template strings as it doesn't actually exist as a file

            # Process templated file name with the output ftype, useful with the "auto" ftype
            self.fname_out = self.fname_out.parent / gguf.fill_templated_filename(
                self.fname_out.name, output_type
            )
            logger.info(f"self.fname_out create here2 : {self.fname_out}")

        logger.info("Set model tokenizer")
        self.set_vocab()

    def set_gguf_parameters(self):
        self.gguf_writer.add_block_count(self.block_count)

        if (
            n_ctx := self.find_hparam(
                ["max_position_embeddings", "n_ctx", "n_positions", "max_length"],
                optional=True,
            )
        ) is not None:
            self.gguf_writer.add_context_length(n_ctx)
            logger.info(f"gguf: context length = {n_ctx}")

        if (
            n_embd := self.find_hparam(["hidden_size", "n_embd", "dim"], optional=True)
        ) is not None:
            self.gguf_writer.add_embedding_length(n_embd)
            logger.info(f"gguf: embedding length = {n_embd}")

        if (
            n_ff := self.find_hparam(
                ["intermediate_size", "n_inner", "hidden_dim"], optional=True
            )
        ) is not None:
            self.gguf_writer.add_feed_forward_length(n_ff)
            logger.info(f"gguf: feed forward length = {n_ff}")

        if (
            n_head := self.find_hparam(
                ["num_attention_heads", "n_head", "n_heads"], optional=True
            )
        ) is not None:
            self.gguf_writer.add_head_count(n_head)
            logger.info(f"gguf: head count = {n_head}")

        if (n_head_kv := self.hparams.get("num_key_value_heads")) is not None:
            self.gguf_writer.add_head_count_kv(n_head_kv)
            logger.info(f"gguf: key-value head count = {n_head_kv}")

        if (rope_theta := self.hparams.get("rope_theta")) is not None:
            self.gguf_writer.add_rope_freq_base(rope_theta)
            logger.info(f"gguf: rope theta = {rope_theta}")
        if (f_rms_eps := self.hparams.get("rms_norm_eps")) is not None:
            self.gguf_writer.add_layer_norm_rms_eps(f_rms_eps)
            logger.info(f"gguf: rms norm epsilon = {f_rms_eps}")
        if (
            f_norm_eps := self.find_hparam(
                ["layer_norm_eps", "layer_norm_epsilon", "norm_epsilon"], optional=True
            )
        ) is not None:
            self.gguf_writer.add_layer_norm_eps(f_norm_eps)
            logger.info(f"gguf: layer norm epsilon = {f_norm_eps}")
        if (n_experts := self.hparams.get("num_local_experts")) is not None:
            self.gguf_writer.add_expert_count(n_experts)
            logger.info(f"gguf: expert count = {n_experts}")
        if (n_experts_used := self.hparams.get("num_experts_per_tok")) is not None:
            self.gguf_writer.add_expert_used_count(n_experts_used)
            logger.info(f"gguf: experts used count = {n_experts_used}")

        if (head_dim := self.hparams.get("head_dim")) is not None:
            self.gguf_writer.add_key_length(head_dim)
            self.gguf_writer.add_value_length(head_dim)

        self.gguf_writer.add_file_type(self.ftype)
        logger.info(f"gguf: file type = {self.ftype}")

    def write_vocab(self):
        if len(self.gguf_writer.tensors) != 1:
            raise ValueError("Splitting the vocabulary is not supported")

        self.prepare_metadata(vocab_only=True)
        self.gguf_writer.write_header_to_file(path=self.fname_out)
        self.gguf_writer.write_kv_data_to_file()
        self.gguf_writer.close()

    def does_token_look_special(self, token: str | bytes) -> bool:
        if isinstance(token, (bytes, bytearray)):
            token_text = token.decode(encoding="utf-8")
        elif isinstance(token, memoryview):
            token_text = token.tobytes().decode(encoding="utf-8")
        else:
            token_text = token

        # Some models mark some added tokens which ought to be control tokens as not special.
        # (e.g. command-r, command-r-plus, deepseek-coder, gemma{,-2})
        seems_special = token_text in (
            "<pad>",  # deepseek-coder
            "<mask>",
            "<2mass>",
            "[@BOS@]",  # gemma{,-2}
        )

        seems_special = seems_special or (
            token_text.startswith("<|") and token_text.endswith("|>")
        )
        seems_special = seems_special or (
            token_text.startswith("<｜") and token_text.endswith("｜>")
        )  # deepseek-coder

        # TODO: should these be marked as UNUSED instead? (maybe not)
        seems_special = seems_special or (
            token_text.startswith("<unused") and token_text.endswith(">")
        )  # gemma{,-2}

        return seems_special

    # used for GPT-2 BPE and WordPiece vocabs
    def get_vocab_base(self) -> tuple[list[str], list[int], str]:
        tokens: list[str] = []
        toktypes: list[int] = []

        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(self.dir_model)
        vocab_size = self.hparams.get("vocab_size", len(tokenizer.vocab))
        assert max(tokenizer.vocab.values()) < vocab_size

        tokpre = self.get_vocab_base_pre(tokenizer)

        reverse_vocab = {
            id_: encoded_tok for encoded_tok, id_ in tokenizer.vocab.items()
        }
        added_vocab = tokenizer.get_added_vocab()

        added_tokens_decoder = tokenizer.added_tokens_decoder

        for i in range(vocab_size):
            if i not in reverse_vocab:
                tokens.append(f"[PAD{i}]")
                toktypes.append(gguf.TokenType.UNUSED)
            else:
                token: str = reverse_vocab[i]
                if token in added_vocab:
                    # The tokenizer in llama.cpp assumes the CONTROL and USER_DEFINED tokens are pre-normalized.
                    # To avoid unexpected issues - we make sure to normalize non-normalized tokens
                    if not added_tokens_decoder[i].normalized:
                        previous_token = token
                        token = tokenizer.decode(
                            tokenizer.encode(token, add_special_tokens=False)
                        )
                        if previous_token != token:
                            logger.info(
                                f"{repr(previous_token)} is encoded and decoded back to {repr(token)} using AutoTokenizer"
                            )

                    if added_tokens_decoder[i].special or self.does_token_look_special(
                        token
                    ):
                        toktypes.append(gguf.TokenType.CONTROL)
                    else:
                        # NOTE: this was added for Gemma.
                        # Encoding and decoding the tokens above isn't sufficient for this case.
                        token = token.replace(
                            b"\xe2\x96\x81".decode("utf-8"), " "
                        )  # pre-normalize user-defined spaces
                        toktypes.append(gguf.TokenType.USER_DEFINED)
                else:
                    toktypes.append(gguf.TokenType.NORMAL)
                tokens.append(token)

        return tokens, toktypes, tokpre

    # NOTE: this function is generated by convert_hf_to_gguf_update.py
    #       do not modify it manually!
    # ref:  https://github.com/ggml-org/llama.cpp/pull/6920
    # Marker: Start get_vocab_base_pre
    def get_vocab_base_pre(self, tokenizer) -> str:
        # encoding this string and hashing the resulting tokens would (hopefully) give us a unique identifier that
        # is specific for the BPE pre-tokenizer used by the model
        # we will use this unique identifier to write a "tokenizer.ggml.pre" entry in the GGUF file which we can
        # use in llama.cpp to implement the same pre-tokenizer

        chktxt = "\n \n\n \n\n\n \t \t\t \t\n  \n   \n    \n     \n🚀 (normal) 😶\u200d🌫️ (multiple emojis concatenated) ✅ 🦙🦙 3 33 333 3333 33333 333333 3333333 33333333 3.3 3..3 3...3 កាន់តែពិសេសអាច😁 ?我想在apple工作1314151天～ ------======= нещо на Български ''''''```````\"\"\"\"......!!!!!!?????? I've been 'told he's there, 'RE you sure? 'M not sure I'll make it, 'D you like some tea? We'Ve a'lL"

        chktok = tokenizer.encode(chktxt)
        chkhsh = sha256(str(chktok).encode()).hexdigest()

        logger.debug(f"chktok: {chktok}")
        logger.debug(f"chkhsh: {chkhsh}")

        res = None

        # NOTE: if you get an error here, you need to update the convert_hf_to_gguf_update.py script
        #       or pull the latest version of the model from Huggingface
        #       don't edit the hashes manually!
        if chkhsh == "0ef9807a4087ebef797fc749390439009c3b9eda9ad1a097abbe738f486c01e5":
            # ref: https://huggingface.co/meta-llama/Meta-Llama-3-8B
            res = "llama-bpe"
        if chkhsh == "049ecf7629871e3041641907f3de7c733e4dbfdc736f57d882ba0b0845599754":
            # ref: https://huggingface.co/deepseek-ai/deepseek-llm-7b-base
            res = "deepseek-llm"
        if chkhsh == "347715f544604f9118bb75ed199f68779f423cabb20db6de6f31b908d04d7821":
            # ref: https://huggingface.co/deepseek-ai/deepseek-coder-6.7b-base
            res = "deepseek-coder"
        if chkhsh == "8aeee3860c56296a157a1fe2fad249ec40aa59b1bb5709f4ade11c4e6fe652ed":
            # ref: https://huggingface.co/tiiuae/falcon-7b
            res = "falcon"
        if chkhsh == "0876d13b50744004aa9aeae05e7b0647eac9d801b5ba4668afc01e709c15e19f":
            # ref: https://huggingface.co/BAAI/bge-small-en-v1.5
            res = "bert-bge"
        if chkhsh == "9d032fcbd5501f4a38150912590928bfb36091efb5df11b8e2124b0390e3fb1e":
            # ref: https://huggingface.co/tiiuae/Falcon3-7B-Base
            res = "falcon3"
        if chkhsh == "8e62295832751ca1e8f92f2226f403dea30dc5165e448b5bfa05af5340c64ec7":
            # ref: https://huggingface.co/BAAI/bge-large-zh-v1.5
            res = "bert-bge-large"
        if chkhsh == "b6dc8df998e1cfbdc4eac8243701a65afe638679230920b50d6f17d81c098166":
            # ref: https://huggingface.co/mosaicml/mpt-7b
            res = "mpt"
        if chkhsh == "35d91631860c815f952d711435f48d356ebac988362536bed955d43bfa436e34":
            # ref: https://huggingface.co/bigcode/starcoder2-3b
            res = "starcoder"
        if chkhsh == "3ce83efda5659b07b1ad37ca97ca5797ea4285d9b9ab0dc679e4a720c9da7454":
            # ref: https://huggingface.co/openai-community/gpt2
            res = "gpt-2"
        if chkhsh == "32d85c31273f8019248f2559fed492d929ea28b17e51d81d3bb36fff23ca72b3":
            # ref: https://huggingface.co/stabilityai/stablelm-2-zephyr-1_6b
            res = "stablelm2"
        if chkhsh == "6221ad2852e85ce96f791f476e0b390cf9b474c9e3d1362f53a24a06dc8220ff":
            # ref: https://huggingface.co/smallcloudai/Refact-1_6-base
            res = "refact"
        if chkhsh == "9c2227e4dd922002fb81bde4fc02b0483ca4f12911410dee2255e4987644e3f8":
            # ref: https://huggingface.co/CohereForAI/c4ai-command-r-v01
            res = "command-r"
        if chkhsh == "e636dc30a262dcc0d8c323492e32ae2b70728f4df7dfe9737d9f920a282b8aea":
            # ref: https://huggingface.co/Qwen/Qwen1.5-7B
            res = "qwen2"
        if chkhsh == "b6dc8df998e1cfbdc4eac8243701a65afe638679230920b50d6f17d81c098166":
            # ref: https://huggingface.co/allenai/OLMo-1.7-7B-hf
            res = "olmo"
        if chkhsh == "a8594e3edff7c29c003940395316294b2c623e09894deebbc65f33f1515df79e":
            # ref: https://huggingface.co/databricks/dbrx-base
            res = "dbrx"
        if chkhsh == "c7699093ba4255a91e702aa38a596aa81669f3525dae06c2953267dde580f448":
            # ref: https://huggingface.co/jinaai/jina-reranker-v1-tiny-en
            res = "jina-v1-en"
        if chkhsh == "0876d13b50744004aa9aeae05e7b0647eac9d801b5ba4668afc01e709c15e19f":
            # ref: https://huggingface.co/jinaai/jina-embeddings-v2-base-en
            res = "jina-v2-en"
        if chkhsh == "171aeeedd6fb548d418a7461d053f11b6f1f1fc9b387bd66640d28a4b9f5c643":
            # ref: https://huggingface.co/jinaai/jina-embeddings-v2-base-es
            res = "jina-v2-es"
        if chkhsh == "27949a2493fc4a9f53f5b9b029c82689cfbe5d3a1929bb25e043089e28466de6":
            # ref: https://huggingface.co/jinaai/jina-embeddings-v2-base-de
            res = "jina-v2-de"
        if chkhsh == "c136ed14d01c2745d4f60a9596ae66800e2b61fa45643e72436041855ad4089d":
            # ref: https://huggingface.co/abacusai/Smaug-Llama-3-70B-Instruct
            res = "smaug-bpe"
        if chkhsh == "c7ea5862a53e4272c035c8238367063e2b270d51faa48c0f09e9d5b54746c360":
            # ref: https://huggingface.co/LumiOpen/Poro-34B-chat
            res = "poro-chat"
        if chkhsh == "7967bfa498ade6b757b064f31e964dddbb80f8f9a4d68d4ba7998fcf281c531a":
            # ref: https://huggingface.co/jinaai/jina-embeddings-v2-base-code
            res = "jina-v2-code"
        if chkhsh == "7fc505bd3104ca1083b150b17d088b59534ede9bde81f0dd2090967d7fe52cee":
            # ref: https://huggingface.co/LumiOpen/Viking-7B
            res = "viking"
        if chkhsh == "b53802fb28e26d645c3a310b34bfe07da813026ec7c7716883404d5e0f8b1901":
            # ref: https://huggingface.co/core42/jais-13b
            res = "jais"
        if chkhsh == "7b3e7548e4308f52a76e8229e4e6cc831195d0d1df43aed21ac6c93da05fec5f":
            # ref: https://huggingface.co/WisdomShell/CodeShell-7B
            res = "codeshell"
        if chkhsh == "63b97e4253352e6f357cc59ea5b583e3a680eaeaf2632188c2b952de2588485e":
            # ref: https://huggingface.co/mistralai/Mistral-Nemo-Base-2407
            res = "tekken"
        if chkhsh == "855059429035d75a914d1eda9f10a876752e281a054a7a3d421ef0533e5b6249":
            # ref: https://huggingface.co/HuggingFaceTB/SmolLM-135M
            res = "smollm"
        if chkhsh == "3c30d3ad1d6b64202cd222813e7736c2db6e1bd6d67197090fc1211fbc612ae7":
            # ref: https://huggingface.co/bigscience/bloom
            res = "bloom"
        if chkhsh == "bc01ce58980e1db43859146dc51b1758b3b88729b217a74792e9f8d43e479d21":
            # ref: https://huggingface.co/TurkuNLP/gpt3-finnish-small
            res = "gpt3-finnish"
        if chkhsh == "4e2b24cc4770243d65a2c9ec19770a72f08cffc161adbb73fcbb6b7dd45a0aae":
            # ref: https://huggingface.co/LGAI-EXAONE/EXAONE-3.0-7.8B-Instruct
            res = "exaone"
        if chkhsh == "fcace8b9cac38ce847670c970cd5892031a753a1ef381abd1d9af00f713da085":
            # ref: https://huggingface.co/microsoft/phi-2
            res = "phi-2"
        if chkhsh == "60824e3c0d9401f89943cbb2fff727f0e2d4c545ba4df2d6e4f09a6db0f5b450":
            # ref: https://huggingface.co/facebook/chameleon-7b
            res = "chameleon"
        if chkhsh == "8b5a93ed704057481f240da0be7e7dca721d7f8f4755263b6807227a2cbeae65":
            # ref: https://huggingface.co/sentence-transformers/stsb-roberta-base
            res = "roberta-bpe"
        if chkhsh == "ad851be1dba641f2e3711822f816db2c265f788b37c63b4e1aeacb9ee92de8eb":
            # ref: https://huggingface.co/ai-sage/GigaChat-20B-A3B-instruct
            res = "gigachat"
        if chkhsh == "d4c8f286ea6b520b3d495c4455483cfa2302c0cfcd4be05d781b6a8a0a7cdaf1":
            # ref: https://huggingface.co/Infinigence/Megrez-3B-Instruct
            res = "megrez"
        if chkhsh == "877081d19cf6996e2c4ff0e1236341e9b7bde288f5311a56a937f0afbbb3aeb5":
            # ref: https://huggingface.co/deepseek-ai/DeepSeek-V3
            res = "deepseek-v3"
        if chkhsh == "b3f499bb4255f8ca19fccd664443283318f2fd2414d5e0b040fbdd0cc195d6c5":
            # ref: https://huggingface.co/deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B
            res = "deepseek-r1-qwen"
        if chkhsh == "ccc2ef013c104be7bae2965776d611e1d7a8a2a9c547dd93a682c9a9fc80352e":
            # ref: https://huggingface.co/Xenova/gpt-4o
            res = "gpt-4o"
        if chkhsh == "7dec86086fcc38b66b7bc1575a160ae21cf705be7718b9d5598190d7c12db76f":
            # ref: https://huggingface.co/UW/OLMo2-8B-SuperBPE-t180k
            res = "superbpe"
        if chkhsh == "1994ffd01900cfb37395608534236ecd63f2bd5995d6cb1004dda1af50240f15":
            # ref: https://huggingface.co/trillionlabs/Trillion-7B-preview
            res = "trillion"
        if chkhsh == "96a5f08be6259352137b512d4157e333e21df7edd3fcd152990608735a65b224":
            # ref: https://huggingface.co/inclusionAI/Ling-lite
            res = "bailingmoe"
        if chkhsh == "d353350c764d8c3b39c763113960e4fb4919bea5fbf208a0e3b22e8469dc7406":
            # ref: https://huggingface.co/meta-llama/Llama-4-Scout-17B-16E-Instruct
            res = "llama4"
        if chkhsh == "0e9433cbbb161f89e264eb32e8e64bfe69e834973ffca5d41d3948a604a3e2a3":
            # ref: https://huggingface.co/mistral-community/pixtral-12b
            res = "pixtral"
        if chkhsh == "d5f1dd6f980fec569fb218a81a7658ac45fc56b38c5a0adeb1c232fbe04ef5ec":
            # ref: https://huggingface.co/ByteDance-Seed/Seed-Coder-8B-Base
            res = "seed-coder"
        if chkhsh == "b6e8e1518dc4305be2fe39c313ed643381c4da5db34a98f6a04c093f8afbe99b":
            # ref: https://huggingface.co/THUDM/glm-4-9b-chat
            res = "chatglm-bpe"
        if chkhsh == "81d72c7348a9f0ebe86f23298d37debe0a5e71149e29bd283904c02262b27516":
            # ref: https://huggingface.co/THUDM/glm-4-9b-chat
            res = "chatglm-bpe"
        if chkhsh == "a1336059768a55c99a734006ffb02203cd450fed003e9a71886c88acf24fdbc2":
            # ref: https://huggingface.co/THUDM/glm-4-9b-hf
            res = "glm4"
        if chkhsh == "1431a23e583c97432bc230bff598d103ddb5a1f89960c8f1d1051aaa944d0b35":
            # ref: https://huggingface.co/sapienzanlp/Minerva-7B-base-v1.0
            res = "minerva-7b"

        if res is None:
            logger.warning("\n")
            logger.warning(
                "**************************************************************************************"
            )
            logger.warning("** WARNING: The BPE pre-tokenizer was not recognized!")
            logger.warning("**          There are 2 possible reasons for this:")
            logger.warning(
                "**          - the model has not been added to convert_hf_to_gguf_update.py yet"
            )
            logger.warning(
                "**          - the pre-tokenization config has changed upstream"
            )
            logger.warning(
                "**          Check your model files and convert_hf_to_gguf_update.py and update them accordingly."
            )
            logger.warning(
                "** ref:     https://github.com/ggml-org/llama.cpp/pull/6920"
            )
            logger.warning("**")
            logger.warning(f"** chkhsh:  {chkhsh}")
            logger.warning(
                "**************************************************************************************"
            )
            logger.warning("\n")
            raise NotImplementedError(
                "BPE pre-tokenizer was not recognized - update get_vocab_base_pre()"
            )

        logger.debug(f"tokenizer.ggml.pre: {repr(res)}")
        logger.debug(f"chkhsh: {chkhsh}")

        return res
        # Marker: End get_vocab_base_pre

    def _set_vocab_none(self) -> None:
        self.gguf_writer.add_tokenizer_model("none")

    def _set_vocab_gpt2(self) -> None:
        tokens, toktypes, tokpre = self.get_vocab_base()
        self.gguf_writer.add_tokenizer_model("gpt2")
        self.gguf_writer.add_tokenizer_pre(tokpre)
        self.gguf_writer.add_token_list(tokens)
        self.gguf_writer.add_token_types(toktypes)

        special_vocab = gguf.SpecialVocab(self.dir_model, load_merges=True)
        special_vocab.add_to_gguf(self.gguf_writer)

    def _set_vocab_qwen(self):
        dir_model = self.dir_model
        hparams = self.hparams
        tokens: list[str] = []
        toktypes: list[int] = []

        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(dir_model, trust_remote_code=True)
        vocab_size = hparams["vocab_size"]
        assert max(tokenizer.get_vocab().values()) < vocab_size

        tokpre = self.get_vocab_base_pre(tokenizer)

        merges = []
        vocab = {}
        mergeable_ranks = tokenizer.mergeable_ranks
        for token, rank in mergeable_ranks.items():
            vocab[QwenModel.token_bytes_to_string(token)] = rank
            if len(token) == 1:
                continue
            merged = QwenModel.bpe(mergeable_ranks, token, max_rank=rank)
            assert len(merged) == 2
            merges.append(" ".join(map(QwenModel.token_bytes_to_string, merged)))

        # for this kind of tokenizer, added_vocab is not a subset of vocab, so they need to be combined
        added_vocab = tokenizer.special_tokens
        reverse_vocab = {
            id_: encoded_tok for encoded_tok, id_ in {**vocab, **added_vocab}.items()
        }

        for i in range(vocab_size):
            if i not in reverse_vocab:
                tokens.append(f"[PAD{i}]")
                toktypes.append(gguf.TokenType.UNUSED)
            elif reverse_vocab[i] in added_vocab:
                tokens.append(reverse_vocab[i])
                toktypes.append(gguf.TokenType.CONTROL)
            else:
                tokens.append(reverse_vocab[i])
                toktypes.append(gguf.TokenType.NORMAL)

        self.gguf_writer.add_tokenizer_model("gpt2")
        self.gguf_writer.add_tokenizer_pre(tokpre)
        self.gguf_writer.add_token_list(tokens)
        self.gguf_writer.add_token_types(toktypes)

        special_vocab = gguf.SpecialVocab(dir_model, load_merges=False)
        special_vocab.merges = merges
        # only add special tokens when they were not already loaded from config.json
        if len(special_vocab.special_token_ids) == 0:
            special_vocab._set_special_token(
                "bos", tokenizer.special_tokens["<|endoftext|>"]
            )
            special_vocab._set_special_token(
                "eos", tokenizer.special_tokens["<|endoftext|>"]
            )
        # this one is usually not in config.json anyway
        special_vocab._set_special_token(
            "unk", tokenizer.special_tokens["<|endoftext|>"]
        )
        special_vocab.add_to_gguf(self.gguf_writer)

    def _set_vocab_sentencepiece(self, add_to_gguf=True):
        tokens, scores, toktypes = self._create_vocab_sentencepiece()

        self.gguf_writer.add_tokenizer_model("llama")
        self.gguf_writer.add_tokenizer_pre("default")
        self.gguf_writer.add_token_list(tokens)
        self.gguf_writer.add_token_scores(scores)
        self.gguf_writer.add_token_types(toktypes)

        special_vocab = gguf.SpecialVocab(self.dir_model, n_vocab=len(tokens))
        special_vocab.add_to_gguf(self.gguf_writer)

    def _create_vocab_sentencepiece(self):
        from sentencepiece import SentencePieceProcessor

        tokenizer_path = self.dir_model / "tokenizer.model"

        if not tokenizer_path.is_file():
            raise FileNotFoundError(f"File not found: {tokenizer_path}")

        tokenizer = SentencePieceProcessor()
        tokenizer.LoadFromFile(str(tokenizer_path))

        vocab_size = (
            self.find_hparam(
                [
                    "vocab_size_per_layer_input",  # gemma3n
                    "vocab_size",
                ],
                optional=True,
            )
            or tokenizer.vocab_size()
        )

        tokens: list[bytes] = [f"[PAD{i}]".encode("utf-8") for i in range(vocab_size)]
        scores: list[float] = [-10000.0] * vocab_size
        toktypes: list[int] = [SentencePieceTokenTypes.UNUSED] * vocab_size

        for token_id in range(tokenizer.vocab_size()):
            if token_id >= vocab_size:
                logger.warning(
                    f"ignore tokens from {token_id}: id is out of range, max={vocab_size - 1}"
                )
                break

            piece = tokenizer.IdToPiece(token_id)
            text = piece.encode("utf-8")
            score = tokenizer.GetScore(token_id)

            toktype = SentencePieceTokenTypes.NORMAL
            if tokenizer.IsUnknown(token_id):
                toktype = SentencePieceTokenTypes.UNKNOWN
            elif tokenizer.IsControl(token_id):
                toktype = SentencePieceTokenTypes.CONTROL
            elif tokenizer.IsUnused(token_id):
                toktype = SentencePieceTokenTypes.UNUSED
            elif tokenizer.IsByte(token_id):
                toktype = SentencePieceTokenTypes.BYTE

            tokens[token_id] = text
            scores[token_id] = score
            toktypes[token_id] = toktype

        added_tokens_file = self.dir_model / "added_tokens.json"
        if added_tokens_file.is_file():
            with open(added_tokens_file, "r", encoding="utf-8") as f:
                added_tokens_json = json.load(f)
                for key in added_tokens_json:
                    token_id = added_tokens_json[key]
                    if token_id >= vocab_size:
                        logger.warning(
                            f"ignore token {token_id}: id is out of range, max={vocab_size - 1}"
                        )
                        continue

                    tokens[token_id] = key.encode("utf-8")
                    scores[token_id] = -1000.0
                    toktypes[token_id] = SentencePieceTokenTypes.USER_DEFINED

        tokenizer_config_file = self.dir_model / "tokenizer_config.json"
        if tokenizer_config_file.is_file():
            with open(tokenizer_config_file, "r", encoding="utf-8") as f:
                tokenizer_config_json = json.load(f)
                added_tokens_decoder = tokenizer_config_json.get(
                    "added_tokens_decoder", {}
                )
                for token_id, token_data in added_tokens_decoder.items():
                    token_id = int(token_id)
                    token: str = token_data["content"]
                    if token_id >= vocab_size:
                        logger.warning(
                            f"ignore token {token_id}: id is out of range, max={vocab_size - 1}"
                        )
                        continue
                    if toktypes[token_id] != SentencePieceTokenTypes.UNUSED:
                        if tokens[token_id] != token.encode("utf-8"):
                            logger.warning(
                                f'replacing token {token_id}: {tokens[token_id].decode("utf-8")!r} -> {token!r}'
                            )
                    if token_data.get("special") or self.does_token_look_special(token):
                        toktypes[token_id] = SentencePieceTokenTypes.CONTROL
                    else:
                        token = token.replace(
                            b"\xe2\x96\x81".decode("utf-8"), " "
                        )  # pre-normalize user-defined spaces
                        toktypes[token_id] = SentencePieceTokenTypes.USER_DEFINED

                    scores[token_id] = -1000.0
                    tokens[token_id] = token.encode("utf-8")

        if vocab_size > len(tokens):
            pad_count = vocab_size - len(tokens)
            logger.debug(
                f"Padding vocab with {pad_count} token(s) - [PAD1] through [PAD{pad_count}]"
            )
            for i in range(1, pad_count + 1):
                tokens.append(bytes(f"[PAD{i}]", encoding="utf-8"))
                scores.append(-1000.0)
                toktypes.append(SentencePieceTokenTypes.UNUSED)

        return tokens, scores, toktypes

    def _set_vocab_llama_hf(self):
        vocab = gguf.LlamaHfVocab(self.dir_model)
        tokens = []
        scores = []
        toktypes = []

        for text, score, toktype in vocab.all_tokens():
            tokens.append(text)
            scores.append(score)
            toktypes.append(toktype)

        assert len(tokens) == vocab.vocab_size

        self.gguf_writer.add_tokenizer_model("llama")
        self.gguf_writer.add_tokenizer_pre("default")
        self.gguf_writer.add_token_list(tokens)
        self.gguf_writer.add_token_scores(scores)
        self.gguf_writer.add_token_types(toktypes)

        special_vocab = gguf.SpecialVocab(self.dir_model, n_vocab=len(tokens))
        special_vocab.add_to_gguf(self.gguf_writer)

    def _set_vocab_rwkv_world(self):
        assert (self.dir_model / "rwkv_vocab_v20230424.txt").is_file()
        vocab_size = self.hparams.get("vocab_size", 65536)

        tokens: list[bytes] = ["<s>".encode("utf-8")]
        toktypes: list[int] = [gguf.TokenType.CONTROL]

        with open(
            self.dir_model / "rwkv_vocab_v20230424.txt", "r", encoding="utf-8"
        ) as f:
            lines = f.readlines()
            for line in lines:
                parts = line.split(" ")
                assert len(parts) >= 3
                token, token_len = ast.literal_eval(" ".join(parts[1:-1])), int(
                    parts[-1]
                )
                token = token.encode("utf-8") if isinstance(token, str) else token
                assert isinstance(token, bytes)
                assert len(token) == token_len
                token_text: str = repr(token)[2:-1]  # "b'\xff'" -> "\xff"
                tokens.append(token_text.encode("utf-8"))
                toktypes.append(gguf.TokenType.NORMAL)
        remainder = vocab_size - len(tokens)
        assert remainder >= 0
        for i in range(len(tokens), vocab_size):
            tokens.append(f"[PAD{i}]".encode("utf-8"))
            toktypes.append(gguf.TokenType.UNUSED)

        self.gguf_writer.add_tokenizer_model("rwkv")
        self.gguf_writer.add_token_list(tokens)
        self.gguf_writer.add_token_types(toktypes)
        special_vocab = gguf.SpecialVocab(self.dir_model, load_merges=False)
        special_vocab.chat_template = "rwkv-world"
        # hack: Add '\n\n' as the EOT token to make it chat normally
        special_vocab._set_special_token("eot", 261)
        # hack: Override these as they have already been set (incorrectly)
        special_vocab.special_token_ids["bos"] = 0
        special_vocab.special_token_ids["eos"] = 0

        special_vocab.add_to_gguf(self.gguf_writer)

    def _set_vocab_builtin(
        self, model_name: Literal["gpt-neox", "llama-spm"], vocab_size: int
    ):
        tokenizer_path = Path(sys.path[0]) / "models" / f"ggml-vocab-{model_name}.gguf"
        logger.warning(
            f"Using tokenizer from '{os.path.relpath(tokenizer_path, os.getcwd())}'"
        )
        vocab_reader = gguf.GGUFReader(tokenizer_path, "r")

        default_pre = "mpt" if model_name == "gpt-neox" else "default"

        field = vocab_reader.get_field(gguf.Keys.Tokenizer.MODEL)
        assert field  # tokenizer model
        self.gguf_writer.add_tokenizer_model(bytes(field.parts[-1]).decode("utf-8"))

        field = vocab_reader.get_field(gguf.Keys.Tokenizer.PRE)
        self.gguf_writer.add_tokenizer_pre(
            bytes(field.parts[-1]).decode("utf-8") if field else default_pre
        )

        field = vocab_reader.get_field(gguf.Keys.Tokenizer.LIST)
        assert field  # token list
        self.gguf_writer.add_token_list(
            [bytes(field.parts[i]) for i in field.data][:vocab_size]
        )

        if model_name == "llama-spm":
            field = vocab_reader.get_field(gguf.Keys.Tokenizer.SCORES)
            assert field  # token scores
            self.gguf_writer.add_token_scores(
                [field.parts[i].tolist()[0] for i in field.data][:vocab_size]
            )

        field = vocab_reader.get_field(gguf.Keys.Tokenizer.TOKEN_TYPE)
        assert field  # token types
        self.gguf_writer.add_token_types(
            [field.parts[i].tolist()[0] for i in field.data][:vocab_size]
        )

        if model_name != "llama-spm":
            field = vocab_reader.get_field(gguf.Keys.Tokenizer.MERGES)
            assert field  # token merges
            self.gguf_writer.add_token_merges(
                [bytes(field.parts[i]) for i in field.data]
            )

        if (field := vocab_reader.get_field(gguf.Keys.Tokenizer.BOS_ID)) is not None:
            self.gguf_writer.add_bos_token_id(field.parts[-1].tolist()[0])
        if (field := vocab_reader.get_field(gguf.Keys.Tokenizer.EOS_ID)) is not None:
            self.gguf_writer.add_eos_token_id(field.parts[-1].tolist()[0])
        if (field := vocab_reader.get_field(gguf.Keys.Tokenizer.UNK_ID)) is not None:
            self.gguf_writer.add_unk_token_id(field.parts[-1].tolist()[0])
        if (field := vocab_reader.get_field(gguf.Keys.Tokenizer.PAD_ID)) is not None:
            self.gguf_writer.add_pad_token_id(field.parts[-1].tolist()[0])
        if (field := vocab_reader.get_field(gguf.Keys.Tokenizer.ADD_BOS)) is not None:
            self.gguf_writer.add_add_bos_token(field.parts[-1].tolist()[0])
        if (field := vocab_reader.get_field(gguf.Keys.Tokenizer.ADD_EOS)) is not None:
            self.gguf_writer.add_add_eos_token(field.parts[-1].tolist()[0])

    def _try_set_pooling_type(self) -> None:
        # get pooling path
        pooling_path = None
        module_path = self.dir_model / "modules.json"
        if module_path.is_file():
            with open(module_path, encoding="utf-8") as f:
                modules = json.load(f)
            for mod in modules:
                if mod["type"] == "sentence_transformers.models.Pooling":
                    pooling_path = mod["path"]
                    break

        # get pooling type
        if pooling_path is not None:
            with open(
                self.dir_model / pooling_path / "config.json", encoding="utf-8"
            ) as f:
                pooling = json.load(f)
            if pooling["pooling_mode_mean_tokens"]:
                pooling_type = gguf.PoolingType.MEAN
            elif pooling["pooling_mode_cls_token"]:
                pooling_type = gguf.PoolingType.CLS
            elif pooling["pooling_mode_lasttoken"]:
                pooling_type = gguf.PoolingType.LAST
            else:
                raise NotImplementedError(
                    "Only MEAN, CLS, and LAST pooling types supported"
                )
            self.gguf_writer.add_pooling_type(pooling_type)


class MmprojModel(ModelBase):
    model_type = ModelType.MMPROJ
    model_arch = gguf.MODEL_ARCH.MMPROJ
    preprocessor_config: dict[str, Any]
    global_config: dict[str, Any]

    n_block_keys = ["n_layers", "num_hidden_layers", "n_layer", "num_layers", "depth"]

    has_vision_encoder: bool = True  # by default
    has_audio_encoder: bool = False

    # for models having multiple encoders, we need to separate their hparams
    hparams_vision: dict[str, Any] | None = None
    hparams_audio: dict[str, Any] | None = None

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        if self.model_arch != gguf.MODEL_ARCH.MMPROJ:
            raise TypeError(
                "MmprojModel must be subclassed with model_arch = gguf.MODEL_ARCH.MMPROJ"
            )

        # get n_embd of the text model
        if "text_config" not in self.hparams:
            self.hparams["text_config"] = {}
        if "audio_config" not in self.hparams:
            self.hparams["audio_config"] = {}
        text_config = {**self.hparams, **self.hparams["text_config"]}
        self.n_embd_text = text_config.get("hidden_size", text_config.get("n_embd", 0))
        assert self.n_embd_text > 0, "n_embd not found in hparams"

        # move vision config to the top level, while preserving the original hparams in global_config
        import copy

        self.global_config = copy.deepcopy(self.hparams)
        self.hparams_vision = self.get_vision_config()
        self.hparams_audio = self.get_audio_config()

        if self.hparams_vision is None and self.hparams_audio is None:
            raise ValueError("vision_config / audio_config not found in hparams")

        # for compat with vision-only models
        self.hparams = self.hparams_vision or self.hparams_audio or self.hparams

        # TODO @ngxson : this is a hack to support both vision and audio encoders
        have_multiple_encoders = self.has_audio_encoder and self.has_vision_encoder
        self.block_count = (
            128 if have_multiple_encoders else self.find_hparam(self.n_block_keys, True)
        )
        self.tensor_map = gguf.get_tensor_name_map(
            gguf.MODEL_ARCH.MMPROJ, self.block_count
        )

        # load preprocessor config
        with open(
            self.dir_model / "preprocessor_config.json", "r", encoding="utf-8"
        ) as f:
            self.preprocessor_config = json.load(f)

    def get_vision_config(self) -> dict[str, Any] | None:
        return self.global_config.get("vision_config")

    def get_audio_config(self) -> dict[str, Any] | None:
        return self.global_config.get("audio_config")

    def set_type(self):
        self.gguf_writer.add_type(gguf.GGUFType.MMPROJ)

    def set_gguf_parameters(self):
        self.gguf_writer.add_file_type(self.ftype)

        if self.has_vision_encoder:
            self.gguf_writer.add_clip_has_vision_encoder(True)
            self.gguf_writer.add_vision_projection_dim(self.n_embd_text)

            # vision config
            self.gguf_writer.add_vision_image_size(self.find_vparam(["image_size"]))
            self.gguf_writer.add_vision_patch_size(self.find_vparam(["patch_size"]))
            self.gguf_writer.add_vision_embedding_length(
                self.find_vparam(["hidden_size"])
            )
            self.gguf_writer.add_vision_feed_forward_length(
                self.find_vparam(["intermediate_size"])
            )
            self.gguf_writer.add_vision_block_count(self.find_vparam(self.n_block_keys))
            self.gguf_writer.add_vision_head_count(
                self.find_vparam(["num_attention_heads"])
            )

            # preprocessor config
            self.gguf_writer.add_vision_image_mean(
                self.preprocessor_config["image_mean"]
            )
            self.gguf_writer.add_vision_image_std(self.preprocessor_config["image_std"])

        if self.has_audio_encoder:
            self.gguf_writer.add_clip_has_audio_encoder(True)
            self.gguf_writer.add_audio_projection_dim(self.n_embd_text)

            # audio config
            self.gguf_writer.add_audio_embedding_length(
                self.find_aparam(["hidden_size"])
            )
            self.gguf_writer.add_audio_feed_forward_length(
                self.find_aparam(["intermediate_size"])
            )
            self.gguf_writer.add_audio_block_count(self.find_aparam(self.n_block_keys))
            self.gguf_writer.add_audio_head_count(
                self.find_aparam(["num_attention_heads"])
            )

        if not self.has_vision_encoder and not self.has_audio_encoder:
            raise ValueError("MmprojModel must have either vision or audio encoder")

    def write_vocab(self):
        raise ValueError("MmprojModel does not support vocab writing")

    def find_vparam(self, keys: Iterable[str], optional: bool = False) -> Any:
        assert self.hparams_vision is not None
        return self._find_param(self.hparams_vision, keys, optional)

    def find_aparam(self, keys: Iterable[str], optional: bool = False) -> Any:
        assert self.hparams_audio is not None
        return self._find_param(self.hparams_audio, keys, optional)

    def _find_param(
        self, obj: dict[str, Any], keys: Iterable[str], optional: bool = False
    ) -> Any:
        key = next((k for k in keys if k in obj), None)
        if key is not None:
            return obj[key]
        if optional:
            return None
        raise KeyError(f"could not find any of: {keys}")


@ModelBase.register("QWenLMHeadModel")
class QwenModel(TextModel):
    model_arch = gguf.MODEL_ARCH.QWEN

    @staticmethod
    def token_bytes_to_string(b):
        from transformers.models.gpt2.tokenization_gpt2 import bytes_to_unicode

        byte_encoder = bytes_to_unicode()
        return "".join([byte_encoder[ord(char)] for char in b.decode("latin-1")])

    @staticmethod
    def bpe(
        mergeable_ranks: dict[bytes, int], token: bytes, max_rank: int | None = None
    ) -> list[bytes]:
        parts = [bytes([b]) for b in token]
        while True:
            min_idx = None
            min_rank = None
            for i, pair in enumerate(zip(parts[:-1], parts[1:])):
                rank = mergeable_ranks.get(pair[0] + pair[1])
                if rank is not None and (min_rank is None or rank < min_rank):
                    min_idx = i
                    min_rank = rank
            if min_rank is None or (max_rank is not None and min_rank >= max_rank):
                break
            assert min_idx is not None
            parts = (
                parts[:min_idx]
                + [parts[min_idx] + parts[min_idx + 1]]
                + parts[min_idx + 2 :]
            )
        return parts

    def set_vocab(self):
        self._set_vocab_qwen()

    def set_gguf_parameters(self):
        self.gguf_writer.add_context_length(self.hparams["max_position_embeddings"])
        self.gguf_writer.add_block_count(self.hparams["num_hidden_layers"])
        self.gguf_writer.add_embedding_length(self.hparams["hidden_size"])
        self.gguf_writer.add_feed_forward_length(self.hparams["intermediate_size"])
        self.gguf_writer.add_rope_freq_base(self.hparams["rotary_emb_base"])
        self.gguf_writer.add_rope_dimension_count(
            self.hparams["hidden_size"] // self.hparams["num_attention_heads"]
        )
        self.gguf_writer.add_head_count(self.hparams["num_attention_heads"])
        self.gguf_writer.add_layer_norm_rms_eps(self.hparams["layer_norm_epsilon"])
        self.gguf_writer.add_file_type(self.ftype)


@ModelBase.register(
    "Qwen2Model", "Qwen2ForCausalLM", "Qwen2AudioForConditionalGeneration"
)
class Qwen2Model(TextModel):
    model_arch = gguf.MODEL_ARCH.QWEN2

    def set_vocab(self):
        try:
            self._set_vocab_sentencepiece()
        except FileNotFoundError:
            self._set_vocab_gpt2()

    def set_gguf_parameters(self):
        super().set_gguf_parameters()
        if (
            self.hparams.get("rope_scaling") is not None
            and "factor" in self.hparams["rope_scaling"]
        ):
            if self.hparams["rope_scaling"].get("type") == "yarn":
                self.gguf_writer.add_rope_scaling_type(gguf.RopeScalingType.YARN)
                self.gguf_writer.add_rope_scaling_factor(
                    self.hparams["rope_scaling"]["factor"]
                )
                self.gguf_writer.add_rope_scaling_orig_ctx_len(
                    self.hparams["rope_scaling"]["original_max_position_embeddings"]
                )
        self.gguf_writer.add_bool("is_hmm", True)
        if self.model_description:
            self.gguf_writer.add_string("model.description", self.model_description)

    def conver_embedding(self, embedding_path: Path):
        # 加载嵌入权重
        torch.serialization.add_safe_globals([torch.nn.modules.sparse.Embedding])
        embedding_weight = torch.load(
            embedding_path, map_location="cpu", weights_only=True
        )
        if isinstance(embedding_weight, dict) and "weight" in embedding_weight:
            embedding_weight = embedding_weight["weight"]
        if isinstance(embedding_weight, torch.nn.Embedding):
            embedding_weight = embedding_weight.weight
        print(f"embedding_weight.dtype: {embedding_weight.dtype}")
        if embedding_weight.dtype == torch.int16:
            print(
                "keep weight to int16 for minimal memory usage. convert to float32 when using in C++"
            )
            embedding_weight = embedding_weight.to(torch.int16)
        elif embedding_weight.dtype == torch.float16:
            print("keep embedding weight as float16")
        else:
            print(f"convert embedding weight to float32 from {embedding_weight.dtype}")
            embedding_weight = embedding_weight.to(torch.float32)
        # 创建空模块并手动注册参数
        m = nn.Module()
        # par = nn.Parameter(embedding_weight)  # 注册为可训练参数
        m.register_buffer("0", embedding_weight)  # 参数名称为"0"
        scripted_model = torch.jit.script(m)
        out_path = self.dir_model / "cpp_quant_embedding.pt"
        scripted_model.save(out_path)
        if isinstance(scripted_model, torch.jit.ScriptModule):
            print("****scripted_model is torch.jit.ScriptModule")
        return out_path

    def get_tensors(self) -> Iterator[tuple[str, Tensor]]:
        # TODO: @guoxing.xu check & test this block
        logger.info(".....Getting tensors")
        for name in [
            "quant_embedding.pt",
            "prefill.hmm",
            "decoder.hmm",
            "prefill_0.hmm",
            "prefill_1.hmm",
            "decoder_0.hmm",
            "decoder_1.hmm",
            f"{model_name_from_arg}_prefill.hmm",
            f"{model_name_from_arg}_decode.hmm",
        ]:
            file_path = self.dir_model / name
            logger.info(f".....Loading tensor from {file_path}")
            if name == "quant_embedding.pt":
                file_path = self.dir_model /"hmquant"/ name
                file_path = self.conver_embedding(file_path)
            if not file_path.is_file():
                logger.warning(f".....{file_path} not found")
                continue
            with open(file_path, "rb") as file:
                binary_data = np.frombuffer(file.read(), dtype=np.uint8)
                logger.info(f".....Tensor size: {binary_data.shape}")
                data = torch.from_numpy(binary_data.copy())
                if name == "quant_embedding.pt":
                    os.remove(file_path)
                #增加名字转换
                if model_name_from_arg in name:
                    logger.info(f"##hm find substring:{model_name_from_arg} in name: {name}")
                    if "prefill" in name:
                        name = "prefill.hmm"
                    elif "decode" in name:
                        name = "decoder.hmm"
                    else:
                        logger.info(f"##error not supported model name found:{name}")

                logger.info(f"#final name is:{name}")
                yield name, data

    def prepare_tensors(self):
        # TODO: @guoxing.xu check & test this block
        # 自定义的tensor，不需要该函数
        # super().prepare_tensors()
        logger.info(".....Preparing tensors")
        for name, data_torch in chain(
            self.generate_extra_tensors(), self.get_tensors()
        ):
            logger.info(
                f".....Adding tensor {name}, {data_torch.shape}, {data_torch.nbytes}"
            )
            self.gguf_writer.add_tensor(
                name, data_torch.numpy(), raw_dtype=gguf.GGMLQuantizationType.I8
            )


@ModelBase.register(
    "Qwen2VLModel",
    "Qwen2VLForConditionalGeneration",
    "Qwen2_5_VLForConditionalGeneration",
    "Qwen2_5OmniModel",
)
class Qwen2VLModel(TextModel):
    model_arch = gguf.MODEL_ARCH.QWEN2VL

    def set_gguf_parameters(self):
        super().set_gguf_parameters()
        mrope_section = self.hparams["rope_scaling"]["mrope_section"]
        mrope_section += [0] * max(0, 4 - len(mrope_section))
        self.gguf_writer.add_rope_dimension_sections(mrope_section)
        self.gguf_writer.add_bool("is_hmm", True)
        self.gguf_writer.add_uint32("hmm.vit.height", self.hmm_vit_height)
        self.gguf_writer.add_uint32("hmm.vit.width", self.hmm_vit_width)
        if self.model_description:
            self.gguf_writer.add_string("model.description", self.model_description)

    def set_vocab(self):
        try:
            self._set_vocab_sentencepiece()
        except FileNotFoundError:
            self._set_vocab_gpt2()

    def conver_embedding(self, embedding_path: Path):
        # 加载嵌入权重
        torch.serialization.add_safe_globals([torch.nn.modules.sparse.Embedding])
        embedding_weight = torch.load(
            embedding_path, map_location="cpu", weights_only=True
        )
        if isinstance(embedding_weight, dict) and "weight" in embedding_weight:
            embedding_weight = embedding_weight["weight"]
        if isinstance(embedding_weight, torch.nn.Embedding):
            embedding_weight = embedding_weight.weight

        print(f"embedding_weight.dtype: {embedding_weight.dtype}")
        if embedding_weight.dtype == torch.int16:
            print(
                "keep weight to int16 for minimal memory usage. convert to float32 when using in C++"
            )
            embedding_weight = embedding_weight.to(torch.int16)
        elif embedding_weight.dtype == torch.float16:
            print("keep embedding weight as float16")
        else:
            print(f"convert embedding weight to float32 from {embedding_weight.dtype}")
            embedding_weight = embedding_weight.to(torch.float32)
        # 创建空模块并手动注册参数
        m = nn.Module()
        m.register_buffer("0", embedding_weight)  # 参数名称为"0""
        scripted_model = torch.jit.script(m)
        out_path = self.dir_model / "cpp_quant_embedding.pt"
        scripted_model.save(out_path)
        if isinstance(scripted_model, torch.jit.ScriptModule):
            print("****scripted_model is torch.jit.ScriptModule")
        return out_path

    def get_tensors(self) -> Iterator[tuple[str, Tensor]]:
        # TODO: @guoxing.xu check & test this block
        logger.info(".....Getting tensors")
        for name in [
            "quant_embedding.pt",
            "prefill.hmm",
            "decoder.hmm",
            "prefill_0.hmm",
            "prefill_1.hmm",
            "decoder_0.hmm",
            "decoder_1.hmm",
            f"{model_name_from_arg}_prefill.hmm",
            f"{model_name_from_arg}_decode.hmm",
        ]:
            file_path = self.dir_model / name
            logger.info(f".....Loading tensor from {file_path}")
            if name == "quant_embedding.pt":
                file_path = self.dir_model /"hmquant"/name
                file_path = self.conver_embedding(file_path)
            if not file_path.is_file():
                logger.warning(f".....{file_path} not found")
                continue
            with open(file_path, "rb") as file:
                binary_data = np.frombuffer(file.read(), dtype=np.uint8)
                logger.info(f".....Tensor size: {binary_data.shape}")
                data = torch.from_numpy(binary_data.copy())
                if name == "quant_embedding.pt":
                    os.remove(file_path)
                yield name, data

    def prepare_tensors(self):
        # TODO: @guoxing.xu check & test this block
        # 自定义的tensor，不需要该函数
        # super().prepare_tensors()
        logger.info(".....Preparing tensors")
        for name, data_torch in chain(
            self.generate_extra_tensors(), self.get_tensors()
        ):
            logger.info(
                f".....Adding tensor {name}, {data_torch.shape}, {data_torch.nbytes}"
            )
            self.gguf_writer.add_tensor(
                name, data_torch.numpy(), raw_dtype=gguf.GGMLQuantizationType.I8
            )


@ModelBase.register(
    "Qwen2VLModel",
    "Qwen2VLForConditionalGeneration",
    "Qwen2_5_VLForConditionalGeneration",
)
class Qwen2VLVisionModel(MmprojModel):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        assert self.hparams_vision is not None
        self.hparams_vision["image_size"] = self.hparams_vision.get("image_size", 560)
        # rename config.json values
        self.hparams_vision["num_attention_heads"] = self.hparams_vision.get(
            "num_heads"
        )
        self.hparams_vision["num_hidden_layers"] = self.hparams_vision.get("depth")
        if "embed_dim" in self.hparams_vision:  # qwen2vl
            self.hparams_vision["intermediate_size"] = self.hparams_vision.get(
                "hidden_size"
            )
            self.hparams_vision["hidden_size"] = self.hparams_vision.get("embed_dim")

    def set_gguf_parameters(self):
        super().set_gguf_parameters()
        assert self.hparams_vision is not None
        hparams = self.hparams_vision
        model_type = self.global_config["model_type"]
        if model_type == "qwen2_vl":
            self.gguf_writer.add_clip_projector_type(gguf.VisionProjectorType.QWEN2VL)
        elif model_type == "qwen2_5_vl" or model_type == "qwen2_5_omni":
            if model_type == "qwen2_5_omni":
                self.gguf_writer.add_clip_projector_type(
                    gguf.VisionProjectorType.QWEN25O
                )
            else:
                self.gguf_writer.add_clip_projector_type(
                    gguf.VisionProjectorType.QWEN25VL
                )
            self.gguf_writer.add_vision_use_silu(True)
            # find n_wa_pattern (window attention pattern)
            fullatt_block_indexes = hparams.get("fullatt_block_indexes")
            assert (
                fullatt_block_indexes is not None
            ), "fullatt_block_indexes is required for qwen2_5_vl"
            n_wa_pattern = fullatt_block_indexes[0] + 1
            # validate n_wa_pattern
            for i in range(1, len(fullatt_block_indexes)):
                if (
                    fullatt_block_indexes[i] - fullatt_block_indexes[i - 1]
                    != n_wa_pattern
                ):
                    raise ValueError(
                        f"Invalid fullatt_block_indexes: {fullatt_block_indexes}"
                    )
            self.gguf_writer.add_vision_n_wa_pattern(n_wa_pattern)
        else:
            raise ValueError(
                f"Unknown QwenVL model type: {self.global_config['model_type']}"
            )
        # default values below are taken from HF tranformers code
        self.gguf_writer.add_vision_attention_layernorm_eps(
            self.global_config.get("rms_norm_eps", 1e-6)
        )
        self.gguf_writer.add_bool("is_hmm", True)
        if self.model_description:
            self.gguf_writer.add_string("model.description", self.model_description)

    def get_tensors(self) -> Iterator[tuple[str, Tensor]]:
        # TODO: @guoxing.xu check & test this block
        logger.info(".....Getting tensors")
        for name in [
            "visual.hmm",
        ]:
            file_path = self.dir_model / name
            logger.info(f".....Loading tensor from {file_path}")
            if not file_path.is_file():
                logger.warning(f".....{file_path} not found")
                continue
            with open(file_path, "rb") as file:
                binary_data = np.frombuffer(file.read(), dtype=np.uint8)
                logger.info(f".....Tensor size: {binary_data.shape}")
                data = torch.from_numpy(binary_data.copy())
                yield name, data

    def prepare_tensors(self):
        # TODO: @guoxing.xu check & test this block
        # 自定义的tensor，不需要该函数
        # super().prepare_tensors()
        logger.info(".....Preparing tensors")
        for name, data_torch in chain(
            self.generate_extra_tensors(), self.get_tensors()
        ):
            logger.info(
                f".....Adding tensor {name}, {data_torch.shape}, {data_torch.nbytes}"
            )
            self.gguf_writer.add_tensor(
                name, data_torch.numpy(), raw_dtype=gguf.GGMLQuantizationType.I8
            )


@ModelBase.register("Qwen2_5OmniModel")
class Qwen25OmniModel(Qwen2VLVisionModel):
    has_vision_encoder = True
    has_audio_encoder = True

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        assert self.hparams_audio is not None
        self.hparams_audio["hidden_size"] = self.hparams_audio["d_model"]
        self.hparams_audio["intermediate_size"] = self.hparams_audio["encoder_ffn_dim"]
        self.hparams_audio["num_attention_heads"] = self.hparams_audio[
            "encoder_attention_heads"
        ]

    def set_gguf_parameters(self):
        super().set_gguf_parameters()
        assert self.hparams_audio is not None
        self.gguf_writer.add_audio_num_mel_bins(self.hparams_audio["num_mel_bins"])
        self.gguf_writer.add_audio_attention_layernorm_eps(
            self.hparams_audio.get("layer_norm_eps", 1e-5)
        )

    def get_vision_config(self) -> dict[str, Any] | None:
        return self.global_config["thinker_config"].get("vision_config")

    def get_audio_config(self) -> dict[str, Any] | None:
        return self.global_config["thinker_config"].get("audio_config")

    def generate_extra_tensors(self) -> Iterable[tuple[str, Tensor]]:
        # SinusoidsPositionEmbedding
        assert self.hparams_audio is not None
        max_timescale = 10000
        length = 1500
        channels = self.hparams_audio["hidden_size"]
        log_timescale_increment = np.log(max_timescale) / (channels // 2 - 1)
        inv_timescales = torch.exp(
            -log_timescale_increment * torch.arange(channels // 2).float()
        )
        scaled_time = (
            torch.arange(length)[:, np.newaxis] * inv_timescales[np.newaxis, :]
        )
        pos_embd = torch.cat(
            [torch.sin(scaled_time), torch.cos(scaled_time)], dim=1
        ).to(dtype=torch.float32)
        yield ("audio_tower.embed_positions.weight", pos_embd)

    def tensor_force_quant(self, name, new_name, bid, n_dims):
        del bid, new_name, n_dims  # unused
        if ".conv" in name and ".weight" in name:
            return gguf.GGMLQuantizationType.F16
        return False

    def modify_tensors(
        self, data_torch: Tensor, name: str, bid: int | None
    ) -> Iterable[tuple[str, Tensor]]:
        if name.startswith("thinker."):
            name = name.replace("thinker.", "")

        if name.startswith("audio_tower"):
            # process audio tensors
            if "conv1.bias" in name or "conv2.bias" in name:
                # transpose conv1 and conv2 bias
                data_torch = data_torch.unsqueeze(-1)
            if "audio_bos_eos_token" in name:
                # this tensor is left unused in transformers code
                # https://github.com/huggingface/transformers/blob/6e3063422c4b1c014aa60c32b9254fd2902f0f28/src/transformers/models/qwen2_5_omni/modular_qwen2_5_omni.py#L1809
                return []
            return [(self.map_tensor_name(name), data_torch)]

        return super().modify_tensors(data_torch, name, bid)


@ModelBase.register("Qwen2MoeForCausalLM")
class Qwen2MoeModel(TextModel):
    model_arch = gguf.MODEL_ARCH.QWEN2MOE

    def set_gguf_parameters(self):
        super().set_gguf_parameters()
        if (n_experts := self.hparams.get("num_experts")) is not None:
            self.gguf_writer.add_expert_count(n_experts)
        if (
            moe_intermediate_size := self.hparams.get("moe_intermediate_size")
        ) is not None:
            self.gguf_writer.add_expert_feed_forward_length(moe_intermediate_size)
            logger.info(f"gguf: expert feed forward length = {moe_intermediate_size}")
        if (
            shared_expert_intermediate_size := self.hparams.get(
                "shared_expert_intermediate_size"
            )
        ) is not None:
            self.gguf_writer.add_expert_shared_feed_forward_length(
                shared_expert_intermediate_size
            )
            logger.info(
                f"gguf: expert shared feed forward length = {shared_expert_intermediate_size}"
            )
        # YaRN is not enabled by default
        # To enable it, please refer to this guide: https://huggingface.co/Qwen/Qwen3-30B-A3B#processing-long-texts
        rope_scaling = self.hparams.get("rope_scaling") or {}
        if (
            rope_scaling.get("rope_type", rope_scaling.get("type")) == "yarn"
            and "factor" in rope_scaling
        ):
            self.gguf_writer.add_rope_scaling_type(gguf.RopeScalingType.YARN)
            self.gguf_writer.add_rope_scaling_factor(rope_scaling["factor"])
            self.gguf_writer.add_rope_scaling_orig_ctx_len(
                rope_scaling["original_max_position_embeddings"]
            )

    _experts: list[dict[str, Tensor]] | None = None

    def modify_tensors(
        self, data_torch: Tensor, name: str, bid: int | None
    ) -> Iterable[tuple[str, Tensor]]:
        # process the experts separately
        if name.find("experts") != -1:
            n_experts = self.hparams["num_experts"]
            assert bid is not None

            if self._experts is None:
                self._experts = [{} for _ in range(self.block_count)]

            self._experts[bid][name] = data_torch

            if len(self._experts[bid]) >= n_experts * 3:
                tensors: list[tuple[str, Tensor]] = []

                # merge the experts into a single 3d tensor
                for w_name in ["down_proj", "gate_proj", "up_proj"]:
                    datas: list[Tensor] = []

                    for xid in range(n_experts):
                        ename = f"model.layers.{bid}.mlp.experts.{xid}.{w_name}.weight"
                        datas.append(self._experts[bid][ename])
                        del self._experts[bid][ename]

                    data_torch = torch.stack(datas, dim=0)

                    merged_name = f"model.layers.{bid}.mlp.experts.{w_name}.weight"

                    new_name = self.map_tensor_name(merged_name)

                    tensors.append((new_name, data_torch))
                return tensors
            else:
                return []

        return [(self.map_tensor_name(name), data_torch)]

    def prepare_tensors(self):
        super().prepare_tensors()

        if self._experts is not None:
            # flatten `list[dict[str, Tensor]]` into `list[str]`
            experts = [k for d in self._experts for k in d.keys()]
            if len(experts) > 0:
                raise ValueError(f"Unprocessed experts: {experts}")


@ModelBase.register("Qwen3ForCausalLM")
class Qwen3Model(Qwen2Model):
    model_arch = gguf.MODEL_ARCH.QWEN3


@ModelBase.register("Qwen3MoeForCausalLM")
class Qwen3MoeModel(Qwen2MoeModel):
    model_arch = gguf.MODEL_ARCH.QWEN3MOE


@ModelBase.register("GPT2LMHeadModel")
class GPT2Model(TextModel):
    model_arch = gguf.MODEL_ARCH.GPT2

    def set_gguf_parameters(self):
        self.gguf_writer.add_block_count(self.hparams["n_layer"])
        self.gguf_writer.add_context_length(self.hparams["n_ctx"])
        self.gguf_writer.add_embedding_length(self.hparams["n_embd"])
        self.gguf_writer.add_feed_forward_length(4 * self.hparams["n_embd"])
        self.gguf_writer.add_head_count(self.hparams["n_head"])
        self.gguf_writer.add_layer_norm_eps(self.hparams["layer_norm_epsilon"])
        self.gguf_writer.add_file_type(self.ftype)

    def modify_tensors(
        self, data_torch: Tensor, name: str, bid: int | None
    ) -> Iterable[tuple[str, Tensor]]:
        del bid  # unused

        tensors: list[tuple[str, Tensor]] = []

        # we don't need these
        if name.endswith((".attn.bias", ".attn.masked_bias")):
            return tensors

        if name.endswith(
            (".c_attn.weight", ".c_proj.weight", ".c_fc.weight", ".c_proj.weight")
        ):
            data_torch = data_torch.transpose(1, 0)

        new_name = self.map_tensor_name(name)

        tensors.append((new_name, data_torch))

        return tensors


@ModelBase.register("DeepseekForCausalLM")
class DeepseekModel(TextModel):
    model_arch = gguf.MODEL_ARCH.DEEPSEEK

    def set_vocab(self):
        try:
            self._set_vocab_sentencepiece()
        except FileNotFoundError:
            self._set_vocab_gpt2()

    def set_gguf_parameters(self):
        super().set_gguf_parameters()
        hparams = self.hparams
        if (rope_dim := hparams.get("head_dim")) is None:
            rope_dim = hparams["hidden_size"] // hparams["num_attention_heads"]

        self.gguf_writer.add_rope_dimension_count(rope_dim)
        self.gguf_writer.add_rope_scaling_type(gguf.RopeScalingType.NONE)
        self.gguf_writer.add_leading_dense_block_count(hparams["first_k_dense_replace"])
        self.gguf_writer.add_vocab_size(hparams["vocab_size"])
        self.gguf_writer.add_expert_feed_forward_length(
            hparams["moe_intermediate_size"]
        )
        self.gguf_writer.add_expert_weights_scale(1.0)
        self.gguf_writer.add_expert_count(hparams["n_routed_experts"])
        self.gguf_writer.add_expert_shared_count(hparams["n_shared_experts"])

    _experts: list[dict[str, Tensor]] | None = None

    @staticmethod
    def permute(weights: Tensor, n_head: int, n_head_kv: int | None):
        if n_head_kv is not None and n_head != n_head_kv:
            n_head = n_head_kv
        return (
            weights.reshape(
                n_head, 2, weights.shape[0] // n_head // 2, *weights.shape[1:]
            )
            .swapaxes(1, 2)
            .reshape(weights.shape)
        )

    def modify_tensors(
        self, data_torch: Tensor, name: str, bid: int | None
    ) -> Iterable[tuple[str, Tensor]]:
        n_head = self.hparams["num_attention_heads"]
        n_kv_head = self.hparams.get("num_key_value_heads")

        if name.endswith(("q_proj.weight", "q_proj.bias")):
            data_torch = DeepseekModel.permute(data_torch, n_head, n_head)
        if name.endswith(("k_proj.weight", "k_proj.bias")):
            data_torch = DeepseekModel.permute(data_torch, n_head, n_kv_head)

        # process the experts separately
        if name.find("mlp.experts") != -1:
            n_experts = self.hparams["n_routed_experts"]
            assert bid is not None

            if self._experts is None:
                self._experts = [{} for _ in range(self.block_count)]

            self._experts[bid][name] = data_torch

            if len(self._experts[bid]) >= n_experts * 3:
                tensors: list[tuple[str, Tensor]] = []

                # merge the experts into a single 3d tensor
                for w_name in ["down_proj", "gate_proj", "up_proj"]:
                    datas: list[Tensor] = []

                    for xid in range(n_experts):
                        ename = f"model.layers.{bid}.mlp.experts.{xid}.{w_name}.weight"
                        datas.append(self._experts[bid][ename])
                        del self._experts[bid][ename]

                    data_torch = torch.stack(datas, dim=0)

                    merged_name = f"model.layers.{bid}.mlp.experts.{w_name}.weight"

                    new_name = self.map_tensor_name(merged_name)

                    tensors.append((new_name, data_torch))
                return tensors
            else:
                return []

        return [(self.map_tensor_name(name), data_torch)]

    def prepare_tensors(self):
        super().prepare_tensors()

        if self._experts is not None:
            # flatten `list[dict[str, Tensor]]` into `list[str]`
            experts = [k for d in self._experts for k in d.keys()]
            if len(experts) > 0:
                raise ValueError(f"Unprocessed experts: {experts}")


@ModelBase.register("DeepseekV2ForCausalLM")
@ModelBase.register("DeepseekV3ForCausalLM")
class DeepseekV2Model(TextModel):
    model_arch = gguf.MODEL_ARCH.DEEPSEEK2

    def set_vocab(self):
        self._set_vocab_gpt2()

    def set_gguf_parameters(self):

        # note: deepseek2 using MLA converts into MQA (ie: GQA with 1 group)
        self.hparams["num_key_value_heads"] = 1

        super().set_gguf_parameters()
        hparams = self.hparams

        self.gguf_writer.add_leading_dense_block_count(hparams["first_k_dense_replace"])
        self.gguf_writer.add_vocab_size(hparams["vocab_size"])
        if "q_lora_rank" in hparams and hparams["q_lora_rank"] is not None:
            self.gguf_writer.add_q_lora_rank(hparams["q_lora_rank"])
        self.gguf_writer.add_kv_lora_rank(hparams["kv_lora_rank"])

        # note: deepseek2 using MLA converts into MQA with larger heads, then decompresses to MHA
        self.gguf_writer.add_key_length(
            hparams["kv_lora_rank"] + hparams["qk_rope_head_dim"]
        )
        self.gguf_writer.add_value_length(hparams["kv_lora_rank"])
        self.gguf_writer.add_key_length_mla(
            hparams["qk_nope_head_dim"] + hparams["qk_rope_head_dim"]
        )
        self.gguf_writer.add_value_length_mla(hparams["v_head_dim"])

        self.gguf_writer.add_expert_feed_forward_length(
            hparams["moe_intermediate_size"]
        )
        self.gguf_writer.add_expert_count(hparams["n_routed_experts"])
        self.gguf_writer.add_expert_shared_count(hparams["n_shared_experts"])
        self.gguf_writer.add_expert_weights_scale(hparams["routed_scaling_factor"])
        self.gguf_writer.add_expert_weights_norm(hparams["norm_topk_prob"])

        if hparams["scoring_func"] == "sigmoid":
            self.gguf_writer.add_expert_gating_func(gguf.ExpertGatingFuncType.SIGMOID)
        elif hparams["scoring_func"] == "softmax":
            self.gguf_writer.add_expert_gating_func(gguf.ExpertGatingFuncType.SOFTMAX)
        else:
            raise ValueError(
                f"Unsupported scoring_func value: {hparams['scoring_func']}"
            )

        self.gguf_writer.add_rope_dimension_count(hparams["qk_rope_head_dim"])

        rope_scaling = self.hparams.get("rope_scaling") or {}
        if (
            rope_scaling.get("rope_type", rope_scaling.get("type")) == "yarn"
            and "factor" in rope_scaling
        ):
            self.gguf_writer.add_rope_scaling_type(gguf.RopeScalingType.YARN)
            self.gguf_writer.add_rope_scaling_factor(rope_scaling["factor"])
            self.gguf_writer.add_rope_scaling_orig_ctx_len(
                rope_scaling["original_max_position_embeddings"]
            )
            self.gguf_writer.add_rope_scaling_yarn_log_mul(
                0.1 * rope_scaling["mscale_all_dim"]
            )

    _experts: list[dict[str, Tensor]] | None = None

    def modify_tensors(
        self, data_torch: Tensor, name: str, bid: int | None
    ) -> Iterable[tuple[str, Tensor]]:
        # rename e_score_correction_bias tensors
        if name.endswith("e_score_correction_bias"):
            name = name.replace("e_score_correction_bias", "e_score_correction.bias")

        # skip Multi-Token Prediction (MTP) layers
        block_count = self.hparams["num_hidden_layers"]
        match = re.match(r"model.layers.(\d+)", name)
        if match and int(match.group(1)) >= block_count:
            return []

        # process the experts separately
        if name.find("mlp.experts") != -1:
            n_experts = self.hparams["n_routed_experts"]
            assert bid is not None

            if self._experts is None:
                self._experts = [{} for _ in range(self.block_count)]

            self._experts[bid][name] = data_torch

            if len(self._experts[bid]) >= n_experts * 3:
                tensors: list[tuple[str, Tensor]] = []

                # merge the experts into a single 3d tensor
                for w_name in ["down_proj", "gate_proj", "up_proj"]:
                    datas: list[Tensor] = []

                    for xid in range(n_experts):
                        ename = f"model.layers.{bid}.mlp.experts.{xid}.{w_name}.weight"
                        datas.append(self._experts[bid][ename])
                        del self._experts[bid][ename]

                    data_torch = torch.stack(datas, dim=0)

                    merged_name = f"model.layers.{bid}.mlp.experts.{w_name}.weight"

                    new_name = self.map_tensor_name(merged_name)

                    tensors.append((new_name, data_torch))
                return tensors
            else:
                return []

        # note: MLA with the absorption optimization, needs these two split and k_b_proj transposed
        if name.endswith("kv_b_proj.weight"):
            name_kb = name.replace("kv_b_proj", "k_b_proj")
            name_vb = name.replace("kv_b_proj", "v_b_proj")

            n_head_kv = self.hparams["num_key_value_heads"]
            v_head_dim = self.hparams["v_head_dim"]
            qk_nope_head_dim = self.hparams["qk_nope_head_dim"]

            assert data_torch.shape[0] == n_head_kv * (v_head_dim + qk_nope_head_dim)

            kv_b = data_torch.view(
                n_head_kv, v_head_dim + qk_nope_head_dim, data_torch.shape[-1]
            )
            k_b, v_b = torch.split(kv_b, [qk_nope_head_dim, v_head_dim], dim=1)
            k_b = k_b.transpose(1, 2)

            return [
                (self.map_tensor_name(name_kb), k_b),
                (self.map_tensor_name(name_vb), v_b),
            ]

        return [(self.map_tensor_name(name), data_torch)]

    def prepare_tensors(self):
        super().prepare_tensors()

        if self._experts is not None:
            # flatten `list[dict[str, Tensor]]` into `list[str]`
            experts = [k for d in self._experts for k in d.keys()]
            if len(experts) > 0:
                raise ValueError(f"Unprocessed experts: {experts}")


@ModelBase.register("Qwen2AudioForConditionalGeneration")
class WhisperEncoderModel(MmprojModel):
    has_vision_encoder = False  # no vision encoder
    has_audio_encoder = True

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.hparams["hidden_size"] = self.hparams["d_model"]
        self.hparams["intermediate_size"] = self.hparams["encoder_ffn_dim"]
        self.hparams["num_attention_heads"] = self.hparams["encoder_attention_heads"]

    def set_gguf_parameters(self):
        super().set_gguf_parameters()
        self.gguf_writer.add_clip_projector_type(gguf.VisionProjectorType.QWEN2A)
        self.gguf_writer.add_audio_num_mel_bins(self.hparams["num_mel_bins"])
        self.gguf_writer.add_audio_attention_layernorm_eps(
            self.hparams.get("layer_norm_eps", 1e-5)
        )

    def tensor_force_quant(self, name, new_name, bid, n_dims):
        del bid, new_name, n_dims  # unused
        if ".conv" in name and ".weight" in name:
            return gguf.GGMLQuantizationType.F16
        return False

    def modify_tensors(
        self, data_torch: Tensor, name: str, bid: int | None
    ) -> Iterable[tuple[str, Tensor]]:
        del bid  # unused

        if name.startswith("language_model."):
            # skip language model tensors
            return []

        # prevent clash naming with vision tensors
        if name.startswith("multi_modal_projector"):
            name = "audio." + name

        if "conv1.bias" in name or "conv2.bias" in name:
            # transpose conv1 and conv2 bias
            data_torch = data_torch.unsqueeze(-1)

        return [(self.map_tensor_name(name), data_torch)]


###### CONVERSION LOGIC ######


# tree of lazy tensors
class LazyTorchTensor(gguf.LazyBase):
    _tensor_type = torch.Tensor
    # to keep the type-checker happy
    dtype: torch.dtype
    shape: torch.Size

    # only used when converting a torch.Tensor to a np.ndarray
    _dtype_map: dict[torch.dtype, type] = {
        torch.float16: np.float16,
        torch.float32: np.float32,
    }

    # used for safetensors slices
    # ref: https://github.com/huggingface/safetensors/blob/079781fd0dc455ba0fe851e2b4507c33d0c0d407/bindings/python/src/lib.rs#L1046
    # TODO: uncomment U64, U32, and U16, ref: https://github.com/pytorch/pytorch/issues/58734
    _dtype_str_map: dict[str, torch.dtype] = {
        "F64": torch.float64,
        "F32": torch.float32,
        "BF16": torch.bfloat16,
        "F16": torch.float16,
        # "U64": torch.uint64,
        "I64": torch.int64,
        # "U32": torch.uint32,
        "I32": torch.int32,
        # "U16": torch.uint16,
        "I16": torch.int16,
        "U8": torch.uint8,
        "I8": torch.int8,
        "BOOL": torch.bool,
        "F8_E4M3": torch.float8_e4m3fn,
        "F8_E5M2": torch.float8_e5m2,
    }

    def numpy(self) -> gguf.LazyNumpyTensor:
        dtype = self._dtype_map[self.dtype]
        return gguf.LazyNumpyTensor(
            meta=gguf.LazyNumpyTensor.meta_with_dtype_and_shape(dtype, self.shape),
            args=(self,),
            func=(lambda s: s.numpy()),
        )

    @classmethod
    def meta_with_dtype_and_shape(
        cls, dtype: torch.dtype, shape: tuple[int, ...]
    ) -> Tensor:
        return torch.empty(size=shape, dtype=dtype, device="meta")

    @classmethod
    def from_safetensors_slice(cls, st_slice: Any) -> Tensor:
        dtype = cls._dtype_str_map[st_slice.get_dtype()]
        shape: tuple[int, ...] = tuple(st_slice.get_shape())
        lazy = cls(
            meta=cls.meta_with_dtype_and_shape(dtype, shape),
            args=(st_slice,),
            func=lambda s: s[:],
        )
        return cast(torch.Tensor, lazy)

    @classmethod
    def from_remote_tensor(cls, remote_tensor: gguf.utility.RemoteTensor):
        dtype = cls._dtype_str_map[remote_tensor.dtype]
        shape = remote_tensor.shape
        meta = cls.meta_with_dtype_and_shape(dtype, shape)
        lazy = cls(
            meta=meta,
            args=(remote_tensor,),
            func=lambda r: torch.frombuffer(r.data(), dtype=dtype).reshape(shape),
        )
        return cast(torch.Tensor, lazy)

    @classmethod
    def __torch_function__(cls, func, types, args=(), kwargs=None):
        del types  # unused

        if kwargs is None:
            kwargs = {}

        if func is torch.Tensor.numpy:
            return args[0].numpy()

        return cls._wrap_fn(func)(*args, **kwargs)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert a huggingface model to a GGML compatible file"
    )
    ####new add start
    parser.add_argument(
        "task_id",
        nargs='?',
        default=None,
        help="Task ID (optional positional argument)"
    )

    parser.add_argument(
        "-name",
        "--model_name",
        required=True,
        type=str,
        help="(required) model name, example: qwen3",
    )
    parser.add_argument(
        "-size",
        "--model_size",
        required=True,
        type=str,
        help="(required) model size, example: 8b, 14b",
    )
    parser.add_argument(
        "-v",
        "--version",
        required=True,
        type=str,
        help="Houmo Dadao software version, example: 0.3.0, 2.4.2",
    )
    parser.add_argument(
        "-t",
        "--target",
        type=str,
        default="xh2",
        help="Houmo backend, support: xh1, xh2.",
    )
    parser.add_argument(
        "-qm",
        "--quant_model_path",
        type=str,
        default="",
        help="Quantized model path, only support Jfrog url.",
    )
    parser.add_argument(
        "-pl",
        "--prefill_length",
        type=int,
        default=256,
        help="Prefill length, recommend to use the default value 256.",
    )
    parser.add_argument(
        "-cl",
        "--context_length",
        type=int,
        default=2048,
        help="Context length, default is 2048(2k).",
    )
    parser.add_argument(
        "-b",
        "--batch",
        type=int,
        default=1,
        help="batch number, default is 1.",
    )
    parser.add_argument(
        "-dn",
        "--device_num",
        type=int,
        default=1,
        help="The number of device, default is 1.",
    )
    parser.add_argument(
        "-cn",
        "--core_num",
        type=int,
        default=1,
        help="The number of core, default is 1. The maximum value is 4.",
    )
    parser.add_argument(
        "-up",
        "--upload",
        action="store_true",
        help="Upload quantized/compiled model to JFrog (default is False).",
    )
    parser.add_argument(
        "-r",
        "--result_dir",
        type=str,
        default="./",
        help="The path for storing the results.",
    )
    # parser.add_argument(
    #     "--hm_compiled_modelpath",
    #     type=str,
    #     required=True,
    #     help="The path  storing the compiled hmm files",
    # )

    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="The path  storing the compiled hmm files",
    )

    ####new end
    parser.add_argument(
        "--vocab-only",
        action="store_true",
        help="extract only the vocab",
    )
    parser.add_argument(
        "--outfile",
        type=Path,
        help="path to write to; default: based on input. {ftype} will be replaced by the outtype.",
    )
    parser.add_argument(
        "--outtype",
        type=str,
        choices=["f32", "f16", "bf16", "q8_0", "tq1_0", "tq2_0", "auto"],
        default="f16",
        help="output format - use f32 for float32, f16 for float16, bf16 for bfloat16, q8_0 for Q8_0, tq1_0 or tq2_0 for ternary, and auto for the highest-fidelity 16-bit float type depending on the first loaded tensor type",
    )
    parser.add_argument(
        "--bigendian",
        action="store_true",
        help="model is executed on big endian machine",
    )

    parser.add_argument(
        "--use-temp-file",
        action="store_true",
        help="use the tempfile library while processing (helpful when running out of memory, process killed)",
    )
    parser.add_argument(
        "--no-lazy",
        action="store_true",
        help="use more RAM by computing all outputs before writing (use in case lazy evaluation is broken)",
    )
    parser.add_argument(
        "--model-name",
        type=str,
        default=None,
        help="name of the model",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="increase output verbosity",
    )
    parser.add_argument(
        "--split-max-tensors",
        type=int,
        default=0,
        help="max tensors in each split",
    )
    parser.add_argument(
        "--split-max-size",
        type=str,
        default="0",
        help="max size per split N(M|G)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="only print out a split plan and exit, without writing any new files",
    )
    parser.add_argument(
        "--no-tensor-first-split",
        action="store_true",
        help="do not add tensors to the first split (disabled by default)",
    )
    parser.add_argument(
        "--metadata",
        type=Path,
        help="Specify the path for an authorship metadata override file",
    )
    parser.add_argument(
        "--hmm-info",
        type=str,
        default="dadao",
        help="Specify Description of the HMM model to be used for conversion",
    )
    parser.add_argument(
        "--hmm-vit-width",
        type=int,
        default=644,
        help="Specify the visual width of the HMM model to be used for qwenvl",
    )
    parser.add_argument(
        "--hmm-vit-height",
        type=int,
        default=364,
        help="Specify the visual height of the HMM model to be used for qwenvl",
    )
    parser.add_argument(
        "--print-supported-models",
        action="store_true",
        help="Print the supported models",
    )
    parser.add_argument(
        "--remote",
        action="store_true",
        help="(Experimental) Read safetensors file remotely without downloading to disk. Config and tokenizer files will still be downloaded. To use this feature, you need to specify Hugging Face model repo name instead of a local directory. For example: 'HuggingFaceTB/SmolLM2-1.7B-Instruct'. Note: To access gated repo, set HF_TOKEN environment variable to your Hugging Face token.",
    )
    parser.add_argument(
        "--mmproj",
        action="store_true",
        help="(Experimental) Export multimodal projector (mmproj) for vision models. This will only work on some vision models. A prefix 'mmproj-' will be added to the output file name.",
    )




    args = parser.parse_args()
    if not args.print_supported_models and args.model is None:
        parser.error("the following arguments are required: model")
    global model_name_from_arg
    global size_label_from_arg
    global contxt_len_from_arg
    global core_num_from_arg
    global device_num_from_arg
    global version_from_arg
    global target_from_arg
    global prefill_len_from_arg
    global unzip_file_dir
    model_name_from_arg=args.model_name
    size_label_from_arg=args.model_size
    contxt_len_from_arg=int(args.context_length/1024)
    core_num_from_arg=args.core_num
    device_num_from_arg=args.device_num
    version_from_arg=args.version
    target_from_arg=args.target
    prefill_len_from_arg=args.prefill_length

    if extract_url(args.model):
        logger.info(f"#hm convert hmm to gguf from url:{args.model}")
        unzip_file_dir=get_current_file_dir()+f'/./{args.task_id}'
        download_and_unzip(args.model, unzip_file_dir)
        args.model=unzip_file_dir+"/compile_results"
        if os.path.isdir(args.model):
            logger.info(f"##hm {args.model} exits")
        else:
            logger.info(f"##hm {args.model} not exits delete compile_results")
            args.model=unzip_file_dir
        logger.info(f"##hm change args.model to:{args.model}")
    else:
        logger.info(f"#hm convert hmm to gguf from local dir:{args.model}")
    return args


def split_str_to_n_bytes(split_str: str) -> int:
    if split_str.endswith("K"):
        n = int(split_str[:-1]) * 1000
    elif split_str.endswith("M"):
        n = int(split_str[:-1]) * 1000 * 1000
    elif split_str.endswith("G"):
        n = int(split_str[:-1]) * 1000 * 1000 * 1000
    elif split_str.isnumeric():
        n = int(split_str)
    else:
        raise ValueError(
            f"Invalid split size: {split_str}, must be a number, optionally followed by K, M, or G"
        )

    if n < 0:
        raise ValueError(f"Invalid split size: {split_str}, must be positive")

    return n


def get_model_architecture(hparams: dict[str, Any], model_type: ModelType) -> str:
    # TODO @ngxson : this won't work correctly if the model has both audio & vision encoders
    # maybe we should fallback to text model's arch in that case, since not many models have both
    text_config = hparams.get("text_config", {})
    vision_config = hparams.get("vision_config", {})
    arch = None
    if (arches := hparams.get("architectures")) is not None and len(arches) > 0:
        arch = arches[0]
    elif "ssm_cfg" in hparams:
        # For non-hf Mamba and Mamba2 models
        arch = hparams["ssm_cfg"].get("layer", "Mamba") + "ForCausalLM"

    # if "architectures" is found in the sub-config, use that instead
    if model_type == ModelType.TEXT and text_config.get("architectures") is not None:
        arch = text_config["architectures"][0]
    elif (
        model_type == ModelType.MMPROJ
        and vision_config.get("architectures") is not None
    ):
        arch = vision_config["architectures"][0]
    if arch is None:
        raise ValueError("Failed to detect model architecture")
    return arch


def main() -> None:
    logging.basicConfig(level=logging.INFO)
    args = parse_args()

    if args.print_supported_models:
        logger.error("Supported models:")
        ModelBase.print_registered_models()
        sys.exit(0)

    if args.verbose:
        logging.basicConfig(level=logging.DEBUG)
    else:
        logging.basicConfig(level=logging.INFO)

    if args.remote:
        hf_repo_id = args.model
        from huggingface_hub import snapshot_download

        local_dir = snapshot_download(
            repo_id=hf_repo_id,
            allow_patterns=["LICENSE", "*.json", "*.md", "*.txt", "tokenizer.model"],
        )
        dir_model = Path(local_dir)
        logger.info(f"Downloaded config and tokenizer to {local_dir}")
    else:
        hf_repo_id = None
        dir_model = Path(args.model)
    current_dir_path = get_current_file_dir()
    # execute_cmd = [
    # 'cp',  # 命令名单独作为第一个元素
    # '-rf',  # 选项作为第二个元素
    # f'{current_dir_path}/model_configs/{args.model_name}_{args.model_size}/*',  # 源路径
    # f'{args.model}/'  # 目标路径
    # ]
    src_pattern = f'{current_dir_path}/model_configs/{args.model_name}_{args.model_size}/*'
    dest_path = f'{args.model}/'

    execute_cmd = f"cp -rf {src_pattern} {dest_path}"
    #shell=True 增加shell对通配符号的处理
    ls_result = execute_linux_command(execute_cmd,shell=True)
    if ls_result['success']:
        logger.info(f"#hm run {execute_cmd} succeed")
    else:
        logger.info(f"#hm run {execute_cmd} failed reason：{ls_result['error']}")

    if not dir_model.is_dir():
        logger.error(f"Error: {dir_model} is not a directory")
        sys.exit(1)

    ftype_map: dict[str, gguf.LlamaFileType] = {
        "f32": gguf.LlamaFileType.ALL_F32,
        "f16": gguf.LlamaFileType.MOSTLY_F16,
        "bf16": gguf.LlamaFileType.MOSTLY_BF16,
        "q8_0": gguf.LlamaFileType.MOSTLY_Q8_0,
        "tq1_0": gguf.LlamaFileType.MOSTLY_TQ1_0,
        "tq2_0": gguf.LlamaFileType.MOSTLY_TQ2_0,
        "auto": gguf.LlamaFileType.GUESSED,
    }

    is_split = args.split_max_tensors > 0 or args.split_max_size != "0"
    if args.use_temp_file and is_split:
        logger.error("Error: Cannot use temp file when splitting")
        sys.exit(1)

    if args.outfile is not None:
        fname_out = args.outfile
    elif hf_repo_id:
        # if remote, use the model ID as the output file name
        fname_out = Path("./" + hf_repo_id.replace("/", "-") + "-{ftype}.gguf")
    else:
        fname_out = dir_model

    logger.info(f"Loading model: {dir_model.name}")

    if args.mmproj:
        if "mmproj" not in fname_out.name:
            fname_out = ModelBase.add_prefix_to_filename(fname_out, "mmproj-")

    with torch.inference_mode():
        output_type = ftype_map[args.outtype]
        model_type = ModelType.MMPROJ if args.mmproj else ModelType.TEXT
        hparams = ModelBase.load_hparams(dir_model)
        model_architecture = get_model_architecture(hparams, model_type)
        logger.info(f"Model architecture: {model_architecture}")
        try:
            model_class = ModelBase.from_model_architecture(
                model_architecture, model_type=model_type
            )
        except NotImplementedError:
            logger.error(f"Model {model_architecture} is not supported")
            sys.exit(1)

        model_instance = model_class(
            dir_model,
            output_type,
            fname_out,
            is_big_endian=args.bigendian,
            use_temp_file=args.use_temp_file,
            eager=args.no_lazy,
            metadata_override=args.metadata,
            model_name=args.model_name,
            split_max_tensors=args.split_max_tensors,
            split_max_size=split_str_to_n_bytes(args.split_max_size),
            dry_run=args.dry_run,
            small_first_shard=args.no_tensor_first_split,
            remote_hf_model_id=hf_repo_id,
            model_description=args.hmm_info,
            hmm_vit_height=args.hmm_vit_height,
            hmm_vit_width=args.hmm_vit_width,
        )

        if args.vocab_only:
            logger.info("Exporting model vocab...")
            model_instance.write_vocab()
            logger.info(
                f"Model vocab successfully exported to {model_instance.fname_out}"
            )
        else:
            logger.info(f"Exporting model...is_split:{is_split}")
            model_instance.write()
            out_path = (
                f"{model_instance.fname_out.parent}{os.sep}"
                if is_split
                else model_instance.fname_out
            )
            logger.info(f"Model successfully exported to {out_path}")

            if args.upload is True:
                logger.info(f"#hm uploading gguf files")
                current_dt = datetime.now()
                current_ts = str(current_dt.timestamp())
                today = current_dt.strftime("%Y%m%d")

                host_result_dir,compiled_file_name=extract_path_and_filename(out_path)
                host_compile_dir=host_result_dir
                host_log_file=host_result_dir + '/' + str(args.task_id)+"_publishmodel.log"
                logger.info(f"##hm host_result_dir:{host_result_dir} compiled_file_name:{compiled_file_name} host_compile_dir:{host_compile_dir} host_log_file:{host_log_file}")
                md5sum_compile, jfrog_path_compile = _publish_model(
                    today,
                    compiled_file_name,
                    host_compile_dir,
                    host_result_dir,
                    args.model_name,
                    host_log_file,
                    [".gguf"],
                )
                if jfrog_path_compile is None or md5sum_compile is None:
                    logger.error(f"Failed to publish compiled model {out_path}.")
                    exit(4)
                else:
                    logger.info(f"upload {out_path} to jfrog success")
                    logger.info(f"#hm upload to jfrog success result jfrog path:{jfrog_path_compile}")

if __name__ == "__main__":
    main()
