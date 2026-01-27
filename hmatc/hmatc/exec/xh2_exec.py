# Copyright 2025 HOUMO AI
#
# File: xh2_exec.py
# Description:
#   XH2 Executor
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
import shutil
import time
import cv2
import numpy as np
import json
import torch
from prettytable import PrettyTable
from datetime import datetime
from ..base.base_exec import BaseExec
from ..infer.onnx_infer import OnnxInfer
from ..infer.xh2_infer import Xh2Infer
from ..infer.xhquant_infer import Xh2HmQuantInfer
from ..utils import logger
from ..utils.dist_metrics import cosine_distance
from ..utils.preprocess import calc_padding_size, convert_bgr_to_yuv
from ..utils.preprocess import xh1_preprocess as resizer_preprocess
from ..utils.utils import (
    SUPPORT_IMAGE_FORMATS,
    compress_files_to_tar_xz_with_progress,
    compress_folder_to_tar_xz_with_progress,
    get_file_md5,
    get_hmquant_xh2_version,
    get_md5,
    get_package_version,
    get_houmo_version,
    load_npz,
    upload_file_to_artifactory,
)


class Xh2Exec(BaseExec):
    """
    Executor class for XH2 target platform.
    Handles quantization, building, checking golden data, and comparison for XH2 hardware.
    """

    def __init__(self, cfg: dict) -> None:
        """
        Initialize the XH2 executor.

        Args:
            cfg (dict): Configuration dictionary containing model and quantization settings
        """
        super().__init__(cfg)
        self.quant_type = self.quant_cfg.get("quant_type", "w8a8h1_sefp")
        self.golden_dir = os.path.join(self.quant_output_dir, "golden")
        self.upgrade_opset_version()

        # Get ONNX output node names
        self.outputs_name = list()
        for name in self.onnx_outputs_info:
            self.outputs_name.append(name)

        # Append dynamic parameter input names to inputs_name
        if self.resizer_mode in [1, 2]:
            input_name = self.inputs_name[0]
            self.inputs_name.append(f"resizer_crop_{input_name}")

    def upgrade_opset_version(self):
        """
        Upgrade the ONNX model opset version to minimum required version (13).
        """
        import onnx
        from onnx import version_converter

        model = onnx.load(self.model_path)
        # Iterate through the opset_import fields in the model (there may be multiple domains)
        opset_version = None
        for opset in model.opset_import:
            if opset.domain == "":  # Main domain (default ONNX operator set)
                opset_version = opset.version
                break
        if opset_version is None:
            logger.warning(f"Not found onnx opset version: {self.model_path}")
            return
        min_opset_version = 13
        if opset_version < min_opset_version:
            new_model_path = self.model_path.replace(
                ".onnx", f"_opset{min_opset_version}.onnx"
            )
            if not os.path.exists(new_model_path):
                new_model = version_converter.convert_version(model, min_opset_version)
                onnx.save(new_model, new_model_path)
                logger.info(
                    f"Upgrade onnx opset {opset_version} to {min_opset_version}, and save new onnx to: {new_model_path}"
                )
            self.model_path = new_model_path

    @staticmethod
    def get_format(toYUV_format):
        """
        Get the format string based on YUV format.

        Args:
            toYUV_format (str): YUV format string

        Returns:
            str: Corresponding format string
        """
        fmt = "yuv420"
        if toYUV_format == "YUV420SP":
            fmt = "yuv420"
        elif toYUV_format == "YUV422SP":
            fmt = "yuv422"
        elif toYUV_format == "YUV444SP":
            fmt = "yuv444"
        elif toYUV_format == "YUV400":
            fmt = "R8"
        return fmt

    def get_quant_cfg(self):
        """
        Get the quantization configuration for XH2 platform.

        Returns:
            dict: Quantization configuration dictionary
        """
        try:
            from xhquant.api import (
                DeviceType,
                QuantScheme,
                ResizerScheme,
                create_quant_config,
            )
        except ImportError as e:
            logger.error(f"{e}")
            logger.error("Not found xhquant module, and please install xhquant.")
            exit(-1)

        input_ppc_config = []
        if self.is_multi_input_model:
            # Multi-input does not support resizer for now
            for input_name in self.inputs_cfg:
                input_ppc_config.append("float16")
        else:
            for input_name in self.inputs_cfg:
                input_cfg = self.inputs_cfg[input_name]
                data_format = input_cfg.get("data_format")
                if data_format is None or "resizer" not in input_cfg:
                    input_ppc_config.append("float16")
                    continue
                _, _, H, W = input_cfg.get("shape")
                mean = input_cfg.get("mean")
                std = input_cfg.get("std")
                resizer_cfg = input_cfg.get("resizer", dict())
                resizer_format = resizer_cfg.get("toYUV_format", "YUV420SP")
                resizer_input_size = resizer_cfg.get("max_input_size", [H, W])
                MAX_H, MAX_W = resizer_input_size
                enable_static_resizer = resizer_cfg.get("enable_static_resizer", True)
                dynamic_crop = not enable_static_resizer
                resize_type = input_cfg["resize_type"]
                crop_offset = (0, 0)
                crop_size = (resizer_input_size[0], resizer_input_size[1])
                pad_value = 0
                if resize_type == 1:
                    padding_values = input_cfg.get("padding_values", [114, 114, 114])
                    pad_value = padding_values[0]
                    # Static is temporarily limited to 0, TODO: modify after compiler fix
                    if enable_static_resizer:
                        pad_value = 0
                pad_size = (0, 0, 0, 0)
                if enable_static_resizer:
                    resizer_crop_size = resizer_cfg.get("crop_size", [0, 0, H, W])
                    CROP_H, CROP_W = (resizer_crop_size[2], resizer_crop_size[3])
                    if CROP_H > MAX_H or CROP_W > MAX_W:
                        _, new_size, _ = calc_padding_size(
                            (CROP_H, CROP_W), (MAX_W, MAX_H), 0
                        )
                        CROP_H, CROP_W = new_size
                    crop_offset = (resizer_crop_size[0], resizer_crop_size[1])
                    crop_size = (CROP_H, CROP_W)
                    # if resize_type == 1:
                    #     padding_mode = input_cfg["padding_mode"]
                    #     pad_size, _, _ = calc_padding_size(
                    #         crop_size, (W, H), padding_mode
                    #     )
                resizer_cfg = ResizerScheme(
                    size=(H, W),  # target size, also ONNX model input size
                    mode="bilinear",  # nearest, not supported for configuration yet
                    align_corners=False,
                    fmt=self.get_format(resizer_format),
                    int_trans=True,  # Default: True  -128 for suit 8bit
                    crop_size=crop_size,  # Default full image crop
                    crop_offset=crop_offset,
                    pad_size=pad_size,
                    pad_value=pad_value,  # Only scalar supported temporarily
                    mean=[v / 255.0 for v in mean],
                    std=[v / 255.0 for v in std],
                    dynamic_crop=dynamic_crop,
                    model_inp_fmt=data_format.lower(),
                ).to_dict()
                input_ppc_config.append(resizer_cfg)
        quant_scheme = QuantScheme(
            target_device=DeviceType.XH2a,
            quant_type=self.quant_type,
            input_ppc_config=input_ppc_config,
        )
        quant_config = create_quant_config(quant_scheme)

        return quant_config

    def get_random_input_data(self):
        """
        Generate random golden input data.

        Returns:
            list: List of input data tensors
        """
        in_datas = list()
        dynamic_inputs = list()
        for input_name in self.inputs_cfg:
            input_cfg = self.inputs_cfg[input_name]
            shape = input_cfg.get("shape")
            dtype_str = self.onnx_inputs_info[input_name]["dtype"]
            if not self.is_image_single_input or "resizer" not in input_cfg:
                in_datas.append(
                    torch.from_numpy(self.gen_random_data(shape, dtype_str))
                )
                continue
            # Input is image and using resizer, need to regenerate input and dynamic input parameters
            resizer_cfg = input_cfg.get("resizer", dict())
            N, C, H, W = shape
            # Resizer operator default input size is model input size
            resizer_input_size = resizer_cfg.get("max_input_size", [H, W])
            toYUV_format = resizer_cfg.get("toYUV_format", "YUV420SP")
            resizer_input_shape = [
                N,
                C,
                resizer_input_size[0],
                resizer_input_size[1],
            ]
            random_bgr_img = self.gen_random_data(resizer_input_shape, "uint8")
            random_yuv_img = convert_bgr_to_yuv(random_bgr_img, toYUV_format)  # HWC
            random_yuv_img = random_yuv_img.view(N, C, H, W)
            in_datas.append(random_yuv_img)
            enable_static_resizer = resizer_cfg.get("enable_static_resizer", True)
            if not enable_static_resizer:
                # Dynamic parameter data defaults to using full image size as valid data
                dynamic_inputs.append(
                    torch.tensor(
                        [
                            [
                                0,
                                0,
                                resizer_input_size[0],
                                resizer_input_size[1],
                                H,
                                W,
                                0,
                                0,
                                0,
                                0,
                            ]
                        ],
                        dtype=torch.int32,
                    )
                )

        in_datas.extend(dynamic_inputs)
        return in_datas

    def get_input_data(self):
        """Generate golden input data.

        Returns:
            list: List of input data tensors"""
        if self.use_random_data:
            logger.info("Use random calib data")
            return self.get_random_input_data()

        calib_data = self.quant_cfg.get("calib_data")
        if not os.path.isdir(calib_data):
            HOUMO_DATASETS_PATH = os.environ.get("HOUMO_DATASETS_PATH", "")
            calib_data = os.path.join(HOUMO_DATASETS_PATH, calib_data)
            if not os.path.isdir(calib_data):
                logger.error(f"Not found calib_data path: {calib_data}")
                exit(-1)
        logger.info(f"calib_data: {calib_data}")
        filenames = os.listdir(calib_data)
        data_list = []
        for filename in filenames:
            _, ext = os.path.splitext(filename)
            if (
                ext not in SUPPORT_IMAGE_FORMATS
                if self.is_image_single_input
                else [".npz"]
            ):
                continue
            data_list.append(os.path.join(calib_data, filename))
        if len(data_list) == 0:
            logger.error(f"Not found calib data in {calib_data}")
            exit(-1)
        data_list.sort()

        data_path = data_list[0]
        logger.info(f"Using data path: {data_path}")
        if self.is_multi_input_model or not self.is_image_single_input:
            in_datas = load_npz(data_path)
            in_datas = [torch.from_numpy(in_datas[k]) for k in in_datas]
        else:
            input_name = self.inputs_name[0]
            input_cfg = self.inputs_cfg[input_name]
            data_format = input_cfg["data_format"]
            input_shape = input_cfg["shape"]
            N, C, H, W = input_shape
            mean = input_cfg["mean"]
            std = input_cfg["std"]
            resize_type = input_cfg["resize_type"]
            padding_mode = input_cfg.get("padding_mode", 1)
            padding_values = input_cfg.get("padding_values", [0, 0, 0])
            cv_image = cv2.imread(data_path)
            if cv_image is None:
                logger.error(f"Failed to load image: {data_path}")
                exit(1)
            resizer_cfg = input_cfg.get("resizer", dict())
            toYUV_format = resizer_cfg.get("toYUV_format", "YUV420SP")
            max_input_size = resizer_cfg.get("max_input_size", [H, W])
            MAX_H, MAX_W = max_input_size
            yuv_im, dyn_info = resizer_preprocess(
                cv_image,
                input_shape,
                max_input_size=max_input_size,
                mean=mean,
                std=std,
                use_resize=self.resizer_mode in [0, 3],
                use_norm=self.resizer_mode == 0,
                use_rgb=data_format == "RGB" and self.resizer_mode == 0,
                resize_type=resize_type,
                padding_mode=padding_mode,
                padding_values=padding_values,
                is_onnx=self.resizer_mode == 0,
                to_YUV=self.resizer_mode in [1, 2, 3],
                fmt=toYUV_format,
                return_dynamic_v1_format=self.resizer_mode == 1,
            )
            in_datas = []
            if self.resizer_mode in [1, 2, 3]:
                yuv_im = yuv_im.view(N, C, MAX_H, MAX_W)
            in_datas.append(yuv_im)
            if self.resizer_mode in [1, 2]:
                in_datas.append(dyn_info)

        return in_datas

    def quantize(self):
        """
        Quantize the ONNX model for XH2 hardware.

        Returns:
            dict: Dictionary containing quantization results and information
        """
        logger.info(f"Quant type: {self.quant_type}")
        # quantize the model
        if not os.path.exists(self.quant_output_dir):
            os.makedirs(self.quant_output_dir)

        if hasattr(self, "ApplicationOnnxOpt"):
            self.ApplicationOnnxOpt.opt()
            if hasattr(self.ApplicationOnnxOpt, "opt_model_path"):
                self.model_path = self.ApplicationOnnxOpt.opt_model_path

        try:
            from xhquant.api import (
                DeviceType,
                HMONNXGoldenInference,
                convert_onnx_to_hmonnx,
            )
        except ImportError as e:
            logger.error(f"{e}")
            logger.error("Not found xhquant module, and please install xhquant.")
            exit(-1)

        quant_cfg = self.get_quant_cfg()
        logger.info(f"quant_cfg: {quant_cfg}")

        in_datas = self.get_input_data()

        quant_onnx_model_path = os.path.join(
            self.quant_output_dir, f"{self.model_name}.onnx"
        )
        # Quantization and HMONNX export
        now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        t_start = time.time()
        convert_onnx_to_hmonnx(
            self.model_path,
            in_datas,
            device_type=DeviceType.XH2a,
            out_hmonnx_file=quant_onnx_model_path,
            quant_config=quant_cfg,
            input_names=self.inputs_name,
            output_names=self.outputs_name,
        )
        # Generate chip required format model
        session = HMONNXGoldenInference(quant_onnx_model_path)
        session.to(self.device)
        session.save_golden = True
        session.golden_dir = self.golden_dir
        if os.path.exists(self.golden_dir):
            shutil.rmtree(self.golden_dir)
        session.step = 0
        # float32 -> float16 and int64 -> int32
        for idx, in_data in enumerate(in_datas):
            if self.inputs_name[idx].startswith("resizer_crop_"):
                continue
            if in_data.dtype == torch.int64:
                in_datas[idx] = in_datas[idx].type(torch.int32).to(self.device)
            elif in_data.dtype == torch.float32:
                in_datas[idx] = in_data.half().to(self.device)
        session(*in_datas)  #
        if os.path.exists(quant_onnx_model_path):
            os.remove(quant_onnx_model_path)
        shutil.copytree(
            os.path.join(self.golden_dir, "step_0"),
            self.quant_output_dir,
            dirs_exist_ok=True,
        )
        shutil.rmtree(self.golden_dir)
        # Compress quantization outputs
        if self.enable_upload and 0:
            logger.info("Compressing quant output...")
            with open(os.path.join(self.quant_output_dir, "VERSION.txt"), "w") as f:
                f.write(f"hmquant_version: {get_hmquant_xh2_version()}\n")
                f.write(f"quant_time: {now}\n")
            filename = f"hmquant_{self.model_dir_name}_xh2_{get_houmo_version()}.tar.xz"
            compress_quant_output_path = os.path.join(self.save_dir, "xh2", filename)
            compress_folder_to_tar_xz_with_progress(
                self.quant_output_dir,
                compress_quant_output_path,
                # exclude=["*_with_act.onnx"],
            )
            logger.info(
                f"MD5: {get_file_md5(compress_quant_output_path)}, save path: {compress_quant_output_path}"
            )
            upload_file_to_artifactory(
                compress_quant_output_path,
                f"models/{self.target.lower()}-{get_houmo_version()}/{self.model_dir_name}/{filename}",
                max_retries=3,
            )
            logger.info(f"Compressing quant output done.")
        span = time.time() - t_start
        res = dict()
        res["time"] = span
        res_info = {"quant": res, "model": self.model_cfg}
        logger.info(f"Quantize done. and save hmonnx: {self.quant_onnx_model_path}")
        return res_info

    def build(self, enable_profile=False):
        """
        Build the quantized model to HMM format for XH2 hardware.

        Args:
            enable_profile (bool): Whether to enable profiling during build, defaults to False

        Returns:
            dict: Dictionary containing build results and information
        """
        self.enable_profile = enable_profile
        if not os.path.exists(self.build_output_dir):
            os.makedirs(self.build_output_dir)

        try:
            import tcim
        except ImportError:
            logger.error("Not found tcim module, and please install tcim first!")
            exit(-1)

        logger.info(f"hmmonnx: {self.quant_onnx_model_path}")
        now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        t_start = time.time()
        tcim.build_from_hmonnx(
            self.quant_onnx_model_path,
            output_name=self.hmm_name,
            ncore=self.build_ncore,
            opt_level=self.build_opt_level,
            target="xh2",
            batch=self.build_batch if self.roi_num == 1 else self.roi_num,
            enable_profile=enable_profile,
            output_dir=self.hmm_save_dir,
            work_dir=self.build_output_dir,
            # enable_dynamic_image_resize=self.resizer_mode in [1],
            one_img_multi_roi=self.roi_num > 1,
            custom_msg=json.dumps(self.custom_msg, ensure_ascii=False),
        )
        span = time.time() - t_start
        # Compress compiled outputs
        if self.enable_upload:
            logger.info("Compressing hmmodel...")
            hmcc_version = get_package_version(f"houmo-tcim-xh2")
            runtime_version = get_package_version(f"houmo_tcim_runtime_xh2")
            with open(os.path.join(self.save_dir, "xh2", "VERSION.txt"), "w") as f:
                f.write(f"hmquant_version: {get_hmquant_xh2_version()}\n")
                f.write(f"tcim_version: {hmcc_version}\n")
                f.write(f"tcim_runtime_version: {runtime_version}\n")
                f.write(f"build_time: {now}\n")
            filename = f"{self.model_dir_name}_xh2_b{self.hmm_batch}_{self.build_ncore}core_{self.build_opt_level}_{get_houmo_version()}.tar.xz"
            compress_hmm_path = os.path.join(
                self.save_dir,
                "xh2",
                filename,
            )
            compress_files_to_tar_xz_with_progress(
                [self.hmm_path, os.path.join(self.save_dir, "xh2", "VERSION.txt")],
                compress_hmm_path,
            )
            logger.info(
                f"MD5: {get_file_md5(compress_hmm_path)}, save path: {compress_hmm_path}"
            )
            upload_file_to_artifactory(
                compress_hmm_path,
                f"models/{self.target.lower()}-{get_houmo_version()}/{self.model_dir_name}/{filename}",
                max_retries=3,
            )
            logger.info(f"Compressing hmmodel done.")
        res_info = {"build": {"time": span}}
        return res_info

    def check_golden(self, device_id=0, enable_layers=False):
        """
        Check the golden data against the hardware model outputs.

        Args:
            device_id (int): Device ID for inference, default is 0.
            enable_layers (bool): Whether to enable layer-by-layer checking

        Returns:
            dict: Dictionary containing comparison results between golden and hardware outputs
        """
        logger.info("Checking golden...")
        if enable_layers:
            # Add all node outputs of quantized ONNX as graph outputs
            self.quant_onnx_model_path = self.add_node_output_as_graph_output(
                self.quant_onnx_model_path
            )
            self.build_output_dir += "_debug"
            self.hmm_name += "_debug"
            self.hmm_path = os.path.join(self.hmm_save_dir, f"{self.hmm_name}.hmm")
            if not os.path.exists(self.hmm_path):
                logger.info("Rebuild hmmodel with all layers output...")
                self.build(enable_profile=False)

        logger.info(f"Build batch: {self.build_batch}")

        xh2 = Xh2Infer()
        xh2.load(self.hmm_path, device_id=device_id)
        in_datas = dict()
        for input_name in self.inputs_cfg:
            new_name = input_name.replace("/", "_")
            golden_input_path = os.path.join(
                self.quant_output_dir,
                f"hmquant_{self.model_name}_{new_name}_input.npy",
            )
            golden_input = np.load(golden_input_path)
            logger.info(f"Load golden: {golden_input_path}")
            logger.info(
                f"[Golden][Input] name: {input_name}, shape: {list(golden_input.shape)}, stype: {golden_input.dtype}"
            )
            if self.resizer_mode in [1, 2, 3]:
                fmt = xh2.inputs_format[input_name]
                golden_input = golden_input.flatten()
                size = golden_input.size
                if fmt == "YUV420SP":
                    size //= 2
                elif fmt == "YUV422SP":
                    size = size * 3 // 2
                golden_input = golden_input[:size].reshape(1, size)
            golden_input = np.repeat(golden_input, self.build_batch, axis=0)
            in_datas[input_name] = golden_input
            if self.resizer_mode == 1:
                resizer_name = f"resizer_crop_{input_name}"
                golden_dyn_info_input_path = os.path.join(
                    self.quant_output_dir,
                    f"hmquant_{self.model_name}_resizer_crop_{new_name}_input.npy",
                )
                golden_dyn_input = np.load(golden_dyn_info_input_path)
                logger.info(f"Load golden: {golden_dyn_info_input_path}")
                logger.info(
                    f"[Golden][DynInfo] name: {resizer_name}, shape: {list(golden_dyn_input.shape)}, stype: {golden_dyn_input.dtype}"
                )
                repeats = 1
                if self.roi_num > 1:
                    # 1 image n boxes
                    repeats = self.roi_num
                elif self.roi_num == 1 and self.build_batch > 1:
                    # n images n boxes, and compilation batch > 1
                    repeats = self.build_batch
                golden_dyn_input = np.repeat(golden_dyn_input, repeats=repeats, axis=0)
                in_datas[resizer_name] = golden_dyn_input

        res_info = dict()
        outputs, _ = xh2.run(in_datas)
        self.save_profile_data(outputs)

        repeats = 1
        if self.build_batch > 1 and self.roi_num == 1:
            # n images n boxes, and compilation batch > 1
            repeats = self.build_batch
        elif self.roi_num > 1:
            # 1 image n boxes
            repeats = self.roi_num

        header = ["name", "cosine_dist"]
        table = PrettyTable(header)
        table.title = "xh2 vs hmquant"
        for output_name in outputs:
            new_name = output_name.replace("/", "_")
            if enable_layers:
                golden_output_path = os.path.join(
                    self.quant_output_dir,
                    f"hmquant_{self.model_name}_with_act",
                    f"{new_name}.npy",
                )
            else:
                golden_output_path = os.path.join(
                    self.quant_output_dir,
                    f"hmquant_{self.model_name}_{new_name}_output.npy",
                )
            golden_output = np.load(golden_output_path)
            logger.info(f"Load golden: {golden_output_path}")
            logger.info(
                f"[Golden][Output] name: {output_name}, shape: {list(golden_output.shape)}, dtype: {golden_output.dtype}"
            )
            golden_output = np.repeat(golden_output, repeats=repeats, axis=0)
            golden_output_md5 = get_md5(golden_output)
            output = outputs[output_name]
            output_md5 = get_md5(output)
            # compare
            dist = cosine_distance(golden_output, output)
            table.add_row([output_name, f"{dist:.6f}"])
            res_info[output_name] = {
                "md5": output_md5,
                "golden_md5": golden_output_md5,
                "cosine_dist": float(dist),
            }
        logger.info(f"\n{table}")
        return res_info

    def compare(self, data_path: str, device_id=0):
        """
        Compare outputs from ONNX, HmQuant and XH2 inference.

        Args:
            data_path (str): Path to input data for comparison
            device_id (int): Device ID for XH2 inference, defaults to 0

        Returns:
            dict: Dictionary containing comparison results between different inference engines
        """
        t_start = datetime.now().strftime("%Y%m%d%H%M%S")
        # onnx
        onnx_infer = OnnxInfer()
        onnx_infer.load(self.model_path)
        # hmquant
        hmquant_infer = Xh2HmQuantInfer()
        hmquant_infer.load(self.quant_onnx_model_path)
        # xh2
        xh2_infer = Xh2Infer()
        xh2_infer.load(self.hmm_path)

        onnx_in_datas = dict()
        hmquant_in_datas = dict()
        xh2_in_datas = dict()
        _, ext = os.path.splitext(os.path.basename(data_path))
        if self.is_image_single_input:
            # Single input image
            input_name = self.inputs_name[0]
            input_cfg = self.inputs_cfg[input_name]
            data_format = input_cfg["data_format"]
            input_shape = input_cfg["shape"]
            N, C, H, W = input_shape
            mean = input_cfg["mean"]
            std = input_cfg["std"]
            resize_type = input_cfg["resize_type"]
            padding_mode = input_cfg.get("padding_mode", 1)
            padding_values = input_cfg.get("padding_values", [114, 114, 114])
            if ext not in SUPPORT_IMAGE_FORMATS:
                logger.error(f"Not support image: {data_path}")
                exit(-1)
            if not os.path.exists(data_path):
                logger.error(f"Not found data_path: {data_path}")
                exit(-1)
            cv_image = cv2.imread(
                data_path,
                cv2.IMREAD_COLOR if data_format != "GRAY" else cv2.IMREAD_GRAYSCALE,
            )
            if cv_image is None:
                logger.error("Failed to decode image")
                exit(-1)
            # resizer preprocess
            resizer_cfg = input_cfg.get("resizer", dict())
            toYUV_format = resizer_cfg.get("toYUV_format", "YUV420SP")
            max_input_size = resizer_cfg.get("max_input_size", [H, W])
            # crop_size = resizer_cfg.get("crop_size", [0, 0, H, W])
            # Get compiled model batch
            hmm_batch = xh2_infer.inputs_info[input_name].shape[0]
            # onnx preprocess
            onnx_data, _ = resizer_preprocess(
                cv_image,
                input_shape,
                max_input_size,
                mean=mean,
                std=std,
                use_resize=True,
                use_norm=True,
                use_rgb=data_format == "RGB",
                resize_type=resize_type,
                padding_mode=padding_mode,
                padding_values=padding_values,
                is_onnx=True,
            )
            onnx_data = np.repeat(
                onnx_data.numpy(), repeats=self.model_input_batch, axis=0
            )
            onnx_in_datas[input_name] = onnx_data  # np.ndarray

            yuv_pad_hwc, dyn_info = resizer_preprocess(
                cv_image,
                input_shape,
                max_input_size,
                mean=mean,
                std=std,
                use_resize=self.resizer_mode in [0, 3],
                use_norm=self.resizer_mode == 0,
                use_rgb=data_format == "RGB" and self.resizer_mode == 0,
                resize_type=resize_type,
                padding_mode=padding_mode,
                padding_values=padding_values,
                is_onnx=self.resizer_mode == 0,
                to_YUV=self.resizer_mode in [1, 2, 3],
                fmt=toYUV_format,
                return_dynamic_v1_format=self.resizer_mode == 1,
                # crop_size=crop_size,
            )

            if self.resizer_mode in [1, 2, 3]:
                # Using resizer
                h, w, c = yuv_pad_hwc.shape
                yuv_pad = yuv_pad_hwc.view(1, c, h, w)
                yuv_pad = yuv_pad.repeat_interleave(self.model_input_batch, dim=0)
                hmquant_in_datas[input_name] = yuv_pad.contiguous()  # torch.Tensor
                # xh2
                yuv_pad = yuv_pad_hwc.detach().cpu().numpy().flatten()
                if toYUV_format == "YUV420SP":
                    valid_len = yuv_pad.size // 2
                elif toYUV_format == "YUV422SP":
                    valid_len = yuv_pad.size * 2 // 3
                elif toYUV_format in ["YUV444SP", "YUV400"]:
                    valid_len = yuv_pad.size
                yuv = yuv_pad[:valid_len].copy()
                yuv = yuv.reshape(1, -1)
                yuv = np.repeat(yuv, repeats=hmm_batch, axis=0)
                xh2_in_datas[input_name] = np.ascontiguousarray(yuv)  # np.ndarray
            elif self.resizer_mode == 0:
                # Disable resizer
                in_data = np.repeat(onnx_data, repeats=self.build_batch, axis=0)
                in_data = in_data.astype(np.float16)
                hmquant_in_datas[input_name] = torch.from_numpy(
                    in_data[0 : self.model_input_batch, ...]
                ).cpu()
                xh2_in_datas[input_name] = np.ascontiguousarray(in_data)
            # dynamic_resizer info
            if self.resizer_mode in [1, 2]:
                if self.roi_num > 1:
                    # 1 image n boxes
                    hmquant_dyn_info = dyn_info
                    dyn_info = dyn_info.repeat_interleave(self.roi_num, dim=0)
                else:
                    # n images n boxes
                    hmquant_dyn_info = dyn_info.repeat_interleave(
                        self.model_input_batch, dim=0
                    )
                    dyn_info = dyn_info.repeat_interleave(hmm_batch, dim=0)
                hmquant_in_datas[f"resizer_crop_{input_name}"] = hmquant_dyn_info
                xh2_in_datas[f"resizer_crop_{input_name}"] = (
                    dyn_info.detach().cpu().numpy()
                )
        else:
            # Single input non-image or multi-input
            in_datas = load_npz(data_path)
            onnx_in_datas = in_datas
            for input_name in in_datas:
                _in_data = in_datas[input_name]
                if _in_data.dtype == np.int64:
                    _in_data = _in_data.astype(np.int32).copy()
                if _in_data.dtype == np.float32:
                    _in_data = _in_data.astype(np.float16).copy()
                hmquant_in_datas[input_name] = torch.from_numpy(_in_data)
                xh2_in_datas[input_name] = np.repeat(_in_data, self.build_batch, axis=0)

        onnx_outputs = onnx_infer.run(onnx_in_datas)
        hmquant_outputs = hmquant_infer.run(hmquant_in_datas)
        xh2_outputs, xh2_outputs_dequanted = xh2_infer.run(xh2_in_datas)
        self.save_profile_data(xh2_outputs)

        repeats = 1
        if self.build_batch > 1 and self.roi_num == 1:
            # n images n boxes, and compilation batch > 1
            repeats = self.build_batch
        elif self.roi_num > 1:
            # 1 image n boxes
            repeats = self.roi_num

        res_info = {"compare": {t_start: dict()}}
        res_info["compare"][t_start]["data_path"] = data_path
        # Calculate similarity
        header = ["name", "onnx vs hmquant", "onnx vs xh2", "hmquant vs xh2"]
        table = PrettyTable(header)
        table.title = "Cosine Distance"
        for output_name in onnx_outputs:
            onnx_output = onnx_outputs[output_name]
            onnx_output = np.repeat(onnx_output, repeats=repeats, axis=0)

            hmquant_output = hmquant_outputs[output_name]
            hmquant_output = np.repeat(hmquant_output, repeats=repeats, axis=0)

            xh2_output_dequanted = xh2_outputs_dequanted[output_name]
            xh2_output_dequanted = np.split(
                xh2_output_dequanted, self.build_batch, axis=0
            )[0]
            onnx_vs_hmquant = cosine_distance(onnx_output, hmquant_output)
            onnx_vs_xh2 = cosine_distance(onnx_output, xh2_output_dequanted)
            hmquant_vs_xh2 = cosine_distance(hmquant_output, xh2_output_dequanted)
            table.add_row(
                [
                    output_name,
                    f"{onnx_vs_hmquant:.6f}",
                    f"{onnx_vs_xh2:.6f}",
                    f"{hmquant_vs_xh2:.6f}",
                ]
            )
            res_info["compare"][t_start][output_name] = {
                "onnx_vs_hmquant": float(onnx_vs_hmquant),
                "onnx_vs_xh2": float(onnx_vs_xh2),
                "hmquant_vs_xh2": float(hmquant_vs_xh2),
            }
        logger.info(f"\n{table}")
        return res_info
