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
import psutil
import torch
from prettytable import PrettyTable
from datetime import datetime
from ..base.base_exec import BaseExec
from ..infer.onnx_infer import OnnxInfer
from ..infer.xh2_infer import Xh2Infer
from ..infer.xhquant_infer import Xh2HmQuantInfer
from ..utils import logger
from ..utils.dist_metrics import cosine_distance
from ..utils.preprocess import calc_padding_size, convert_bgr_to_yuv, default_preprocess
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
    find_input_files,
    find_output_files,
    gen_random_data,
)


class Xh2Exec(BaseExec):
    """
    Executor class for XH2 target platform.
    Handles quantization, building, checking golden data, and comparison for XH2 hardware.

    XH2 Resizer Constraints:
        - Padding must be symmetric (center padding mode, padding_mode=1)
        - Padding pixels cannot exceed 32 pixels on each side
        - Resizer input width cannot exceed 1024 pixels
        - Resizer input height cannot exceed 4096 pixels
    """

    MAX_PADDING_SIZE = 32  # Maximum padding pixels on each side
    MAX_RESIZER_INPUT_W = 1024  # Maximum resizer input width
    MAX_RESIZER_INPUT_H = 4096  # Maximum resizer input height

    def __init__(self, cfg: dict) -> None:
        super().__init__(cfg)
        self.quant_type = self.quant_cfg.get("quant_type", "w8a8h1_sefp")
        self.golden_dir = os.path.abspath(os.path.join(self.quant_output_dir, "golden"))
        self.upgrade_opset_version()
        self._validate_resizer_constraints()

        # Get ONNX output node names
        self.outputs_name = list(self.onnx_outputs_info.keys())

        # Append dynamic parameter input names for inputs with dynamic resizer (mode 1 or 2)
        for input_name in self.inputs_cfg:
            if self.resizer_modes.get(input_name, 0) in [1, 2]:
                self.inputs_name.append(f"resizer_crop_{input_name}")

    # ==================== Helper Methods ====================

    @staticmethod
    def _ensure_hwc_uint8(data):
        """Ensure data is HWC BGR uint8 format."""
        if data.dtype != np.uint8:
            data = data.astype(np.uint8)
        if len(data.shape) == 4:
            data = data[0]
        return data

    @staticmethod
    def _get_input_preprocess_params(input_cfg: dict):
        """Extract preprocessing parameters from input config."""
        N, C, H, W = input_cfg["shape"]
        return {
            "shape": input_cfg["shape"],
            "N": N,
            "C": C,
            "H": H,
            "W": W,
            "mean": input_cfg.get("mean", [0, 0, 0]),
            "std": input_cfg.get("std", [1, 1, 1]),
            "resize_type": input_cfg.get("resize_type", 0),
            "padding_mode": input_cfg.get("padding_mode", 1),
            "padding_values": input_cfg.get("padding_values", [114, 114, 114]),
            "data_format": input_cfg.get("data_format", "BGR"),
        }

    def _has_resizer(self, input_name: str) -> bool:
        """Check if input has resizer configured."""
        input_cfg = self.inputs_cfg[input_name]
        return "resizer" in input_cfg and input_cfg.get("data_format") is not None

    def _get_resizer_params(self, input_cfg: dict):
        """Extract resizer parameters from input config."""
        resizer_cfg = input_cfg.get("resizer", {})
        _, _, H, W = input_cfg["shape"]
        return {
            "toYUV_format": resizer_cfg.get("toYUV_format", "YUV420SP"),
            "max_input_size": resizer_cfg.get("max_input_size", [H, W]),
            "enable_static_resizer": resizer_cfg.get("enable_static_resizer", True),
            "crop_size": resizer_cfg.get("crop_size", [0, 0, H, W]),
        }

    def _preprocess_for_resizer(self, cv_image, input_cfg: dict, input_mode: int):
        """Preprocess image through resizer pipeline.

        Returns:
            tuple: (yuv_im, dyn_info) - yuv image and dynamic info (if applicable)
        """
        params = self._get_input_preprocess_params(input_cfg)
        resizer_params = self._get_resizer_params(input_cfg)
        MAX_H, MAX_W = resizer_params["max_input_size"]

        yuv_im, dyn_info = resizer_preprocess(
            cv_image,
            params["shape"],
            max_input_size=resizer_params["max_input_size"],
            mean=params["mean"],
            std=params["std"],
            use_resize=input_mode in [0, 3],
            use_norm=input_mode == 0,
            use_rgb=params["data_format"] == "RGB" and input_mode == 0,
            resize_type=params["resize_type"],
            padding_mode=params["padding_mode"],
            padding_values=params["padding_values"],
            is_onnx=input_mode == 0,
            to_YUV=input_mode in [1, 2, 3],
            fmt=resizer_params["toYUV_format"],
            return_dynamic_v1_format=input_mode == 1,
        )

        if input_mode in [1, 2, 3]:
            yuv_im = yuv_im.view(params["N"], params["C"], MAX_H, MAX_W)

        return yuv_im, dyn_info

    def _preprocess_for_onnx(self, cv_image, input_cfg: dict):
        """Preprocess image for ONNX inference (resize + normalize, no YUV)."""
        params = self._get_input_preprocess_params(input_cfg)
        return default_preprocess(
            cv_image,
            size=(params["W"], params["H"]),
            mean=params["mean"],
            std=params["std"],
            use_norm=True,
            use_rgb=(params["data_format"] == "RGB"),
            use_resize=True,
            resize_type=params["resize_type"],
            padding_value=params["padding_values"],
            padding_mode=params["padding_mode"],
            to_YUV=False,
        )

    # ==================== Validation Methods ====================

    def _validate_resizer_constraints(self):
        """Validate XH2 resizer constraints:
        - padding_mode must be 1 (center/symmetric padding)
        - padding values cannot exceed MAX_PADDING_SIZE
        - resizer input width cannot exceed MAX_RESIZER_INPUT_W
        - resizer input height cannot exceed MAX_RESIZER_INPUT_H
        """
        for input_name, input_cfg in self.inputs_cfg.items():
            mode = self.resizer_modes.get(input_name, 0)
            if mode == 0:
                continue

            padding_mode = input_cfg.get("padding_mode", 1)
            if padding_mode != 1 and mode in [1, 2]:
                logger.error(
                    f"XH2 dynamic resizer constraint violation: input '{input_name}' has padding_mode={padding_mode}. "
                    f"XH2 dynamic resizer requires symmetric padding (padding_mode=1)."
                )
                exit(-1)

            _, _, H, W = input_cfg["shape"]
            resizer_cfg = input_cfg.get("resizer", {})
            max_input_size = resizer_cfg.get("max_input_size", [H, W])
            max_H, max_W = max_input_size

            # Validate resizer input size constraints
            if max_W > self.MAX_RESIZER_INPUT_W:
                logger.error(
                    f"XH2 resizer constraint violation: input '{input_name}' has max_input_width "
                    f"({max_W}) exceeding maximum allowed ({self.MAX_RESIZER_INPUT_W})."
                )
                exit(-1)

            if max_H > self.MAX_RESIZER_INPUT_H:
                logger.error(
                    f"XH2 resizer constraint violation: input '{input_name}' has max_input_height "
                    f"({max_H}) exceeding maximum allowed ({self.MAX_RESIZER_INPUT_H})."
                )
                exit(-1)

            # Validate padding size
            if input_cfg.get("resize_type", 0) == 1:
                if abs(W / H - max_W / max_H) > 0.001:
                    padding_size, _, _ = calc_padding_size(
                        (max_H, max_W), (W, H), padding_mode=1
                    )
                    top, left, bottom, right = padding_size

                    for side, val in [
                        ("top", top),
                        ("bottom", bottom),
                        ("left", left),
                        ("right", right),
                    ]:
                        if val > self.MAX_PADDING_SIZE:
                            logger.error(
                                f"XH2 resizer constraint violation: input '{input_name}' has {side} padding "
                                f"({val}) exceeding maximum allowed ({self.MAX_PADDING_SIZE}). "
                                f"Max input size ({max_W}x{max_H}) with target size ({W}x{H})."
                            )
                            exit(-1)

    def upgrade_opset_version(self):
        """Upgrade the ONNX model opset version to minimum required version (13)."""
        import onnx
        from onnx import version_converter

        model = onnx.load(self.model_path)
        opset_version = None
        for opset in model.opset_import:
            if opset.domain == "":
                opset_version = opset.version
                break

        if opset_version is None:
            logger.warning(f"Not found onnx opset version: {self.model_path}")
            return

        min_version = 13
        if opset_version < min_version:
            new_model_path = self.model_path.replace(
                ".onnx", f"_opset{min_version}.onnx"
            )
            if not os.path.exists(new_model_path):
                new_model = version_converter.convert_version(model, min_version)
                onnx.save(new_model, new_model_path)
                logger.info(
                    f"Upgrade onnx opset {opset_version} to {min_version}, save to: {new_model_path}"
                )
            self.model_path = new_model_path

    @staticmethod
    def get_format(toYUV_format):
        """Get the format string based on YUV format."""
        fmt_map = {
            "YUV420SP": "yuv420",
            "YUV422SP": "yuv422",
            "YUV444SP": "yuv444",
            "YUV400": "R8",
        }
        return fmt_map.get(toYUV_format, "yuv420")

    # ==================== Quantization Methods ====================

    def get_quant_cfg(self):
        """Get the quantization configuration for XH2 platform."""
        try:
            from xhquant.api import (
                DeviceType,
                QuantScheme,
                ResizerScheme,
                create_quant_config,
            )
        except ImportError as e:
            logger.error(f"{e}\nNot found xhquant module, please install xhquant.")
            exit(-1)

        input_ppc_config = []
        for input_name, input_cfg in self.inputs_cfg.items():
            if not self._has_resizer(input_name):
                input_ppc_config.append("float16")
                continue

            params = self._get_input_preprocess_params(input_cfg)
            resizer_params = self._get_resizer_params(input_cfg)
            MAX_H, MAX_W = resizer_params["max_input_size"]

            crop_offset = (0, 0)
            crop_size = (MAX_H, MAX_W)
            pad_value = 0
            pad_size = (0, 0, 0, 0)

            if params["resize_type"] == 1:
                pad_value = params["padding_values"][0]
                if resizer_params["enable_static_resizer"]:
                    pad_value = 0  # Static is temporarily limited to 0

            if resizer_params["enable_static_resizer"]:
                crop_offset = (
                    resizer_params["crop_size"][0],
                    resizer_params["crop_size"][1],
                )
                CROP_H, CROP_W = (
                    resizer_params["crop_size"][2],
                    resizer_params["crop_size"][3],
                )
                if CROP_H > MAX_H or CROP_W > MAX_W:
                    _, (CROP_H, CROP_W), _ = calc_padding_size(
                        (CROP_H, CROP_W), (MAX_W, MAX_H), 0
                    )
                crop_size = (CROP_H, CROP_W)

            input_ppc_config.append(
                ResizerScheme(
                    size=(params["H"], params["W"]),
                    mode="bilinear",
                    align_corners=False,
                    fmt=self.get_format(resizer_params["toYUV_format"]),
                    int_trans=True,
                    crop_size=crop_size,
                    crop_offset=crop_offset,
                    pad_size=pad_size,
                    pad_value=pad_value,
                    mean=[v / 255.0 for v in params["mean"]],
                    std=[v / 255.0 for v in params["std"]],
                    dynamic_crop=not resizer_params["enable_static_resizer"],
                    model_inp_fmt=params["data_format"].lower(),
                ).to_dict()
            )

        return create_quant_config(
            QuantScheme(
                target_device=DeviceType.XH2a,
                quant_type=self.quant_type,
                input_ppc_config=input_ppc_config,
            )
        )

    # ==================== Input Data Methods ====================

    def get_random_input_data(self):
        """Generate random input data for calibration."""
        in_datas = []
        dynamic_inputs = []

        for input_name, input_cfg in self.inputs_cfg.items():
            shape = input_cfg["shape"]
            dtype_str = self.onnx_inputs_info[input_name]["dtype"]

            if not self._has_resizer(input_name):
                in_datas.append(torch.from_numpy(gen_random_data(shape, dtype_str)))
                continue

            resizer_params = self._get_resizer_params(input_cfg)
            N, C, H, W = shape
            MAX_H, MAX_W = resizer_params["max_input_size"]

            random_bgr = torch.from_numpy(
                gen_random_data([N, C, MAX_H, MAX_W], "uint8")
            )
            random_yuv = convert_bgr_to_yuv(
                random_bgr, resizer_params["toYUV_format"]
            ).view(N, C, MAX_H, MAX_W)
            in_datas.append(random_yuv)

            if not resizer_params["enable_static_resizer"]:
                dynamic_inputs.append(
                    torch.tensor(
                        [[0, 0, MAX_H, MAX_W, H, W, 0, 0, 0, 0]], dtype=torch.int32
                    )
                )

        in_datas.extend(dynamic_inputs)
        return in_datas

    def get_input_data(self):
        """Load or generate input data for calibration."""
        if self.use_random_data:
            logger.info("Use random calib data")
            return self.get_random_input_data()

        calib_data = self._resolve_calib_data_path()

        # Multi-input models only support NPZ format
        if self.is_multi_input_model:
            data_path = self._find_data_file(calib_data)
            logger.info(f"Using NPZ data path: {data_path}")
            return self._load_npz_data(data_path)

        # Single input
        data_path = self._find_data_file(calib_data)
        logger.info(f"Using data path: {data_path}")

        if not self.is_image_single_input:
            return self._load_npz_data(data_path)
        return self._load_single_image(data_path)

    def _resolve_calib_data_path(self):
        """Resolve calibration data path."""
        calib_data = self.quant_cfg.get("calib_data")
        if not os.path.isdir(calib_data):
            calib_data = os.path.join(
                os.environ.get("HOUMO_DATASETS_PATH", ""), calib_data
            )
            if not os.path.isdir(calib_data):
                logger.error(f"Not found calib_data path: {calib_data}")
                exit(-1)
        logger.info(f"calib_data: {calib_data}")
        return calib_data

    def _find_data_file(self, calib_data):
        """Find data file in calibration directory."""
        valid_exts = SUPPORT_IMAGE_FORMATS if self.is_image_single_input else [".npz"]
        data_list = sorted(
            [
                os.path.join(calib_data, f)
                for f in os.listdir(calib_data)
                if os.path.splitext(f)[1] in valid_exts
            ]
        )
        if not data_list:
            logger.error(f"Not found calib data in {calib_data}")
            exit(-1)
        return data_list[0]

    def _load_npz_data(self, data_path):
        """Load multi-input data from NPZ file."""
        npz_data = load_npz(data_path)
        in_datas = []
        dynamic_params = {}

        for input_name in self.inputs_cfg:
            if input_name not in npz_data:
                logger.error(f"Input '{input_name}' not found in NPZ file: {data_path}")
                exit(-1)

            input_mode = self.resizer_modes.get(input_name, 0)
            input_cfg = self.inputs_cfg[input_name]
            raw_data = npz_data[input_name]

            if input_mode != 0:
                # Input with resizer
                cv_image = self._ensure_hwc_uint8(raw_data)
                yuv_im, dyn_info = self._preprocess_for_resizer(
                    cv_image, input_cfg, input_mode
                )
                in_datas.append(yuv_im)

                if input_mode in [1, 2]:
                    dyn_param_name = f"resizer_crop_{input_name}"
                    dynamic_params[input_name] = (
                        torch.from_numpy(npz_data[dyn_param_name])
                        if dyn_param_name in npz_data
                        else dyn_info
                    )
            else:
                # mode == 0: Resizer disabled
                data_format = input_cfg.get("data_format")
                if data_format is not None:
                    # Image input with resizer disabled - still needs preprocessing
                    cv_image = self._ensure_hwc_uint8(raw_data)
                    preprocessed = self._preprocess_for_onnx(cv_image, input_cfg)
                    in_datas.append(torch.from_numpy(preprocessed))
                else:
                    # Non-image input - use raw data directly
                    in_datas.append(torch.from_numpy(raw_data))

        # Append dynamic params at the end
        for input_name in self.inputs_cfg:
            if input_name in dynamic_params:
                in_datas.append(dynamic_params[input_name])

        return in_datas

    def _load_single_image(self, data_path: str):
        """Load single image and preprocess for resizer."""
        input_name = self.inputs_name[0]
        input_cfg = self.inputs_cfg[input_name]
        input_mode = self.resizer_modes.get(input_name, 0)

        cv_image = cv2.imread(data_path)
        if cv_image is None:
            logger.error(f"Failed to load image: {data_path}")
            exit(1)

        yuv_im, dyn_info = self._preprocess_for_resizer(cv_image, input_cfg, input_mode)
        in_datas = [yuv_im]

        if input_mode in [1, 2]:
            in_datas.append(dyn_info)

        return in_datas

    def _load_multi_input_images(self, images_dir: str):
        """Load multi-input images from directory."""
        image_files = sorted(
            [
                os.path.join(images_dir, f)
                for f in os.listdir(images_dir)
                if any(f.lower().endswith(ext) for ext in SUPPORT_IMAGE_FORMATS)
            ]
        )

        if not image_files:
            logger.error(f"No images found in {images_dir}")
            exit(-1)

        logger.info(f"Loading {len(image_files)} images from {images_dir}")
        cv_image = cv2.imread(image_files[0])
        if cv_image is None:
            logger.error(f"Failed to load image: {image_files[0]}")
            exit(-1)

        in_datas = []
        dynamic_inputs = []

        for input_name, input_cfg in self.inputs_cfg.items():
            if not self._has_resizer(input_name):
                shape = input_cfg["shape"]
                dtype_str = self.onnx_inputs_info[input_name]["dtype"]
                in_datas.append(torch.from_numpy(gen_random_data(shape, dtype_str)))
                continue

            input_mode = self.resizer_modes.get(input_name, 0)
            yuv_im, dyn_info = self._preprocess_for_resizer(
                cv_image, input_cfg, input_mode
            )
            in_datas.append(yuv_im)

            resizer_params = self._get_resizer_params(input_cfg)
            if not resizer_params["enable_static_resizer"]:
                dynamic_inputs.append(dyn_info)

        in_datas.extend(dynamic_inputs)
        return in_datas

    def _find_named_image_files(self, data_dir: str):
        """Find image files named by input (e.g., images_0.jpg)."""
        named_files = {}
        for input_name in self.inputs_cfg:
            for f in os.listdir(data_dir):
                if f.startswith(input_name + "_"):
                    ext = os.path.splitext(f)[1].lower()
                    if ext in SUPPORT_IMAGE_FORMATS or ext in [".npy", ".npz"]:
                        named_files.setdefault(input_name, []).append(
                            os.path.join(data_dir, f)
                        )

        for name in named_files:
            named_files[name].sort()

        return named_files if named_files else None

    def _load_named_images(self, named_files: dict):
        """Load calibration data from named image files."""
        min_samples = min(len(files) for files in named_files.values())
        logger.info(f"Loading {min_samples} samples from named files")

        in_datas = []
        dynamic_inputs = []

        for input_name, input_cfg in self.inputs_cfg.items():
            has_resizer = self._has_resizer(input_name)

            if input_name not in named_files:
                shape = input_cfg["shape"]
                dtype_str = self.onnx_inputs_info[input_name]["dtype"]
                in_datas.append(torch.from_numpy(gen_random_data(shape, dtype_str)))
                continue

            file_path = named_files[input_name][0]
            ext = os.path.splitext(file_path)[1].lower()

            if ext in SUPPORT_IMAGE_FORMATS:
                cv_image = cv2.imread(file_path)
                if cv_image is None:
                    logger.error(f"Failed to load image: {file_path}")
                    exit(-1)

                if has_resizer:
                    input_mode = self.resizer_modes.get(input_name, 0)
                    yuv_im, dyn_info = self._preprocess_for_resizer(
                        cv_image, input_cfg, input_mode
                    )
                    in_datas.append(yuv_im)

                    resizer_params = self._get_resizer_params(input_cfg)
                    if not resizer_params["enable_static_resizer"]:
                        dynamic_inputs.append(dyn_info)
                else:
                    # Preprocess for float16 input
                    data_format = input_cfg.get("data_format", "BGR")
                    params = self._get_input_preprocess_params(input_cfg)
                    im = default_preprocess(
                        cv_image,
                        (params["W"], params["H"]),
                        mean=params["mean"],
                        std=params["std"],
                        use_norm=True,
                        use_rgb=data_format == "RGB",
                        use_resize=True,
                        resize_type=params["resize_type"],
                    )
                    in_datas.append(torch.from_numpy(im.astype(np.float16)))

            elif ext in [".npy", ".npz"]:
                arr = (
                    np.load(file_path)[list(np.load(file_path).keys())[0]]
                    if ext == ".npz"
                    else np.load(file_path)
                )
                if arr.dtype == np.int64:
                    arr = arr.astype(np.int32)
                elif arr.dtype == np.float32:
                    arr = arr.astype(np.float16)
                in_datas.append(torch.from_numpy(arr))

                if (
                    has_resizer
                    and not self._get_resizer_params(input_cfg)["enable_static_resizer"]
                ):
                    dynamic_inputs.append(torch.zeros(1, 10, dtype=torch.int32))

        in_datas.extend(dynamic_inputs)
        return in_datas

    # ==================== Quantize and Build ====================

    @staticmethod
    def _log_section(title: str, char: str = "=", width: int = 60):
        """Log a section banner."""
        logger.info(f"{char * width}")
        logger.info(f" {title}")
        logger.info(f"{char * width}")

    @staticmethod
    def _log_kv(key: str, value, indent: int = 2):
        """Log a key-value pair with consistent formatting."""
        logger.info(f"{' ' * indent}{key}: {value}")

    def quantize(self):
        """Quantize the ONNX model for XH2 hardware."""
        self._log_section(f"Quantize: {self.model_name}")

        t_start = time.time()

        self._log_kv("model", self.model_path)
        self._log_kv("device", self.device)
        self._log_kv("quant_type", self.quant_type)
        self._log_kv("target", "XH2a")

        if not os.path.exists(self.quant_output_dir):
            os.makedirs(self.quant_output_dir)

        if hasattr(self, "ApplicationOnnxOpt"):
            self.ApplicationOnnxOpt.opt()
            if hasattr(self.ApplicationOnnxOpt, "opt_model_path"):
                self.model_path = self.ApplicationOnnxOpt.opt_model_path

        try:
            from xhquant.api import (
                xhquant_init,
                DeviceType,
                HMONNXGoldenInference,
                convert_onnx_to_hmonnx,
            )

            xhquant_init(logger=logger)
            os.environ.setdefault("PYDEVD_DISABLE_FILE_VALIDATION", "1")
        except ImportError as e:
            logger.error(f"{e}\nNot found xhquant module, please install xhquant.")
            exit(-1)

        # Load calibration data
        logger.info("")
        in_datas = self.get_input_data()

        quant_onnx_model_path = os.path.join(
            self.quant_output_dir, f"{self.model_name}.onnx"
        )

        # Quantization step
        logger.info("Converting ONNX to hmonnx...")
        convert_onnx_to_hmonnx(
            self.model_path,
            in_datas,
            device_type=DeviceType.XH2a,
            out_hmonnx_file=quant_onnx_model_path,
            quant_config=self.get_quant_cfg(),
            input_names=self.inputs_name,
            output_names=self.outputs_name,
        )

        # Generate chip required format model
        try:
            logger.info("Generating golden data...")
            session = HMONNXGoldenInference(quant_onnx_model_path)
            session.to(self.device)
            session.save_golden = True
            session.golden_dir = self.golden_dir
            if os.path.exists(self.golden_dir):
                shutil.rmtree(self.golden_dir)
            session.step = 0

            # Convert dtypes for inference
            for idx, in_data in enumerate(in_datas):
                if not self.inputs_name[idx].startswith("resizer_crop_"):
                    if in_data.dtype == torch.int64:
                        in_datas[idx] = in_data.type(torch.int32).to(self.device)
                    elif in_data.dtype == torch.float32:
                        in_datas[idx] = in_data.half().to(self.device)

            session(*in_datas)
        except Exception as e:
            logger.error(f"Error occurred while generating golden data: \n{e}")

        if os.path.exists(quant_onnx_model_path):
            os.remove(quant_onnx_model_path)
        shutil.copytree(
            os.path.join(os.path.abspath(self.golden_dir), "step_0"),
            self.quant_output_dir,
            dirs_exist_ok=True,
        )
        shutil.rmtree(self.golden_dir)

        span = time.time() - t_start
        self._log_section("Quantize Complete", char="-")
        self._log_kv("hmonnx", self.quant_onnx_model_path)
        self._log_kv("time", f"{span:.2f}s")

        return {
            "quant": {
                "success": True,
                "time": round(span, 2),
                "quant_type": self.quant_type,
                "hmonnx": self.quant_onnx_model_path,
            }
        }

    def build(self, enable_profile=False, upload_dir_name=None, file_prefix=None):
        """Build the quantized model to HMM format for XH2 hardware.

        Args:
            enable_profile: Enable profiling during build
            upload_dir_name: Optional external upload_dir_name for upload directory
            file_prefix: Optional file prefix for compressed file name
        """
        self._log_section(f"Build: {self.model_name}")

        t_start = time.time()

        self.enable_profile = enable_profile
        self._log_kv("hmonnx", self.quant_onnx_model_path)
        self._log_kv("ncore", self.build_ncore)
        self._log_kv("opt_level", f"{self.build_opt_level}")
        self._log_kv("batch", self.build_batch if self.roi_num == 1 else self.roi_num)
        self._log_kv("target", "xh2")

        if not os.path.exists(self.build_output_dir):
            os.makedirs(self.build_output_dir)

        try:
            import tcim
        except ImportError:
            logger.error("Not found tcim module, please install tcim first!")
            exit(-1)

        logger.info("Building HMM model...")
        tcim.build_from_hmonnx(
            self.quant_onnx_model_path,
            output_name=self.hmm_name,
            ncore=self.build_ncore,
            opt_level=f"{self.build_opt_level}",
            target="xh2",
            batch=self.build_batch if self.roi_num == 1 else self.roi_num,
            enable_profile=enable_profile,
            output_dir=self.hmm_save_dir,
            work_dir=self.build_output_dir,
            one_img_multi_roi=self.roi_num > 1,
            j=psutil.cpu_count(logical=False),
            custom_msg=json.dumps(self.custom_msg, ensure_ascii=False),
        )

        span = time.time() - t_start
        self._log_section("Build Complete", char="-")
        self._log_kv("hmm", self.hmm_path)
        self._log_kv("time", f"{span:.2f}s")

        res_info = {
            "build": {
                "success": True,
                "time": round(span, 2),
                "ncore": self.build_ncore,
                "opt_level": self.build_opt_level,
                "batch": self.build_batch if self.roi_num == 1 else self.roi_num,
                "hmm": self.hmm_path,
            }
        }

        # Compress and upload compiled outputs
        self.upload_hmm(upload_dir_name, file_prefix)

        return res_info

    def upload_hmm(self, upload_dir_name=None, file_prefix=None):
        """Upload compiled HMM model to artifactory.

        Args:
            upload_dir_name: Optional external upload_dir_name to override self.upload_dir_name (for upload directory)
            file_prefix: Optional file prefix for compressed file name (defaults to upload_dir_name)
        """
        if not self.enable_upload:
            logger.info("Upload is disabled.")
            return

        # Use external upload_dir_name if provided (for upload directory)
        if upload_dir_name is not None:
            self.upload_dir_name = upload_dir_name

        # Use file_prefix for filename, defaults to upload_dir_name
        actual_file_prefix = (
            file_prefix if file_prefix is not None else self.upload_dir_name
        )

        if not os.path.exists(self.hmm_path):
            logger.error(f"HMM file not found: {self.hmm_path}")
            return

        logger.info("Compressing hmmodel...")
        now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        hmcc_version = get_package_version("houmo-tcim-xh2")
        runtime_version = get_package_version("houmo_tcim_runtime_xh2")
        with open(os.path.join(self.save_dir, "xh2", "VERSION.txt"), "w") as f:
            f.write(f"hmquant_version: {get_hmquant_xh2_version()}\n")
            f.write(f"tcim_version: {hmcc_version}\n")
            f.write(f"tcim_runtime_version: {runtime_version}\n")
            f.write(f"build_time: {now}\n")
        # Use actual_file_prefix for filename
        filename = f"{actual_file_prefix}_xh2_b{self.hmm_batch}_{self.build_ncore}core_{self.build_opt_level}_{get_houmo_version()}.tar.xz"
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
        # Upload path uses upload_dir_name, filename uses actual_file_prefix
        upload_file_to_artifactory(
            compress_hmm_path,
            f"models/{self.target.lower()}-{get_houmo_version()}/{self.upload_dir_name}/{filename}",
            max_retries=3,
        )
        logger.info("Upload hmmodel done.")

    # ==================== Check and Compare ====================

    def check_golden(self, device_id=0, enable_layers=False):
        """Check the golden data against the hardware model outputs."""
        self._log_section(f"Check Golden: {self.model_name}")

        t_start = time.time()

        if enable_layers:
            self.quant_onnx_model_path = self.add_node_output_as_graph_output(
                self.quant_onnx_model_path, "xh2"
            )
            self.build_output_dir += "_debug"
            self.hmm_name += "_debug"
            self.hmm_path = os.path.join(self.hmm_save_dir, f"{self.hmm_name}.hmm")
            if not os.path.exists(self.hmm_path):
                logger.info("Rebuilding hmmodel with all layers output...")
                self.build(enable_profile=False)

        self._log_kv("hmm", self.hmm_path)
        self._log_kv("batch", self.build_batch)
        self._log_kv("device_id", device_id)

        xh2 = Xh2Infer()
        xh2.load(self.hmm_path, device_id=device_id)
        in_datas = {}

        # Load golden inputs
        logger.info("Loading golden inputs:")
        for input_name in self.inputs_cfg:
            new_name = input_name.replace("/", "_")
            golden_input_path = os.path.join(
                self.quant_output_dir, f"hmquant_{self.model_name}_{new_name}_input.npy"
            )
            golden_input = np.load(golden_input_path)
            self._log_kv(
                f"{input_name}",
                f"shape={list(golden_input.shape)}, dtype={golden_input.dtype}",
            )

            input_mode = self.resizer_modes.get(input_name, 0)
            if input_mode in [1, 2, 3]:
                fmt = xh2.inputs_format[input_name]
                golden_input = golden_input.flatten()
                size = (
                    golden_input.size // 2
                    if fmt == "YUV420SP"
                    else (
                        golden_input.size * 3 // 2
                        if fmt == "YUV422SP"
                        else golden_input.size
                    )
                )
                golden_input = golden_input[:size].reshape(1, size)

            golden_input = np.repeat(golden_input, self.build_batch, axis=0)
            in_datas[input_name] = golden_input

            # Handle dynamic crop parameters
            if input_mode in [1, 2]:
                resizer_name = f"resizer_crop_{input_name}"
                golden_dyn_path = os.path.join(
                    self.quant_output_dir,
                    f"hmquant_{self.model_name}_resizer_crop_{new_name}_input.npy",
                )
                if os.path.exists(golden_dyn_path):
                    golden_dyn_input = np.load(golden_dyn_path)
                    self._log_kv(
                        f"{resizer_name}",
                        f"shape={list(golden_dyn_input.shape)}, dtype={golden_dyn_input.dtype}",
                    )
                    repeats = (
                        self.roi_num
                        if self.roi_num > 1
                        else self.build_batch if self.build_batch > 1 else 1
                    )
                    in_datas[resizer_name] = np.repeat(
                        golden_dyn_input, repeats=repeats, axis=0
                    )

        # Run inference
        logger.info("Running XH2 inference...")
        outputs, _ = xh2.run(in_datas)
        self.save_profile_data(outputs)

        repeats = (
            self.build_batch
            if self.build_batch > 1 and self.roi_num == 1
            else self.roi_num if self.roi_num > 1 else 1
        )

        # Compare outputs
        logger.info("Loading golden outputs:")
        table = PrettyTable(["name", "cosine_dist"])
        table.title = "xh2 vs hmquant"
        outputs_result = {}

        for output_name in outputs:
            new_name = output_name.replace("/", "_")
            golden_dir = (
                f"hmquant_{self.model_name}_with_act"
                if enable_layers
                else self.quant_output_dir
            )
            golden_output_path = (
                os.path.join(golden_dir, f"{new_name}.npy")
                if enable_layers
                else os.path.join(
                    self.quant_output_dir,
                    f"hmquant_{self.model_name}_{new_name}_output.npy",
                )
            )
            golden_output = np.load(golden_output_path)
            self._log_kv(
                f"{output_name}",
                f"shape={list(golden_output.shape)}, dtype={golden_output.dtype}",
            )

            golden_output = np.repeat(golden_output, repeats=repeats, axis=0)
            output = outputs[output_name]
            dist = cosine_distance(golden_output, output)

            table.add_row([output_name, f"{dist:.6f}"])
            outputs_result[output_name] = {
                "cosine_dist": float(dist),
                "md5": get_md5(output),
                "golden_md5": get_md5(golden_output),
            }

        span = time.time() - t_start
        logger.info(f"\n{table}")
        self._log_section("Check Golden Complete", char="-")
        self._log_kv("time", f"{span:.2f}s")

        return {"outputs": outputs_result}

    def compare(self, data_path: str, device_id=0):
        """Compare outputs from ONNX, HmQuant and XH2 inference."""
        self._log_section(f"Compare: {self.model_name}")

        t_start = time.time()

        self._log_kv("data_path", data_path)
        self._log_kv("device_id", device_id)

        # Load models
        logger.info("\nLoading models:")
        onnx_infer = OnnxInfer()
        onnx_infer.load(self.model_path)
        self._log_kv("onnx", self.model_path)

        hmquant_infer = Xh2HmQuantInfer()
        hmquant_infer.load(self.quant_onnx_model_path)
        self._log_kv("hmquant", self.quant_onnx_model_path)

        xh2_infer = Xh2Infer()
        xh2_infer.load(self.hmm_path)
        self._log_kv("xh2", self.hmm_path)

        onnx_in_datas = {}
        hmquant_in_datas = {}
        xh2_in_datas = {}
        _, ext = os.path.splitext(os.path.basename(data_path))

        if self.is_image_single_input and ext in SUPPORT_IMAGE_FORMATS:
            logger.info("Preparing single image input...")
            self._prepare_single_image_compare(
                data_path, onnx_in_datas, hmquant_in_datas, xh2_in_datas, xh2_infer
            )
        else:
            logger.info("Preparing multi-input data...")
            self._prepare_multi_input_compare(
                data_path, onnx_in_datas, hmquant_in_datas, xh2_in_datas
            )

        # Run inference
        logger.info("Running inference:")
        logger.info("  ONNX inference...")
        onnx_outputs = onnx_infer.run(onnx_in_datas)
        logger.info("  HmQuant inference...")
        hmquant_outputs = hmquant_infer.run(hmquant_in_datas)
        logger.info("  XH2 inference...")
        xh2_outputs, xh2_outputs_dequanted = xh2_infer.run(xh2_in_datas)
        self.save_profile_data(xh2_outputs)

        repeats = (
            self.build_batch
            if self.build_batch > 1 and self.roi_num == 1
            else self.roi_num if self.roi_num > 1 else 1
        )

        # Compare results
        table = PrettyTable(
            ["name", "onnx vs hmquant", "onnx vs xh2", "hmquant vs xh2"]
        )
        table.title = "Cosine Distance"
        outputs_result = {}

        for output_name in onnx_outputs:
            onnx_out = np.repeat(onnx_outputs[output_name], repeats=repeats, axis=0)
            hmquant_out = np.repeat(
                hmquant_outputs[output_name], repeats=repeats, axis=0
            )
            xh2_out = np.split(
                xh2_outputs_dequanted[output_name], self.build_batch, axis=0
            )[0]

            onnx_vs_hmquant = cosine_distance(onnx_out, hmquant_out)
            onnx_vs_xh2 = cosine_distance(onnx_out, xh2_out)
            hmquant_vs_xh2 = cosine_distance(hmquant_out, xh2_out)

            table.add_row(
                [
                    output_name,
                    f"{onnx_vs_hmquant:.6f}",
                    f"{onnx_vs_xh2:.6f}",
                    f"{hmquant_vs_xh2:.6f}",
                ]
            )
            outputs_result[output_name] = {
                "onnx_vs_hmquant": float(onnx_vs_hmquant),
                "onnx_vs_xh2": float(onnx_vs_xh2),
                "hmquant_vs_xh2": float(hmquant_vs_xh2),
            }

        span = time.time() - t_start
        logger.info(f"\n{table}")
        self._log_section("Compare Complete", char="-")
        self._log_kv("time", f"{span:.2f}s")

        return {
            "compare": {
                "success": True,
                "data_path": data_path,
                "outputs": outputs_result,
            }
        }

    def _prepare_single_image_compare(
        self, data_path, onnx_in_datas, hmquant_in_datas, xh2_in_datas, xh2_infer
    ):
        """Prepare data for single image comparison."""
        input_name = self.inputs_name[0]
        input_mode = self.resizer_modes.get(input_name, 0)
        input_cfg = self.inputs_cfg[input_name]
        params = self._get_input_preprocess_params(input_cfg)
        resizer_params = self._get_resizer_params(input_cfg)

        if not os.path.exists(data_path):
            logger.error(f"Not found data_path: {data_path}")
            exit(-1)

        cv_image = cv2.imread(
            data_path,
            (
                cv2.IMREAD_COLOR
                if params["data_format"] != "GRAY"
                else cv2.IMREAD_GRAYSCALE
            ),
        )
        if cv_image is None:
            logger.error("Failed to decode image")
            exit(-1)

        hmm_batch = xh2_infer.inputs_info[input_name].shape[0]

        # ONNX preprocessing
        onnx_data, _ = resizer_preprocess(
            cv_image,
            params["shape"],
            resizer_params["max_input_size"],
            mean=params["mean"],
            std=params["std"],
            use_resize=True,
            use_norm=True,
            use_rgb=params["data_format"] == "RGB",
            resize_type=params["resize_type"],
            padding_mode=params["padding_mode"],
            padding_values=params["padding_values"],
            is_onnx=True,
        )
        onnx_data = np.repeat(onnx_data.numpy(), self.model_input_batch, axis=0)
        onnx_in_datas[input_name] = onnx_data

        # Resizer preprocessing
        yuv_im, dyn_info = self._preprocess_for_resizer(cv_image, input_cfg, input_mode)

        if input_mode in [1, 2, 3]:
            # yuv_im is NCHW format: (N, C, H, W)
            n, c, h, w = yuv_im.shape
            yuv_pad = yuv_im.repeat_interleave(self.model_input_batch, dim=0)
            hmquant_in_datas[input_name] = yuv_pad.contiguous()

            # XH2 format - flatten to 1D
            yuv_flat = yuv_im.detach().cpu().numpy().flatten()
            valid_len = (
                yuv_flat.size // 2
                if resizer_params["toYUV_format"] == "YUV420SP"
                else (
                    yuv_flat.size * 2 // 3
                    if resizer_params["toYUV_format"] == "YUV422SP"
                    else yuv_flat.size
                )
            )
            yuv = np.repeat(yuv_flat[:valid_len].reshape(1, -1), hmm_batch, axis=0)
            xh2_in_datas[input_name] = np.ascontiguousarray(yuv)
        else:
            # Disable resizer
            in_data = np.repeat(onnx_data, self.build_batch, axis=0).astype(np.float16)
            hmquant_in_datas[input_name] = torch.from_numpy(
                in_data[: self.model_input_batch]
            ).cpu()
            xh2_in_datas[input_name] = np.ascontiguousarray(in_data)

        # Dynamic resizer info
        if input_mode in [1, 2]:
            hmquant_dyn = (
                dyn_info
                if self.roi_num > 1
                else dyn_info.repeat_interleave(self.model_input_batch, dim=0)
            )
            xh2_dyn = dyn_info.repeat_interleave(
                self.roi_num if self.roi_num > 1 else hmm_batch, dim=0
            )
            hmquant_in_datas[f"resizer_crop_{input_name}"] = hmquant_dyn
            xh2_in_datas[f"resizer_crop_{input_name}"] = xh2_dyn.detach().cpu().numpy()

    def _prepare_multi_input_compare(
        self, data_path, onnx_in_datas, hmquant_in_datas, xh2_in_datas
    ):
        """Prepare data for multi-input comparison."""
        in_datas = load_npz(data_path)

        for input_name in self.inputs_cfg:
            if input_name not in in_datas:
                logger.error(f"Input '{input_name}' not found in NPZ file: {data_path}")
                exit(-1)

            raw_data = in_datas[input_name]
            input_mode = self.resizer_modes.get(input_name, 0)
            input_cfg = self.inputs_cfg[input_name]

            if input_mode != 0:
                # Input with resizer
                cv_image = self._ensure_hwc_uint8(raw_data)
                params = self._get_input_preprocess_params(input_cfg)

                # ONNX preprocessing
                onnx_data = self._preprocess_for_onnx(cv_image, input_cfg)
                onnx_in_datas[input_name] = onnx_data

                # Resizer preprocessing
                yuv_im, dyn_info = self._preprocess_for_resizer(
                    cv_image, input_cfg, input_mode
                )
                yuv_data = (
                    yuv_im.detach().cpu().numpy()
                    if hasattr(yuv_im, "detach")
                    else yuv_im
                )

                hmquant_in_datas[input_name] = torch.from_numpy(yuv_data[0:1].copy())
                xh2_in_datas[input_name] = np.repeat(yuv_data, self.build_batch, axis=0)

                if input_mode in [1, 2]:
                    hmquant_in_datas[f"resizer_crop_{input_name}"] = dyn_info
                    xh2_in_datas[f"resizer_crop_{input_name}"] = (
                        dyn_info.detach().cpu().numpy()
                    )
            else:
                # mode == 0: Resizer disabled
                data_format = input_cfg.get("data_format")

                if data_format is not None:
                    # Image input with resizer disabled
                    cv_image = self._ensure_hwc_uint8(raw_data)

                    # Preprocess for all backends
                    preprocessed = self._preprocess_for_onnx(cv_image, input_cfg)
                    onnx_in_datas[input_name] = preprocessed

                    hmquant_data = preprocessed.astype(np.float16)
                    hmquant_in_datas[input_name] = torch.from_numpy(hmquant_data)
                    xh2_in_datas[input_name] = np.repeat(
                        hmquant_data, self.build_batch, axis=0
                    )
                else:
                    # Non-image input
                    proc_data = raw_data.copy()
                    if proc_data.dtype == np.int64:
                        proc_data = proc_data.astype(np.int32)
                    elif proc_data.dtype == np.float32:
                        proc_data = proc_data.astype(np.float16)

                    onnx_in_datas[input_name] = (
                        raw_data.astype(np.float32)
                        if raw_data.dtype == np.float16
                        else raw_data
                    )
                    hmquant_in_datas[input_name] = torch.from_numpy(proc_data.copy())
                    xh2_in_datas[input_name] = np.repeat(
                        proc_data, self.build_batch, axis=0
                    )

    # ==================== Static Utility Methods ====================

    @staticmethod
    def check_golden_from_hmm(hmm, golden_dir, enable_layers=False, device_id=0):
        """Check model inference results against golden data consistency."""
        if not os.path.exists(hmm):
            logger.error(f"Not found hmm model: {hmm}")
            return {}
        if not os.path.exists(golden_dir):
            logger.error(f"Not found golden data directory: {golden_dir}")
            return {}

        try:
            xh2 = Xh2Infer()
            xh2.load(hmm, device_id=device_id)
        except Exception as e:
            logger.error(f"Failed to load hmm model: {e}")
            return {}

        input_names = (
            list(xh2.inputs_info.keys()) if hasattr(xh2, "inputs_info") else []
        )
        output_names = []
        if hasattr(xh2, "engine") and xh2.engine:
            output_names = [
                xh2.engine.get_output_name(i)
                for i in range(xh2.engine.get_num_outputs())
            ]

        logger.info(f"Model input names: {input_names}")
        logger.info(f"Model output names: {output_names}")

        input_files_map = find_input_files(golden_dir, input_names)
        output_files_map = find_output_files(golden_dir, output_names)

        missing = [n for n, f in input_files_map.items() if not f] + [
            n for n, f in output_files_map.items() if not f
        ]
        if missing:
            logger.error(f"Missing files: {missing}")
            return {}

        input_data = {}
        for name, paths in input_files_map.items():
            if paths:
                try:
                    data = np.load(paths[0])
                    input_data[name] = data
                    logger.info(
                        f"Loaded input: {name}, shape={data.shape}, from={paths[0]}"
                    )
                except Exception as e:
                    logger.error(f"Failed to load {paths[0]}: {e}")
                    return {}

        golden_outputs = {}
        for name, paths in output_files_map.items():
            if paths:
                try:
                    golden_outputs[name] = np.load(paths[0])
                    logger.info(
                        f"Loaded golden output: {name}, shape={golden_outputs[name].shape}"
                    )
                except Exception as e:
                    logger.error(f"Failed to load {paths[0]}: {e}")
                    return {}

        try:
            logger.info("Running inference...")
            outputs, _ = xh2.run(input_data)
            logger.info("Inference completed")
        except Exception as e:
            logger.error(f"Inference failed: {e}")
            return {}

        logger.info("Calculating cosine similarity...")
        similarity_results = {}
        max_name_len = max(len(n) for n in outputs.keys()) if outputs else 15

        for output_name, output_data in outputs.items():
            if output_name in golden_outputs:
                golden_data = golden_outputs[output_name]
                if golden_data.shape != output_data.shape:
                    logger.error(
                        f"Shape mismatch for {output_name}: {golden_data.shape} vs {output_data.shape}"
                    )
                    similarity_results[output_name] = {
                        "cosine_similarity": None,
                        "reason": "Shape mismatch",
                    }
                    continue

                similarity = cosine_distance(golden_data, output_data)
                similarity_results[output_name] = {
                    "cosine_similarity": float(similarity)
                }
                logger.info(
                    f"{output_name:<{max_name_len}}: Cosine similarity = {similarity:.6f}"
                )
            else:
                similarity_results[output_name] = {
                    "cosine_similarity": None,
                    "reason": "No golden data",
                }

        valid = {
            k: v
            for k, v in similarity_results.items()
            if v["cosine_similarity"] is not None
        }
        if valid:
            avg = np.mean([v["cosine_similarity"] for v in valid.values()])
            logger.info(
                f"Average cosine similarity: {avg:.6f} ({len(valid)}/{len(similarity_results)} outputs)"
            )

        return similarity_results

    @staticmethod
    def gen_golden(
        hmonnx: str,
        output: str,
        data_path=None,
        enable_layers=False,
        target="xh2",
        **kwargs,
    ):
        """Generate golden data from hmonnx model."""
        if not os.path.exists(hmonnx):
            logger.error(f"Not found hmonnx model: {hmonnx}")
            return
        if not os.path.exists(output):
            os.makedirs(output)

        try:
            if enable_layers:
                hmonnx = BaseExec.add_node_output_as_graph_output(hmonnx, target)

            model = Xh2HmQuantInfer()
            model.load(hmonnx)

            if data_path is not None:
                if os.path.splitext(data_path)[1] != ".npz":
                    logger.error(f"Invalid data file: {data_path}")
                    return
                in_datas = load_npz(data_path)
            else:
                in_datas = {}
                for name in model.input_names:
                    in_datas[name] = model.get_random_input_data(name)
                    npy_path = os.path.join(output, f"{name}.npy")
                    np.save(npy_path, in_datas[name])
                    logger.info(
                        f"Generated input: {name}, shape={in_datas[name].shape}, to={npy_path}"
                    )

            outputs = model.run(in_datas, dequant=False)
            for name, data in outputs.items():
                npy_path = os.path.join(output, f"{name}.npy")
                np.save(npy_path, data)
                logger.info(
                    f"Generated output: {name}, shape={data.shape}, to={npy_path}"
                )

        except Exception as e:
            logger.error(f"Failed to process hmonnx: {e}")

    def build_from_hmonnx(
        self,
        hmonnx,
        hmm_name=None,
        output="output",
        ncore=1,
        opt_level=2,
        batch=1,
        enable_profile=False,
        roi_num=1,
        flash_attn=0,
        llm_opt=False,
        enable_common_subgraph=False,
        skip_mlir_compile=False,
        subgraph_repeat_hint=20,
        target="xh2",
        **kwargs,
    ):
        """Build HMM model from hmonnx."""
        try:
            import tcim
        except ImportError:
            logger.error("Not found tcim module, please install tcim first!")
            return

        if hmm_name is None:
            hmm_name = os.path.splitext(os.path.basename(hmonnx))[0]

        output_dir = os.path.join(output, "xh2")
        work_dir = os.path.join(output_dir, "tcim")
        custom_msg = {
            "opt_level": opt_level,
            "ncore": ncore,
            "target": "xh2",
            "llm_opt": llm_opt,
            "skip_mlir_compile": skip_mlir_compile,
            "enable_common_subgraph": enable_common_subgraph,
            "subgraph_repeat_hint": subgraph_repeat_hint,
            "flash_attention": flash_attn,
        }

        tcim.build_from_hmonnx(
            hmonnx,
            output_name=hmm_name,
            ncore=ncore,
            opt_level=f"O{opt_level}",
            target="xh2",
            batch=batch,
            enable_profile=enable_profile,
            output_dir=output_dir,
            work_dir=work_dir,
            one_img_multi_roi=False,
            llm_opt=llm_opt,
            skip_mlir_compile=skip_mlir_compile,
            enable_common_subgraph=enable_common_subgraph,
            subgraph_repeat_hint=subgraph_repeat_hint,
            flash_attention=flash_attn,
            j=psutil.cpu_count(logical=False),
            custom_msg=json.dumps(custom_msg, ensure_ascii=False),
        )

        return os.path.join(output_dir, f"{hmm_name}.hmm")
