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
import traceback
import torch
from prettytable import PrettyTable
from datetime import datetime
from ..base.base_exec import BaseExec
from ..infer.onnx_infer import OnnxInfer
from ..infer.xh2_infer import Xh2Infer
from ..infer.xhquant_infer import Xh2HmQuantInfer
from ..utils import logger
from ..utils.dist_metrics import cosine_distance
from ..utils.preprocess import (
    calc_padding_size,
    convert_bgr_to_yuv,
    default_preprocess,
    resizer_preprocess,
)
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

    XH2 Resizer Constraints (1.3.0):
        Common constraints:
        - H/W方向的输入图片/crop_size/crop_start/输出 都要是2的倍数
        - 输出H最大4096，输出W最大1024

        STATIC mode (resizer_mode=3):
        - 输入W最大1024
        - 缩放倍数范围 [1/32, 16]

        DYNAMIC_V2 mode (resizer_mode=1):
        - 输入W最大4096
        - 放大倍数最大16，H缩小倍数最大32，W缩小倍数最大8(YUV444最大4)
        - Padding支持H或W单方向，pad大小需是偶数且不超过16
        - 不支持one image multi roi (roi_num必须为1)
    """

    def __init__(self, cfg: dict) -> None:
        super().__init__(cfg)
        self.quant_type = self.quant_cfg.get("quant_type", "w8a8h1_sefp")
        self.mix_search_cfg = self.quant_cfg.get(
            "mix_search"
        )  # Optional mix_search config
        self.golden_dir = os.path.abspath(os.path.join(self.quant_output_dir, "golden"))
        self.upgrade_opset_version()

        # Get ONNX output node names
        self.outputs_name = list(self.onnx_outputs_info.keys())

        # Append dynamic parameter input names for inputs with dynamic resizer (mode 1 or 2)
        for input_name in self.inputs_cfg:
            if self.resizer_modes[input_name] in [1, 2]:
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

    def _get_resizer_cfg(self, input_cfg: dict):
        """Extract resizer parameters from input config."""
        resizer_cfg = input_cfg.get("resizer")
        if resizer_cfg is None:
            return dict()
        return resizer_cfg

    def _preprocess_for_resizer(
        self, cv_image: np.ndarray, input_cfg: dict, resizer_mode: int
    ):
        """Preprocess image through resizer pipeline.

        Returns:
            tuple: (yuv_im, dyn_info) - yuv image and dynamic info (if applicable)
        """
        N, C, H, W = input_cfg["shape"]
        resizer_cfg = self._get_resizer_cfg(input_cfg)
        toYUV_format = resizer_cfg.get("toYUV_format", "YUV420SP")
        resizer_input_size = resizer_cfg.get("resizer_input_size", [H, W])
        resizer_input_h, resizer_input_w = resizer_input_size
        resizer_crop = resizer_cfg.get(
            "resizer_crop", [0, 0, resizer_input_h, resizer_input_w]
        )
        yuv_im: torch.Tensor
        dyn_info: torch.Tensor
        yuv_im, dyn_info = resizer_preprocess(
            cv_image,
            input_cfg["shape"],
            resizer_input_size=resizer_input_size,
            resizer_crop=resizer_crop,
            resizer_mode=resizer_mode,
            mean=input_cfg["mean"],
            std=input_cfg["std"],
            use_resize=resizer_mode in [0, 3],
            use_norm=resizer_mode == 0,
            use_rgb=input_cfg["data_format"] == "RGB" and resizer_mode == 0,
            resize_type=input_cfg["resize_type"],
            padding_mode=input_cfg.get("padding_mode"),
            padding_values=input_cfg.get("padding_values"),
            is_onnx=resizer_mode == 0,
            to_YUV=resizer_mode in [1, 2, 3],
            fmt=toYUV_format,
        )
        return yuv_im, dyn_info

    def _preprocess_for_onnx(self, cv_image, input_cfg: dict):
        """Preprocess image for ONNX inference (resize + normalize, no YUV)."""
        data_format = input_cfg["data_format"]
        resize_type = input_cfg["resize_type"]
        N, C, H, W = input_cfg["shape"]
        return default_preprocess(
            cv_image,
            size=(W, H),
            mean=input_cfg["mean"],
            std=input_cfg["std"],
            use_norm=True,
            use_rgb=(data_format == "RGB"),
            use_resize=True,
            resize_type=resize_type,
            padding_mode=input_cfg.get("padding_mode"),
            padding_value=input_cfg.get("padding_values"),
            to_YUV=False,
        )

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
            logger.fatal(
                f"{traceback.format_exc()}\nNot found xhquant module, please install xhquant."
            )

        input_ppc_config = []
        for input_name in self.inputs_cfg:
            input_cfg = self.inputs_cfg[input_name]
            resizer_mode = self.resizer_modes[input_name]
            if resizer_mode == 0:
                input_ppc_config.append("float16")
                continue
            # Resizer
            _, _, H, W = input_cfg["shape"]
            mean, std = input_cfg["mean"], input_cfg["std"]
            data_format = input_cfg["data_format"]
            resize_type = input_cfg["resize_type"]
            # padding_mode = input_cfg.get("padding_mode")
            padding_values = input_cfg.get("padding_values")

            resizer_cfg = self._get_resizer_cfg(input_cfg)
            toYUV_format = resizer_cfg.get("toYUV_format", "YUV420SP")
            resizer_input_size = resizer_cfg.get("resizer_input_size", [H, W])
            resizer_input_h, resizer_input_w = resizer_input_size
            resizer_crop = resizer_cfg.get(
                "resizer_crop", [0, 0, resizer_input_h, resizer_input_w]
            )
            crop_y, crop_x, crop_h, crop_w = resizer_crop
            crop_offset = (crop_y, crop_x)
            crop_size = (crop_h, crop_w)
            pad_value = 0
            pad_size = (0, 0, 0, 0)

            if resize_type == 1 and resizer_mode == 1:
                pad_value = padding_values[0]

            input_ppc_config.append(
                ResizerScheme(
                    size=(H, W),
                    mode="bilinear",
                    align_corners=False,
                    fmt=self.get_format(toYUV_format),
                    int_trans=True,
                    crop_size=crop_size,
                    crop_offset=crop_offset,
                    pad_size=pad_size,
                    pad_value=pad_value,
                    mean=[v / 255.0 for v in mean],
                    std=[v / 255.0 for v in std],
                    dynamic_crop=resizer_mode in [1, 2],
                    model_inp_fmt=data_format.lower(),
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

        for input_name in self.inputs_cfg:
            input_cfg = self.inputs_cfg[input_name]
            resizer_mode = self.resizer_modes[input_name]
            shape = input_cfg["shape"]
            dtype_str = self.onnx_inputs_info[input_name]["dtype"]
            if resizer_mode == 0:
                in_datas.append(torch.from_numpy(gen_random_data(shape, dtype_str)))
                continue

            N, C, H, W = input_cfg["shape"]
            resizer_cfg = self._get_resizer_cfg(input_cfg)
            toYUV_format = resizer_cfg.get("toYUV_format", "YUV420SP")
            resizer_input_size = resizer_cfg.get("resizer_input_size", [H, W])
            resizer_input_h, resizer_input_w = resizer_input_size

            random_bgr = torch.from_numpy(
                gen_random_data([N, C, resizer_input_h, resizer_input_w], "uint8")
            )
            random_yuv = convert_bgr_to_yuv(random_bgr, toYUV_format, to_NCHW=True)
            in_datas.append(random_yuv)

            if resizer_mode == 1:
                dynamic_inputs.append(
                    torch.tensor(
                        [[0, 0, resizer_input_h, resizer_input_w, H, W, 0, 0, 0, 0]],
                        dtype=torch.int32,
                    )
                )
            elif resizer_mode == 2:
                dynamic_inputs.append(
                    torch.tensor(
                        [[0, 0, resizer_input_h, resizer_input_w]], dtype=torch.int32
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
                logger.fatal(f"Not found calib_data path: {calib_data}")
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
            logger.fatal(f"Not found calib data in {calib_data}")
        return data_list[0]

    def _load_npz_data(self, data_path):
        """Load multi-input data from NPZ file."""
        npz_data = load_npz(data_path)
        in_datas = []
        dynamic_params = {}

        for input_name in self.inputs_cfg:
            if input_name not in npz_data:
                logger.fatal(f"Input '{input_name}' not found in NPZ file: {data_path}")

            resizer_mode = self.resizer_modes[input_name]
            input_cfg = self.inputs_cfg[input_name]
            raw_data = npz_data[input_name]

            if resizer_mode != 0:
                # Input with resizer
                cv_image = self._ensure_hwc_uint8(raw_data)
                yuv_im, dyn_info = self._preprocess_for_resizer(
                    cv_image, input_cfg, resizer_mode
                )
                in_datas.append(yuv_im)

                if resizer_mode in [1, 2]:
                    dynamic_params[input_name] = dyn_info
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
        resizer_mode = self.resizer_modes[input_name]

        cv_image = cv2.imread(data_path)
        if cv_image is None:
            logger.fatal(f"Failed to load image: {data_path}")

        yuv_im, dyn_info = self._preprocess_for_resizer(
            cv_image, input_cfg, resizer_mode
        )
        in_datas = [yuv_im]

        if resizer_mode in [1, 2]:
            in_datas.append(dyn_info)

        return in_datas

    # ==================== Quantize and Build ====================

    @staticmethod
    def _log_section(title: str, char: str = "=", width: int = 60):
        """Log a section banner."""
        logger.info(f"{char * width}")
        logger.info(f" {title}")
        logger.info(f"{char * width}")

    def quantize(self):
        """Quantize the ONNX model for XH2 hardware."""
        self._log_section(f"Quantize: {self.model_name}")

        t_start = time.time()

        logger.info(f"  model: {self.model_path}")
        logger.info(f"  device: {self.device}")
        logger.info(f"  quant_type: {self.quant_type}")
        logger.info(f"  target: XH2a")

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
            logger.fatal(
                f"{traceback.format_exc()}\nNot found xhquant module, please install xhquant."
            )

        # Load calibration data
        logger.info("")
        in_datas = self.get_input_data()

        quant_onnx_model_path = os.path.join(
            self.quant_output_dir, f"{self.model_name}.onnx"
        )

        # Quantization step
        logger.info("Converting ONNX to hmonnx...")
        if self.mix_search_cfg is not None:
            logger.info(f"  mix_search: enabled")
            logger.info(f" {self.mix_search_cfg}")
        convert_onnx_to_hmonnx(
            self.model_path,
            in_datas,
            device_type=DeviceType.XH2a,
            out_hmonnx_file=quant_onnx_model_path,
            quant_config=self.get_quant_cfg(),
            input_names=self.inputs_name,
            output_names=self.outputs_name,
            mix_search=self.mix_search_cfg,
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
            logger.fatal(
                f"Error occurred while generating golden data: \n{traceback.format_exc()}"
            )

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
        logger.info(f"  hmonnx: {self.quant_onnx_model_path}")
        logger.info(f"  time: {span:.2f}s")

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
        logger.info(f"  hmonnx: {self.quant_onnx_model_path}")
        logger.info(f"  ncore: {self.build_ncore}")
        logger.info(f"  opt_level: {self.build_opt_level}")
        logger.info(
            f"  batch: {self.build_batch if self.roi_num == 1 else self.roi_num}"
        )
        logger.info(f"  target: xh2")

        if not os.path.exists(self.build_output_dir):
            os.makedirs(self.build_output_dir)

        try:
            import tcim
        except ImportError:
            logger.fatal("Not found tcim module, please install tcim first!")

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
            j=self.build_parallel_jobs,
            custom_msg=json.dumps(self.custom_msg, ensure_ascii=False),
        )

        span = time.time() - t_start
        self._log_section("Build Complete", char="-")
        logger.info(f"  hmm: {self.hmm_path}")
        logger.info(f"  time: {span:.2f}s")

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
    @staticmethod
    def _get_yuv_valid_len(size, toYUV_format):
        if toYUV_format == "YUV420SP":
            return size // 2
        elif toYUV_format == "YUV422SP":
            return size * 2 // 3
        else:
            return size

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

        logger.info(f"  hmm: {self.hmm_path}")
        logger.info(f"  batch: {self.build_batch}")
        logger.info(f"  device_id: {device_id}")

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
            logger.info(
                f"  {input_name}: shape={list(golden_input.shape)}, dtype={golden_input.dtype}"
            )

            hmm_batch = xh2.inputs_info[input_name].shape[0]

            resizer_mode = self.resizer_modes[input_name]

            bs = golden_input.shape[0]
            if resizer_mode in [1, 2, 3]:
                # TODO 多batch可能有问题
                fmt = xh2.inputs_format[input_name]
                golden_input = golden_input.flatten()
                size = self._get_yuv_valid_len(golden_input.size, fmt)
                golden_input = golden_input[:size].reshape(bs, size // bs)

            golden_input = np.repeat(
                golden_input, hmm_batch // golden_input.shape[0], axis=0
            )
            in_datas[input_name] = golden_input

            # Handle dynamic resizer parameters
            if resizer_mode in [1, 2]:
                resizer_name = f"resizer_crop_{input_name}"
                hmm_batch = xh2.inputs_batch[resizer_name]
                golden_dyn_path = os.path.join(
                    self.quant_output_dir,
                    f"hmquant_{self.model_name}_resizer_crop_{new_name}_input.npy",
                )
                if not os.path.exists(golden_dyn_path):
                    logger.fatal(f"Dynamic resizer input not found: {golden_dyn_path}")

                golden_dyn_input = np.load(golden_dyn_path)
                logger.info(
                    f"  {resizer_name}: shape={list(golden_dyn_input.shape)}, dtype={golden_dyn_input.dtype}"
                )
                golden_dyn_input_batch = golden_dyn_input.shape[0]
                golden_dyn_input = np.repeat(
                    golden_dyn_input, hmm_batch // golden_dyn_input_batch, axis=0
                )
                in_datas[resizer_name] = golden_dyn_input

        # Run inference
        logger.info("Running XH2 inference...")
        outputs, _ = xh2.run(in_datas)
        self.save_profile_data(outputs)

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
                os.path.join(self.quant_output_dir, golden_dir, f"{new_name}.npy")
                if enable_layers
                else os.path.join(
                    self.quant_output_dir,
                    f"hmquant_{self.model_name}_{new_name}_output.npy",
                )
            )
            golden_output = np.load(golden_output_path)
            logger.info(
                f"  {output_name}: shape={list(golden_output.shape)}, dtype={golden_output.dtype}"
            )

            hmm_batch = xh2.outputs_batch[output_name]
            golden_output = np.repeat(
                golden_output, repeats=hmm_batch // golden_output.shape[0], axis=0
            )
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
        logger.info(f"  time: {span:.2f}s")

        return {"outputs": outputs_result}

    def compare(self, data_path: str, device_id=0):
        """Compare outputs from ONNX, HmQuant and XH2 inference."""
        self._log_section(f"Compare: {self.model_name}")

        t_start = time.time()

        logger.info(f"  data_path: {data_path}")
        logger.info(f"  device_id: {device_id}")

        # Load models
        logger.info("Loading models:")
        onnx_infer = OnnxInfer()
        onnx_infer.load(self.model_path)
        logger.info(f"  onnx: {self.model_path}")

        hmquant_infer = Xh2HmQuantInfer()
        hmquant_infer.load(self.quant_onnx_model_path)
        logger.info(f"  hmquant: {self.quant_onnx_model_path}")

        xh2_infer = Xh2Infer()
        xh2_infer.load(self.hmm_path)
        logger.info(f"  xh2: {self.hmm_path}")

        onnx_in_datas = {}
        hmquant_in_datas = {}
        xh2_in_datas = {}
        _, ext = os.path.splitext(os.path.basename(data_path))

        if not os.path.exists(data_path):
            logger.fatal(f"Not found data_path: {data_path}")

        logger.info("Preparing data...")
        for input_name in self.inputs_cfg:
            input_cfg = self.inputs_cfg[input_name]
            resizer_mode = self.resizer_modes[input_name]
            hmm_batch = xh2_infer.inputs_batch[input_name]
            hmonnx_batch = hmquant_infer.inputs_batch[input_name]
            onnx_batch = onnx_infer.inputs_batch[input_name]
            fmt = xh2_infer.inputs_format[input_name]
            data_format = input_cfg.get("data_format")
            if data_format is not None:
                if len(self.inputs_cfg) == 1:
                    if ext not in SUPPORT_IMAGE_FORMATS:
                        logger.fatal(f"Unsupported image format: {ext}")
                    cv_image = cv2.imread(
                        data_path,
                        (
                            cv2.IMREAD_COLOR
                            if data_format != "GRAY"
                            else cv2.IMREAD_GRAYSCALE
                        ),
                    )
                    if cv_image is None:
                        logger.fatal("Failed to decode image")
                else:
                    in_datas = load_npz(data_path)
                    if input_name not in in_datas:
                        logger.fatal(f"Input data not found: {input_name}")

                    cv_image = in_datas[input_name]  # BGR
                onnx_data: np.ndarray = self._preprocess_for_onnx(cv_image, input_cfg)
                yuv_im: torch.Tensor
                yuv_im, dyn_info = self._preprocess_for_resizer(
                    cv_image, input_cfg, resizer_mode
                )
                onnx_in_datas[input_name] = np.repeat(
                    onnx_data, repeats=onnx_batch // onnx_data.shape[0], axis=0
                )
                if resizer_mode == 0:
                    if onnx_batch != hmm_batch or onnx_batch != hmonnx_batch:
                        logger.fatal(
                            f"Batch size mismatch, expected onnx: {onnx_batch}, got hmm: {hmm_batch} and hmonnx: {hmonnx_batch}"
                        )
                    hmquant_in_datas[input_name] = torch.from_numpy(
                        onnx_in_datas[input_name].astype(np.float16)
                    ).cpu()
                    xh2_in_datas[input_name] = onnx_in_datas[input_name].astype(
                        np.float16
                    )
                elif resizer_mode in [1, 2, 3]:
                    if onnx_batch != hmonnx_batch:
                        logger.fatal(
                            f"Batch size mismatch, expected onnx: {onnx_batch}, got hmonnx: {hmonnx_batch}"
                        )
                    yuv_im = yuv_im.to(torch.float16)
                    hmquant_in_datas[input_name] = yuv_im.repeat_interleave(
                        hmonnx_batch, dim=0
                    ).contiguous()
                    yuv = yuv_im.detach().cpu().numpy().flatten()
                    valid_len = self._get_yuv_valid_len(yuv.size, fmt)
                    yuv = np.repeat(yuv[:valid_len].reshape(1, -1), hmm_batch, axis=0)
                    xh2_in_datas[input_name] = np.ascontiguousarray(yuv)
                # Dynamic resizer info
                if resizer_mode in [1, 2]:
                    resizer_name = f"resizer_crop_{input_name}"
                    hmonnx_batch = hmquant_infer.inputs_batch[resizer_name]
                    hmm_batch = xh2_infer.inputs_batch[resizer_name]
                    hmquant_dyn = dyn_info.repeat_interleave(hmonnx_batch, dim=0)
                    xh2_dyn = dyn_info.repeat_interleave(hmm_batch, dim=0)
                    hmquant_in_datas[resizer_name] = hmquant_dyn
                    xh2_in_datas[resizer_name] = xh2_dyn.detach().cpu().numpy()
            else:
                in_datas = load_npz(data_path)
                in_data: np.ndarray = in_datas[input_name].copy()
                if in_data.dtype == np.int64:
                    in_data = in_data.astype(np.int32)
                elif in_data.dtype == np.float32:
                    in_data = in_data.astype(np.float16)
                onnx_in_datas[input_name] = in_datas[input_name].copy()
                hmquant_in_datas[input_name] = torch.from_numpy(in_data.copy())
                xh2_in_datas[input_name] = np.repeat(
                    in_data, repeats=hmm_batch // in_data.shape[0], axis=0
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

        # Compare results
        table = PrettyTable(
            ["name", "onnx vs hmquant", "onnx vs xh2", "hmquant vs xh2"]
        )
        table.title = "Cosine Distance"
        outputs_result = {}

        for output_name in onnx_outputs:
            onnx_batch = onnx_infer.outputs_batch[output_name]
            onnx_out = np.repeat(
                onnx_outputs[output_name],
                repeats=onnx_batch // onnx_outputs[output_name].shape[0],
                axis=0,
            )
            hmquant_batch = hmquant_infer.outputs_batch[output_name]
            hmquant_out = np.repeat(
                hmquant_outputs[output_name],
                repeats=hmquant_batch // hmquant_outputs[output_name].shape[0],
                axis=0,
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
        logger.info(f"  time: {span:.2f}s")

        return {
            "compare": {
                "success": True,
                "data_path": data_path,
                "outputs": outputs_result,
            }
        }

    # ==================== Static Utility Methods ====================

    @staticmethod
    def check_golden_from_hmm(hmm, golden_dir, device_id=0):
        """Check model inference results against golden data consistency."""
        if not os.path.exists(hmm):
            logger.fatal(f"Not found hmm model: {hmm}")
        if not os.path.exists(golden_dir):
            logger.fatal(f"Not found golden data directory: {golden_dir}")

        try:
            xh2 = Xh2Infer()
            xh2.load(hmm, device_id=device_id)
        except Exception as e:
            logger.fatal(f"Failed to load hmm model: \n{traceback.format_exc()}")

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
                    hmm_batch = xh2.inputs_batch[name]
                    data = np.repeat(data, repeats=hmm_batch // data.shape[0], axis=0)
                    input_data[name] = data
                    logger.info(
                        f"Loaded input: {name}, shape={data.shape}, from={paths[0]}"
                    )
                except Exception as e:
                    logger.fatal(
                        f"Failed to load {paths[0]}: \n{traceback.format_exc()}"
                    )

        golden_outputs = {}
        for name, paths in output_files_map.items():
            if paths:
                try:
                    golden_outputs[name] = np.load(paths[0])
                    golden_batch = xh2.outputs_batch[name]
                    golden_outputs[name] = np.repeat(
                        golden_outputs[name],
                        repeats=golden_batch // golden_outputs[name].shape[0],
                        axis=0,
                    )
                    logger.info(
                        f"Loaded golden output: {name}, shape={golden_outputs[name].shape}"
                    )
                except Exception as e:
                    logger.fatal(
                        f"Failed to load {paths[0]}: \n{traceback.format_exc()}"
                    )

        try:
            logger.info("Running inference...")
            outputs, _ = xh2.run(input_data)
            logger.info("Inference completed")
        except Exception as e:
            logger.fatal(f"Inference failed: \n{traceback.format_exc()}")

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

    @staticmethod
    def build_from_hmonnx(
        hmonnx,
        hmm_name=None,
        output="output",
        ncore=1,
        opt_level=2,
        batch=1,
        llm_batch=1,
        enable_profile=False,
        roi_num=1,
        flash_attn=0,
        llm_opt=False,
        enable_xh2_stable_output=False,
        context_length=None,
        prefill_length=None,
        ndevice=1,
        is_prefill=False,
        enable_common_subgraph=False,
        skip_mlir_compile=False,
        subgraph_repeat_hint=20,
        all_logits=False,
        work_dir=None,
        cpp_backend="v1",
        target="xh2",
        **kwargs,
    ):
        """Build HMM model from hmonnx."""
        try:
            import tcim
        except ImportError:
            logger.error("Not found tcim module, please install tcim first!")
            return

        if not hmonnx or not os.path.exists(hmonnx):
            logger.warning(
                f"HMONNX file not found, please check the file path: {hmonnx}."
            )
            return

        if hmm_name is None:
            hmm_name = os.path.splitext(os.path.basename(hmonnx))[0]
        if (batch > 1 and roi_num > 1) or batch < 0 or roi_num < 0:
            logger.fatal(f"Invalid combination of batch{batch} and roi_num{roi_num }")
        # output_dir = os.path.join(output, target)
        output_dir = output
        work_dir = (
            os.path.join(output_dir, "tcim", hmm_name) if work_dir is None else work_dir
        )

        # Build kwargs for tcim
        build_kwargs = {}

        # LLM modification parameters
        build_kwargs["modify_llm"] = {}
        if is_prefill:
            if llm_batch > 1:
                logger.warning(
                    "batch is ignored for prefill model. "
                    "Prefill model uses fill-length instead of batch."
                )
            if prefill_length is not None:
                build_kwargs["modify_llm"]["fill-length"] = prefill_length
            if context_length is not None:
                build_kwargs["modify_llm"]["context-length"] = context_length
        else:
            if prefill_length is not None:
                logger.warning(
                    "prefill_length is ignored for decode model. "
                    "Decode model uses batch instead of fill-length."
                )
            if llm_batch > 1:
                build_kwargs["modify_llm"]["batch"] = llm_batch
            if context_length is not None:
                build_kwargs["modify_llm"]["context-length"] = context_length
        if all_logits:
            build_kwargs["modify_llm"]["all-logits"] = all_logits

        # Multi-device
        if ndevice > 1:
            build_kwargs["ndevice"] = ndevice

        # Flash attention with context_length check
        if flash_attn > 0 and context_length is not None and context_length < 2048:
            logger.warning("Flash attention disabled: context_length < 2048")
            flash_attn = 0

        custom_msg = {
            "target": "xh2",
            "ndevice": ndevice,
            "ncore": ncore,
            "batch": batch,
            "llm_batch": llm_batch,
            "context_length": context_length,
            "prefill_length": prefill_length,
            "all_logits": all_logits,
            "flash_attention": flash_attn,
            "llm_opt": llm_opt,
            "enable_xh2_stable_output": enable_xh2_stable_output,
            "enable_common_subgraph": enable_common_subgraph,
            "subgraph_repeat_hint": subgraph_repeat_hint,
            "skip_mlir_compile": skip_mlir_compile,
            "cpp_backend": cpp_backend,
            "opt_level": opt_level,
            "is_prefill": is_prefill,
        }

        # Merge kwargs with build_kwargs
        merged_kwargs = dict(kwargs or {})
        parallel_jobs = merged_kwargs.pop(
            "parallel_jobs", psutil.cpu_count(logical=False)
        )
        input_modify_llm = merged_kwargs.get("modify_llm") or {}
        build_modify_llm = build_kwargs.get("modify_llm") or {}
        merged_modify_llm = {**input_modify_llm, **build_modify_llm}
        merged_kwargs.update(build_kwargs)
        if merged_modify_llm:
            merged_kwargs["modify_llm"] = merged_modify_llm
        logger.info(f"==> {hmm_name} build start, kwargs: {merged_kwargs}")

        tcim.build_from_hmonnx(
            hmonnx,
            output_name=hmm_name,
            ncore=ncore,
            opt_level=f"O{opt_level}",
            target=target,
            batch=batch,
            enable_profile=enable_profile,
            output_dir=output_dir,
            work_dir=work_dir,
            one_img_multi_roi=roi_num > 1,
            llm_opt=llm_opt,
            enable_xh2_stable_output=enable_xh2_stable_output,
            skip_mlir_compile=skip_mlir_compile,
            enable_common_subgraph=enable_common_subgraph,
            subgraph_repeat_hint=subgraph_repeat_hint,
            flash_attention=flash_attn,
            cpp_backend=cpp_backend,
            j=parallel_jobs,
            custom_msg=json.dumps(custom_msg, ensure_ascii=False),
            **merged_kwargs,
        )

        return os.path.join(output_dir, f"{hmm_name}.hmm")
