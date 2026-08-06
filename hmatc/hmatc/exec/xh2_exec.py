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
import re
import shutil
import time
import numpy as np
import json
import psutil
import traceback
import torch
from prettytable import PrettyTable
from datetime import datetime
from ..base.base_exec import BaseExec
from ..dataloaders.factory import create_dataloader
from ..dataloaders.loaders import validate_sample
from ..utils import logger
from ..utils.dist_metrics import cosine_distance
from ..utils.bfp import cast_fp_data_to_act_hmfp_data
from ..utils.utils import (
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
)


class Xh2Exec(BaseExec):
    """
    Executor class for XH2 target platform.
    Handles quantization, building, checking golden data, and comparison for XH2 hardware.

    XH2 Resizer Constraints:
        Common constraints:
        - W方向: max 4096, >2048时32对齐, <=2048时2对齐
        - H方向: max 4096, 2对齐
        - crop: 4参数(y, x, h, w), 全部2对齐
        - pad: 4参数(top, left, bottom, right), 全部2对齐
        - pad可支持任意规格，不限制仅上下或左右单方向

        STATIC mode (resizer_mode=3):
        - 缩放倍数范围 [1/32, 16]

        DYNAMIC_V2 mode (resizer_mode=1):
        - 放大倍数最大16，H缩小倍数最大32，W缩小倍数最大8(YUV444最大4)
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

    def _get_resizer_cfg(self, input_cfg: dict):
        """Extract resizer parameters from input config."""
        resizer_cfg = input_cfg.get("resizer")
        if resizer_cfg is None:
            return dict()
        return resizer_cfg

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

    def get_input_data(self):
        """Load or generate input data for calibration."""
        data_dir = None if self.use_random_data else self._resolve_calib_data_path()
        num = self.quant_cfg.get("num", self.quant_cfg.get("calib_num", 0))
        for input_name, input_info in self.onnx_inputs_info.items():
            if input_name in self.model_cfg["inputs"]:
                self.model_cfg["inputs"][input_name]["dtype"] = input_info["dtype"]
        dataloader = create_dataloader(
            self.model_cfg,
            data_dir=data_dir,
            stage="quant",
            num=num,
        )
        if len(dataloader) == 0:
            logger.fatal("No calibration data found")
        sample = validate_sample(dataloader[0], self.inputs_cfg)
        return self._sample_to_quant_inputs(sample)

    def _sample_to_quant_inputs(self, sample):
        """Convert a standard DataLoader sample to xhquant input list."""
        inputs = sample["hmonnx_inputs"]
        dyn_info = sample.get("meta", {}).get("dyn_info", {}) or {}
        in_datas = []
        dynamic_inputs = []

        for input_name in self.inputs_cfg:
            in_datas.append(torch.from_numpy(inputs[input_name]))
            if input_name in dyn_info:
                value = dyn_info[input_name]
                if isinstance(value, np.ndarray):
                    value = torch.from_numpy(value)
                dynamic_inputs.append(value)

        in_datas.extend(dynamic_inputs)
        return in_datas

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
            from tcim.builder.api import build_from_hmonnx
        except ImportError:
            logger.fatal("Not found tcim module, please install tcim first!")

        logger.info("Building HMM model...")
        build_from_hmonnx(
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
            cpp_backend=self.cpp_backend,
            custom_msg=json.dumps(self.custom_msg, ensure_ascii=False),
            dump_compiled_mlir=self.dump_compiled_mlir,
            skip_check=True,
            flash_attention=self.flash_attention,
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
        hmcc_version = get_package_version("houmo_tcim_xh2")
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

        try:
            import tcim_lite
            from ..infer.xh2_infer import Xh2Infer
        except ModuleNotFoundError:
            logger.warning("tcim_lite not found, skipping golden check.")
            exit(0)

        if tcim_lite.runtime.get_device_num() < 1:
            logger.warning("No available devices found, skipping golden check.")
            exit(0)

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
        table.title = "hmm vs hmonnx"
        outputs_result = {}

        for output_name in outputs:
            new_name = output_name.replace("/", "_")
            if new_name in ["auto_profile_data.bin", "primitive_profile_data.bin"]:
                continue
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
        from ..infer.onnx_infer import OnnxInfer
        from ..infer.hmonnx_infer import HmonnxInfer
        from ..infer.xh2_infer import Xh2Infer

        onnx_infer = OnnxInfer()
        onnx_infer.load(self.model_path)
        logger.info(f"  onnx: {self.model_path}")

        hmonnx_infer = HmonnxInfer()
        hmonnx_infer.load(self.quant_onnx_model_path)
        logger.info(f"  hmonnx: {self.quant_onnx_model_path}")

        xh2_infer = Xh2Infer()
        xh2_infer.load(self.hmm_path)
        logger.info(f"  hmm: {self.hmm_path}")

        if not os.path.exists(data_path):
            logger.fatal(f"Not found data_path: {data_path}")

        logger.info("Preparing data with DataLoader...")
        dataloader = create_dataloader(
            self.model_cfg,
            data_dir=data_path,
            stage="compare",
            num=1,
        )
        if len(dataloader) == 0:
            logger.fatal("No compare data found")
        sample = validate_sample(dataloader[0], self.inputs_cfg)
        onnx_in_datas, hmonnx_in_datas, xh2_in_datas = self._sample_to_compare_inputs(
            sample,
            onnx_infer,
            hmonnx_infer,
            xh2_infer,
        )

        # Run inference
        logger.info("Running inference:")
        logger.info("  ONNX inference...")
        onnx_outputs = onnx_infer.run(onnx_in_datas)
        logger.info("  Hmonnx inference...")
        hmonnx_outputs = hmonnx_infer.run(hmonnx_in_datas)
        logger.info("  Hmm inference...")
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
            hmquant_batch = hmonnx_infer.outputs_batch[output_name]
            hmquant_out = np.repeat(
                hmonnx_outputs[output_name],
                repeats=hmquant_batch // hmonnx_outputs[output_name].shape[0],
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

    def _sample_to_compare_inputs(
        self,
        sample,
        onnx_infer,
        hmonnx_infer,
        xh2_infer,
    ):
        onnx_in_datas = {}
        hmonnx_in_datas = {}
        xh2_in_datas = {}
        inputs = sample["inputs"]
        hmonnx_inputs = sample["hmonnx_inputs"]
        dyn_info = sample.get("meta", {}).get("dyn_info", {}) or {}

        for input_name in self.inputs_cfg:
            resizer_mode = self.resizer_modes[input_name]
            onnx_data = inputs[input_name]
            hmonnx_data = hmonnx_inputs[input_name]
            onnx_batch = onnx_infer.inputs_batch[input_name]
            hmonnx_batch = hmonnx_infer.inputs_batch[input_name]
            hmm_batch = xh2_infer.inputs_batch[input_name]

            onnx_in_datas[input_name] = np.repeat(
                onnx_data,
                repeats=onnx_batch // onnx_data.shape[0],
                axis=0,
            )

            if resizer_mode == 0:
                if onnx_batch != hmm_batch or onnx_batch != hmonnx_batch:
                    logger.fatal(
                        "Batch size mismatch, "
                        f"expected onnx: {onnx_batch}, got hmm: {hmm_batch} "
                        f"and hmonnx: {hmonnx_batch}"
                    )
                hmonnx_data = self._cast_runtime_input(hmonnx_data)
                hmonnx_in_datas[input_name] = torch.from_numpy(hmonnx_data.copy())
                xh2_in_datas[input_name] = np.repeat(
                    hmonnx_data,
                    repeats=hmm_batch // hmonnx_data.shape[0],
                    axis=0,
                )
                continue

            if onnx_batch != hmonnx_batch:
                logger.fatal(
                    "Batch size mismatch, "
                    f"expected onnx: {onnx_batch}, got hmonnx: {hmonnx_batch}"
                )
            hmonnx_tensor = torch.from_numpy(hmonnx_data).to(torch.float16)
            hmonnx_in_datas[input_name] = hmonnx_tensor.repeat_interleave(
                hmonnx_batch, dim=0
            ).contiguous()

            fmt = xh2_infer.inputs_format[input_name]
            xh2_data = hmonnx_data.astype(np.float16).flatten()
            valid_len = self._get_yuv_valid_len(xh2_data.size, fmt)
            xh2_in_datas[input_name] = np.ascontiguousarray(
                np.repeat(xh2_data[:valid_len].reshape(1, -1), hmm_batch, axis=0)
            )

            if resizer_mode in [1, 2]:
                if input_name not in dyn_info:
                    logger.fatal(
                        f"Missing dynamic resizer params for input: {input_name}"
                    )
                resizer_name = f"resizer_crop_{input_name}"
                dyn_tensor = torch.from_numpy(dyn_info[input_name])
                hmonnx_dyn_batch = hmonnx_infer.inputs_batch[resizer_name]
                hmm_dyn_batch = xh2_infer.inputs_batch[resizer_name]
                hmonnx_in_datas[resizer_name] = dyn_tensor.repeat_interleave(
                    hmonnx_dyn_batch, dim=0
                )
                xh2_in_datas[resizer_name] = dyn_tensor.repeat_interleave(
                    hmm_dyn_batch, dim=0
                ).numpy()

        return onnx_in_datas, hmonnx_in_datas, xh2_in_datas

    @staticmethod
    def _cast_runtime_input(data):
        if data.dtype == np.int64:
            return data.astype(np.int32)
        if data.dtype == np.float32:
            return data.astype(np.float16)
        return data

    # ==================== Static Utility Methods ====================

    @staticmethod
    def check_golden_from_hmm(hmm, golden_dir, device_id=0):
        """Check model inference results against golden data consistency."""
        if not os.path.exists(hmm):
            logger.fatal(f"Not found hmm model: {hmm}")
        if not os.path.exists(golden_dir):
            logger.fatal(f"Not found golden data directory: {golden_dir}")

        try:
            from ..infer.xh2_infer import Xh2Infer

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

                    # Convert fp16 golden data to hmfp format when kvcache
                    # input expects int8 (hmfp-packed) data.
                    # kcache: name contains both k/key and cache
                    # vcache: name contains both v/value and cache
                    name_lower = name.lower()
                    has_k = bool(re.search(r"(?:^|_|\.)k(?:ey)?(?:$|_|\.)", name_lower))
                    has_v = bool(
                        re.search(r"(?:^|_|\.)v(?:alue)?(?:$|_|\.)", name_lower)
                    )
                    has_cache = bool(re.search(r"cache", name_lower))
                    is_kcache = has_k and has_cache
                    is_vcache = has_v and has_cache
                    if (
                        (is_kcache or is_vcache)
                        and hasattr(xh2, "inputs_info")
                        and name in xh2.inputs_info
                    ):
                        input_dtype = np.dtype(xh2.inputs_info[name].dtype).name
                        if input_dtype == "int8":
                            # kcache: pack along last axis
                            # vcache: pack along context-length axis (second-to-last)
                            pack_axis = -2 if is_vcache else -1
                            logger.info(
                                f"Converting {name} from fp16 to hmfp "
                                f"(pack_axis={pack_axis}, expected_dtype={input_dtype})"
                            )
                            data = cast_fp_data_to_act_hmfp_data(
                                data, "g32e8", pack_axis
                            )

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

            from ..infer.hmonnx_infer import HmonnxInfer

            model = HmonnxInfer()
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
        subgraph_repeat_hint=None,
        all_logits=False,
        work_dir=None,
        cpp_backend="v2",
        target="xh2",
        **kwargs,
    ):
        """Build HMM model from hmonnx."""
        try:
            from tcim.builder.api import build_from_hmonnx
        except ImportError:
            logger.fatal("Not found tcim module, please install tcim first!")

        if not hmonnx or not os.path.exists(hmonnx):
            logger.warning(
                f"HMONNX file not found, please check the file path: {hmonnx}."
            )
            return

        if hmm_name is None:
            hmm_name = os.path.splitext(os.path.basename(hmonnx))[0]
        if (batch > 1 and roi_num > 1) or batch < 0 or roi_num < 0:
            logger.fatal(f"Invalid combination of batch{batch} and roi_num{roi_num}")
        if batch > 1 and llm_batch > 1:
            logger.fatal(
                f"Configuring both batch{batch} and llm_batch{llm_batch} greater than 1 is not supported."
            )

        output_dir = output
        work_dir = (
            os.path.join(output_dir, "tcim", hmm_name) if work_dir is None else work_dir
        )

        # Check for reuse of compile results
        reuse_compile_results = os.environ.get("REUSE_COMPILE_RESULTS", "")
        if reuse_compile_results.strip().lower() in {"on", "true"} and os.path.isdir(
            output_dir
        ):
            candidate_files = sorted(
                os.path.join(output_dir, file_name)
                for file_name in os.listdir(output_dir)
                if os.path.isfile(os.path.join(output_dir, file_name))
                and os.path.splitext(file_name)[0] == hmm_name
            )
            if candidate_files:
                logger.info(
                    f"Reuse compile results, result file is: {candidate_files[0]}"
                )
                return candidate_files[0]

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
        if subgraph_repeat_hint is not None:
            build_kwargs["subgraph_repeat_hint"] = subgraph_repeat_hint
        if skip_mlir_compile is True:
            build_kwargs["skip_mlir_compile"] = skip_mlir_compile

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

        build_from_hmonnx(
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
            enable_common_subgraph=enable_common_subgraph,
            flash_attention=flash_attn,
            cpp_backend=cpp_backend,
            j=parallel_jobs,
            custom_msg=json.dumps(custom_msg, ensure_ascii=False),
            skip_check=True,
            **merged_kwargs,
        )

        return os.path.join(output_dir, f"{hmm_name}.hmm")
