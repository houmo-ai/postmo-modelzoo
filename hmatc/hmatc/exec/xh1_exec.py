# Copyright 2025 HOUMO AI
#
# File: xh1_exec.py
# Description:
#   XH1 executor
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
import json
import os
import time
import cv2
import numpy as np
import torch
from datetime import datetime
from prettytable import PrettyTable
from ..base.base_exec import BaseExec
from ..infer.hmquant_infer import HmQuantInfer
from ..infer.onnx_infer import OnnxInfer
from ..infer.xh1_infer import Xh1Infer
from ..utils import logger
from ..utils.dist_metrics import cosine_distance
from ..utils.preprocess import xh1_preprocess, calc_padding_size
from ..utils.utils import (
    get_hmquant_xh1_version,
    get_package_version,
    upload_file_to_artifactory,
)
from ..utils.utils import (
    SUPPORT_IMAGE_FORMATS,
    compress_files_to_tar_xz_with_progress,
    compress_folder_to_tar_xz_with_progress,
    get_file_md5,
    get_md5,
    load_npz,
    get_houmo_version,
    gen_random_data,
)


class Xh1Exec(BaseExec):
    """
    Executor class for XH1 target platform.
    Handles quantization, building, checking golden data, and comparison for XH1 hardware.
    """

    def __init__(self, cfg: dict) -> None:
        """
        Initialize the XH1 executor.

        Args:
            cfg (dict): Configuration dictionary containing model and quantization settings
        """
        super().__init__(cfg)
        self.quant_sequencer_model_path = os.path.join(
            self.quant_output_dir,
            f"{self.model_name}_xh1_b{self.model_input_batch}.pkl",
        )
        mix_search_cfg = self.quant_advance_cfg.get("mix_search", dict())
        if "activation" in mix_search_cfg:
            activation_cfg = mix_search_cfg["activation"]
            if "method" in activation_cfg:
                activation_method_cfg = activation_cfg["method"]
                method_name = activation_method_cfg.get("name")
                if method_name is None:
                    self.quant_advance_cfg["activation"]["method"] = {"name": "auto"}
                if method_name not in ["all", "auto"]:
                    logger.error("activation_mix method must be all or auto")
                    exit(-1)
            else:
                self.quant_advance_cfg["activation"]["method"] = {"name": "auto"}

    def get_quant_cfg(self) -> dict:
        """
        Get the quantization configuration for XH1 platform.

        Returns:
            dict: Quantization configuration dictionary
        """
        # Set quantization log output
        log_dir = os.path.join(self.save_dir, "xh1", "logs")
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)
        quant_cfg = {
            "inputs_cfg": dict(),
            "quant_cfg": self.quant_advance_cfg,
            "extra_cfg": {
                "with_label": False,
                "log_dir": log_dir,
            },
        }
        for input_name in self.inputs_cfg:
            input_cfg = self.inputs_cfg[input_name]
            input_shape = input_cfg["shape"]
            data_format = input_cfg["data_format"]
            quant_cfg["inputs_cfg"][input_name] = {
                "data_format": self.dtype_transform(
                    self.onnx_inputs_info[input_name]["dtype"]
                ),
                "first_layer_weight_denorm_mean": None,
                "first_layer_weight_denorm_std": None,
            }

            # Skip if not single image input or resizer_mode is 0
            if not self.is_image_single_input or self.resizer_mode == 0:
                continue
            _, _, H, W = input_shape
            new_input_cfg = dict()
            new_input_cfg["data_format"] = data_format
            # mean/std
            mean_values = input_cfg["mean"]
            std_values = input_cfg["std"]
            mean_values = [v / 255.0 for v in mean_values]
            std_values = [v / 255.0 for v in std_values]
            new_input_cfg["first_layer_weight_denorm_mean"] = mean_values
            new_input_cfg["first_layer_weight_denorm_std"] = std_values
            # toYUV_format
            resizer_cfg = input_cfg["resizer"]
            max_input_size = resizer_cfg.get("max_input_size", [H, W])
            toYUV_format = resizer_cfg.get("toYUV_format", "YUV420SP")
            new_input_cfg["toYUV_format"] = toYUV_format[0:6]  # Remove SP
            insert_pad_scatter = resizer_cfg.get("insert_pad_scatter", False)
            if insert_pad_scatter not in [False, True]:
                logger.error(
                    f"Not support insert_pad_scatter: {insert_pad_scatter} yet"
                )
                exit(-1)
            new_input_cfg["insert_pad_scatter"] = insert_pad_scatter
            # resizer_resize
            new_input_cfg["resizer_resize"] = {
                "height": H,
                "width": W,
                "align_corners": False,
                "method": "bilinear",
            }
            resize_type = input_cfg["resize_type"]
            max_height, max_width = max_input_size
            nh, nw = H, W
            # ONNX input larger than max_input_size, affecting static_resizer crop
            if H > max_height or W > max_width:
                # Scale proportionally within max input size
                _, size, _ = calc_padding_size(
                    (H, W), (max_width, max_height), padding_mode=0
                )
                nh, nw = size
            new_input_cfg["resizer_crop"] = {
                "top": 0,
                "left": 0,
                "height": nh,
                "width": nw,
            }
            new_input_cfg["dynamic_crop"] = self.resizer_mode in [1, 2]
            new_input_cfg["fold"] = self.resizer_mode in [
                2,
                3,
            ]  # Internal quantization decision
            if resize_type == 1 and self.resizer_mode == 1:
                # padding
                padding_values = input_cfg["padding_values"]
                if len(padding_values) == 3:
                    R, G, B = padding_values
                    if data_format == "BGR":
                        B, G, R = padding_values
                    Y = round(min(max(0.299 * R + 0.587 * G + 0.114 * B, 0), 255))
                    U = round(
                        min(max(-0.169 * R - 0.331 * G + 0.500 * B + 128, 0), 255)
                    )
                    V = round(min(max(0.500 * R - 0.419 * G - 0.081 * B + 128, 0), 255))
                    padding_values = [Y - 128, U - 128, V - 128]
                elif len(padding_values) == 1:
                    Y = padding_values[0]
                    padding_values = [Y - 128]
                new_input_cfg["resizer_pad"] = {"value": padding_values}
            quant_cfg["inputs_cfg"][input_name].update(new_input_cfg)
        return quant_cfg

    def get_quant_dataset(self):
        """
        Provide quantization data for calibration.

        Returns:
            tuple: A tuple containing:
                - calib_datasets (list): List of calibration datasets
                - onnx_datasets (list): List of ONNX datasets
        """
        # Provide quantization data
        onnx_datasets = list()
        calib_datasets = list()
        input_name = self.inputs_name[0]
        input_cfg = self.inputs_cfg[input_name]
        calib_num = self.quant_cfg.get("calib_num")
        data_format = input_cfg.get("data_format")
        filenames = (
            ["random"] * calib_num
            if self.use_random_data
            else os.listdir(self.calib_data)
        )
        filenames.sort()
        if self.use_random_data:
            logger.warning("Using random data for calibration")
        if self.is_image_single_input:
            # Single input and input is image
            input_name = self.inputs_name[0]
            input_cfg = self.inputs_cfg[input_name]
            input_shape = input_cfg["shape"]
            N, C, H, W = input_shape
            mean = input_cfg["mean"]
            std = input_cfg["std"]
            resize_type = input_cfg["resize_type"]
            padding_mode = input_cfg.get("padding_mode")
            padding_values = input_cfg.get("padding_values")
            resizer_cfg = input_cfg.get("resizer", dict())
            max_input_size = resizer_cfg.get("max_input_size", (H, W))
            max_height, max_width = max_input_size
            # Pad images
            padding_len = len(filenames) % N
            for idx in range(padding_len):
                filenames.append(filenames[0])
            # Split images
            actual_calib_num = len(filenames) // N
            if actual_calib_num < calib_num:
                logger.warning(
                    f"The number of calibration data is less than the number of calibration samples"
                )
                calib_num = actual_calib_num
            for idx in range(calib_num):
                batch_filenames = filenames[idx * N : (idx + 1) * N]
                batch_datas = list()
                dyn_infos = list()
                batch_onnx_datas = list()
                in_datas = dict()
                logger.info(f"Processing calibration data {idx}...")
                for filename in batch_filenames:
                    if not self.use_random_data:
                        _, ext = os.path.splitext(filename)
                        if ext not in SUPPORT_IMAGE_FORMATS:
                            logger.warning(f"Not supported ext: {ext}")
                            continue
                        filepath = os.path.join(self.calib_data, filename)
                        if not os.path.exists(filepath):
                            logger.warning(f"{filepath} not exists")
                            continue
                        cv_image = cv2.imread(filepath)
                        if cv_image is None:
                            logger.warning(f"{filepath} not exists or decode failed")
                            continue
                    else:
                        cv_image = np.random.randint(
                            low=0,
                            high=255,
                            size=(max_height, max_width, C),
                            dtype=np.uint8,
                        )

                    im, dyn_info = xh1_preprocess(
                        cv_image,
                        input_shape,
                        max_input_size,
                        mean=mean,
                        std=std,
                        use_norm=self.resizer_mode == 0,
                        use_resize=self.resizer_mode in [0, 3],
                        use_rgb=data_format == "RGB",
                        resize_type=resize_type,
                        padding_mode=padding_mode,
                        padding_values=padding_values,
                        is_onnx=self.resizer_mode == 0,
                    )
                    # onnx
                    onnx_im, _ = xh1_preprocess(
                        cv_image,
                        input_shape,
                        max_input_size,
                        mean=mean,
                        std=std,
                        use_norm=True,
                        use_resize=True,
                        use_rgb=data_format == "RGB",
                        resize_type=resize_type,
                        padding_mode=padding_mode,
                        padding_values=padding_values,
                        is_onnx=True,
                    )
                    onnx_im = onnx_im.detach().cpu().numpy()
                    batch_onnx_datas.append(onnx_im)
                    dyn_infos.append(dyn_info)
                    batch_datas.append(im)
                onnx_datasets.append(
                    {input_name: np.concatenate(batch_onnx_datas, axis=0)}
                )
                in_datas[input_name] = torch.cat(batch_datas, dim=0)
                if self.resizer_mode in [1, 2]:
                    batch_dyninfos = torch.cat(dyn_infos, dim=0)
                    in_datas[f"resizer_crop_{input_name}"] = batch_dyninfos
                calib_datasets.append(in_datas)
        else:
            # Single input and input is non-image or multi-input
            if len(filenames) < calib_num:
                logger.warning(
                    f"The number of calibration data is less than the number of calibration samples"
                )
                calib_num = len(filenames)
            for idx in range(calib_num):
                if self.use_random_data:
                    in_datas = dict()
                    for input_name in self.inputs_cfg:
                        dtype = self.onnx_inputs_info[input_name]["dtype"]
                        input_shape = self.inputs_cfg[input_name]["shape"]
                        data = gen_random_data(input_shape, dtype)
                        in_datas[input_name] = torch.from_numpy(data)
                else:
                    filename = filenames[idx]
                    data_path = os.path.join(self.calib_data, filename)
                    in_datas = load_npz(data_path)
                    for input_name in in_datas:
                        in_data = in_datas[input_name]
                        batch = in_data.shape[0]
                        onnx_dtype = self.onnx_inputs_info[input_name]["dtype"]
                        assert (
                            onnx_dtype == in_data.dtype
                        ), "npz data dtype must be equal to onnx input dtype"
                        assert (
                            batch == self.model_inputs_batch[input_name]
                        ), "npz data batch must be equal to onnx input batch"
                        in_datas[input_name] = torch.from_numpy(in_datas[input_name])
                calib_datasets.append(in_datas)
                onnx_datasets.append(
                    {key: in_datas[key].detach().cpu().numpy() for key in in_datas}
                )
        return calib_datasets, onnx_datasets

    def quantize(self):
        """
        Quantize the ONNX model for XH1 hardware.

        Returns:
            dict: Dictionary containing quantization results and information
        """
        # quantize the model
        # quant info
        logger.info(f"Using device: {self.device}")
        if self.quant_cfg is None:
            logger.error("quant info not found")
            return dict()
        if self.calib_data is not None:
            HOUMO_DATASETS_PATH = os.environ.get(
                "HOUMO_DATASETS_PATH", "/usr/local/src/houmo-modelzoo/data/datasets"
            )
            HM_calib_data = os.path.join(HOUMO_DATASETS_PATH, self.calib_data)
            if not os.path.isdir(self.calib_data) and not os.path.isdir(HM_calib_data):
                logger.error("calib_data must be a exist directory")
                return dict()
            if not os.path.isdir(self.calib_data):
                self.calib_data = HM_calib_data
            logger.info(f"calib_data: {self.calib_data}")

        if hasattr(self, "ApplicationOnnxOpt"):
            self.ApplicationOnnxOpt.opt()
            if hasattr(self.ApplicationOnnxOpt, "opt_model_path"):
                self.model_path = self.ApplicationOnnxOpt.opt_model_path

        try:
            from hmquant.api import (
                quant_single_onnx_network,
                generate_golden,
                convert_profiling,
                quantize_profiling,
            )
        except ImportError:
            logger.error("Not found hmquant module, and please install hmquant first")
            exit(-1)

        calib_datasets, onnx_datasets = self.get_quant_dataset()
        in_datas = calib_datasets[0]
        onnx_in_datas = onnx_datasets[0]

        logger.info(f"Using {self.device} quantization...")
        now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        t_start = time.time()
        sequencer = quant_single_onnx_network(
            cfg=self.get_quant_cfg(),
            calibration_data=calib_datasets,
            onnx_model_or_path=self.model_path,
            device=self.device,
        )

        # Print frontend conversion comparison results
        logger.info("Frontend conversion profiling results:")
        convert_profiling(
            self.model_path,
            onnx_input=[onnx_in_datas],
            cfg=self.get_quant_cfg(),
            sequencer_input=[in_datas],
            save_tmp_path=self.quant_output_dir,
            device=self.device,
        )
        span = time.time() - t_start
        # Print quantization before and after comparison results
        logger.info("Quantization profiling results:")
        res = quantize_profiling(
            sequencer,
            [in_datas],
            device=self.device,
            mode=0,  # 0: cumulative error  1: single operator comparison
            quant_mode="quant_forward",
            return_o_metric=True,
        )
        res = {
            out_name: {
                k: float(v) if isinstance(v, np.float64) else v
                for k, v in metrics.items()
            }
            for out_name, metrics in res.items()
        }
        if not os.path.exists(self.quant_output_dir):
            os.makedirs(self.quant_output_dir)

        logger.info(f"Generate golden data to {self.quant_output_dir}")
        generate_golden(
            sequencer=sequencer,
            calibset=in_datas,
            save_path=self.quant_output_dir,
            model_name=self.model_name,
            batch_size=self.model_inputs_batch,
            device=self.device,
            mode="hardware_forward",
            input_types=["int8"],
            output_types=["int8"],
            separate_weight=False,
            save_output=True,
            use_cache_hard_drive=False,
            save_model_output=False,
            set_golden_filename_prefix=False,
            save_special_onnx=False,
        )
        sequencer.save_pkl(
            self.quant_output_dir, f"{self.model_name}_xh1_b{self.model_input_batch}"
        )
        res["time"] = span
        res_info = {"quant": res, "model": self.model_cfg}
        # Compress quantization outputs
        if self.enable_upload and 0:
            logger.info("Compressing quant output...")
            # Write version information
            with open(os.path.join(self.quant_output_dir, "VERSION.txt"), "w") as f:
                f.write(f"hmquant_version: {get_hmquant_xh1_version()}\n")
                f.write(f"quant_time: {now}\n")
            filename = f"hmquant_{self.model_dir_name}_xh1_{get_houmo_version()}.tar.xz"
            compress_quant_output_path = os.path.join(self.save_dir, "xh1", filename)
            compress_folder_to_tar_xz_with_progress(
                self.quant_output_dir,
                compress_quant_output_path,
                exclude=["*.pkl", "tmp_model.onnx"],
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
        return res_info

    def build(self, enable_profile=False):
        """
        Build the quantized model to HMM format for XH1 hardware.

        Args:
            enable_profile (bool): Whether to enable profiling during build, defaults to False

        Returns:
            dict: Dictionary containing build results and information
        """
        if not os.path.exists(self.build_output_dir):
            os.makedirs(self.build_output_dir)
        try:
            import tcim
        except ImportError:
            logger.error("Not found tcim module, and please install tcim first!")
            exit(-1)
        now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        t_start = time.time()
        tcim.build_from_hmonnx(
            self.quant_onnx_model_path,
            output_name=self.hmm_name,
            ncore=self.build_ncore,
            opt_level=self.build_opt_level,
            target="xh1",
            batch=self.build_batch if self.roi_num == 1 else self.roi_num,
            output_dir=self.hmm_save_dir,
            work_dir=self.build_output_dir,
            enable_dynamic_image_resize=self.resizer_mode in [1],
            one_img_multi_roi=self.roi_num > 1,
            enable_profile=enable_profile,
            custom_msg=json.dumps(self.custom_msg, ensure_ascii=False),
        )
        span = time.time() - t_start
        res_info = {"build": {"time": span}}
        # Compress compiled outputs
        if self.enable_upload:
            logger.info("Compressing hmmodel...")
            hmcc_version = get_package_version(f"houmo-tcim-xh1")
            runtime_version = get_package_version(f"houmo_tcim_runtime_xh1")
            with open(os.path.join(self.save_dir, "xh1", "VERSION.txt"), "w") as f:
                f.write(f"hmquant_version: {get_hmquant_xh1_version()}\n")
                f.write(f"tcim_version: {hmcc_version}\n")
                f.write(f"tcim_runtime_version: {runtime_version}\n")
                f.write(f"build_time: {now}\n")
            filename = f"{self.model_dir_name}_xh1_b{self.hmm_batch}_{self.build_ncore}core_{self.build_opt_level}_{get_houmo_version()}.tar.xz"
            compress_hmm_path = os.path.join(
                self.save_dir,
                "xh1",
                filename,
            )
            compress_files_to_tar_xz_with_progress(
                [self.hmm_path, os.path.join(self.save_dir, "xh1", "VERSION.txt")],
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
        return res_info

    def check_golden(self, device_id=0, enable_layers=False):
        """
        Check the golden data against the hardware model outputs.

        Args:
            device_id (int): Device ID for inference, default is 0
            enable_layers (bool): Whether to enable layer-by-layer checking

        Returns:
            dict: Dictionary containing comparison results between golden and hardware outputs
        """
        logger.info("Checking golden...")
        if enable_layers:
            self.quant_onnx_model_path = self.add_node_output_as_graph_output(
                self.quant_onnx_model_path, "xh1"
            )
            self.build_output_dir += "_debug"
            self.hmm_name += "_debug"
            self.hmm_path = os.path.join(self.hmm_save_dir, f"{self.hmm_name}.hmm")
            if not os.path.exists(self.hmm_path):
                logger.info("Rebuild hmmodel with all layers output...")
                self.build(enable_profile=False)

        xh1 = Xh1Infer()
        xh1.load(self.hmm_path, device_id=device_id)
        in_datas = dict()
        for input_name in self.inputs_cfg:
            new_input_name = input_name.replace("/", "_")
            golden_input_path = os.path.join(
                self.quant_output_dir,
                f"hmquant_{self.model_name}_{new_input_name}_input.npy",
            )
            golden_input = np.load(golden_input_path)
            # Multiple batch during compilation, need to copy data
            if self.build_batch > 1:
                golden_input = np.repeat(golden_input, self.build_batch, axis=0)
            in_datas[input_name] = golden_input
            if self.resizer_mode in [1, 2]:
                # Multiple batch in compiled model, dynamic_resizer data needs to be copied
                golden_dyn_info_input_path = os.path.join(
                    self.quant_output_dir,
                    f"hmquant_{self.model_name}_resizer_crop_{new_input_name}_input.npy",
                )
                golden_dyn_input = np.load(golden_dyn_info_input_path)
                repeats = 1
                if self.roi_num > 1:  # 1 image n boxes
                    repeats = self.roi_num
                elif (
                    self.roi_num == 1 and self.build_batch > 1
                ):  # n images n boxes, and compilation batch > 1
                    repeats = self.build_batch
                golden_dyn_input = np.repeat(golden_dyn_input, repeats=repeats, axis=0)
                in_datas[f"resizer_crop_{input_name}"] = golden_dyn_input

        res_info = dict()
        outputs, outputs_dequanted = xh1.run(in_datas)
        self.save_profile_data(outputs)
        repeats = 1
        if self.build_batch > 1 and self.roi_num == 1:
            # n images n boxes, and compilation batch > 1
            repeats = self.build_batch
        elif self.roi_num > 1:
            # 1 image n boxes
            repeats = self.roi_num
        header = [
            "name",
            "cosine_dist",
            "MD5",
            "cosine_dist[dequanted]",
            "MD5[dequanted]",
        ]
        table = PrettyTable(header)
        table.title = "xh1 vs hmquant"
        for output_name in outputs_dequanted:
            new_output_name = output_name.replace("/", "_")
            if enable_layers:
                golden_output_path = os.path.join(
                    self.quant_output_dir,
                    f"hmquant_{self.model_name}_with_act",
                    f"{new_output_name}.npy",
                )
                golden_outputs = np.load(golden_output_path, allow_pickle=True).item()
                out_idx = ""
                if "split_size_or_sections" in golden_outputs:
                    out_idx = int(new_output_name.split("_")[-1])
                output_granularity = golden_outputs[f"output{out_idx}_granularity"]
                golden_output = golden_outputs["output_tensor"]
                zero_point = np.array(
                    golden_outputs[f"output{out_idx}_zero_point"], dtype=np.float32
                )
                scale = np.array(
                    golden_outputs[f"output{out_idx}_scale"], dtype=np.float32
                )
                if output_granularity != "tensor":
                    assert output_granularity.startswith("dim")
                    dim = int(output_granularity[-1])
                    output_tensor_shape = golden_output.shape
                    shape = [
                        1 if i != dim else output_tensor_shape[i]
                        for i in range(len(output_tensor_shape))
                    ]
                    zero_point = np.reshape(zero_point, shape)
                    scale = np.reshape(scale, shape)
                golden_output_dequanted = (golden_output - zero_point) * scale
            else:
                golden_output_path = os.path.join(
                    self.quant_output_dir,
                    f"hmquant_{self.model_name}_{new_output_name}_output.npy",
                )
                golden_output_dequant_path = os.path.join(
                    self.quant_output_dir,
                    f"hmquant_{self.model_name}_{new_output_name}_dequant_output.npy",
                )
                golden_output = np.load(golden_output_path)
                golden_output_dequanted = np.load(golden_output_dequant_path)

            golden_output = np.repeat(golden_output, repeats=repeats, axis=0)
            golden_output_md5 = get_md5(golden_output)
            output = outputs[output_name]
            output_md5 = get_md5(output)
            output_dequanted = outputs_dequanted[output_name]
            output_dequanted_md5 = get_md5(output_dequanted)
            golden_output_dequanted = np.repeat(
                golden_output_dequanted, repeats=repeats, axis=0
            )
            golden_output_dequanted_md5 = get_md5(golden_output_dequanted)
            # compare
            dist = cosine_distance(golden_output, output)
            dist_dequanted = cosine_distance(golden_output_dequanted, output_dequanted)
            table.add_row(
                [
                    output_name,
                    f"{dist:.6f}",
                    "ok" if output_md5 == golden_output_md5 else "fail",
                    f"{dist_dequanted:.6f}",
                    (
                        "ok"
                        if output_dequanted_md5 == golden_output_dequanted_md5
                        else "fail"
                    ),
                ]
            )
            res_info[output_name] = {
                "md5": output_md5,
                "dequanted_md5": output_dequanted_md5,
                "golden_md5": golden_output_md5,
                "golden_dequanted_md5": golden_output_dequanted_md5,
                "cosine_dist": float(dist),
                "dequanted_cosine_dist": float(dist_dequanted),
                "md5_ok": output_md5 == golden_output_md5,
                "dequanted_md5_ok": output_dequanted_md5 == golden_output_dequanted_md5,
            }
        logger.info(f"\n{table}")
        return res_info

    def compare(self, data_path: str, device_id=0):
        """
        Compare outputs from ONNX, HmQuant and XH1 inference.

        Args:
            data_path (str): Path to input data for comparison
            device_id (int): Device ID for XH1 inference, defaults to 0

        Returns:
            dict: Dictionary containing comparison results between different inference engines
        """
        t_start = datetime.now().strftime("%Y%m%d%H%M%S")
        # onnx
        onnx_infer = OnnxInfer()
        onnx_infer.load(self.model_path)
        # hmquant
        hmquant_infer = HmQuantInfer()
        hmquant_infer.load(self.quant_sequencer_model_path)
        # xh1
        xh1_infer = Xh1Infer()
        xh1_infer.load(self.hmm_path, device_id)

        onnx_in_datas = dict()
        hmquant_in_datas = dict()
        xh1_in_datas = dict()
        _, ext = os.path.splitext(os.path.basename(data_path))
        if self.is_image_single_input:
            # Single input image
            input_name = self.inputs_name[0]
            input_cfg = self.inputs_cfg[input_name]
            input_shape = input_cfg["shape"]
            N, C, H, W = input_shape
            data_format = input_cfg["data_format"]
            mean = input_cfg["mean"]
            std = input_cfg["std"]
            resize_type = input_cfg["resize_type"]
            padding_mode = input_cfg.get("padding_mode")
            padding_values = input_cfg.get("padding_values")
            reszier_cfg = input_cfg.get("resizer", dict())
            toYUV_format = reszier_cfg.get("toYUV_format", "YUV420SP")
            if self.resizer_mode != 0:
                max_input_size = self.max_inputs_size[input_name]
                max_height, max_width = max_input_size
            else:
                max_height, max_width = H, W
                max_input_size = (max_height, max_width)
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
            # Get compiled model batch
            hmm_batch = xh1_infer.inputs_info[input_name].shape[0]
            # onnx
            onnx_data, _ = xh1_preprocess(
                cv_image.copy(),
                input_shape,
                max_input_size,
                mean=mean,
                std=std,
                use_norm=True,
                use_resize=True,
                use_rgb=data_format == "RGB",
                resize_type=resize_type,
                padding_mode=padding_mode,
                padding_values=padding_values,
                is_onnx=True,
            )
            onnx_data = np.repeat(
                onnx_data.detach().cpu().numpy(), repeats=self.model_input_batch, axis=0
            )
            onnx_in_datas[input_name] = onnx_data  # np.ndarray

            yuv_pad_hwc, dyn_info = xh1_preprocess(
                cv_image,
                input_shape,
                max_input_size,
                mean=mean,
                std=std,
                use_norm=self.resizer_mode == 0,
                use_resize=self.resizer_mode in [0, 3],
                use_rgb=data_format == "RGB" and self.resizer_mode == 0,
                resize_type=resize_type,
                padding_mode=padding_mode,
                padding_values=padding_values,
                is_onnx=self.resizer_mode == 0,
                to_YUV=self.resizer_mode in [1, 2, 3],
                fmt=toYUV_format,
            )
            if self.resizer_mode in [1, 2, 3]:
                # Using resizer
                h, w, c = yuv_pad_hwc.shape
                yuv_pad = yuv_pad_hwc.view(1, c, h, w)
                yuv_pad = yuv_pad.repeat_interleave(self.model_input_batch, dim=0)
                hmquant_in_datas[input_name] = yuv_pad.contiguous()  # torch.Tensor
                # xh1
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
                xh1_in_datas[input_name] = np.ascontiguousarray(yuv)  # np.ndarray
            elif self.resizer_mode == 0:
                # Disable resizer
                in_data = np.repeat(onnx_data, repeats=self.build_batch, axis=0)
                in_data_quanted = xh1_infer.quantize(input_name, in_data)
                hmquant_in_datas[input_name] = torch.from_numpy(
                    in_data_quanted[0 : self.model_input_batch, ...]
                )
                xh1_in_datas[input_name] = np.ascontiguousarray(in_data_quanted)

            # dynamic_resizer info
            if self.resizer_mode in [1, 2]:
                if self.roi_num > 1:
                    # 1 image n boxes
                    hmquant_dyn_info = dyn_info
                    xh1_dyn_info = dyn_info.repeat_interleave(self.roi_num, dim=0)
                else:
                    # n images n boxes
                    hmquant_dyn_info = dyn_info.repeat_interleave(
                        self.model_input_batch, dim=0
                    )
                    xh1_dyn_info = dyn_info.repeat_interleave(hmm_batch, dim=0)
                hmquant_in_datas[f"resizer_crop_{input_name}"] = hmquant_dyn_info
                xh1_in_datas[f"resizer_crop_{input_name}"] = (
                    xh1_dyn_info.detach().cpu().numpy()
                )
        else:
            # Single input non-image or multi-input
            if ext != ".npz":
                logger.error(f"{data_path} is not npz file")
                exit(-1)
            in_datas = load_npz(data_path)
            onnx_in_datas = in_datas
            for input_name in in_datas:
                in_data = in_datas[input_name]
                in_data = np.repeat(in_data, repeats=self.build_batch, axis=0)
                in_data_quanted = xh1_infer.quantize(input_name, in_data)
                logger.info(
                    f"Hmquant input[{input_name}] quantize data_dtype: {in_data.dtype} -> {in_data_quanted.dtype}"
                )
                hmquant_in_datas[input_name] = torch.from_numpy(
                    in_data_quanted[0 : self.model_inputs_batch[input_name], ...]
                ).type(torch.int64)
                xh1_in_datas[input_name] = in_data_quanted

        # Save input data
        input_data_dir = os.path.join(self.save_dir, "xh1", "datas", "input")
        if not os.path.exists(input_data_dir):
            os.makedirs(input_data_dir)

        def save_input(runner, datas):
            for key in datas:
                bin_name = f"{runner}_input_{key}.bin"
                txt_name = f"{runner}_input_{key}.txt"
                npy_name = f"{runner}_input_{key}.npy"
                data = datas[key]
                if isinstance(data, torch.Tensor):
                    data = data.detach().cpu().numpy()
                data.tofile(os.path.join(input_data_dir, bin_name))
                data.tofile(os.path.join(input_data_dir, txt_name), sep="\n")
                np.save(os.path.join(input_data_dir, npy_name), data)

        output_data_dir = os.path.join(self.save_dir, "xh1", "datas", "output")
        if not os.path.exists(output_data_dir):
            os.makedirs(output_data_dir)

        def save_output(runner, key, data):
            bin_name = f"{runner}_output_{key}.bin"
            txt_name = f"{runner}_output_{key}.txt"
            npy_name = f"{runner}_output_{key}.npy"
            data.tofile(os.path.join(output_data_dir, bin_name))
            data.tofile(os.path.join(output_data_dir, txt_name), sep="\n")
            np.save(os.path.join(output_data_dir, npy_name), data)

        save_input("onnx", onnx_in_datas)
        save_input("xh1", xh1_in_datas)
        save_input("hmquant", hmquant_in_datas)

        onnx_outputs = onnx_infer.run(onnx_in_datas)
        hmquant_outputs = hmquant_infer.run(hmquant_in_datas)
        xh1_outputs, xh1_outputs_dequanted = xh1_infer.run(xh1_in_datas)
        self.save_profile_data(xh1_outputs)

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
        header = [
            "name",
            "onnx vs hmquant",
            "onnx vs xh1",
            "hmquant vs xh1",
            "MD5[hmquant vs xh1]",
        ]
        table = PrettyTable(header)
        table.title = "Cosine Distance"
        for output_name in onnx_outputs:
            new_output_name = output_name.replace("/", "_")
            # onnx
            onnx_output = onnx_outputs[output_name]
            onnx_output = np.repeat(onnx_output, repeats=repeats, axis=0)
            save_output("onnx", new_output_name, onnx_output)
            # hmquant
            hmquant_output = hmquant_outputs[output_name]
            hmquant_output = np.repeat(hmquant_output, repeats=repeats, axis=0)
            hmquant_output_dequanted = xh1_infer.dequantize(output_name, hmquant_output)
            save_output("hmquant", new_output_name, hmquant_output_dequanted)
            logger.info(
                f"Hmquant output[{output_name}] quantize data_dtype: {hmquant_output.dtype} -> {hmquant_output_dequanted.dtype}"
            )
            # xh1
            xh1_output_dequanted = xh1_outputs_dequanted[output_name]
            save_output("xh1", new_output_name, xh1_output_dequanted)
            # compare
            onnx_vs_hmquant = cosine_distance(onnx_output, hmquant_output_dequanted)
            onnx_vs_xh1 = cosine_distance(onnx_output, xh1_output_dequanted)
            hmquant_vs_xh1 = cosine_distance(
                hmquant_output_dequanted, xh1_output_dequanted
            )
            table.add_row(
                [
                    output_name,
                    f"{onnx_vs_hmquant:.6f}",
                    f"{onnx_vs_xh1:.6f}",
                    f"{hmquant_vs_xh1:.6f}",
                    (
                        "ok"
                        if get_md5(hmquant_output_dequanted)
                        == get_md5(xh1_output_dequanted)
                        else "fail"
                    ),
                ]
            )

            res_info["compare"][t_start][output_name] = {
                "onnx_vs_hmquant": float(onnx_vs_hmquant),
                "onnx_vs_xh1": float(onnx_vs_xh1),
                "hmquant_vs_xh1": float(hmquant_vs_xh1),
                "MD5": (
                    "ok"
                    if get_md5(hmquant_output_dequanted)
                    == get_md5(xh1_output_dequanted)
                    else "fail"
                ),
            }
        logger.info(f"Compare...\n{table}")
        return res_info
