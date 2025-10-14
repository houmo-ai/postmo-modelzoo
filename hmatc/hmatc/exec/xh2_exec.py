import os
import shutil
import time
from datetime import datetime
from importlib.metadata import PackageNotFoundError, version

import cv2
import numpy as np
import torch
from prettytable import PrettyTable

from ..base.base_exec import BaseExec
from ..infer.onnx_infer import OnnxInfer
from ..infer.xh2_infer import Xh2Infer
from ..infer.xhquant_infer import Xh2HmQuantInfer
from ..optimizer.onnx_opt_engine import HMAppOnnxOptConvert
from ..utils import logger
from ..utils.dist_metrics import cosine_distance
from ..utils.preprocess import default_preprocess
from ..utils.utils import (
    SUPPORT_IMAGE_FORMATS,
    compress_files_to_tar_xz_with_progress,
    compress_folder_to_tar_xz_with_progress,
    get_file_md5,
    get_hmquant_xh2_version,
    get_md5,
    get_package_version,
    load_npz,
    str_to_torch_dtype,
    upload_file_to_artifactory,
)


class Xh2Exec(BaseExec):
    def __init__(self, cfg: dict) -> None:
        super().__init__(cfg)
        self.quant_type = "w8a8h1_sefp"
        self.hmm_batch = self.build_batch * self.model_input_batch
        self.hmm_name = f"{self.model_name}_xh2_b{self.hmm_batch}_{self.build_ncore}core_{self.build_opt_level}"
        self.hmm_path = os.path.join(self.hmm_save_dir, f"{self.hmm_name}.hmm")
        self.hmonnx_name = f"{self.model_name}"
        self.quant_onnx_model_path = os.path.join(
            self.quant_output_dir, f"{self.hmonnx_name}.onnx"
        )
        self.new_quant_onnx_model_path = os.path.join(
            self.quant_output_dir, f"hmquant_{self.hmonnx_name}_with_act.onnx"
        )
        self.golden_dir = os.path.join(self.quant_output_dir, "golden")
        self.quant_advance_cfg = self.quant_cfg.get("config", dict())
        self.upgrade_opset_version()
        # hmatc onnx optimizer initialization
        if "app_onnx_opt" in cfg["model"]:
            self.ApplicationOnnxOpt = HMAppOnnxOptConvert(cfg)

    def get_quant_cfg(self) -> dict:
        return dict()

    def get_quant_dataset(self):
        """提供量化数据"""
        return dict()

    def upgrade_opset_version(self):
        import onnx
        from onnx import version_converter

        model = onnx.load(self.model_path)
        # 遍历模型中的 opset_import 字段（可能有多个域）
        opset_version = None
        for opset in model.opset_import:
            if opset.domain == "":  # 主域（默认的 ONNX operator set）
                opset_version = opset.version
                break
        if opset_version is None:
            logger.warning(f"Not found onnx opset version: {self.model_path}")
            return
        min_opset_version = 11
        if opset_version < min_opset_version:
            new_model_path = self.model_path.replace(".onnx", "_opset11.onnx")
            if not os.path.exists(new_model_path):
                new_model = version_converter.convert_version(model, min_opset_version)
                onnx.save(new_model, new_model_path)
                logger.info(
                    f"Upgrade onnx opset {opset_version} to {min_opset_version}, and save new onnx to: {new_model_path}"
                )
            self.model_path = new_model_path

    def quantize(self):
        """quantize the model"""
        if not os.path.exists(self.quant_output_dir):
            os.makedirs(self.quant_output_dir)
        # 检查opset_version

        if hasattr(self, "ApplicationOnnxOpt"):
            self.ApplicationOnnxOpt.opt()
            if hasattr(self.ApplicationOnnxOpt, "opt_model_path"):
                self.model_path = self.ApplicationOnnxOpt.opt_model_path

        try:
            from xhquant.api import (
                DeviceType,
                HMONNXGoldenInference,
                HMONNXInference,
                QuantScheme,
                convert_onnx_to_hmonnx,
                create_quant_config,
            )
        except ImportError:
            logger.error("Not found xhquant module, and please install xhquant.")
            exit(-1)

        quant_scheme = QuantScheme(
            target_device=DeviceType.XH2a, quant_type=self.quant_type
        )
        quant_config = create_quant_config(quant_scheme)

        in_datas = list()
        for input_name in self.inputs_cfg:
            input_cfg = self.inputs_cfg[input_name]
            shape = input_cfg["shape"]
            dtype_str = self.onnx_inputs_info[input_name]["dtype"]
            in_datas.append(torch.from_numpy(self.gen_random_data(shape, dtype_str)))
        # 量化以及HMONNX导出
        now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        t_start = time.time()
        convert_onnx_to_hmonnx(
            self.model_path,
            in_datas,
            device_type=DeviceType.XH2a,
            out_hmonnx_file=self.quant_onnx_model_path,
            quant_config=quant_config,
            input_names=self.inputs_name,
            output_names=self.outputs_name,
        )
        # 生成芯片所需格式模型
        session = HMONNXGoldenInference(self.quant_onnx_model_path)
        session.to(self.device)
        session.save_golden = True
        session.golden_dir = self.golden_dir
        if os.path.exists(self.golden_dir):
            shutil.rmtree(self.golden_dir)
        session.step = 0
        # float32 -> float16 and int64 -> int32
        for idx, in_data in enumerate(in_datas):
            if in_data.dtype == torch.int64:
                in_datas[idx] = in_datas[idx].type(torch.int32).to(self.device)
            elif in_data.dtype == torch.float32:
                in_datas[idx] = in_data.half().to(self.device)
        session(*in_datas)  #
        if os.path.exists(self.quant_onnx_model_path):
            os.remove(self.quant_onnx_model_path)
        shutil.copytree(
            os.path.join(self.golden_dir, "step_0"),
            self.quant_output_dir,
            dirs_exist_ok=True,
        )
        shutil.rmtree(self.golden_dir)
        # 压缩量化产物
        compress = os.environ.get("HMATC_COMPRESS", "0")
        if compress == "1" and self.enable_upload:
            logger.info("Compressing quant output...")
            runtime_version = get_package_version(f"houmo_tcim_runtime_xh2")
            runtime_version = runtime_version.split(".dev")[0]
            with open(os.path.join(self.quant_output_dir, "VERSION.txt"), "w") as f:
                f.write(f"hmquant_version: {get_hmquant_xh2_version()}\n")
                f.write(f"quant_time: {now}\n")
            filename = f"hmquant_{self.model_dir_name}_xh2_v{runtime_version}.tar.xz"
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
                f"models/v{runtime_version}/{self.model_dir_name}/{filename}",
                max_retries=3,
            )
            logger.info(f"Compressing quant output done.")
        span = time.time() - t_start
        res = dict()
        res["time"] = span
        res_info = {"quant": res, "model": self.model_cfg}
        logger.info(f"Quantize done. and save hmonnx: {self.new_quant_onnx_model_path}")
        return res_info

    def build(self, enable_profile=False):
        self.enable_profile = enable_profile
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
            self.new_quant_onnx_model_path,
            output_name=self.hmm_name,
            ncore=self.build_ncore,
            opt_level=self.build_opt_level,
            target="xh2",
            batch=self.build_batch,
            legacy=True,
            enable_profile=enable_profile,
            output_dir=self.hmm_save_dir,
            work_dir=self.build_output_dir,
            enable_dynamic_image_resize=False,
            # custom_msg=self.custom_msg,
        )
        span = time.time() - t_start
        # 压缩编译后产物
        compress = os.environ.get("HMATC_COMPRESS", "0")
        if compress == "1" and self.enable_upload:
            logger.info("Compressing hmmodel...")
            hmcc_version = get_package_version(f"houmo-tcim-xh2")
            runtime_version = get_package_version(f"houmo_tcim_runtime_xh2")
            with open(os.path.join(self.save_dir, "xh2", "VERSION.txt"), "w") as f:
                f.write(f"hmquant_version: {get_hmquant_xh2_version()}\n")
                f.write(f"tcim_version: {hmcc_version}\n")
                f.write(f"tcim_runtime_version: {runtime_version}\n")
                f.write(f"build_time: {now}\n")
            runtime_version = runtime_version.split(".dev")[0]
            filename = f"{self.model_dir_name}_xh2_b{self.hmm_batch}_{self.build_ncore}core_{self.build_opt_level}_v{runtime_version}.tar.xz"
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
                f"models/v{runtime_version}/{self.model_dir_name}/{filename}",
                max_retries=3,
            )
            logger.info(f"Compressing hmmodel done.")
        res_info = {"build": {"time": span}}
        return res_info

    def check_golden(self, device_id=0):
        xh2 = Xh2Infer()
        xh2.load(self.hmm_path, device_id=device_id)
        in_datas = dict()
        for input_name in self.inputs_cfg:
            new_name = input_name.replace("/", "_")
            golden_input_path = os.path.join(
                self.quant_output_dir,
                f"hmquant_{self.model_name}_{input_name}_input.npy",
            )
            golden_input = np.load(golden_input_path)
            logger.info(f"Load golden: {golden_input_path}")
            logger.info(
                f"[input] name: {input_name}, shape: {list(golden_input.shape)}, stype: {golden_input.dtype}"
            )
            golden_input = np.repeat(golden_input, self.build_batch, axis=0)
            in_datas[input_name] = golden_input

        res_info = dict()
        outputs, _ = xh2.run(in_datas)
        self.save_profile_data(outputs)
        header = ["name", "cosine_dist"]
        table = PrettyTable(header)
        table.title = "xh2 vs hmquant"
        for output_name in outputs:
            new_name = output_name.replace("/", "_")
            golden_output_path = os.path.join(
                self.quant_output_dir,
                f"hmquant_{self.model_name}_{new_name}_output.npy",
            )
            golden_output = np.load(golden_output_path)
            logger.info(f"Load golden: {golden_output_path}")
            logger.info(
                f"[output] name: {output_name}, shape: {list(golden_output.shape)}, dtype: {golden_output.dtype}"
            )
            golden_output = np.repeat(golden_output, repeats=self.build_batch, axis=0)
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
        logger.info(f"Check golden...\n{table}")
        return res_info

    def compare(self, data_path: str, device_id=0):
        t_start = datetime.now().strftime("%Y%m%d%H%M%S")
        # onnx
        onnx_infer = OnnxInfer()
        onnx_infer.load(self.model_path)
        # hmquant
        hmquant_infer = Xh2HmQuantInfer()
        hmquant_infer.load(self.new_quant_onnx_model_path)
        # xh2
        xh2_infer = Xh2Infer()
        xh2_infer.load(self.hmm_path)

        onnx_in_datas = dict()
        hmquant_in_datas = dict()
        xh2_in_datas = dict()
        _, ext = os.path.splitext(os.path.basename(data_path))
        if self.is_image_single_input:
            # 单输入图像
            input_name = self.inputs_name[0]
            input_cfg = self.inputs_cfg[input_name]
            data_format = input_cfg["data_format"]
            input_shape = input_cfg["shape"]
            mean = input_cfg["mean"]
            std = input_cfg["std"]
            resize_type = input_cfg["resize_type"]
            padding_mode = input_cfg.get("padding_mode")
            padding_values = input_cfg.get("padding_values")
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
            # 获取编译后模型batch
            hmm_batch = xh2_infer.inputs_info[input_name].shape[0]
            # preprocess
            N, C, H, W = input_shape
            onnx_data = default_preprocess(
                cv_image,
                (W, H),
                mean=mean,
                std=std,
                use_norm=True,
                use_resize=True,
                use_rgb=data_format == "RGB",
                resize_type=resize_type,
                padding_mode=padding_mode,
                padding_value=padding_values,
            )
            # onnx
            onnx_data = np.repeat(onnx_data, repeats=self.model_input_batch, axis=0)
            onnx_in_datas[input_name] = onnx_data  # np.ndarray
            onnx_data_fp16 = onnx_data.astype(np.float16).copy()
            # hmquant
            hmquant_in_datas[input_name] = torch.from_numpy(onnx_data_fp16).cpu()
            # xh2
            xh2_in_datas[input_name] = np.repeat(
                onnx_data_fp16, repeats=self.build_batch, axis=0
            )
        else:
            # 单输入非图像or多输入
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

        res_info = {"compare": {t_start: dict()}}
        res_info["compare"][t_start]["data_path"] = data_path
        # 计算相似度
        header = ["name", "onnx vs hmquant", "onnx vs xh2", "hmquant vs xh2"]
        table = PrettyTable(header)
        table.title = "Cosine Distance"
        for output_name in onnx_outputs:
            onnx_output = onnx_outputs[output_name]
            hmquant_output = hmquant_outputs[output_name]
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
