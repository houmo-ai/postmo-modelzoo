import os
import cv2
import shutil
import time
import torch
import numpy as np
from datetime import datetime
from prettytable import PrettyTable
from ..utils import logger
from ..utils.utils import get_md5, SUPPORT_IMAGE_FORMATS, load_npz, str_to_torch_dtype
from ..utils.preprocess import xh1_preprocess, default_preprocess 
from ..utils.dist_metrics import cosine_distance
from ..base.base_exec import BaseExec
from ..infer.xh2_infer import Xh2Infer
from ..infer.onnx_infer import OnnxInfer
from ..infer.hmquant_infer import HmQuantInfer


class Xh2Exec(BaseExec):
    def __init__(self, cfg: dict) -> None:
        super().__init__(cfg)
        self.hmm_batch = self.build_batch * self.model_input_batch
        self.hmm_name = f"{self.model_name}_xh2_b{self.hmm_batch}"
        self.hmm_save_dir = os.path.join(self.save_dir, "xh2")
        if not os.path.exists(self.hmm_save_dir):
            os.makedirs(self.hmm_save_dir)
        self.hmm_path = os.path.join(self.hmm_save_dir, f"{self.hmm_name}.hmm")
        self.quant_output_dir = os.path.join(self.save_dir, "xh2", "hmquant")
        self.build_output_dir = os.path.join(self.save_dir, "xh2", "tcim")
        self.quant_sequencer_model_path = os.path.join(self.quant_output_dir, f"{self.model_name}_xh2_b{self.model_input_batch}.pkl")
        self.quant_onnx_model_path = os.path.join(self.save_dir, "xh2", f"{self.model_name}_xh2.onnx")
        self.golden_dir = os.path.join(self.quant_output_dir, "golden")
        self.quant_advance_cfg = self.quant_cfg.get("config", dict())
  
    def get_quant_cfg(self) -> dict:
        return dict()
    
    def get_quant_dataset(self):
        """提供量化数据"""
        return dict()
        
    def quantize(self):
        """quantize the model"""
        if not os.path.exists(self.quant_output_dir):
            os.makedirs(self.quant_output_dir)
        from xhquant.api import (
            DeviceType,
            HMONNXGoldenInference,
            HMONNXInference,
            QuantScheme,
            convert_onnx_to_hmonnx,
            create_quant_config
        )
        quant_type = "w8a8h1_sefp"
        quant_scheme = QuantScheme(
            target_device=DeviceType.XH2a, 
            quant_type=quant_type)
        quant_config = create_quant_config(quant_scheme)
        
        in_datas = list()
        for input_name in self.inputs_cfg:
            input_cfg = self.inputs_cfg[input_name]
            shape = input_cfg["shape"]
            dtype_str = self.onnx_inputs_info[input_name]["dtype"]
            in_datas.append(torch.randn(shape, dtype=str_to_torch_dtype(dtype_str)))
        # 量化以及HMONNX导出
        t_start = time.time()
        convert_onnx_to_hmonnx(
            self.model_path,
            in_datas,
            device_type=DeviceType.XH2a,
            out_hmonnx_file=self.quant_onnx_model_path,
            quant_config=quant_config,
            input_names=self.inputs_name
        )
        # 生成芯片所需格式模型
        session = HMONNXGoldenInference(self.quant_onnx_model_path)
        session.to(self.device)
        session.save_golden = True
        session.golden_dir = self.golden_dir
        if not os.path.exists(self.golden_dir):
            os.makedirs(self.golden_dir)
        else:
            shutil.rmtree(self.golden_dir)
        session.step = 0
        # to float16
        for idx, in_data in enumerate(in_datas):
            in_datas[idx] = in_data.half().to(self.device)
        session(*in_datas)   # 
        span = time.time() - t_start
        res = dict()
        res["time"] = span
        res_info = {"quant": res, "model": self.model_cfg}
        logger.info(f"Quantize done. and save hmonnx: {self.quant_onnx_model_path}")
        return res_info

    def build(self):
        if self.build_batch > 1 and self.resizer_mode in [1, 2]:
            logger.error("Not support multi-batch, when enable dynamic resizer")
            exit(-1)
        if not os.path.exists(self.build_output_dir):
            os.makedirs(self.build_output_dir)
        import tcim
        t_start = time.time()
        tcim.build_from_hmonnx(
            self.quant_onnx_model_path,
            output_name=self.hmm_name,
            ncore=self.build_ncore,
            opt_level=self.build_opt_level,
            target="xh2",
            batch=self.build_batch,
            legacy=True,
            output_dir=self.hmm_save_dir,
            work_dir=self.build_output_dir,
            enable_dynamic_image_resize=False
        )
        span = time.time() - t_start
        res_info = {"build": {"time": span}}
        return res_info
    
    def check_golden(self):
        xh2 = Xh2Infer()
        xh2.load(self.hmm_path)
        in_datas = dict()
        for input_name in self.inputs_cfg:
            golden_input_path = os.path.join(self.golden_dir, "step_0", f"hmquant_{self.model_name}_xh2_{input_name}_input.npy")
            golden_input = np.load(golden_input_path)
            logger.info(f"Load golden: {golden_input_path}")
            logger.info(f"[input] name: {input_name}, shape: {list(golden_input.shape)}, stype: {golden_input.dtype}")
            if self.build_batch > 1:
                # 静态resizer且是编译时多batch
                golden_input = np.repeat(golden_input, self.build_batch, axis=0)
            in_datas[input_name] = golden_input

        res_info = dict()
        # TODO 图像输入目前暂不支持多batch
        outputs, _ = xh2.run(in_datas)
        header = ["name",  "cosine_dist"]
        table = PrettyTable(header)
        table.title = "xh2 vs hmquant"
        for output_name in outputs:
            golden_output_path = os.path.join(self.golden_dir, "step_0", f"hmquant_{self.model_name}_xh2_{output_name}_output.npy")
            golden_output = np.load(golden_output_path)
            logger.info(f"Load golden: {golden_output_path}")
            logger.info(f"[output] name: {output_name}, shape: {list(golden_output.shape)}, dtype: {golden_output.dtype}")
            if self.build_batch > 1:
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
       
    def compare(self, data_path: str):
        raise NotImplementedError
        t_start = datetime.now().strftime("%Y%m%d%H%M%S")
        # onnx
        onnx_infer = OnnxInfer()
        onnx_infer.load(self.model_path)
        # hmquant
        hmquant_infer = HmQuantInfer()
        hmquant_infer.load(self.quant_sequencer_model_path)
        # xh1
        xh1_infer = Xh1Infer()
        xh1_infer.load(self.hmm_path)
        
        onnx_in_datas = dict()
        hmquant_in_datas = dict()
        xh1_in_datas = dict()
        _, ext = os.path.splitext(os.path.basename(data_path))
        if self.is_image_single_input:
            # 单输入图像
            input_name = self.inputs_name[0]
            input_cfg = self.inputs_cfg[input_name]
            data_format = input_cfg["data_format"]
            use_rgb = True if data_format == "RGB" else False
            input_shape = input_cfg["shape"]
            max_input_size = input_cfg["max_input_size"]
            mean = input_cfg["mean"]
            std = input_cfg["std"]
            resize_type = input_cfg["resize_type"]
            padding_mode = input_cfg.get("padding_mode")
            padding_values = input_cfg.get("padding_values")
            toYUV_format = input_cfg.get("toYUV_format")
            if ext not in SUPPORT_IMAGE_FORMATS:
                logger.error(f"Not support image: {data_path}")
                exit(-1)
            if not os.path.exists(data_path):
                logger.error(f"Not found data_path: {data_path}")
                exit(-1)
            cv_image = cv2.imread(data_path, cv2.IMREAD_COLOR if data_format != "GRAY" else cv2.IMREAD_GRAYSCALE)
            if cv_image is None:
                logger.error("Failed to decode image")
                exit(-1)
            # 获取编译后模型batch
            hmm_batch = xh1_infer.inputs_info[input_name].shape[0]
            # onnx
            N, C, H, W = input_shape
            onnx_data = default_preprocess(
                cv_image, 
                (W, H), 
                mean=mean, 
                std=std, 
                use_norm=True, 
                use_resize=True, 
                use_rgb=use_rgb,
                resize_type=resize_type, 
                padding_mode=padding_mode, 
                padding_value=padding_values,
            )
            if self.model_input_batch > 1:
                onnx_data = np.repeat(onnx_data, repeats=self.model_input_batch, axis=0)
            onnx_in_datas[input_name] = onnx_data  # np.ndarray
            
            # hmquant
            if self.resizer_mode in [1, 2]:
                yuv_pad_hwc, dyn_info = xh1_preprocess(
                    cv_image, 
                    input_shape, 
                    max_input_size, 
                    mean=mean, 
                    std=std, 
                    use_norm=False, 
                    use_resize=False, 
                    use_rgb=False,
                    resize_type=resize_type, 
                    padding_mode=padding_mode, 
                    padding_values=padding_values,
                    is_onnx=False, 
                    to_YUV=True,
                    fmt=toYUV_format
                )
            elif self.resizer_mode == 3:
                yuv_pad_hwc = default_preprocess(
                    cv_image, 
                    (W, H), 
                    mean=None,
                    std=None, 
                    use_norm=False, 
                    use_resize=True, 
                    use_rgb=False,
                    resize_type=0,
                    to_YUV=True,
                    fmt=toYUV_format
                )
            h, w, c = yuv_pad_hwc.shape
            yuv_pad = yuv_pad_hwc.view(1, c, h, w)
            if self.model_input_batch > 1:
                yuv_pad = yuv_pad.repeat_interleave(self.model_input_batch, dim=0)
            hmquant_in_datas[input_name] = yuv_pad.contiguous() # torch.Tensor
            
            # xh1
            yuv_pad = yuv_pad.detach().cpu().numpy().flatten()
            if toYUV_format == "YUV420SP":
                valid_len = yuv_pad.size // 2
            elif toYUV_format == "YUV422SP":
                valid_len = yuv_pad.size * 2 // 3
            elif toYUV_format in ["YUV444SP", "YUV400"]:
                valid_len = yuv_pad.size
            yuv = yuv_pad[:valid_len].copy()
            yuv = yuv.reshape(1, -1)
            if hmm_batch > 1:
                yuv = np.repeat(yuv, repeats=hmm_batch, axis=0) 
            xh1_in_datas[input_name] = np.ascontiguousarray(yuv)   # np.ndarray

            if self.resizer_mode in [1, 2]:
                hmquant_in_datas[f"resizer_crop_{input_name}"] = dyn_info
                xh1_in_datas[f"resizer_crop_{input_name}"] = dyn_info.detach().cpu().numpy()
        else:
            # 单输入非图像or多输入
            in_datas = load_npz(data_path)
            onnx_in_datas = in_datas
            for input_name in in_datas:
                in_data_quanted = xh1_infer.quantize(input_name, in_datas[input_name])
                hmquant_in_datas[input_name] = torch.from_numpy(in_data_quanted)
                xh1_in_datas[input_name] = np.repeat(in_data_quanted, self.build_batch, axis=0)
        
        onnx_outputs = onnx_infer.run(onnx_in_datas)
        hmquant_outputs = hmquant_infer.run(hmquant_in_datas)
        _, xh1_outputs_dequanted = xh1_infer.run(xh1_in_datas)
        
        res_info = {"compare": {t_start: dict()}}
        res_info["compare"][t_start]["data_path"] = data_path
        # 计算相似度
        header = ["name", "onnx vs hmquant", "onnx vs xh1", "hmquant vs xh1"]
        table = PrettyTable(header)
        table.title = "Cosine Distance"
        for output_name in onnx_outputs:
            onnx_output = onnx_outputs[output_name]
            hmquant_output = hmquant_outputs[output_name]
            xh1_output_dequanted = xh1_outputs_dequanted[output_name]
            if self.is_image_single_input and self.build_batch > 1:
                xh1_output_dequanted = np.split(xh1_output_dequanted, self.build_batch, axis=0)[0]
            onnx_vs_hmquant = cosine_distance(onnx_output, hmquant_output)
            onnx_vs_xh1 = cosine_distance(onnx_output, xh1_output_dequanted)
            hmquant_vs_xh1 = cosine_distance(hmquant_output, xh1_output_dequanted)
            table.add_row([output_name, f"{onnx_vs_hmquant:.6f}", f"{onnx_vs_xh1:.6f}", f"{hmquant_vs_xh1:.6f}"])
            res_info["compare"][t_start][output_name] = {
                "onnx_vs_hmquant": float(onnx_vs_hmquant),
                "onnx_vs_xh1": float(onnx_vs_xh1),
                "hmquant_vs_xh1": float(hmquant_vs_xh1),
            }
        logger.info(f"\n{table}")
        return res_info
        
    def perf(self, warmup_num, sample_num, loop_num=1, device_num=1, thread_num=1):
        from ..python import perf
        inputs_hw = dict()  # 收集图像输入的HW
        for input_name in self.inputs_cfg:
            input_cfg = self.inputs_cfg[input_name]
            if self.is_image_single_input:
                inputs_hw[input_name] = input_cfg["shape"][2:]
        
        # TODO 使用golden数据
        perf_info = perf.CModelRunner(
            self.hmm_path, sample_num, thread_num, 
            device_num, loop_num, warmup_num, inputs_hw)
        t_start = datetime.now().strftime("%Y%m%d%H%M%S")
        res_info = {
            "perf": {
                t_start: {
                    "params": {
                        "hmm_path": self.hmm_path,
                        "thread_num": thread_num,
                        "device_num": device_num,
                        "loop_num": loop_num,
                        "warmup_num": warmup_num,
                        "sample_num": sample_num,
                    },
                    "perf_info": {
                        "input_avg_latency": perf_info.input_avg_latency,
                        "input_max_latency": perf_info.input_max_latency,
                        "infer_avg_latency": perf_info.infer_avg_latency,
                        "infer_max_latency": perf_info.infer_max_latency,
                        "output_avg_latency": perf_info.output_avg_latency,
                        "output_max_latency": perf_info.output_max_latency,
                        "avg_cost": perf_info.avg_cost,
                        "qps": perf_info.qps
                    }
                }
            }
        }
        return res_info
    
    def demo(self, backend):
        """Demo入口"""
        if not self.demo_cfg:
            logger.error("demo config not found")
            exit(-1)
        data_dir = self.demo_cfg.get("data_dir", "")
        if not os.path.exists(data_dir):
            logger.error(f"demo data_dir not exist -> {data_dir}")
            exit(-1)
        if not os.path.isdir(data_dir):
            logger.error(f"demo data_dir is not a directory -> {data_dir}")
            exit(-1)
        test_num = self.demo_cfg.get("num", 0)
        if not isinstance(test_num, int):
            logger.error(f"test_num must be int -> {test_num}")
            exit(-1)
        if test_num < 0:
            logger.error(f"test_num must >= 0 -> {test_num}")
            exit(-1)
        filenames = os.listdir(data_dir)
        data_num = len(filenames)
        if test_num > 0 and test_num < data_num:
            filenames = filenames[:test_num]
        filepaths = list()
        for filename in filenames:
            filepath = os.path.join(data_dir, filename)
            if not os.path.exists(filepath):
                logger.warning(f"filepath not exists -> {filepath}")
                continue
            filepaths.append(filepath)
        model = self.get_model(backend)
        model_path = self.model_path if backend == "onnx" else self.hmm_path
        model.load(model_path)
        model.demo(filepaths)
        model.unload()
    
    def evaluate(self, backend):
        """评估入口"""
        if not self.eval_cfg:
            logger.error("demo config not found")
            exit(-1)
        data_dir = self.eval_cfg.get("data_dir", "")
        if not os.path.exists(data_dir):
            logger.error(f"data_dir not exist -> {data_dir}")
            exit(-1)
        if not os.path.isdir(data_dir):
            logger.error(f"data_dir is not a directory -> {data_dir}")
            exit(-1)
        num = self.eval_cfg.get("num", 0)
        if not isinstance(num, int):
            logger.error(f"eval test_num must be int -> {num}")
            exit(-1)
        if num < 0:
            logger.error(f"eval test_num must >= 0 -> {num}")
            exit(-1)
        # 获取dataset
        dataset = self.get_dataset()
        # 获取模型
        model = self.get_model(backend)
        model_path = self.model_path if backend == "onnx" else self.hmm_path
        model.load(model_path)
        res = model.evaluate(dataset, num)
        model.unload()
        logger.info(f"{res}")
