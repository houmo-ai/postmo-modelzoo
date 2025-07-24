import os
import abc
import time
import numpy as np
import torch
from typing import Dict, Any
from ..utils import logger
from ..utils.utils import SUPPORT_BACKEND
from ..utils.preprocess import xh1_preprocess, default_preprocess
from ..infer.xh1_infer import Xh1Infer
from ..infer.onnx_infer import OnnxInfer


class BaseModel(object, metaclass=abc.ABCMeta):
    def __init__(self, **kwargs):
        self.time_span = 0
        self.total = 0
        self.engine = None
        self.inputs_cfg = kwargs["inputs_cfg"]
        self.inputs_name = list(self.inputs_cfg.keys())
        self.resizer_mode = kwargs.get("resizer_mode", 0)
        self.roi_num = kwargs.get("roi_num", 1)
        self.backend = kwargs["backend"]
        if self.backend not in SUPPORT_BACKEND:
            logger.error(f"backend not in {SUPPORT_BACKEND}")
            exit(-1)
        if self.backend == "onnx":
            self.engine = OnnxInfer()
        elif self.backend == "xh1":
            self.engine = Xh1Infer()
        elif self.backend == "xh2":
            logger.error("Xh2 not implemented")
            exit(-1)
        else:
            logger.error(f"Not support backend: {self.backend}")
            exit(-1)
        
    def load(self, model_path: str):
        """模型加载"""
        model_name = os.path.basename(model_path)
        _, ext = os.path.splitext(model_name)
        if ext != self.engine.model_ext:
            logger.error(f"{model_name} is not {self.engine.model_ext}")
            exit(-1)
        self.engine.load(model_path)

    def preprocess(self, in_datas: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        """模型前处理"""
        if self.resizer_mode not in [1, 2, 3]:
            # 单输入非图像or多输入，输入数据为外部预处理后数据
            if self.backend == "onnx":
                return in_datas
            if self.backend in ["xh1"]:
                # TODO 需要量化
                raise NotImplementedError
        else:
            new_datas = dict()
            # 单输入图像，可由内部预处理来支持
            in_name = list(in_datas.keys())[0]
            cv_image = in_datas[in_name]
            input_cfg = self.inputs_cfg[in_name]
            input_shape = input_cfg["shape"]
            data_format = input_cfg["data_format"]
            mean = input_cfg["mean"]
            std = input_cfg["std"]
            toYUV_format = input_cfg["toYUV_format"]
            max_input_size = input_cfg["max_input_size"]
            resize_type = input_cfg["resize_type"]
            padding_mode = input_cfg.get("padding_mode")
            padding_values = input_cfg.get("padding_values")
            N, C, H, W = input_shape
            if self.resizer_mode in [1, 2] or self.backend == "onnx":
                # 动态resizer
                im, dyn_info = xh1_preprocess(
                    cv_image, 
                    input_shape, 
                    max_input_size,
                    mean=mean, 
                    std=std,
                    use_resize=self.backend in ["onnx", "xh2"], 
                    use_norm=self.backend == "onnx", 
                    use_rgb=(self.backend in ["onnx", "xh2"] and data_format == "RGB"), 
                    resize_type=resize_type, 
                    padding_mode=padding_mode,
                    padding_values=padding_values, 
                    is_onnx=self.backend in ["onnx", "xh2"],
                    to_YUV=self.backend == "xh1",
                    fmt=toYUV_format
                )
            elif self.resizer_mode == 3:
                # 静态resizer
                im = default_preprocess(
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
                dyn_info = None
            if self.backend in ["onnx", "xh2"]:
                new_datas[in_name] = np.ascontiguousarray(im)
            elif self.backend == "xh1":
                yuv_pad = im.detach().cpu().numpy().flatten()
                if toYUV_format == "YUV420SP":
                    valid_len = yuv_pad.size // 2
                elif toYUV_format == "YUV422SP":
                    valid_len = yuv_pad.size * 2 // 3
                elif toYUV_format in ["YUV444SP", "YUV400"]:
                    valid_len = yuv_pad.size
                yuv = yuv_pad[:valid_len].copy().reshape(1, -1)
                new_datas[in_name] = np.ascontiguousarray(yuv)
                if self.resizer_mode in [1, 2]:
                    dyn_info = dyn_info.detach().cpu().numpy()
                    if self.resizer_mode == 2:
                        dyn_info = dyn_info.flatten()
                    new_datas[f"resizer_crop_{in_name}"] = dyn_info
            elif self.backend == "xh2":
                raise NotImplementedError
            return new_datas
        
    def run(self, in_datas: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        """模型推理"""
        prerpcessed_in_datas = self.preprocess(in_datas)
        # 多batch直接复制数据，以及后续reszier信息的复制
        for name in prerpcessed_in_datas:
            in_data = prerpcessed_in_datas[name]
            if self.backend == "xh1" and name.startswith("resizer_crop_") and self.roi_num > 1:
                prerpcessed_in_datas[name] = np.repeat(in_data, repeats=self.roi_num, axis=0)
                continue
            batch = self.engine.inputs_batch[name]
            prerpcessed_in_datas[name] = np.repeat(in_data, repeats=batch, axis=0)
        t = time.time()
        # 推理
        outs = self.engine.run(prerpcessed_in_datas)
        self.time_span += (time.time() - t)
        # xh1同时输出量化和反量化结果，只取反量化后的
        if isinstance(outs, tuple):
            outs = outs[1]
        # 后处理前只取batch0
        for name in outs:
            out = outs[name][0:1, ...]
            outs[name] = out.copy()
        outs = self.postprocess(outs, in_datas)
        self.total += 1
        return outs
    
    @abc.abstractmethod
    def postprocess(self, outs: Dict[str, np.ndarray], in_datas: Dict[str, np.ndarray]) -> Any:
        """模型后处理"""
        pass

    def unload(self):
        """模型卸载"""
        pass
    
    @abc.abstractmethod
    def demo(self, filepaths: list):
        """模型演示"""
        pass
    
    @abc.abstractmethod
    def evaluate(self, dataset, num=0):
        """模型评估"""
        pass

    @property
    def ave_latency_ms(self):
        if self.total == 0:
            return 0
        return (self.time_span / self.total) * 1000