import os
import cv2
import time
import torch
import numpy as np
from datetime import datetime
from prettytable import PrettyTable
from ..utils import logger
from ..utils.utils import get_md5, SUPPORT_IMAGE_FORMATS, load_npz
from ..utils.preprocess import xh1_preprocess, default_preprocess 
from ..utils.dist_metrics import cosine_distance
from ..base.base_exec import BaseExec
from ..infer.xh1_infer import Xh1Infer
from ..infer.onnx_infer import OnnxInfer
from ..infer.hmquant_infer import HmQuantInfer


class Xh1Exec(BaseExec):
    def __init__(self, cfg: dict) -> None:
        super().__init__(cfg)
        self.hmm_batch = self.build_batch * self.model_input_batch
        self.hmm_name = f"{self.model_name}_xh1_b{self.hmm_batch}_{self.build_ncore}core_{self.build_opt_level}"
        self.hmm_save_dir = os.path.join(self.save_dir, "xh1")
        if not os.path.exists(self.hmm_save_dir):
            os.makedirs(self.hmm_save_dir)
        self.hmm_path = os.path.join(self.hmm_save_dir, f"{self.hmm_name}.hmm")
        self.quant_output_dir = os.path.join(self.save_dir, "xh1", "hmquant")
        self.build_output_dir = os.path.join(self.save_dir, "xh1", "tcim")
        self.quant_sequencer_model_path = os.path.join(
            self.quant_output_dir, f"{self.model_name}_xh1_b{self.model_input_batch}.pkl")
        self.quant_onnx_model_path = os.path.join(
            self.quant_output_dir, "hmquant_" + self.model_name + "_with_act.onnx")
        self.quant_advance_cfg = self.quant_cfg.get("config", dict())
        self.enable_static_resizers = list()
        self.max_inputs_size = dict()
        self.resizers_cfg = list()
        for input_name in self.inputs_cfg:
            data_format = self.inputs_cfg[input_name].get("data_format")
            if data_format is None:
                continue
            resizer_cfg = self.inputs_cfg[input_name].get("resizer", dict())
            self.resizers_cfg.append(resizer_cfg)
            max_input_size = resizer_cfg.get("max_input_size", list())
            enable_static_resizer = resizer_cfg.get("enable_static_resizer", False)
            self.enable_static_resizers.append(enable_static_resizer)
            self.max_inputs_size[input_name] = max_input_size
        # xh1 resizer工作模式
        # 0 - 输入为非图像数据or多输入情况，禁用resizer，相当于非图像输入
        # 1 - 全动态resizer，参数为10个值[y, x, height, width, h, w, top, left, bottom, right]
        # 2 - crop部分动态resizer, 参数为4个值[y, x, height, width]
        # 3 - 静态resizer，使用场景几乎没有，不建议用
        self.resizer_mode = 0
        if self.is_image_single_input and len(self.resizers_cfg[0]) != 0:
            # 单输入图像且设置了resizer参数
            if self.resize_types[0] == 0:
                self.resizer_mode = 2 if not self.enable_static_resizers[0] else 3
            elif self.resize_types[0] == 1:
                self.resizer_mode = 1
        logger.info(f"resizer_mode: {self.resizer_mode}")
        # roi模式
        # 0 - 1图n框
        # 1 - n图n框，每图1框，比如：1图1框、2图2框、...
        self.roi_num = self.build_cfg.get("roi_num", 1)
        if not isinstance(self.roi_num, int) or self.roi_num < 1:
            logger.error("roi_num must be int, and >= 1")
            exit(-1)
        if self.roi_num > 1 and self.hmm_batch > 1:
            logger.error("Not support roi_num > 1, when model_input_batch > 1 or build_batch > 1")
            exit(-1)
        # 暂不支持
        if self.resizer_mode == 2 and (self.roi_num > 1 or self.build_batch > 1 or self.model_input_batch > 1):
            logger.error("Not support roi_num > 1 or batch > 1 yet, when resizer_mode == 2")
            exit(-1)

    def get_quant_cfg(self) -> dict:
        # 设置量化日志输出
        log_dir = os.path.join(self.save_dir, "xh1", "logs")
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)
        quant_cfg = {
            "inputs_cfg": dict(),
            "quant_cfg": self.quant_advance_cfg,
            "extra_cfg": {
                "with_label": False,
                "log_dir": log_dir,
            }
        }
        for input_name in self.inputs_cfg:
            input_cfg = self.inputs_cfg[input_name]
            input_shape = input_cfg["shape"]
            data_format = input_cfg["data_format"]
            quant_cfg["inputs_cfg"] = {
                input_name: {
                    "data_format": self.dtype_transform(self.onnx_inputs_info[input_name]["dtype"]),
                    "first_layer_weight_denorm_mean": None,
                    "first_layer_weight_denorm_std": None,
                }
            }
            # 非图像or多输入跳过
            if not self.is_image_single_input or self.resizer_mode == 0:
                continue
            
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
            toYUV_format = input_cfg["resizer"]["toYUV_format"]
            new_input_cfg["toYUV_format"] = toYUV_format[0:6]  # 去掉SP
            N, C, H, W = input_shape
            insert_pad_scatter = input_cfg.get("insert_pad_scatter", False)
            if insert_pad_scatter not in [False, True]:
                logger.error(f"Not support insert_pad_scatter: {insert_pad_scatter} yet")
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
            new_input_cfg["resizer_crop"] = {"top": 0, "left": 0, "height": H, "width": W}
            new_input_cfg["dynamic_crop"] = self.resizer_mode in [1, 2]
            new_input_cfg["fold"] = self.resizer_mode in [2, 3]  # 可量化内部判断
            if resize_type == 1:
                # padding 
                padding_values = input_cfg.get["padding_values"]
                padding_values = [v - 128 for v in padding_values]  # 需要减128、
                new_input_cfg["resizer_pad"] = {"value": padding_values}
            quant_cfg["inputs_cfg"][input_name].update(new_input_cfg)
        return quant_cfg
    
    def get_quant_dataset(self):
        """提供量化数据"""
        input_name = self.inputs_name[0]
        input_cfg = self.inputs_cfg[input_name]
        calib_num = self.quant_cfg.get("calib_num")
        data_format = input_cfg.get("data_format")            
        if self.is_image_single_input:
            # 单输入且输入为图像
            input_name = self.inputs_name[0]
            input_cfg = self.inputs_cfg[input_name]
            input_shape = input_cfg["shape"]
            N, C, H, W = input_shape
            if self.resizer_mode != 0:
                max_input_size = self.max_inputs_size[input_name]
                max_height, max_width = max_input_size
            else:
                max_height, max_width = H, W
            mean = input_cfg["mean"]
            std = input_cfg["std"]
            resize_type = input_cfg["resize_type"]
            padding_mode = input_cfg.get("padding_mode")
            padding_values = input_cfg.get("padding_values")
            
            if N > 1 and self.resizer_mode in [1, 2]:
                logger.error(f"model_input_batch > 1 is not supported dynamic resizer")
                exit(-1)
            if self.use_random_data:
                # 随机图像，不管动态静态都以max_input_size来crop
                in_datas = dict()
                for idx in range(calib_num):
                    in_data = torch.randint(low=0, high=255, size=(N, C, max_height, max_width), dtype=torch.uint8)
                    if self.resizer_mode == 1:
                        in_datas[f"resizer_crop_{input_name}"] = \
                            torch.Tensor([0, 0, max_height, max_width, H, W, 0, 0, 0, 0]).type(torch.int32).view(1, -1)
                    elif self.resizer_mode == 2:
                        in_datas[f"resizer_crop_{input_name}"] = torch.Tensor([0, 0, max_height, max_width]).type(torch.int32)
                    if self.resizer_mode == 0:
                        cv_image = in_data[0].permute(1, 2, 0).contiguous().detach().cpu().numpy()
                        im = default_preprocess(
                                cv_image,
                                (W, H),
                                mean=mean, 
                                std=std, 
                                use_norm=True, 
                                use_resize=True,
                                use_rgb=data_format == "RGB",  # 对灰度无效
                                resize_type=resize_type,
                                padding_mode=padding_mode,
                                padding_value=padding_values
                            )
                        in_data = torch.from_numpy(im)
                    in_datas[input_name] = in_data
                    logger.info(f"Processing calibration random data {idx}...")
                    yield in_datas
            else:
                # 真实图像
                filenames = os.listdir(self.calib_data)
                # 填充图片
                padding_len = len(filenames) % N
                for idx in range(padding_len):
                    filenames.append(filenames[0])
                # 切分图片
                actual_calib_num = len(filenames) // N
                if actual_calib_num < calib_num:
                    logger.warning(f"The number of calibration data is less than the number of calibration samples")
                    calib_num = actual_calib_num
                in_datas = dict()
                for idx in range(calib_num):
                    batch_filenames = filenames[idx * N: (idx + 1) * N]
                    batch_datas = list()
                    dyn_infos = list()
                    logger.info(f"Processing calibration data {idx}...")
                    for filename in batch_filenames:
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
                        if self.resizer_mode == 0:
                            im = default_preprocess(
                                cv_image,
                                (W, H),
                                mean=mean, 
                                std=std, 
                                use_norm=True, 
                                use_resize=True,
                                use_rgb=data_format == "RGB",  # 对灰度无效
                                resize_type=resize_type,
                                padding_mode=padding_mode,
                                padding_value=padding_values
                            )
                            im = torch.from_numpy(im)
                            dyn_info = None
                        elif self.resizer_mode in [1, 2]:
                            im, dyn_info = xh1_preprocess(
                                cv_image, 
                                input_shape, 
                                max_input_size, 
                                mean=None, 
                                std=None, 
                                use_norm=False, 
                                use_resize=False,
                                use_rgb=True,  # 对灰度无效
                                resize_type=resize_type, 
                                padding_mode=padding_mode, 
                                is_onnx=False
                            )
                        elif self.resizer_mode == 3:
                            # 直接将数据缩放只输入size
                            im = default_preprocess(
                                cv_image,
                                (W, H),
                                mean=None, 
                                std=None, 
                                use_norm=False, 
                                use_resize=True,
                                use_rgb=True,  # 对灰度无效
                                resize_type=0
                            )
                            im = torch.from_numpy(im)
                            dyn_info = None
                        dyn_infos.append(dyn_info)
                        batch_datas.append(im)
                    in_datas[input_name] = torch.cat(batch_datas, dim=0)
                    if self.resizer_mode in [1, 2]:
                        batch_dyninfos = torch.cat(dyn_infos, dim=0)
                        # dyn_info暂时不支持batch，先去掉batch维度
                        if self.resizer_mode == 2:
                            batch_dyninfos = batch_dyninfos.view(-1)
                        in_datas[f"resizer_crop_{input_name}"] = batch_dyninfos
                    yield in_datas
        else:
            # 单输入且输入为非图像 or 多输入
            in_datas = dict()
            if self.use_random_data:
                for idx in range(calib_num):
                    for input_name in self.inputs_cfg:
                        dtype = self.onnx_inputs_info[input_name]["dtype"]
                        input_shape = self.inputs_cfg[input_name]["shape"]
                        data = self.gen_random_data(input_shape, dtype)
                        in_datas[input_name] = torch.from_numpy(data)
                    logger.info(f"Processing calibration random data {idx}...")
                    yield in_datas
            else:
                filenames = os.listdir(self.calib_data)
                if len(filenames) < calib_num:
                    logger.warning(f"The number of calibration data is less than the number of calibration samples")
                    calib_num = len(filenames)
                for idx in range(calib_num):
                    filename = filenames[idx]
                    data_path = os.path.join(self.calib_data, filename)
                    in_datas = load_npz(data_path)
                    for input_name in in_datas:
                        in_data = in_datas[input_name]
                        batch = in_data.shape[0]
                        onnx_dtype = self.onnx_inputs_info[input_name]["dtype"]
                        assert onnx_dtype == in_data.dtype, "npz data dtype must be equal to onnx input dtype"
                        assert batch == self.model_inputs_batch[input_name], "npz data batch must be equal to onnx input batch"
                        in_datas[input_name] = torch.from_numpy(in_datas[input_name])
                    yield in_datas
        
    def quantize(self):
        """quantize the model"""
        # quant info
        if self.quant_cfg is None:
            logger.error("quant info not found")
            exit(-1)
        calib_data = self.quant_cfg.get("calib_data")
        if calib_data is not None:
            if not os.path.isdir(calib_data):
                logger.error("calib_data must be a directory")
                exit(-1)
            if not os.path.exists(calib_data):
                logger.error("calib_data not exist")
                exit(-1)
            
        from hmquant.api import quant_single_onnx_network, generate_golden, quantize_profiling
        t_start = time.time()
        sequencer = quant_single_onnx_network(
            cfg=self.get_quant_cfg(),
            calibration_data=self.get_quant_dataset(),
            onnx_model_or_path=self.model_path,
            device=self.device
        )
        span = time.time() - t_start
        calib_dataset = self.get_quant_dataset()
        in_datas = next(calib_dataset)
        res = quantize_profiling(
            sequencer, 
            [in_datas], 
            device="cpu",
            mode=0, # 0：累积误差  1：单算子对比
            quant_mode="quant_forward",
            return_o_metric=True
        )
        res = {out_name: {k: float(v) if isinstance(v, np.float64) else v \
            for k, v in metrics.items()} for out_name, metrics in res.items()}
        if not os.path.exists(self.quant_output_dir):
            os.makedirs(self.quant_output_dir)
        generate_golden(
            sequencer=sequencer,
            calibset=in_datas,
            save_path=self.quant_output_dir,
            model_name=self.model_name,
            batch_size=self.model_inputs_batch,
            device="cpu",
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
        sequencer.save_pkl(self.quant_output_dir, f"{self.model_name}_xh1_b{self.model_input_batch}")
        res["time"] = span
        res_info = {"quant": res, "model": self.model_cfg}
        return res_info

    def build(self):
        if not os.path.exists(self.build_output_dir):
            os.makedirs(self.build_output_dir)
        import tcim
        t_start = time.time()
        tcim.build_from_hmonnx(
            self.quant_onnx_model_path,
            output_name=self.hmm_name,
            ncore=self.build_ncore,
            opt_level=self.build_opt_level,
            target="xh1",
            batch=self.build_batch if self.roi_num == 1 else self.roi_num,
            legacy=True,
            output_dir=self.hmm_save_dir,
            work_dir=self.build_output_dir,
            enable_dynamic_image_resize=self.resizer_mode in [1],
            one_img_multi_roi=self.roi_num > 1,
        )
        span = time.time() - t_start
        res_info = {"build": {"time": span}}
        return res_info
    
    def check_golden(self):
        xh1 = Xh1Infer()
        xh1.load(self.hmm_path)
        in_datas = dict()
        for input_name in self.inputs_cfg:
            golden_input_path = os.path.join(self.quant_output_dir, f"hmquant_{self.model_name}_{input_name}_input.npy")
            golden_input = np.load(golden_input_path)
            # 编译时多batch，需要复制数据
            if self.build_batch > 1:
                golden_input = np.repeat(golden_input, self.build_batch, axis=0)
            in_datas[input_name] = golden_input
            if self.resizer_mode in [1, 2]:
                # 编译后模型多batch，dynamic_resizer数据需要复制
                golden_dyn_info_input_path = os.path.join(self.quant_output_dir, f"hmquant_{self.model_name}_resizer_crop_{input_name}_input.npy")
                golden_dyn_input = np.load(golden_dyn_info_input_path)
                repeats = 1
                if self.roi_num > 1:  # 1图n框
                    repeats = self.roi_num
                elif self.roi_num == 1 and self.build_batch > 1:  # n图n框，且编译batch>1
                    repeats = self.build_batch
                golden_dyn_input = np.repeat(golden_dyn_input, repeats=repeats, axis=0)
                in_datas[f"resizer_crop_{input_name}"] = golden_dyn_input
        
        res_info = dict()       
        # TODO 图像输入目前暂不支持多batch
        outputs, outputs_dequanted = xh1.run(in_datas)
        header = ["name",  "cosine_dist", "MD5", "cosine_dist[dequanted]", "MD5[dequanted]"]
        table = PrettyTable(header)
        table.title = "xh1 vs hmquant"
        for output_name in outputs:
            golden_output_path = os.path.join(self.quant_output_dir, f"hmquant_{self.model_name}_{output_name}_output.npy")
            golden_output_dequant_path = os.path.join(self.quant_output_dir, f"hmquant_{self.model_name}_{output_name}_dequant_output.npy")
            golden_output = np.load(golden_output_path)
            golden_output_dequanted = np.load(golden_output_dequant_path)
            repeats = 1
            if self.build_batch > 1 and self.roi_num == 1:
                # n图n框，且编译batch>1
                repeats = self.build_batch
            elif self.roi_num > 1:
                # 1图n框
                repeats = self.roi_num
            golden_output = np.repeat(golden_output, repeats=repeats, axis=0)
            golden_output_dequanted = np.repeat(golden_output_dequanted, repeats=repeats, axis=0)
            golden_output_md5 = get_md5(golden_output)
            golden_output_dequanted_md5 = get_md5(golden_output_dequanted)
            output = outputs[output_name]
            output_dequanted = outputs_dequanted[output_name]
            output_md5 = get_md5(output)
            output_dequanted_md5 = get_md5(output_dequanted)
            # compare
            dist = cosine_distance(golden_output, output)
            dist_dequanted = cosine_distance(golden_output_dequanted, output_dequanted)
            table.add_row([
                output_name,
                f"{dist:.6f}", 
                "ok" if output_md5 == golden_output_md5 else "fail",
                f"{dist_dequanted:.6f}",
                "ok" if output_dequanted_md5 == golden_output_dequanted_md5 else "fail"
            ])
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
        logger.info(f"Check golden...\n{table}")
        return res_info
       
    def compare(self, data_path: str):
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
            
            if self.resizer_mode in [1, 2]:
                max_input_size = input_cfg["max_input_size"]
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
            
            if self.resizer_mode in [1, 2, 3]:
                # 使用resizer
                h, w, c = yuv_pad_hwc.shape
                yuv_pad = yuv_pad_hwc.view(1, c, h, w)
                if self.model_input_batch > 1:
                    yuv_pad = yuv_pad.repeat_interleave(self.model_input_batch, dim=0)
                hmquant_in_datas[input_name] = yuv_pad.contiguous() # torch.Tensor
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
                if hmm_batch > 1:
                    yuv = np.repeat(yuv, repeats=hmm_batch, axis=0) 
                xh1_in_datas[input_name] = np.ascontiguousarray(yuv)   # np.ndarray
            elif self.resizer_mode == 0:
                # 禁用resizer
                in_data = np.repeat(onnx_data, repeats=self.build_batch, axis=0)
                in_data_quanted = xh1_infer.quantize(input_name, in_data)
                hmquant_in_datas[input_name] = torch.from_numpy(in_data_quanted[0:self.model_input_batch, ...])
                xh1_in_datas[input_name] = np.ascontiguousarray(in_data_quanted)
            
            # dynamic_resizer info
            if self.resizer_mode in [1, 2]:
                if self.roi_num > 1:
                    # 1图n框
                    hmquant_dyn_info = dyn_info
                    xh1_dyn_info = dyn_info.repeat_interleave(self.roi_num, dim=0)
                else:
                    # n图n框
                    hmquant_dyn_info = dyn_info.repeat_interleave(self.model_input_batch, dim=0)
                    xh1_dyn_info = dyn_info.repeat_interleave(hmm_batch, dim=0)
                # dyn_info暂时不支持batch，先去掉batch维度
                if self.resizer_mode == 2:
                    hmquant_dyn_info = hmquant_dyn_info.view(-1)
                    xh1_dyn_info = xh1_dyn_info.view(-1)
                hmquant_in_datas[f"resizer_crop_{input_name}"] = hmquant_dyn_info
                xh1_in_datas[f"resizer_crop_{input_name}"] = xh1_dyn_info.detach().cpu().numpy()
        else:
            # 单输入非图像or多输入
            if ext != ".npz":
                logger.error(f"{data_path} is not npz file")
                exit(-1)
            in_datas = load_npz(data_path)
            onnx_in_datas = in_datas
            for input_name in in_datas:
                in_data = in_datas[input_name]
                in_data = np.repeat(in_data, repeats=self.build_batch, axis=0)
                in_data_quanted = xh1_infer.quantize(input_name, in_data)
                hmquant_in_datas[input_name] = torch.from_numpy(in_data_quanted[0:self.model_inputs_batch[input_name], ...])
                xh1_in_datas[input_name] = in_data_quanted

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
            xh1_output_dequanted = xh1_output_dequanted[0:self.model_input_batch, ...]
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
        
        