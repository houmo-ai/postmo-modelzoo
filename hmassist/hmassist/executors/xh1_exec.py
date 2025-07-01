#!/usr/bin/env python  
import pickle
import time
import os
import yaml
import numpy as np
import torch
import re
from datetime import datetime
from abc import ABC
from ..utils import logger
from ..utils.utils import get_random_data
from ..utils.parser import read_yaml_to_dict, save_dict_to_yaml
from .base_exec import BaseExec


class XH1Exec(BaseExec, ABC):
    def __init__(self, cfg: dict):
        super(XH1Exec, self).__init__(cfg)
        self.model_path = os.path.join(self.model_dir, self.model_name)
        self.summary_result_path = cfg["result_path"]
        self.sequencer = None
        # save model info to result.yaml
        res = dict()
        if os.path.exists(self.summary_result_path):
            res = read_yaml_to_dict(self.summary_result_path)
        res["model"] = {
            "framework": cfg["model"]["framework"],
            "name": cfg["model"]["name"],
            "path": cfg["model"]["weight"],
            "inputs": cfg["model"]["inputs"]
        }
        save_dict_to_yaml(res, self.summary_result_path)
        self.model_input_batch = self.inputs[0]["shape"][0]

    def quantize(self, get_input_datas):
        import platform
        arch = platform.machine()
        if arch != "x86_64":
            logger.error(f"quant not support platform: {arch}")
            exit(0)
        if not os.path.exists(self.weight):
            weight = os.path.join(os.getenv("HOUMO_MODEL_PATH", "."), self.weight)
            if not os.path.exists(weight):
                raise RuntimeError(f"{self.weight} or {weight} not exist.")
            self.weight = weight
        logger.info("################  ptq quantize started  ######################")
        t_start = time.time()
        calib_files = []
        calib_dataset = []
        dynamic_resize = self.model_cfg.get("dynamic_resize")
        calib_num = self.quant_cfg.get("calib_num")
        calib_method = self.quant_cfg.get("calib_method")
        precision = self.quant_cfg.get("precision")

        quanttool_config = {'inputs_cfg': {}}
        # quanttool_config['graph_opt_cfg'] = {}

        # 准备量化数据集
        calib_dir = self.quant_cfg["calib_dir"]
        
        for _input in self.inputs:
            name = _input["name"]
            shape = self.shape_dict[name]
            n, c, h, w = shape

            if self.quant_cfg["ptq_cfg_path"] == "none":
                # 准备量化参数
                logger.info("using quanttool_config from config.yml")
                quanttool_config['inputs_cfg'][name] = {}
                input_cfg = quanttool_config['inputs_cfg'][name]
                if dynamic_resize:
                    input_cfg['fold'] = False
                input_cfg['data_format'] = _input["format"]
                input_cfg['first_layer_weight_denorm_mean'] = _input["mean"]
                input_cfg['first_layer_weight_denorm_std'] = _input["std"]
                if "image" in self.inputs[0]:
                    if "size" in self.inputs[0]["image"] and self.inputs[0]["image"]["size"]:
                        image_size = self.inputs[0]["image"]["size"]
                    else:
                        image_size = [h, w]
                    if "crop" in self.inputs[0]["image"] and self.inputs[0]["image"]["crop"]:
                        image_crop = self.inputs[0]["image"]["crop"]
                    else:
                        image_crop = [0, 0, image_size[0], image_size[1]]
                    input_cfg['resizer_crop'] = {'top': image_crop[0], 'left': image_crop[1], 'height': image_crop[2], 'width': image_crop[3]}
                    input_cfg['resizer_resize'] = {'width': w, 'height': h, 'align_corners': False, 'method': 'bilinear'}
                    input_cfg['toYUV_format'] = _input["image"]["format"]

            # 未配置量化数据，采用随机数据
            if calib_dir is None:
                if calib_num <= 0 or calib_num > 100:
                    logger.warning(f"calib_num can't be {calib_num} while calib_dir is None, reset to 1.")
                    calib_num = 1
                logger.info(f"calib num: {calib_num}")
                logger.warning("calibrate will use random data while calib_dir is None.")
                for _ in range(calib_num):
                    dtype = _input["dtype"]
                    input_shape = n, c, image_size[0], image_size[1]
                    calib_data[name] = torch.tensor(get_random_data(name, dtype, input_shape))
                    calib_dataset.append(calib_data)
            else:
                if os.path.isdir(calib_dir):
                    filelist = sorted(os.listdir(calib_dir))  # 保证每次取的数据一致
                elif os.path.isfile(calib_dir):
                    filelist = [calib_dir]
                    calib_num = 1
                else:
                    logger.error(f"unknown calib_dir: {calib_dir}")
                    exit(-1)
                for filename in filelist:
                    _, ext = os.path.splitext(filename)
                    if ext in [".jpg", ".JPEG", ".bmp", ".png", ".jpeg", ".BMP", ".bin"]:
                        calib_files.append(filename)
                        if calib_num > 0 and len(calib_files) >= calib_num:
                            break
                if len(calib_files) < self.quant_cfg["calib_num"]:
                    logger.warning("calib_dir only has {} files, but calib_num is {}."
                        .format(len(calib_files), self.quant_cfg["calib_num"]))
                calib_num = len(calib_files)

                logger.debug(f"calib file: {calib_files}")
                new_calib_num = calib_num // n
                logger.info(f"calib num: {new_calib_num if calib_num > n else 1}")                                
                for c in range(new_calib_num):
                    batch_datas = dict()
                    for i in range(n):
                        idx = c * n + i
                        in_datas = get_input_datas(calib_dir, calib_files[idx])  # {"input_name": np.ndarray} NCHW
                        for key in in_datas:
                            if key in batch_datas:
                                batch_datas[key].append(in_datas[key])
                            else:
                                batch_datas[key] = [in_datas[key]]
                    calib_data = dict()            
                    for key in batch_datas:
                        calib_data[key] = torch.from_numpy(np.concatenate(batch_datas[key], axis=0))
                    calib_dataset.append(calib_data)
                if calib_num < n:
                    # 数据不足1batch，复制最后1份数据
                    batch_datas = dict()
                    for idx in range(calib_num):
                        in_datas = get_input_datas(calib_dir, calib_files[idx])  # {"input_name": np.ndarray} NCHW
                        for key in in_datas:
                            if key in batch_datas:
                                batch_datas[key].append(in_datas[key])
                            else:
                                batch_datas[key] = [in_datas[key]]
                    for key in batch_datas:
                        last_data = batch_datas[key][-1].copy()
                        for idx in range(n - calib_num):
                            batch_datas[key].append(last_data)
                        calib_data = dict()
                        calib_data[key] = torch.from_numpy(np.concatenate(batch_datas[key], axis=0))
                        calib_dataset.append(calib_data)

        if self.quant_cfg["ptq_cfg_path"] != "none":
            logger.info("using quanttool_config from {}".format(self.quant_cfg["ptq_cfg_path"]))
            quanttool_config = self.quant_cfg["ptq_cfg_path"]
        logger.info(quanttool_config)

        from hmquant.api import quant_single_onnx_network
        sequencer = quant_single_onnx_network(
            cfg=quanttool_config,
            calibration_data=calib_dataset,  # 输入的batch可决定量化后模型输入batch
            onnx_model_or_path=self.weight,
            device="cuda" if torch.cuda.is_available() else "cpu",
            debug=None,
            model_name=self.model_name,
            calib_method=calib_method,
            mix_search=False if precision == 'int8' else True,
            method="all" if precision == 'int16' else "smart",
            use_gptq=True if precision == 'auto' else False,
            mix_calib_samples=4,
        )

        logger.info("################  ptq quantize finished  ######################")
        self.quantize_span = time.time() - t_start

        # gen golden data
        data_path = self.quant_cfg.get("data_path")
        input_datas = {}
        if data_path == "default":
            input_datas = calib_dataset[0]
        else:
            inputs = get_input_datas("", data_path)
            for key in inputs:
                n, c, h, w = self.shape_dict[key]
                input_datas[key] = torch.from_numpy(np.concatenate([inputs[key].astype(np.float32) for _ in range(n)], axis=0))
                
        t_start = time.time()
        if self.quant_cfg["debug_level"] == 1:
            from hmquant.api import quantize_profiling
            res = quantize_profiling(
                sequencer=sequencer, 
                sequencer_input=[input_datas], 
                device="cuda" if torch.cuda.is_available() else "cpu", 
                mode=0,
                quant_mode="quant_forward",
                fix_topk=True,
                only_onodes=False,
                return_o_metric=True
            )
            res = {out_name: {k: float(v) if isinstance(v, np.float64) else v for k, v in metrics.items()} for out_name, metrics in res.items()}     
            new_res = dict()
            if os.path.exists(self.summary_result_path):
                new_res = read_yaml_to_dict(self.summary_result_path)
            new_res["quant"] = res
            new_res["quant"]["time"] = self.quantize_span
            save_dict_to_yaml(new_res, self.summary_result_path)    
        self.layer_compare_span = time.time() - t_start

        from hmquant.api import generate_golden
        sequencer.save_pkl(self.quant_dir, self.model_name)
        golden_input_path, _, golden_onnx_path = generate_golden(
            sequencer=sequencer,
            calibset=input_datas,
            save_path=self.quant_dir,
            model_name=self.model_name,
            batch_size=self.model_input_batch,
            device="cuda" if torch.cuda.is_available() else "cpu",
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

        logger.info(f"golden data saved in -> {self.golden_data_path}")
        logger.info(f"quantize cost {self.quantize_span:.3f} s, layer compare cost {self.layer_compare_span:.3f} s")

    def build(self):
        import platform
        arch = platform.machine()
        if arch != "x86_64":
            logger.error(f"build not support platform: {arch}")
            exit(0)
        dynamic_resize = self.model_cfg.get("dynamic_resize")
        ncore = self.build_cfg.get("ncore")
        opt_level = self.build_cfg.get("opt_level")
        logger.info("################  build started  ######################")
        import tcim
        t_start = time.time()
        kwargs = {}
        if self.j:
            kwargs["j"] = self.j
        tcim.build_from_hmonnx(
            self.quant_model_path,
            output_name=self.model_name,
            ncore=ncore,
            batch=self.batch,
            output_dir=self.model_dir,
            work_dir=self.build_dir,
            opt_level=f"O{opt_level}",
            enable_dynamic_image_resize=dynamic_resize,
            **kwargs
        )

        logger.info('{} saved in {}'.format(self.model_name, self.model_dir))
        logger.info("################  build finished  ######################")
        self.build_span = time.time() - t_start
        new_res = dict()
        if os.path.exists(self.summary_result_path):
            new_res = read_yaml_to_dict(self.summary_result_path)
        new_res["build"] = dict()
        new_res["build"]["time"] = self.build_span
        save_dict_to_yaml(new_res, self.summary_result_path)
        logger.info(f"build cost {self.build_span:.3f} s")

    def load(self):
        import tcim_lite
        self.module = tcim_lite.runtime.load(os.path.join(self.model_dir, self.model_name + ".hmm"))
        self.input_infos = self.get_input_info()
        self.output_infos = self.get_output_info()
        logger.info("{} model loaded".format(self.model_name))
            
    def infer(self, input_datas):
        """ infer one time """
        for name in self.input_infos:
            if name == "dyn_info":
                shape = self.inputs[0]["shape"]  # 默认单图像输入，取图像shape
                crop = [0, 0, shape[2], shape[3]]  # y1, x1, h, w
                resize = [shape[2], shape[3]]  # h, w
                pad = [0, 0, 0, 0]  # top, left, bottom, right
                input_data = np.concatenate((crop, resize, pad))
            else:
                if isinstance(input_datas, dict):
                    input_data = input_datas[name]  
                else:
                    input_data = input_datas
            self.module.set_input(name, input_data)
        self.module.run()
        self.module.sync()
        outputs = {}
        output_num = self.module.get_num_outputs()
        for idx in range(0, output_num):
            name = self.module.get_output_name(idx)
            if self.is_fixed_out:
                output_data = self.module.get_output(name).numpy()
            else:
                output_data = self.module.get_output(name).astype(np.float32).numpy()
            outputs[name] = output_data

        return outputs

    def perf(self, test_num):
        HOUMO_MODELZOO_PATH = os.getenv('HOUMO_MODELZOO_PATH')
        model_path = os.path.join(self.model_dir, self.model_name + ".hmm")
        exec = "tcim_perf"
        if os.environ.get("HDPL_PLATFORM") == "ISIM":
            test_num = 1
            logger.warning("test num set to 1 because HDPL_PLATFORM=ISIM may take a lot of time.")
        save_dir = os.path.join(self.cur_dir, "output")
        cmd = "cd {}/utils/{} && ./{} --model {} --data {} --samples {} --threads {} --batch {} --output {}".format(
            HOUMO_MODELZOO_PATH, exec, exec, model_path, self.build_dir, test_num, self.perf_cfg["thread_num"], self.batch * self.model_input_batch,
            save_dir)
        if self.perf_cfg['infer_only']:
            cmd += " --infer_only true"
        logger.info(cmd)
        run_time = datetime.now().strftime("%Y%m%d%H%M%S")
        os.system(cmd)
        # save data to result.yaml
        perf_txt_path = os.path.join(save_dir, "hmperf.txt")
        if not os.path.exists(perf_txt_path):
            logger.error(f"{perf_txt_path} not exist")
            exit(-1)
        with open(perf_txt_path, "r") as f:
            lines = f.readlines()
        new_res = dict()
        if os.path.exists(self.summary_result_path):
            new_res = read_yaml_to_dict(self.summary_result_path)
        if "perf" not in new_res:
            new_res["perf"] = dict()
        new_res["perf"][run_time] = dict()
        new_res["perf"][run_time]["batch"] = int(lines[0].strip().split(" ")[-1])
        new_res["perf"][run_time]["thread_num"] = int(lines[1].strip().split(" ")[-1])
        new_res["perf"][run_time]["loop_num"] = int(lines[3].strip().split(" ")[-1])
        new_res["perf"][run_time]["sample_num"] = int(lines[4].strip().split(" ")[-1])
        new_res["perf"][run_time]["avg_latency"] = float(lines[5].strip().split(" ")[-1]) 
        new_res["perf"][run_time]["max_latency"] = float(lines[6].strip().split(" ")[-1])
        new_res["perf"][run_time]["qps"] = float(lines[7].strip().split(" ")[-1])
        new_res["perf"][run_time]["cmd"] = cmd
        save_dict_to_yaml(new_res, self.summary_result_path)
            
    def _preprocess(self, inputs):
        datas = {}
        for _input in self.inputs:
            dtype = self.input_infos[_input["name"]].dtype
            if _input["image"]["format"] in ["YUV420", "YUV422", "YUV444"]:
                data = torch.from_numpy(inputs[_input["name"]].astype(np.float32))  # NHWC float32
                data = torch.squeeze(data, dim=0)  # HWC float32
                _format = re.sub("YUV", "", _input["image"]["format"])
                from ..utils.transform import RGB2YUV, BGR2YUV
                to_yuv_func = RGB2YUV(fmt=_format) if _input["format"] == "RGB" else BGR2YUV(fmt=_format)
                image = torch.unsqueeze(to_yuv_func(data), 0).numpy()  # NHWC float32
                datas[_input["name"]] = image.astype(dtype)
            else:
                datas[_input["name"]] = inputs[_input["name"]].astype(dtype)
        return datas

    def get_golden_inputs(self):
        datas = {}
        for input in self.inputs:
            input_data_path = os.path.join(self.quant_dir, 'hmquant_' + self.model_name + '_' + input["name"] + '_input.npy')
            if os.path.exists(input_data_path):
                input_data = np.load(input_data_path)
                logger.info("golden input[{}] shape = {}, dtype = {}".format(input["name"], input_data.shape, input_data.dtype))
                input_data = input_data.astype(self.input_infos[input["name"]].dtype)
                input_data = np.concatenate([input_data for i in range(self.batch)], axis=0)
                datas[input["name"]] = input_data
            else:
                logger.warning(f"compare canceled while golden input not found -> {input_data_path}")
                return None 
        return datas

    def get_golden_output(self, name):
        golden_output_path = os.path.join(self.golden_data_path, 'hmquant_' + self.model_name 
                                          + '_' + name + '_output.npy')
        if os.path.exists(golden_output_path):
            output_data = np.load(golden_output_path)
            output_data = np.concatenate([output_data for i in range(self.batch)], axis=0)
            return output_data
        else:
            logger.warning(f"compare canceled while golden output not found -> {golden_output_path}")
            return None

    def gen_golden(self, input_tensors: list):
        if self.sequencer is None:
            sequencer_model_path = f"{os.path.join(self.quant_dir, self.model_name)}.pkl"
            if not os.path.exists(sequencer_model_path):
                logger.error(f"Sequencer model not exist: {sequencer_model_path}")
                exit(-1)
            from hmquant.api import quant_single_onnx_network
            with open(sequencer_model_path, "rb") as f:
                self.sequencer = pickle.load(f)
            logger.info("load quantized model successfully.")
        if not self.is_fixed_out:
            self.sequencer.set_ops_mode("quant_forward")
            # self.sequencer.set_ops_mode("hardware_forward") # without dequant
        outputs = self.sequencer.forward(*input_tensors, get_output_dict=True)
        return {key:outputs[key].detach().cpu().numpy() for key in outputs}

    def get_version(self):
        raise NotImplemented

    def get_input_info(self):
        input_infos = {}
        input_num = self.module.get_num_inputs()
        for id in range(input_num):
            name = self.module.get_input_name(id)
            input_info = self.module.get_input_info(name)
            input_infos[name] = input_info
        return input_infos

    def get_output_info(self):
        output_infos = {}
        output_num = self.module.get_num_outputs()
        for id in range(output_num):
            output_info = {}
            name = self.module.get_output_name(id)
            output_info = self.module.get_output_info(name)
            output_infos[name] = output_info
        return output_infos
    
    def print_input_info(self):
        input_num = len(self.input_infos)
        logger.info("{} input num = {}:".format(self.target, input_num))
        for name, _input in self.input_infos.items():
            _input = self.input_infos[name]
            logger.info("{} input[{}] shape = {}, dtype = {}, format = {}".format(
                self.target, name, _input.shape, _input.dtype, _input.format.name))

    def print_output_info(self):
        output_num = len(self.output_infos)
        logger.info("{} output num = {}:".format(self.target, output_num))
        for name, _output in self.output_infos.items():
            logger.info("{} output[{}] shape = {}, dtype = {}, format = {}".format(
                self.target, name, _output.shape, _output.dtype, _output.format.name))

    @property
    def freq(self):
        return {"H30": 1024}

    def get_relay_mac(self):
        raise NotImplemented

    def get_profile_info(self):
        raise NotImplemented

    def get_device_type(self):
        raise NotImplemented
