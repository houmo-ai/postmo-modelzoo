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
from ..utils.utils import get_random_data, load_npz
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
        self.is_npz = False
        if len(self.inputs) > 1 or self.inputs[0]["format"] in ["Int8Feature", "Uint8Feature", "Int16Feature", 
                                   "Float16Feature", "Float32Feature", "Float64Feature"]:
            self.is_npz = True
    
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
        dynamic_resize = self.model_cfg.get("dynamic_resize")
        quanttool_config = {'inputs_cfg': dict()}
        # quanttool_config['graph_opt_cfg'] = {}

        # 准备量化数据集
        ptq_cfg_path = self.quant_cfg.get("ptq_cfg_path")
        if ptq_cfg_path != "none":
            logger.info("using quanttool_config from {}".format(ptq_cfg_path))
            quanttool_config = ptq_cfg_path
        else:
            logger.info("using quanttool_config from config.yml")
            inputs_cfg = dict()
            for _input in self.inputs:
                name = _input["name"]
                shape = self.shape_dict[name]
                data_format = _input["format"]
                input_cfg = dict()
                input_cfg["first_layer_weight_denorm_mean"] = None
                input_cfg["first_layer_weight_denorm_std"] = None
                input_cfg['data_format'] = data_format
                if self.is_npz:
                    logger.info(f"input[name] is feature_or_tensor")
                    inputs_cfg[name] = input_cfg
                    continue
                N, C, H, W = shape
                input_cfg['first_layer_weight_denorm_mean'] = _input["mean"]
                input_cfg['first_layer_weight_denorm_std'] = _input["std"]
                yuv_format = _input["image"]["format"]
                input_cfg['fold'] = not dynamic_resize
                if "image" in _input:
                    resizer_info = _input["image"]
                    resizer_size = resizer_info.get("size")
                    resizer_crop = resizer_info.get("crop")
                    if resizer_size is None or not isinstance(resizer_size, list) \
                        or len(resizer_size) != 2:
                        resizer_size = [H, W]
                    if resizer_crop is None or not isinstance(resizer_crop, list) \
                        or len(resizer_crop) != 4:
                        resizer_crop = [0, 0, H, W]
                    input_cfg['resizer_crop'] = {'top': resizer_crop[0], 'left': resizer_crop[1], 'height': resizer_crop[2], 'width': resizer_crop[3]}
                    input_cfg['resizer_resize'] = {'width': W, 'height': H, 'align_corners': False, 'method': 'bilinear'}
                    input_cfg['toYUV_format'] = yuv_format
                inputs_cfg[name] = input_cfg
            quanttool_config["inputs_cfg"].update(inputs_cfg)

        calib_num = self.quant_cfg.get("calib_num")
        calib_method = self.quant_cfg.get("calib_method")
        precision = self.quant_cfg.get("precision")
        calib_files = []
        calib_dataset = []
        calib_dir = self.quant_cfg["calib_dir"]     
        if calib_dir == "default":
            if calib_num <= 0 or calib_num > 100:
                logger.warning(f"calib_num can't be {calib_num} while calib_dir is None, reset to 1.")
                calib_num = 1
            logger.info(f"calib num: {calib_num}")
            logger.warning("calibrate will use random data while calib_dir is None.")
            
            for _ in range(calib_num):
                calib_data = dict()
                for _input in self.inputs:
                    name = _input["name"]
                    dtype = _input["dtype"]
                    shape = _input["shape"]
                    if self.is_npz:
                        calib_data[name] = torch.from_numpy(get_random_data(dtype, shape))
                    else:
                        N, C, H, W = shape
                        resizer_crop = quanttool_config["inputs_cfg"][name]["resizer_crop"]
                        input_shape = [N, C, resizer_crop["height"], resizer_crop["width"]]
                        calib_data[name] = torch.from_numpy(get_random_data(name, dtype, input_shape))
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
                if ext.lower() in [".jpg", ".JPEG", ".bmp", ".png", ".jpeg", ".BMP", ".npz"]:
                    calib_files.append(filename)
                    if calib_num > 0 and len(calib_files) >= calib_num:
                        break                   
            if len(calib_files) < calib_num:
                logger.warning("calib_dir only has {} files, but calib_num is {}.".format(len(calib_files), calib_num))
            calib_num = len(calib_files)
            
            if not self.is_npz:
                # 单输入图像
                padd_len = calib_num % self.model_input_batch
                for _ in range(padd_len):
                    calib_files.append(calib_files[-1])
                calib_num = len(calib_files) // self.model_input_batch
                logger.debug(f"calib file: {calib_files}")
                logger.info(f"calib num: {calib_num}")
                for c in range(calib_num):
                    batch_datas = dict()
                    for i in range(self.model_input_batch):
                        idx = c * self.model_input_batch + i
                        in_datas = get_input_datas(calib_dir, calib_files[idx])  # {"input_name": np.ndarray} NCHW
                        for key in in_datas:
                            if key in batch_datas:
                                batch_datas[key].append(in_datas[key])
                            else:
                                batch_datas[key] = [in_datas[key]]
                        batch_datas[key] = torch.from_numpy(np.concatenate(batch_datas[key], axis=0))
                    calib_dataset.append(batch_datas)
            else:
                logger.debug(f"calib file: {calib_files}")
                logger.info(f"calib num: {calib_num}")
                for c in range(calib_num):
                    # 单输入非图像 or 多输入都直接读npz(处理后数据)
                    batch_datas = load_npz(os.path.join(calib_dir, calib_files[c]))            
                    calib_data = dict()            
                    for key in batch_datas:
                        calib_data[key] = torch.from_numpy(batch_datas[key])
                    calib_dataset.append(calib_data)
                
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
            if self.is_npz:
                # 单输入输入为非图像 or 多输入
                _, ext = os.path.splitext(os.path.basename(data_path))
                assert ext == ".npz", f"data_path: {data_path}"
                input_datas = load_npz(data_path)
            else:
                # 单输入且输入为图像
                inputs = get_input_datas("", data_path)
                for key in inputs:
                    input_datas[key] = np.concatenate([inputs[key] for _ in range(self.model_input_batch)], axis=0)
            input_datas = {key: torch.from_numpy(input_datas[key]) for key in input_datas}
                
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
        for _input in self.inputs:
            name = _input["name"]
            input_data_path = os.path.join(self.quant_dir, 'hmquant_' + self.model_name + '_' + name + '_input.npy')
            if os.path.exists(input_data_path):
                input_data = np.load(input_data_path)
                logger.info("golden input[{}] shape = {}, dtype = {}".format(name, input_data.shape, input_data.dtype))
                input_data = input_data.astype(self.input_infos[name].dtype)
                input_data = np.concatenate([input_data for i in range(self.batch)], axis=0)
                datas[name] = input_data
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
