#!/usr/bin/env python  

import time
import os
import numpy as np
from abc import ABC
import torch
import re
from ..utils import logger
from ..utils.utils import get_random_data
from .base_exec import BaseExec


class XH1Exec(BaseExec, ABC):
    def __init__(self, cfg: dict):
        super(XH1Exec, self).__init__(cfg)
        self.model_path = os.path.join(self.model_dir, self.model_name)

    def quantize(self, get_input_datas):
        import platform
        arch = platform.machine()
        if arch != "x86_64":
            logger.error(f"quant not support platform: {arch}")
            exit(0)
        if not os.path.exists(self.weight):
            weight = os.path.join(os.getenv("HOUMO_MODEL_PATH", default=""), self.weight)
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
        mix_search = False
        method = "smart"
        if precision in ["auto", "int16"]:
            mix_search = True
        if precision == "int16":
            method = "all"

        quanttool_config = {'inputs_cfg': {}}
        # quanttool_config['graph_opt_cfg'] = {}

        # 准备量化数据集
        calib_dir = self.quant_cfg["calib_dir"]
        if calib_dir:
            if os.path.isdir(calib_dir):
                filelist = os.listdir(calib_dir)
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
            logger.info(f"calib num: {calib_num}")
            logger.debug(f"calib file: {calib_files}")
            for id in range(calib_num):
                inputs = get_input_datas(calib_dir, calib_files[id])
                calib_data = {}
                for name, data in inputs.items():
                    calib_data[name] = torch.tensor(data.astype(np.float32))
                    calib_dataset.append(calib_data)

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
                    input_cfg['resizer_crop'] = {'top': image_crop[0],
                                                 'left': image_crop[1],
                                                 'height': image_crop[2],
                                                 'width': image_crop[3]}
                    input_cfg['resizer_resize'] = {'width': w,
                                                   'height': h,
                                                   'align_corners': False,
                                                   'method': 'bilinear'}
                    input_cfg['toYUV_format'] = _input["image"]["format"]

            if calib_dir is None:
                if calib_num <= 0 or calib_num > 100:
                    logger.warning(f"calib_num can't be {calib_num} while calib_dir is None, reset to 1.")
                    calib_num = 1
                logger.info(f"calib num: {calib_num}")
                logger.warning("calibrate will use random data while calib_dir is None.")
                for id in range(calib_num):
                    dtype = _input["dtype"]
                    input_shape = n, c, image_size[0], image_size[1]
                    calib_data[name] = torch.tensor(get_random_data(name, dtype, input_shape))
                    calib_dataset.append(calib_data)

        if self.quant_cfg["ptq_cfg_path"] != "none":
            logger.info("using quanttool_config from {}".format(self.quant_cfg["ptq_cfg_path"]))
            quanttool_config = self.quant_cfg["ptq_cfg_path"]
        logger.info(quanttool_config)

        from hmquant.api import quant_single_onnx_network
        sequencer = quant_single_onnx_network(
            cfg=quanttool_config,
            calibration_data=calib_dataset,
            onnx_model_or_path=self.weight,
            device='cpu',
            debug=None,
            model_name=self.model_name,
            mix_search=mix_search,
            calib_method=calib_method,
            method=method,
            # use_gptq=True,
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
            for name, data in inputs.items():
                input_datas[name] = torch.tensor(data.astype(np.float32))

        t_start = time.time()
        if self.quant_cfg["debug_level"] == 1:
            from hmquant.api import quantize_profiling
            quantize_profiling(sequencer, [input_datas])
        self.layer_compare_span = time.time() - t_start

        from hmquant.api import generate_golden
        generate_golden(
            sequencer=sequencer,
            calibset=input_datas,
            save_path=self.quant_dir,
            model_name=self.model_name,
            batch_size=1,
            device="cpu"
        )
        sequencer.save_pkl(self.quant_dir, self.model_name)

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
        tcim.build_from_hmonnx(
            self.quant_model_path,
            output_name=self.model_name,
            ncore=ncore,
            batch=self.batch,
            output_dir=self.model_dir,
            work_dir=self.build_dir,
            opt_level=f"O{opt_level}",
            enable_dynamic_image_resize=dynamic_resize
        )
        logger.info('{} saved in {}'.format(self.model_name, self.model_dir))
        logger.info("################  build finished  ######################")
        self.build_span = time.time() - t_start
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
        for id in range(0, output_num):
            name = self.module.get_output_name(id)
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
        cmd = "cd {}/utils/{} && ./{} --model {} --data {} --samples {} --threads {} --batch {} --output {}".format(
            HOUMO_MODELZOO_PATH, exec, exec, model_path, self.build_dir, test_num, self.perf_cfg["thread_num"], self.batch,
            os.path.join(self.cur_dir, "output"))
        if self.perf_cfg['infer_only']:
            cmd += " --infer_only true"
        logger.info(cmd)
        os.system(cmd)

    def _preprocess(self, inputs):
        datas = {}
        for input in self.inputs:
            dtype = self.input_infos[input["name"]].dtype
            if input["image"]["format"] in ["YUV420", "YUV422", "YUV444"]:
                data = torch.tensor(inputs[input["name"]].astype(np.float32))  # NHWC float32
                data = torch.squeeze(data, 0)  # HWC float32
                format = re.sub("YUV", "", input["image"]["format"])
                from ..utils.transform import RGB2YUV
                rgb2yuv_func = RGB2YUV(fmt=format)
                image = torch.unsqueeze(rgb2yuv_func(data), 0).numpy()  # NHWC float32
                datas[input["name"]] = image.astype(dtype)
            else:
                datas[input["name"]] = inputs[input["name"]].astype(dtype)
        return datas

    def get_golden_inputs(self):
        datas = {}
        for input in self.inputs:
            input_data_path = os.path.join(self.quant_dir, 'hmquant_' + self.model_name 
                                           + '_' + input["name"] + '_input.npy')
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

    def gen_golden(self, inputs):
        from hmodel.utils.general import load_pkl_model
        qmodel = os.path.join(self.quant_dir, self.model_name)
        sequencer = load_pkl_model(qmodel)
        from hmquant.api import generate_golden
        generate_golden(
            sequencer=sequencer,
            calibset=inputs,
            save_path=self.test_dir,
            model_name=self.model_name,
            batch_size=1,
            device="cpu"
        )

        logger.info("golden data saved in -> {}".format(self.test_dir))

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
            logger.info("{} input[{}] shape = {}, dtype = {}, format = {}".format(self.target, name,
                                                                                  _input.shape, _input.dtype,
                                                                                  _input.format.name))

    def print_output_info(self):
        output_num = len(self.output_infos)
        logger.info("{} output num = {}:".format(self.target, output_num))
        for name, _output in self.output_infos.items():
            logger.info("{} output[{}] shape = {}, dtype = {}, format = {}".format(self.target, name,
                                                                                   _output.shape, _output.dtype,
                                                                                   _output.format.name))

    @property
    def freq(self):
        return {"H30": 1024}

    def get_relay_mac(self):
        raise NotImplemented

    def get_profile_info(self):
        raise NotImplemented

    def get_device_type(self):
        raise NotImplemented
