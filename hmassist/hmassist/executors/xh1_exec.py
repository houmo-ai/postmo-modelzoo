#!/usr/bin/env python  

import time
import os
import numpy as np
from abc import ABC
import onnx
import torch
import re
import tcim
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
        logger.info("################  ptq quantize started  ######################")
        t_start = time.time()
        calib_files = []
        calib_dataset = [dict() for i in range(self.quant_cfg["calib_num"])]
        calib_num = self.quant_cfg["calib_num"]
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
                    if len(calib_files) == calib_num:
                        break
            if len(calib_files) < self.quant_cfg["calib_num"]:
                logger.warning("calib_dir only has {} files, but calib_num is {}."
                    .format(len(calib_files), self.quant_cfg["calib_num"]))
                calib_num = len(calib_files)
                logger.info("calib num: {}".format(calib_num))
            for id in range(calib_num):
                logger.debug("calib file: {}".format(calib_files[id]))
                inputs = get_input_datas(calib_dir, calib_files[id])
                for name, data in inputs.items():
                    # data = np.transpose(data, (2, 0, 1))  # CHW
                    # data = np.expand_dims(data, axis=0)  # NCHW
                    calib_dataset[id][name] = torch.tensor(data.astype(np.float32))

        for _input in self.inputs:
            name = _input["name"]
            shape = self.shape_dict[name]
            n, c, h, w = shape

            if "ptq_cfg_path" not in self.quant_cfg or not self.quant_cfg["ptq_cfg_path"]:
                # 准备量化参数
                logger.info("using quanttool_config from config.yml")
                quanttool_config['inputs_cfg'][name] = {}
                input_cfg = quanttool_config['inputs_cfg'][name]
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
                for id in range(calib_num):
                    dtype = _input["dtype"]
                    input_shape = n, c, image_size[0], image_size[1]
                    logger.warning("data[{}] will use random data".format(name))
                    calib_dataset[id][name] = torch.tensor(
                        get_random_data(name, dtype, input_shape))

        if "ptq_cfg_path" in self.quant_cfg and self.quant_cfg["ptq_cfg_path"]:
            logger.info("using quanttool_config from {}".format(self.quant_cfg["ptq_cfg_path"]))
            quanttool_config = self.quant_cfg["ptq_cfg_path"]
        logger.info(quanttool_config)

        # 删除列表中的空项
        del calib_dataset[calib_num:self.quant_cfg["calib_num"]]

        from hmquant.api import quant_single_onnx_network
        sequencer = quant_single_onnx_network(
            cfg=quanttool_config,
            calibration_data=calib_dataset,
            onnx_model_or_path=self.weight,
            device='cpu',
            debug=None,
            model_name=self.model_name,
            with_label=False,
            requant_dispatch=True,
        )

        # sequencer.save_pkl(self.result_dir, self.model_name)
        # sequencer.save_onnx(
        #     self.quant_model_path,
        #     save_special_onnx=True
        # )

        logger.info("################  ptq quantize finished  ######################")
        self.quantize_span = time.time() - t_start

        # gen golden data
        inputs = {}
        # for _input in self.inputs:
        #     inputs[_input["name"]] = calib_dataset[0]["name"]

        from hmquant.api import generate_golden
        generate_golden(
            sequencer=sequencer,
            calibset=calib_dataset[0],
            save_path=self.result_dir,
            model_name=self.model_name,
            batch_size=1,
            device="cpu"
        )

        logger.info("golden data saved in -> {}".format(self.golden_data_path))

        t_start = time.time()
        if self.quant_cfg["debug_level"] == 1:
            from hmquant.api import quantize_profiling
            quantize_profiling(sequencer, [calib_dataset[0]])
        self.layer_compare_span = time.time() - t_start
        logger.info("quantize cost {}s, layer compare cost {}s".format(self.quantize_span, self.layer_compare_span))

    def build(self, build_options=None):
        logger.info("################  build started  ######################")
        t_start = time.time()
        onnx_model = onnx.load(self.quant_model_path)
        if not build_options:
            build_options = {}
        if self.batch % 4 == 0:
            build_options["tcim.core_num"] = 4
        input_cfg = {}
        inputs = onnx_model.graph.input
        for input in inputs:
            dims = input.type.tensor_type.shape.dim
            input_shape = [dim.dim_value for dim in dims]
            input_shape[0] *= self.batch
            input_cfg[input.name] = tcim.HMInput(shape=input_shape)

        logger.info("build_options={}".format(build_options))

        hdplcc_options = []
        # hdplcc_options.append("-O2")

        if os.getenv("TCIM_CROSS_COMPILE") == '1':
            logger.info("cross compile enabled as aarch64 while TCIM_CROSS_COMPILE=1")
            model_name = self.model_name + "_aarch64"
            hdplcc_options.append("--target=aarch64-linux-gnu")
            hdplcc_options.append("--sysroot=/opt/gcc-linaro-7.5.0-2019.12-x86_64_aarch64-linux-gnu/aarch64-linux-gnu/libc")
            hdplcc_options.append("-I/opt/gcc-linaro-7.5.0-2019.12-x86_64_aarch64-linux-gnu/aarch64-linux-gnu/include/c++/7.5.0")
            hdplcc_options.append("-I/opt/gcc-linaro-7.5.0-2019.12-x86_64_aarch64-linux-gnu/aarch64-linux-gnu/include/c++/7.5.0/aarch64-linux-gnu")
            tcim.build.build_from_hmonnx(onnx_model, model_name=model_name, inputs=input_cfg, compiler_cfg=build_options,
                                         target_host="arm64", hdplcc_options=hdplcc_options)
        else:
            model_name = self.model_name
            tcim.build.build_from_hmonnx(onnx_model, model_name=model_name, inputs=input_cfg, compiler_cfg=build_options,
                                         hdplcc_options=hdplcc_options)  # output_layout={"layout": "NCHW"}
        print(model_name + ' build completed.')

        logger.info('{} saved in {}'.format(self.model_name, self.model_dir))

        logger.info("################  build finished  ######################")
        self.build_span = time.time() - t_start
        logger.info("build cost {}s".format(self.build_span))

    def load(self):
        self.module = tcim.runtime.load(self.model_name + ".hmm.so")
        self.input_info = self.get_input_info()
        self.output_info = self.get_output_info()
        logger.info("{} model loaded".format(self.model_name))

    def infer(self, inputs):
        """ infer one time """
        for input in self.inputs:
            if isinstance(inputs, dict):
                input_data = inputs[input["name"]]
            else:
                input_data = inputs
            # print(input_data.shape, input_data.dtype)
            self.module.set_input(input["name"], input_data)
        self.module.run()
        self.module.sync()
        outputs = {}
        output_num = self.module.get_num_outputs()
        for id in range(0, output_num):
            name = self.module.get_output_name(id)
            if self.is_fixed_out:
                output_data = self.module.get_output(name, is_quanted=True)
            else:
                output_data = self.module.get_output(name, is_quanted=False)
            if (len(output_data.shape) == 4):  # toolchain output is NHWC
                output_data = np.transpose(output_data, (0, 3, 1, 2))
            outputs[name] = output_data

        return outputs

    def perf(self, test_num):
        modelzoo_path = os.getenv('MODELZOO_PATH')
        model_path = os.path.join(self.cur_dir, self.model_name + ".hmm")
        exec = "tcim_perf"
        if os.environ.get("HDPL_PLATFORM") == "ISIM":
            test_num = 1
            logger.warning("test num set to 1 because HDPL_PLATFORM=ISIM may take a lot of time.")
        cmd = "cd {}/utils/{} && ./{} --model {} --samples {} --threads {} --batch {} --output {}".format(
            modelzoo_path, exec, exec, model_path, test_num, self.perf_cfg["thread_num"], self.batch,
            os.path.join(self.cur_dir, "output/hmperf.txt"))
        logger.info(cmd)
        os.system(cmd)

    def _preprocess(self, inputs):
        datas = {}
        for i, input in enumerate(self.inputs):
            dtype = self.input_info[i]["dtype"]
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
        for i, input in enumerate(self.inputs):
            input_data_path = os.path.join(self.result_dir, 'hmquant_' + self.model_name 
                                           + '_' + input["name"] + '_input.npy')
            if os.path.exists(input_data_path):
                input_data = np.load(input_data_path)
                logger.info("golden input[{}] shape = {}, dtype = {}".format(input["name"], input_data.shape, input_data.dtype))
                input_data = input_data.astype(self.input_info[i]["dtype"])
                input_data = np.concatenate([input_data for i in range(self.batch)], axis=0)
                datas[input["name"]] = input_data
            else:
                logger.warning("compare canceled while golden input not found -> {}".format(input_data_path))
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
            logger.warning("compare canceled while golden output not found -> {}".format(golden_output_path))
            return None

    def gen_golden(self, inputs):
        from hmodel.utils.general import load_pkl_model
        qmodel = os.path.join(self.result_dir, self.model_name)
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
        input_infos = []
        input_num = self.module.get_num_inputs()
        for id in range(0, input_num):
            input_info = {}
            name = self.module.get_input_name(id)
            input_data = self.module.get_input_info(name)
            input_info["name"] = name
            input_info["shape"] = input_data.shape
            input_info["dtype"] = input_data.dtype
            input_info["format"] = input_data.format
            input_infos.append(input_info)
        return input_infos

    def get_output_info(self):
        output_infos = []
        output_num = self.module.get_num_outputs()
        for id in range(0, output_num):
            output_info = {}
            name = self.module.get_output_name(id)
            output_data = self.module.get_output_info(name, is_quanted=True)
            output_info["name"] = name
            output_info["shape"] = output_data.shape
            output_info["dtype"] = output_data.dtype
            output_info["format"] = output_data.format
            output_infos.append(output_info)
        return output_infos

    @property
    def freq(self):
        return {"H30": 1024}

    def get_relay_mac(self):
        raise NotImplemented

    def get_profile_info(self):
        raise NotImplemented

    def get_device_type(self):
        raise NotImplemented
