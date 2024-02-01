#!/usr/bin/env python  

import time
import os
import numpy as np
from abc import ABC
import onnx
import torch
import re
import tvm
import tvm.relay as relay
import tvm.tcim as tcim
from ..utils import logger
from ..utils.utils import get_random_data
from .base_exec import BaseExec

class H30Exec(BaseExec, ABC):
    def __init__(self, cfg: dict):
        super(H30Exec, self).__init__(cfg)
        self.model_path = os.path.join(self.model_dir, self.model_name)

    def quantize(self, get_input_datas):
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
        type_dict = {}
        shape_dict = {}
        convert_config = {'layout': 'NHWC'}
        t_start = time.time()
        onnx_model = onnx.load(self.quant_model_path)
        for input in onnx_model.graph.input:
            dims = input.type.tensor_type.shape.dim
            for i in self.inputs:
                if i["name"] == input.name:
                    type_dict[input.name] = i["dtype"]
            input_shape = (
                self.batch * dims[0].dim_value, dims[1].dim_value,
                dims[2].dim_value, dims[3].dim_value,
            )
            shape_dict[input.name] = input_shape

        mod = relay.frontend.from_hmonnx(
            onnx_model, shape_dict, type_dict, resizer_attr=None,
            convert_config=convert_config,
        )

        if not build_options:
            build_options = {"tcim.fuse_strategy": 1, "tcim.codegen_pic": True, "tcim.for_benchmark": True}
            # "tcim.mem_plan_strategy": "linearscan"

        logger.info("build_options={}".format(build_options))

        if self.build_mode == "AOT":
            executor = relay.backend.Executor('aot')
            target = tvm.target.Target('hdpl', host='c')

            with tvm.transform.PassContext(opt_level=3, config=build_options):
                graph, lib, params = relay.build(
                    mod, target, executor=executor, mod_name=self.model_name,
                )

            if os.getenv("TCIM_CROSS_COMPILE") == 1:
                logger.info("cross compile enabled as aarch64")
                model_name = self.model_name + "_aarch64"
                tcim.store_so(model_name, lib, self.result_dir + "/../build", hdplcc_options=['-O2'], host_target="arm64")
            else:
                model_name = self.model_name
                tcim.store_so(model_name, lib, self.result_dir + "/../build", hdplcc_options=['-O2'])
            logger.info('{} saved as aot model in {}'.format(self.model_name, self.model_dir))

        elif self.build_mode == "JIT":
            with relay.build_config(opt_level=3):
                graph, lib, params = tvm.relay.build(mod, 'hdpl --host=llvm')
            # store model as one fusedop
            rt_opt = '-resizer'
            tcim.store_as_fusedop(self.model_name, graph, params, shape_dict, lib, rt_opt)
            logger.info(self.model_name, ' saved as one fusedop model')

        logger.info("################  build finished  ######################")
        self.build_span = time.time() - t_start
        logger.info("build cost {}s".format(self.build_span))

    def load(self):
        if self.build_mode == "AOT":
            self.module = tcim.load_so(self.model_name)
        elif self.build_mode == "JIT":
            self.module = tcim.load_model(self.model_name)
        else:
            logger.error("unsuported build mode ", self.build_mode)
            return -1
        self.input_info = self.get_input_info()
        self.output_info = self.get_output_info()
        logger.info("H30 model loaded")

    def infer(self, inputs):
        """ infer one time """
        for input in self.inputs:
            if isinstance(inputs, dict):
                input_data = inputs[input["name"]]
            else:
                input_data = inputs
            # print(input_data.shape, input_data.dtype)
            if self.build_mode == "AOT":
                if input["image"]["format"] in ["YUV420", "YUV422", "YUV444"]:
                    image_format = input["image"]["format"] + 'SP'
                else:
                    image_format = input["image"]["format"]
                self.module.set_input(input["name"], input_data, image_format)
            else:
                self.module.set_input(input["name"], input_data)
        self.module.run()
        outputs = {}
        output_num = self.module.get_num_outputs()
        for id in range(0, output_num):
            name = self.module.get_output_name_by_index(id)
            if self.is_fixed_out:
                output_data = self.module.get_output_by_name(name).numpy()
            else:
                output_data = self.module.get_float_output_by_name(name).numpy()
            if (len(output_data.shape) == 4):  # toolchain output is NHWC
                output_data = np.transpose(output_data, (0, 3, 1, 2))
            outputs[name] = output_data

        return outputs

    def perf(self, test_num):
        modelzoo_path = os.getenv('MODELZOO_PATH')
        if self.build_mode == "AOT":
            model_path = os.path.join(self.cur_dir, "tcim_" + self.model_name)
            exec = "aottcimexec"
        else:
            model_path = os.path.join(self.cur_dir, self.model_name)
            exec = "tcimexec"
        if os.environ.get("HDPL_PLATFORM") == "ISIM":
            test_num = 1
            logger.warning("test num set to 1 because HDPL_PLATFORM=ISIM may take a lot of time.")
        cmd = "cd {}/utils/{} && ./tcimexec --model {} --iterations {}".format(
            modelzoo_path, exec, model_path, test_num)
        logger.info(cmd)
        os.system(cmd)

    def _preprocess(self, inputs):
        datas = {}
        for input in self.inputs:
            if input["image"]["format"] in ["YUV420", "YUV422", "YUV444"]:
                data = torch.tensor(inputs[input["name"]].astype(np.float32))  # NHWC float32
                data = torch.squeeze(data, 0)  # HWC float32
                format = re.sub("YUV", "", input["image"]["format"])
                if input["format"] == "BGR":
                    from ..utils.transform import BGR2YUV
                    rgb2yuv_func = BGR2YUV(fmt=format)
                elif input["format"] == "RGB":
                    from ..utils.transform import RGB2YUV
                    rgb2yuv_func = RGB2YUV(fmt=format)
                else:
                    logger.error("not support format {}".format(input["image"]))
                image = torch.unsqueeze(rgb2yuv_func(data), 0).numpy()  # NHWC float32
                datas[input["name"]] = image.astype(np.uint8)
            else:
                datas[input["name"]] = inputs[input["name"]]
        return datas

    def get_golden_inputs(self):
        datas = {}
        for input in self.inputs:
            input_data_path = os.path.join(self.result_dir, 'hmquant_' + self.model_name 
                                           + '_' + input["name"] + '_input.npy')
            if os.path.exists(input_data_path):
                input_data = np.load(input_data_path).astype(np.uint8)
                logger.info("golden input[{}] shape = {}, dtype = {}".format(input["name"], input_data.shape, input_data.dtype))
                datas[input["name"]] = input_data
            else:
                logger.warning("compare canceled while golden input not found -> {}".format(input_data_path))
                return None
        return datas

    def get_golden_output(self, name):
        golden_output_path = os.path.join(self.golden_data_path, name + '.npy')
        if os.path.exists(golden_output_path):
            golden_output = np.load(golden_output_path, allow_pickle=True).item().get("output_tensor")
            return golden_output
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
            name = self.module.get_input_name_by_index(id)
            # input_data = self.module.get_input(id).numpy()
            input_info["name"] = name
            input_info["shape"] = None
            input_info["dtype"] = None
            input_infos.append(input_info)
        return input_infos

    def get_output_info(self):
        output_infos = []
        output_num = self.module.get_num_outputs()
        for id in range(0, output_num):
            output_info = {}
            name = self.module.get_output_name_by_index(id)
            output_data = self.module.get_output_by_name(name).numpy()
            output_info["name"] = name
            output_info["shape"] = output_data.shape
            output_info["dtype"] = output_data.dtype
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
