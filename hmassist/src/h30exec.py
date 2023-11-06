#!/usr/bin/env python  

import time
import os
import numpy as np
from abc import ABC
from prettytable import PrettyTable
from collections import OrderedDict
from utils import logger
from utils.enum_type import PaddingMode
from .base_hmexec import Basehmexec
import onnx
import torch
import tvm.relay as relay
import tvm.tcim as tcim


class H30Exec(Basehmexec, ABC):
    def __init__(self, cfg: dict):
        super(H30Exec, self).__init__(cfg)

        self.model_path = os.path.join(self.model_dir, "{}".format(self.model_name))

    @staticmethod
    def set_env():
        pass

    def quantize(self, data_transform):
        logger.info("################  ptq quantize started  ######################")
        t_start = time.time()
        calib_files = {}
        calib_dataset = [dict() for i in range(self.quant["calib_num"])]
        calib_num = self.quant["calib_num"]

        for _input in self.inputs:
            name = _input["name"]
            calib_files[name] = []
            dtype = _input["dtype"]
            shape = self.shape_dict[name]
            calib_dir = None
            if self.quant["calib_dir"]:
                if name in self.quant["calib_dir"]:
                    calib_dir = self.quant["calib_dir"][name]
            if calib_dir:
                filelist = os.listdir(calib_dir)
                for filename in filelist:
                    _, ext = os.path.splitext(filename)
                    if ext in [".jpg", ".JPEG", ".bmp", ".png", ".jpeg", ".BMP", ".bin"]:
                        calib_files[name].append(os.path.join(calib_dir, filename))
                        if len(calib_files[name]) == calib_num:
                            break
                if len(calib_files[name]) < self.quant["calib_num"]:
                    logger.warning("calib_dir[{}] only has {} files, but calib_num is {}."
                                .format(name, len(calib_files[name]), self.quant["calib_num"]))
                    calib_num = len(calib_files[name])
                for i in range(calib_num):
                    calib_dataset[i][name] = torch.tensor(
                        self.get_data(name, dtype, shape, calib_files[name][i], data_transform))
            else:
                for i in range(calib_num):
                    calib_dataset[i][name] = torch.tensor(
                        self.get_data(name, dtype, shape, None, None))

        if not "ptq_cfg_path" in self.quant or not self.quant["ptq_cfg_path"]:
            n, c, h, w = self.inputs[0]["shape"]
            quanttool_config = {
                'inputs_cfg': {
                    self.inputs[0]["name"]: {
                        'data_format': self.inputs[0]["format"],
                        'first_layer_weight_denorm_mean': self.inputs[0]["mean"],
                        'first_layer_weight_denorm_std': self.inputs[0]["std"],
                        'resizer_crop': {'top': 0, 'left': 0, 'height': h, 'width': w},
                        'resizer_resize': {
                            'height': h,
                            'width': w,
                            'align_corners': False,
                            'method': 'bilinear',
                        },
                        'toYUV_format': self.inputs[0]["src_format"],
                    },
                },
                'graph_opt_cfg': {},
            }
        else:
            quanttool_config = self.quant["ptq_cfg_path"]

        # 删除列表中的空项
        del calib_dataset[calib_num:self.quant["calib_num"]]

        from hmquant.api import quant_single_onnx_network
        sequencer = quant_single_onnx_network(
            cfg=quanttool_config,
            calibration_data=calib_dataset,
            onnx_model_or_path=self.weight,
            device='cpu',
            analyze=True,
            debug=None,
            model_name=self.model_name,
            with_label=False,
            requant_dispatch=False,
        )

        sequencer.save_onnx(
            self.quant_model_path,
            save_out_tensor=False,
            save_params_npy=True,
            save_special_onnx=True
        )

        logger.info("################  ptq quantize finished  ######################")
        self.quantize_span = time.time() - t_start

        # gen golden data
        in_datas = {}
        for _input in self.inputs:
            name = _input["name"]
            dtype = _input["dtype"]
            shape = self.shape_dict[name]
            if "data_path" in _input:
                file_path = _input["data_path"]
            else:
                file_path = None
            in_datas[name] = self.get_data(name, dtype, shape, file_path, data_transform).astype("float32")

        quant_datas = {}
        for name, data in in_datas.items():
            # golden_input_path = os.path.join(golden_data_path, name + '.npy')
            # np.save(golden_data_path + name + '.npy', data)
            # data.tofile(golden_data_path + name + '.bin', data)
            quant_datas[name] = torch.tensor(data)

        from hmquant.api import generate_golden
        generate_golden(
            sequencer=sequencer,
            calibset=quant_datas,
            save_path=self.result_dir,
            model_name=self.model_name,
            batch_size=1,
            device="cpu"
        )

        logger.info("golden data saved in -> {}".format(self.golden_data_path))

        t_start = time.time()
        if self.quant["debug_level"] == 1:
            from hmquant.api import convert_profiling, quantize_profiling
            convert_profiling(self.weight, [in_datas], quanttool_config, [in_datas])
            quantize_profiling(sequencer, [in_datas])
        self.layer_compare_span = time.time() - t_start
        logger.info("quantize cost {}s, layer compare cost {}s".format(self.quantize_span, self.layer_compare_span))

    def build(self, config=None):
        """build relay quant
        @param in_datas:  infer data
        @return:
        """
        logger.info("################  build started  ######################")
        type_dict = {}
        shape_dict = {}
        convert_config = {}
        layout_dict = {}
        for input in self.inputs:
            layout_dict[input["name"]] = "NHWC"
            type_dict[input["name"]] = input["dtype"]
            shape_dict[input["name"]] = input["shape"]
        convert_config = {'layout': 'NHWC'}
        # convert_config["transpose_axes"] = (0, 2, 3, 1)
        t_start = time.time()
        onnx_model = onnx.load(self.quant_model_path)
        mod = relay.frontend.from_hmonnx(
            onnx_model, shape_dict, type_dict, layout_dict, resizer_attr=None,
            convert_config=convert_config,
        )

        if not config:
            config = {"tcim.sync_strategy": 0, "tcim.for_benchmark": True}

        logger.info("build_config={}".format(config))

        if self.build_mode == "AOT":
            from tvm.relay.backend import Executor
            import tvm
            executor = Executor('aot')
            target = tvm.target.Target('hdpl', host='c')

            with tvm.transform.PassContext(opt_level=3, config=config):
                graph, lib, params = relay.build(
                    mod, target, executor=executor, mod_name=self.model_name,
                )
            if os.getenv("TCIM_CROSS_COMPILE"):
                logger.info("cross compile enabled as aarch64")
                model_name = self.model_name + "_aarch64"
                tcim.store_so(model_name, lib, hdplcc_options=['-O2'], host_target="arm64")
            else:
                model_name = self.model_name
                tcim.store_so(model_name, lib, hdplcc_options=['-O2'])
            logger.info('{} saved as aot model in {}'.format(self.model_name, self.model_dir))

        elif self.build_mode == "JIT":
            with relay.build_config(opt_level=3):
                graph, lib, params = relay.build(mod, 'hdpl --host=llvm')
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
            logger.error("unsupoorted build mode ", self.build_mode)

    def infer(self, in_datas):
        """ infer one time """
        for input in self.inputs:
            if type(in_datas) == dict:
                input_data = in_datas[input["name"]]
            else:
                input_data = in_datas
            if self.build_mode == "AOT":
                if input["src_format"] in ["YUV420", "YUV422", "YUV444"]:
                    src_format = input["src_format"] + 'SP'
                else:
                    src_format = input["src_format"]
                self.module.set_input(input["name"], input_data, src_format)
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
            if (len(output_data.shape) == 4):
                # toolchain output is NHWC
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
        cmd = "cd {}/utils/{} && ./tcimexec --model {} --iterations {}".format(
            modelzoo_path, exec, model_path, test_num)
        logger.info(cmd)
        os.system(cmd)

    def get_golden_inputs(self):
        datas = {}
        for input in self.inputs:
            input_data_path = os.path.join(self.result_dir, 'hmquant_' + self.model_name 
                                           + '_' + input["name"] + '_input.npy')
            input_data = np.load(input_data_path).astype('uint8')
            # if input["layout"] == "NCHW":
                # NCHW -> NHWC
                # input_data = np.transpose(input_data, (0, 2, 3, 1))
            logger.info("golden input[{}] shape = {}, dtype = {}".format(input["name"], input_data.shape, input_data.dtype))
            datas[input["name"]] = input_data
        return datas

    def get_golden_output(self, name):
        golden_output_path = os.path.join(self.golden_data_path, name + '.npy')
        if os.path.exists(golden_output_path):
            golden_output = np.load(golden_output_path, allow_pickle=True).item().get("output_tensor")
            # if (len(golden_output.shape) == 4):
            #     # toolchain output is NHWC
            #     golden_output = np.transpose(golden_output, (0, 2, 3, 1))
            return golden_output
        else:
            logger.warning("compare canceled while golden data not found -> {}".format(golden_output_path))
            return None

    def get_version(self):
        raise NotImplemented

    def print_input_info(self):
        input_num = self.module.get_num_inputs()
        logger.info("model input num = {}".format(input_num))
        for id in range(0, input_num):
            name = self.module.get_input_name_by_index(id)
            # name = self.module.get_input_name(id)
            input_data = self.module.get_input_by_name(name).numpy()
        logger.info("model input[{}] shape = {}, dtype = {}".format(name, input_data.shape, input_data.dtype))

    def print_output_info(self):
        output_num = self.module.get_num_outputs()
        logger.info("model output num = {}".format(output_num))
        for id in range(0, output_num):
            name = self.module.get_output_name_by_index(id)
            output_data = self.module.get_output_by_name(name).numpy()
        logger.info("model output[{}] shape = {}, dtype = {}".format(name, output_data.shape, output_data.dtype))

    @property
    def freq(self):
        return {"H30": 1024}

    def get_relay_mac(self):
        raise NotImplemented

    def get_profile_info(self):
        raise NotImplemented

    def get_device_type(self):
        raise NotImplemented
