#!/usr/bin/env python

import os
import numpy as np
from abc import ABC
import onnx
import onnxruntime
import torch
import torchvision
from .base_exec import BaseExec
from ..utils import logger


class OnnxExec(BaseExec, ABC):
    def __init__(self, cfg: dict):
        super(OnnxExec, self).__init__(cfg)

    def load(self):
        self.module = onnxruntime.InferenceSession(self.weight)
        self.input_names = self.get_input_name()
        self.output_names = self.get_output_name()

    def infer(self, inputs):
        """ infer one time """
        outputs = {}
        datas = self.module.run(None, inputs)
        for id, name in enumerate(self.output_names):
            outputs[name] = datas[id]
        return outputs

    def _preprocess(self, inputs):
        datas = {}
        for input in self.inputs:
            if input["image"]["crop"]:
                pass
            if input["image"]["size"]:
                pass
            if input["image"]["format"] in ["YUV420", "YUV422", "YUV444"]:
                datas[input["name"]] = np.expand_dims(inputs[input["name"]], 0).astype(np.uint8)  # NHWC
            else:
                datas[input["name"]] = inputs[input["name"]]
        return datas

    def get_golden_inputs(self):
        datas = {}
        for input in self.inputs:
            input_data_path = os.path.join(self.result_dir, 'hmquant_' + self.model_name 
                                           + '_' + input["name"] + '_input.npy')
            input_data = np.load(input_data_path).astype(np.float32)
            logger.info("golden input[{}] shape = {}, dtype = {}".format(input["name"], input_data.shape, input_data.dtype))
            datas[input["name"]] = input_data
        return datas

    def get_golden_output(self, name):
        golden_output_path = os.path.join(self.golden_data_path, name + '.npy')
        if os.path.exists(golden_output_path):
            golden_output = np.load(golden_output_path, allow_pickle=True).item().get("output_tensor")
            return golden_output
        else:
            logger.warning("compare canceled while golden data not found -> {}".format(golden_output_path))
            return None

    def print_input_info(self):
        logger.info("input_names: {}".format(self.input_names))

    def print_output_info(self):
        logger.info("output_names: {}".format(self.output_names))

    def get_input_name(self):
        input_name = []
        for node in self.module.get_inputs():
            input_name.append(node.name)
        return input_name

    def get_output_name(self):
        output_name = []
        for node in self.module.get_outputs():
            output_name.append(node.name)
        return output_name
