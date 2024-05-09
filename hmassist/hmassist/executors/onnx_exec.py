#!/usr/bin/env python

import os
import numpy as np
from abc import ABC
import onnxruntime
from .base_exec import BaseExec
from ..utils import logger


from enum import IntEnum


class Format(IntEnum):
    YUV420SP = 0
    YUV422SP = 1
    YUV444SP = 2
    RGB888 = 3
    BGR888 = 4
    ND = 5


class OnnxExec(BaseExec, ABC):
    def __init__(self, cfg: dict):
        super(OnnxExec, self).__init__(cfg)

    def load(self):
        self.module = onnxruntime.InferenceSession(self.weight)
        self.input_info = self.get_input_info()
        self.output_info = self.get_output_info()
        logger.info("onnx model loaded")

    def infer(self, inputs):
        """ infer one time """
        outputs = {}
        datas = self.module.run(None, inputs)
        for id, output in enumerate(self.output_info):
            outputs[output["name"]] = datas[id]
        return outputs

    def _preprocess(self, inputs):
        datas = {}
        for input in self.inputs:
            # TODO: raw image crop and resize
            # if input["image"]["crop"]:
            #     pass
            # if input["image"]["size"]:
            #     pass
            data = inputs[input["name"]].astype(np.float32)
            data /= 255
            if input["format"] == "BGR":
                import cv2
                data = cv2.cvtColor(data, cv2.COLOR_BGR2RGB)
            if input["mean"]:
                for ch in range(input["shape"][1]):
                    data[0][ch] -= input["mean"][ch]
            if input["std"]:
                for ch in range(input["shape"][1]):
                    data[0][ch] /= input["std"][ch]
            datas[input["name"]] = data
        return datas

    def get_input_info(self):
        input_infos = []
        for node in self.module.get_inputs():
            input_info = {}
            input_info["name"] = node.name
            input_info["dtype"] = node.type
            input_info["shape"] = node.shape
            input_info["format"] = Format.RGB888
            input_infos.append(input_info)
        return input_infos

    def get_output_info(self):
        output_infos = []
        for node in self.module.get_outputs():
            output_info = {}
            output_info["name"] = node.name
            output_info["dtype"] = node.type
            output_info["shape"] = node.shape
            output_info["format"] = Format.ND
            output_infos.append(output_info)
        return output_infos
