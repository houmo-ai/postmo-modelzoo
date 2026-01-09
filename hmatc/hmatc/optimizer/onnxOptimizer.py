#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
* Copyright 2025 HOUMO AI
*
* File: onnxOptimizer.py
* Description:
*   Graph Optimizer.
*
* Licensed under the Apache License, Version 2.0 (the "License");
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at
*
*     https://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.
*
* SPDX-License-Identifier: Apache-2.0
*
"""
import sys
import onnx
import logging
import collections
import copy

from .onnxBaseOpt.onnxConfigController import OnnxCfg
from .onnxBaseOpt.onnxDebugger import OnnxDebugger
from .onnxBaseOpt.onnxOptimizerManager import OnnxOptimizerManager
from .onnxBaseOpt.onnxRuntimeEngine import OnnxRuntimeEngine
from ..utils import logger

class OnnxOptimizer(object):
    def __init__(self, cfg):
        self.print_logo()
        OnnxCfg.set_cfg(cfg)
    
    def print_logo(self):
        logging.info("=========================================================================")
        logging.info(" _     _                 __         __          ____              _     ")
        logging.info("| |   | | ____  _    _  |  \       /  | ____   / __ \  _____   __| |__  ")
        logging.info("| |___| |/ __ \| |  | | |   \     /   |/ __ \ / /  \ \|  __  \|__   __| ")
        logging.info("|  ___  | /  \ | |  | | | |\ \   / /| | /  \ | (    ) | |__)  )  | |  _ ")
        logging.info("| |   | | \__/ | \__/ |_| | \ \_/ / | | \__/ |\ \__/ /| |___ /   | \_/ |")
        logging.info("|_|   |_|\____/ \____/\_|_|  \___/  |_|\____/  \____/ | |         \__ / ")
        logging.info("                                                      |_|               ")
        logging.info("======================Copywrite by Houmo 2025.07.30======================")
    
    def run(self, onnx_model):
        if not self.need_opt(onnx_model):
            return onnx_model, False
        OnnxDebugger.set_work_mode()
        OnnxDebugger.set_logging()

        self.stats_display(onnx_model)

        onnx_model = self.run_opt(onnx_model, "base_opt")
        onnx_model_origin = copy.deepcopy(onnx_model)
        onnx_model = self.run_opt(onnx_model, "general_opt")
        if OnnxDebugger.work_mode == "product":
            OnnxRuntimeEngine().ort_check_precision(onnx_model_origin, onnx_model)
        #self.check_opt_model(onnx_model)
        self.stats_display(onnx_model)
        return onnx_model, True

    def run_opt(self, onnx_model, name, *args):
        if OnnxCfg.get_val(name, True):
            onnx_model = OnnxOptimizerManager.get(name).opt(onnx_model, *args)
        return onnx_model  
    
    def need_opt(self, model:onnx.ModelProto):
        if not hasattr(model, 'metadata_props'):
            return False
        metadata_props = {entry.key: entry.value for entry in model.metadata_props}
        set_flag = metadata_props.get('__set_flag__', None) #getattr(metadata_props, "__set_flag__", None)
        producer = metadata_props.get('__producer__', None) #getattr(metadata_props, "__producer__", None)
        if set_flag is None or producer is None or set_flag != "HMAppOpt" or producer != "hmatc":
            return True
        return False
    
    def onnx_model_statistics(self, model:onnx.ModelProto):
        model_node = collections.defaultdict(int)
        model_size = sys.getsizeof(model.SerializeToString())
        for node in model.graph.node:
            model_node[node.op_type] += 1
        return (model_node, model_size)
    
    def stats_display(self, model):
        stats = self.onnx_model_statistics(model)
        size_name = ['bytes', 'KiB', 'MiB', 'GiB']
        count_size = 0
        display_size = stats[1]
        while display_size / 1024 > 1:
            display_size /= 1024.0
            count_size += 1
        size_string = "{:.2f}{}".format(display_size, size_name[count_size])
        key_len = [len(key) for key in stats[0].keys()]
        node_name_max_len= max(max(key_len), len("Model Size")) + 2
        count_col_len = max(len("nodeCount"), len(size_string) + 1) + 2
        logger.info(f"+{'-' * node_name_max_len}+{'-' * count_col_len}+")
        # print Table header
        logger.info("|{:^{}}|{:^{}}|".format("NodeName", node_name_max_len,
                                              "nodeCount", count_col_len))

        logger.info(f"+{'-' * node_name_max_len}+{'-' * count_col_len}+")
        # print stats
        for key in stats[0].keys():
            logger.info("|{:^{}}|{:<{}}|".format(key, node_name_max_len,
                                                  f" {stats[0][key]}", count_col_len))
        logger.info("|{:^{}}|{:<{}}|".format("Model Size", node_name_max_len,
                                              f" {size_string}", count_col_len))
        logger.info(f"+{'-' * node_name_max_len}+{'-' * count_col_len}+")