"""

Author: Nan Xu
Maintainer: Nan Xu
Date: 2025/07/30
Company: Houmo

"""

from typing import List, Dict, Optional
TensorShape = List[int]
TensorShapes = Dict[Optional[str], TensorShape]


class OnnxOptimizerManager(object):
    OPTIMIZER_DICT={}
    def __init__(self,art_onnx_optimizer_method):
        self.current_art_onnx_optimizer_method=art_onnx_optimizer_method
        
    def __call__(self,art_onnx_optimizer_ptr):
        # print("[Art Onnx Optimizer registed]:",self.current_art_onnx_optimizer_method)
        OnnxOptimizerManager.OPTIMIZER_DICT[self.current_art_onnx_optimizer_method]=art_onnx_optimizer_ptr
        return art_onnx_optimizer_ptr

    @staticmethod
    def get(art_onnx_optimizer_method):
        return OnnxOptimizerManager.OPTIMIZER_DICT[art_onnx_optimizer_method]


