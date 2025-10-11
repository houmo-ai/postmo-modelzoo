#import copy
from ....utils import logger
#import math

import numpy as np  # type: ignore
import onnx  # type: ignore

#from ..onnxBaseOpt.onnxRuntimeEngine import OnnxRuntimeEngine
#from ..onnxGeneralManager.onnxGeneralDeleteFunctions import delete_shape_useless_node
from ...onnxBaseOpt.onnxDebugger import OnnxDebugger
#from ..onnxBaseOpt.onnxBaseFunctions import infer_shapes
from ...onnxBaseOpt.onnxBaseOptimizer import OnnxBaseOptimizer
from ...onnxUtils.onnxBasicUtils import *