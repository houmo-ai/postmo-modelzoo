"""

Author: Nan Xu
Maintainer: Nan Xu
Date: 2025/07/14
Company: Houmo

"""
# import numpy as np  # type: ignore
# from onnx import TensorProto
import onnx.mapping

# ONNX_DTYPE = {
#     0: TensorProto.FLOAT,
#     1: TensorProto.FLOAT,
#     2: TensorProto.UINT8,
#     3: TensorProto.INT8,
#     4: TensorProto.UINT16,
#     5: TensorProto.INT16,
#     6: TensorProto.INT32,
#     7: TensorProto.INT64,
#     8: TensorProto.STRING,
#     9: TensorProto.BOOL,
#     10: TensorProto.FLOAT16,
#     11: TensorProto.DOUBLE,
#     12: TensorProto.UINT32,
#     13: TensorProto.UINT64,
# }

# NUMPY_DTYPE = {
#     0: np.float32,
#     1: np.float,
#     2: np.uint8,
#     3: np.int8,
#     4: np.uint16,
#     5: np.int16,
#     6: np.int32,
#     7: np.int64,
#     8: np.str,
#     9: np.bool_,
#     10: np.float16,
#     11: np.float64,
#     12: np.uint32,
#     13: np.uint64,
# }

NPDTYPE_2_ONNXDTYPE = onnx.mapping.NP_TYPE_TO_TENSOR_TYPE

ONNXDTYPE_2_NPDTYPE = onnx.mapping.TENSOR_TYPE_TO_NP_TYPE

# SHAPE_IRRELEVENT_OP_TYPE = ["Relu", "Sigmoid", "Tanh", "Clip", "Abs", "LeakyRelu", "PRelu", "Softplus",
#                             "BatchNormalization"]

# SHAPE_INDEPENDENT_OP_TYPE = ["Relu", "Sigmoid", "HardSigmoid", "Tanh", "Clip", "Erf", "Sqrt", "Neg", "Not",
#                              "Sin", "Cos", "Log", "Exp", "Abs", "Floor", "Ceil", "Cast"]

# REDUCEX_OP_TYPE = ["ReduceL1", "ReduceL2", "ReduceLogSum", "ReduceLogSumExp", "ReduceMax", "ReduceMean", "ReduceMin",
#                    "ReduceProd", "ReduceSum", "ReduceSumSquare"]

# LOGICAL_OP_TYPE = ["Greater", "Less", "Equal", "GreaterOrEqual", "LessOrEqual"]