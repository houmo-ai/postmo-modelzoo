"""
ONNX Model Profiler

A single-file tool for profiling ONNX models and calculating MACs (Multiply-Accumulate operations).
Extracted from onnx_tool for minimal dependency.

Usage:
    from hmatc.utils.onnx_profile import model_profile

    macs = model_profile("model.onnx", mcfg={"verbose": True})
    print(f"Total MACs: {macs}")
"""

import os
import time
import math
import warnings
import pathlib
from typing import List

import numpy
import onnx

# ==============================================================================
# Constants and Version
# ==============================================================================

VERSION = "0.9.0"

# Tensor types
STATIC_TENSOR = 0
DYNAMIC_TENSOR = 1

# MACs constants (based on x86 instruction costs)
MUL_MACS = 1
ADD_MACS = 1
CMP_MACS = 1
DIV_MACS = 4
EXP_MACS = 32
POW_MACS = EXP_MACS
LOG_MACS = 43
SIN_MACS = 39
COS_MACS = 39
TANH_MACS = EXP_MACS + 2 * ADD_MACS + DIV_MACS
SQRT_MACS = 24
RESIZE_LINEAR_MACS = 4
RESIZE_CUBIC_MACS = 8

# Default ops to exclude from profiling (no computation)
NoMacsOps = (
    "Identity", "Constant", "Shape", "Squeeze", "Unsqueeze", "Reshape",
    "ConstantOfShape", "Cast", "Pad", "Concat", "Slice", "Gather",
)


# ==============================================================================
# Utility Classes and Functions
# ==============================================================================

class timer:
    """Simple timer for performance measurement."""
    def __init__(self):
        self._startt = time.time()

    def start(self):
        self._startt = time.time()

    def stop(self):
        return time.time() - self._startt


class ModelConfig:
    """Configuration for model loading and processing."""
    def __init__(self, mcfg=None):
        if mcfg is None:
            mcfg = {}
        self.cfg = mcfg
        self._set_default('constant_folding', False)
        self._set_default('node_rename', False)
        self._set_default('if_fixed_branch', None)
        self._set_default('fixed_topk', 0)
        self._set_default('verbose', False)

    def _set_default(self, name, default):
        setattr(self, name, self.cfg.get(name, default))


class Registry:
    """Simple registry for node type registration."""
    def __init__(self, name):
        self._name = name
        self._obj_map = {}

    def register(self, obj=None):
        if obj is None:
            def deco(func_or_class):
                self._obj_map[func_or_class.__name__] = func_or_class
                return func_or_class
            return deco
        self._obj_map[obj.__name__] = obj
        return obj

    def get(self, name):
        return self._obj_map.get(name)

    def __contains__(self, name):
        return name in self._obj_map


NODE_REGISTRY = Registry('NODE')


def volume(shape):
    """Calculate the volume (total elements) of a shape."""
    if not shape:
        return 0
    val = 1
    for v in shape:
        val *= v
    return val


def tuple2str(t, splitch=','):
    """Convert tuple to string representation."""
    return splitch.join(str(v) for v in t)


def onnxdtype2npdtype(data_type):
    """Convert ONNX data type to numpy dtype."""
    dtype_map = {
        onnx.TensorProto.FLOAT: numpy.float32,
        onnx.TensorProto.DOUBLE: numpy.float64,
        onnx.TensorProto.FLOAT16: numpy.float16,
        onnx.TensorProto.INT32: numpy.int32,
        onnx.TensorProto.INT16: numpy.int16,
        onnx.TensorProto.INT64: numpy.int64,
        onnx.TensorProto.INT8: numpy.int8,
        onnx.TensorProto.UINT8: numpy.uint8,
        onnx.TensorProto.BOOL: numpy.bool_,
    }
    return dtype_map.get(data_type, numpy.float32)


def npdtype2onnxdtype(npdtype):
    """Convert numpy dtype to ONNX data type."""
    if npdtype == numpy.float32: return onnx.TensorProto.FLOAT
    if npdtype == numpy.float64: return onnx.TensorProto.DOUBLE
    if npdtype == numpy.float16: return onnx.TensorProto.FLOAT16
    if npdtype == numpy.int32: return onnx.TensorProto.INT32
    if npdtype == numpy.int64: return onnx.TensorProto.INT64
    if npdtype == numpy.int8: return onnx.TensorProto.INT8
    if npdtype == numpy.uint8: return onnx.TensorProto.UINT8
    return onnx.TensorProto.FLOAT


def get_attribute_data(att):
    """Extract data from ONNX attribute."""
    if att.type == att.INTS:
        return list(att.ints)
    elif att.type == att.INT:
        return att.i
    elif att.type == att.FLOAT:
        return att.f
    elif att.type == att.STRING:
        return att.s
    elif att.type == att.FLOATS:
        return list(att.floats)
    elif att.type == att.TENSOR:
        return tensorproto2ndarray(att.t)
    return None


def tensorproto2ndarray(initial):
    """Convert ONNX TensorProto to numpy array."""
    shape = list(initial.dims)
    ndtype = onnxdtype2npdtype(initial.data_type)

    if initial.raw_data == b'':
        if ndtype == numpy.float32:
            arr = numpy.fromiter(initial.float_data, dtype=ndtype)
        elif ndtype == numpy.int32:
            arr = numpy.fromiter(initial.int32_data, dtype=ndtype)
        elif ndtype == numpy.int64:
            arr = numpy.fromiter(initial.int64_data, dtype=ndtype)
        elif ndtype == numpy.float16:
            raw = numpy.fromiter(initial.int32_data, dtype=numpy.uint16)
            arr = numpy.frombuffer(raw.tobytes(), dtype=numpy.float16)
        else:
            arr = numpy.zeros(shape, ndtype)
    else:
        arr = numpy.frombuffer(initial.raw_data, dtype=ndtype)

    # Always reshape to preserve scalar shape ([])
    return arr.reshape(shape)


def shape_of_tensor(tensor):
    """Get shape from ONNX ValueInfoProto."""
    shape = []
    for nb in tensor.type.tensor_type.shape.dim:
        if nb.HasField('dim_value'):
            shape.append(nb.dim_value)
        elif nb.HasField('dim_param'):
            shape.append(nb.dim_param)
    return shape


def create_ndarray_f32(shape):
    """Create float32 numpy array filled with ones."""
    return numpy.ones(shape, dtype=numpy.float32)


# ==============================================================================
# Tensor Class
# ==============================================================================

class Tensor:
    """Represents a tensor in the ONNX graph."""

    def __init__(self, t):
        if isinstance(t, str):
            self.name = t
            self.proto = None
            self.shape = []
            self.numpy = None
            self.type = DYNAMIC_TENSOR if t != '' else STATIC_TENSOR
            self.dtype = numpy.float32
        elif isinstance(t, onnx.ValueInfoProto):
            self.name = t.name
            self.proto = t
            self.shape = shape_of_tensor(t)
            self.numpy = None
            self.type = DYNAMIC_TENSOR
            self.dtype = onnxdtype2npdtype(t.type.tensor_type.elem_type)
        elif isinstance(t, onnx.TensorProto):
            self.name = t.name
            self.proto = t
            self.numpy = tensorproto2ndarray(t)
            self.shape = list(self.numpy.shape)
            self.type = STATIC_TENSOR
            self.dtype = self.numpy.dtype.type
        else:
            raise ValueError(f"Unsupported tensor type: {type(t)}")

        self.sparsity = None

    def update_shape(self, shape):
        self.shape = list(shape) if not isinstance(shape, list) else shape

    def update_dtype(self, dtype):
        self.dtype = dtype

    def get_shape(self):
        return [int(s) if not isinstance(s, str) else s for s in self.shape]

    def get_numpy(self):
        if self.numpy is not None:
            return self.numpy
        self.numpy = numpy.zeros(self.shape, dtype=self.dtype)
        return self.numpy

    def get_elementsize(self):
        if self.numpy is None:
            return numpy.dtype(self.dtype).itemsize
        return self.numpy.dtype.itemsize

    def get_memsize(self):
        return volume(self.get_shape()) * self.get_elementsize()

    def update_tensor(self, data):
        if not isinstance(data, numpy.ndarray):
            data = numpy.array(data)
        self.numpy = data
        self.update_shape(data.shape)
        self.update_dtype(data.dtype.type)


# ==============================================================================
# Node Classes
# ==============================================================================

def _max_shape(shapes):
    """Get the shape with maximum volume."""
    maxshape = shapes[0]
    maxvol = volume(maxshape)
    for shape in shapes:
        vol = volume(shape)
        if vol > maxvol or (vol == maxvol and len(shape) > len(maxshape)):
            maxshape = shape
            maxvol = vol
    return maxshape


def _axes_neg2pos(ndim, axes):
    """Convert negative axes to positive."""
    return [ax if ax >= 0 else ndim + ax for ax in axes]


def _conv_output_shape(xin, pad, ksize, stride, dilation):
    """Calculate convolution output shape."""
    return int((xin + pad - dilation * (ksize - 1) - 1) / stride + 1)


def _pooling_shape_calc(inshape, pad, kshape, dilation, stride, ceilmode):
    """Calculate pooling output shape."""
    outshape = (inshape + pad - ((kshape - 1) * dilation + 1)) / stride + 1
    return math.ceil(outshape) if ceilmode else math.floor(outshape)


class Node:
    """Base class for all ONNX operator nodes."""

    def __init__(self, n: onnx.NodeProto):
        self.name = n.name
        self.op_type = n.op_type
        self.nextnodes = []
        self.prevnodes = []
        self.output = []
        self.input = []
        self.proto = n
        self.shape_calc = False
        self.attr = {}

        for att in n.attribute:
            self.attr[att.name] = onnx.helper.get_attribute_value(att)
            setattr(self, att.name, get_attribute_data(att))
            if att.name == 'axes' and isinstance(self.axes, list):
                self.axes = tuple(self.axes)

    def add_default_value(self, name, default):
        if not hasattr(self, name):
            setattr(self, name, default)

    def shape_infer(self, intensors: List[Tensor], outtensors: List[Tensor]):
        """Infer output tensor shapes from input tensors."""
        pass

    def profile(self, intensors: List[Tensor], outtensors: List[Tensor]):
        """Calculate MACs for this node. Returns [forward_macs, backward_macs]."""
        return [0, 0]


class PWNode(Node):
    """Point-wise operation node base class."""

    def __init__(self, n):
        super().__init__(n)
        self.op_mac = ADD_MACS
        self.ratio = max(1, len(self.input) - 1)

    def shape_infer(self, intensors, outtensors):
        inshapes = [t.get_shape() for t in intensors]
        outtensors[0].update_shape(_max_shape(inshapes))
        outtensors[0].update_dtype(intensors[0].dtype)

    def profile(self, intensors, outtensors):
        return [volume(outtensors[0].get_shape()) * self.ratio * self.op_mac, 0]


class NpMathBase(Node):
    """Base for numpy-style broadcasting operations."""

    def __init__(self, n):
        super().__init__(n)
        self.op_mac = ADD_MACS
        self.ratio = max(1, len(self.input) - 1)

    def shape_infer(self, intensors, outtensors):
        maxlen = max(len(t.get_shape()) for t in intensors)
        inshapes = []
        for t in intensors:
            shape = t.get_shape()
            shape = [1] * (maxlen - len(shape)) + shape
            inshapes.append(shape)

        outshape = [max(shape[i] for shape in inshapes) for i in range(maxlen)]
        outtensors[0].update_shape(outshape)
        outtensors[0].update_dtype(intensors[0].dtype)

    def profile(self, intensors, outtensors):
        return [volume(outtensors[0].get_shape()) * self.ratio * self.op_mac, 0]


# Register math operation nodes
@NODE_REGISTRY.register()
class AddNode(NpMathBase): pass

@NODE_REGISTRY.register()
class SubNode(NpMathBase): pass

@NODE_REGISTRY.register()
class MulNode(NpMathBase):
    def __init__(self, n):
        super().__init__(n)
        self.op_mac = MUL_MACS

@NODE_REGISTRY.register()
class DivNode(NpMathBase):
    def __init__(self, n):
        super().__init__(n)
        self.op_mac = DIV_MACS


@NODE_REGISTRY.register()
class ReluNode(PWNode):
    def __init__(self, n):
        super().__init__(n)
        self.op_mac = CMP_MACS

@NODE_REGISTRY.register()
class LeakyReluNode(PWNode):
    def __init__(self, n):
        super().__init__(n)
        self.op_mac = MUL_MACS + CMP_MACS

@NODE_REGISTRY.register()
class SigmoidNode(PWNode):
    def __init__(self, n):
        super().__init__(n)
        # Sigmoid: 1/(1+exp(-x)), simplified to EXP_MACS for consistency with onnx-tool
        self.op_mac = EXP_MACS

@NODE_REGISTRY.register()
class TanhNode(PWNode):
    def __init__(self, n):
        super().__init__(n)
        self.op_mac = TANH_MACS

@NODE_REGISTRY.register()
class HardSigmoidNode(PWNode):
    def __init__(self, n):
        super().__init__(n)
        # HardSigmoid: max(0, min(1, alpha * x + beta))
        self.op_mac = MUL_MACS + ADD_MACS + CMP_MACS * 2

@NODE_REGISTRY.register()
class HardSwishNode(PWNode):
    def __init__(self, n):
        super().__init__(n)
        # HardSwish: x * ReLU6(x + 3) / 6
        # Equivalent to: x * max(0, min(1, x/6 + 0.5))
        self.op_mac = MUL_MACS * 2 + ADD_MACS + CMP_MACS * 2

@NODE_REGISTRY.register()
class SoftmaxNode(PWNode):
    def __init__(self, n):
        super().__init__(n)
        self.op_mac = EXP_MACS + DIV_MACS

@NODE_REGISTRY.register()
class ExpNode(PWNode):
    def __init__(self, n):
        super().__init__(n)
        self.op_mac = EXP_MACS

@NODE_REGISTRY.register()
class LogNode(PWNode):
    def __init__(self, n):
        super().__init__(n)
        self.op_mac = LOG_MACS

@NODE_REGISTRY.register()
class SqrtNode(PWNode):
    def __init__(self, n):
        super().__init__(n)
        self.op_mac = SQRT_MACS

@NODE_REGISTRY.register()
class ErfNode(PWNode):
    def __init__(self, n):
        super().__init__(n)
        # Erf has 0 MACs (element-wise operation, counted as memory access only)
        self.op_mac = 0

@NODE_REGISTRY.register()
class PowNode(PWNode):
    def __init__(self, n):
        super().__init__(n)
        self.op_mac = POW_MACS

@NODE_REGISTRY.register()
class SinNode(PWNode):
    def __init__(self, n):
        super().__init__(n)
        self.op_mac = SIN_MACS

@NODE_REGISTRY.register()
class CosNode(PWNode):
    def __init__(self, n):
        super().__init__(n)
        self.op_mac = COS_MACS

@NODE_REGISTRY.register()
class ClipNode(PWNode):
    def __init__(self, n):
        super().__init__(n)
        self.op_mac = CMP_MACS * 2


@NODE_REGISTRY.register()
class GemmNode(Node):
    """General Matrix Multiply node."""

    def __init__(self, n):
        super().__init__(n)
        self.add_default_value('transA', 0)
        self.add_default_value('transB', 0)

    def shape_infer(self, intensors, outtensors):
        xshape = intensors[0].get_shape()
        wshape = intensors[1].get_shape()

        if self.transA:
            xshape = xshape[::-1]
        if self.transB:
            yshape = xshape[:-1] + [wshape[-2]]
        else:
            yshape = xshape[:-1] + [wshape[-1]]

        outtensors[0].update_shape(yshape)
        outtensors[0].update_dtype(intensors[0].dtype)

    def profile(self, intensors, outtensors):
        yshape = outtensors[0].get_shape()
        wshape = intensors[1].get_shape()

        macs = volume(yshape)
        if self.transB:
            macs *= wshape[-1]
        else:
            macs *= wshape[-2]

        if len(intensors) == 3:
            macs += volume(yshape) * ADD_MACS

        return [macs, 0]


@NODE_REGISTRY.register()
class MatMulNode(GemmNode):
    """Matrix multiplication node."""
    pass


@NODE_REGISTRY.register()
class ConvNode(Node):
    """Convolution node."""

    def __init__(self, n):
        super().__init__(n)
        self.add_default_value('auto_pad', None)
        self.add_default_value('pads', (0, 0, 0, 0))
        self.add_default_value('strides', (1, 1))
        self.add_default_value('dilations', (1, 1))
        self.add_default_value('group', 1)

    def shape_infer(self, intensors, outtensors):
        xshape = intensors[0].get_shape()
        wshape = intensors[1].get_shape()

        if self.auto_pad and self.auto_pad != b'NOTSET':
            if self.auto_pad in [b'SAME_LOWER', b'SAME_UPPER']:
                shape = [xshape[0], wshape[0], math.ceil(xshape[2] / self.strides[0])]
                if len(xshape) == 4:
                    shape.append(math.ceil(xshape[3] / self.strides[1]))
            elif self.auto_pad == b'VALID':
                oh = math.ceil((xshape[2] - wshape[2] + 1) / self.strides[0])
                shape = [xshape[0], wshape[0], oh]
                if len(xshape) == 4:
                    ow = math.ceil((xshape[3] - wshape[3] + 1) / self.strides[1])
                    shape.append(ow)
            else:
                raise ValueError(f"Unknown auto_pad: {self.auto_pad}")
        else:
            if len(xshape) == 4:
                oh = _conv_output_shape(xshape[2], self.pads[0] + self.pads[2],
                                        wshape[2], self.strides[0], self.dilations[0])
                ow = _conv_output_shape(xshape[3], self.pads[1] + self.pads[3],
                                        wshape[3], self.strides[1], self.dilations[1])
                shape = [xshape[0], wshape[0], oh, ow]
            elif len(xshape) == 3:
                oh = _conv_output_shape(xshape[2], self.pads[0] + self.pads[1],
                                        wshape[2], self.strides[0], self.dilations[0])
                shape = [xshape[0], wshape[0], oh]
            else:
                shape = [1]

        outtensors[0].update_shape(shape)
        outtensors[0].update_dtype(intensors[0].dtype)

    def profile(self, intensors, outtensors):
        kernel_shape = intensors[1].get_shape()
        outvol = volume(outtensors[0].get_shape())
        reduce_vol = volume(kernel_shape[1:])

        macs = outvol * reduce_vol * MUL_MACS
        if len(intensors) > 2:
            macs += outvol * ADD_MACS

        return [macs, 0]


@NODE_REGISTRY.register()
class ConvTransposeNode(Node):
    """Transposed convolution node."""

    def __init__(self, n):
        super().__init__(n)
        self.add_default_value('pads', (0, 0, 0, 0))
        self.add_default_value('output_padding', (0, 0))
        self.add_default_value('strides', (1, 1))
        self.add_default_value('dilations', (1, 1))
        self.add_default_value('group', 1)

    def shape_infer(self, intensors, outtensors):
        xshape = intensors[0].get_shape()
        wshape = intensors[1].get_shape()
        outc = self.group * wshape[1]

        if len(xshape) == 4:
            ow = (xshape[2] - 1) * self.strides[0] - (self.pads[0] + self.pads[2]) + \
                 self.dilations[0] * (wshape[2] - 1) + self.output_padding[0] + 1
            oh = (xshape[3] - 1) * self.strides[1] - (self.pads[1] + self.pads[3]) + \
                 self.dilations[1] * (wshape[3] - 1) + self.output_padding[1] + 1
            shape = [xshape[0], outc, ow, oh]
        else:
            shape = [xshape[0], outc, 1]

        outtensors[0].update_shape(shape)
        outtensors[0].update_dtype(intensors[0].dtype)

    def profile(self, intensors, outtensors):
        kernel_shape = intensors[1].get_shape()
        outvol = volume(outtensors[0].get_shape())
        reduce_vol = volume(kernel_shape[1:])

        macs = outvol * reduce_vol * MUL_MACS
        if len(intensors) > 2:
            macs += outvol * ADD_MACS

        return [macs, 0]


@NODE_REGISTRY.register()
class BatchNormalizationNode(Node):
    """Batch normalization node."""

    def __init__(self, n):
        super().__init__(n)
        self.add_default_value('epsilon', 1e-05)

    def shape_infer(self, intensors, outtensors):
        outtensors[0].update_shape(intensors[0].get_shape())
        outtensors[0].update_dtype(intensors[0].dtype)

    def profile(self, intensors, outtensors):
        base = volume(outtensors[0].get_shape())
        return [base * (ADD_MACS + SQRT_MACS + DIV_MACS + ADD_MACS + MUL_MACS), 0]


@NODE_REGISTRY.register()
class LayerNormalizationNode(Node):
    """Layer normalization node."""

    def __init__(self, n):
        super().__init__(n)
        self.add_default_value('axis', -1)
        self.add_default_value('epsilon', 1e-05)

    def shape_infer(self, intensors, outtensors):
        outtensors[0].update_shape(intensors[0].get_shape())
        outtensors[0].update_dtype(intensors[0].dtype)

    def profile(self, intensors, outtensors):
        tshape = intensors[0].get_shape()
        axis = self.axis if self.axis >= 0 else len(tshape) + self.axis
        vol = volume(tshape)
        # vol2 is volume with axis dimension set to 1
        tshape_axis1 = list(tshape)
        tshape_axis1[axis] = 1
        vol2 = volume(tshape_axis1)
        return [vol * (MUL_MACS * 3 + ADD_MACS * 4) + vol2 * (ADD_MACS + SQRT_MACS + DIV_MACS), 0]


@NODE_REGISTRY.register()
class PoolBase(Node):
    """Base class for pooling operations."""

    def __init__(self, n):
        super().__init__(n)
        self.add_default_value('kernel_shape', (3, 3))
        self.add_default_value('ceil_mode', 0)
        self.add_default_value('pads', (0, 0, 0, 0))
        self.add_default_value('strides', (1, 1))
        self.add_default_value('dilations', (1, 1))
        self.add_default_value('auto_pad', None)

    def shape_infer(self, intensors, outtensors):
        inshape = intensors[0].get_shape()

        # Handle pads with different lengths (1D vs 2D pooling)
        pads = list(self.pads)
        if len(pads) == 2:
            pads = [pads[0], pads[1], pads[0], pads[1]]  # symmetric padding
        elif len(pads) < 4:
            pads = pads + [0] * (4 - len(pads))  # pad with zeros

        if self.auto_pad and self.auto_pad != b'NOTSET':
            if self.auto_pad in [b'SAME_LOWER', b'SAME_UPPER']:
                if len(inshape) >= 3:
                    outshape = inshape[:2] + [math.ceil(inshape[2] / self.strides[0])]
                    if len(inshape) == 4:
                        outshape.append(math.ceil(inshape[3] / self.strides[1]))
                else:
                    outshape = list(inshape)
            else:
                if len(inshape) >= 3:
                    outshape = inshape[:2] + [math.ceil((inshape[2] - self.kernel_shape[0] + 1) / self.strides[0])]
                else:
                    outshape = list(inshape)
        else:
            if len(inshape) >= 3:
                oh = _pooling_shape_calc(inshape[2], pads[0] + pads[2],
                                         self.kernel_shape[0], self.dilations[0],
                                         self.strides[0], self.ceil_mode)
                outshape = inshape[:2] + [oh]
                if len(inshape) == 4:
                    ow = _pooling_shape_calc(inshape[3], pads[1] + pads[3],
                                             self.kernel_shape[1], self.dilations[1],
                                             self.strides[1], self.ceil_mode)
                    outshape.append(ow)
            else:
                outshape = list(inshape)

        outtensors[0].update_shape(outshape)
        outtensors[0].update_dtype(intensors[0].dtype)

    def profile(self, intensors, outtensors):
        outvol = volume(outtensors[0].get_shape())
        macs = outvol * CMP_MACS * self.kernel_shape[0]
        if len(self.kernel_shape) > 1:
            macs *= self.kernel_shape[1]
        return [macs, 0]


@NODE_REGISTRY.register()
class MaxPoolNode(PoolBase): pass

@NODE_REGISTRY.register()
class AveragePoolNode(PoolBase): pass

@NODE_REGISTRY.register()
class GlobalAveragePoolNode(Node):
    def shape_infer(self, intensors, outtensors):
        inshape = intensors[0].get_shape()
        shape = list(inshape[:2]) + [1] * (len(inshape) - 2)
        outtensors[0].update_shape(shape)
        outtensors[0].update_dtype(intensors[0].dtype)

    def profile(self, intensors, outtensors):
        vol = volume(intensors[0].get_shape())
        return [vol * ADD_MACS + volume(outtensors[0].get_shape()) * DIV_MACS, 0]


@NODE_REGISTRY.register()
class ReshapeNode(Node):
    def shape_infer(self, intensors, outtensors):
        srcshape = intensors[0].get_shape()
        if intensors[1].get_numpy() is None:
            outtensors[0].update_shape([1])
        else:
            shape = intensors[1].get_numpy()
            # When shape has 0, it means to keep the corresponding dimension from srcshape
            # But if shape has more dimensions than srcshape, we need to handle index out of range
            newshape = []
            for i, s in enumerate(shape):
                if s == 0:
                    if i < len(srcshape):
                        newshape.append(int(srcshape[i]))
                    else:
                        # shape has more dims than srcshape, 0 means 1 (empty dimension)
                        newshape.append(1)
                else:
                    newshape.append(int(s))
            if -1 in newshape:
                total = volume(srcshape)
                known = volume([s for s in newshape if s > 0])
                newshape[newshape.index(-1)] = total // known
            outtensors[0].update_shape(newshape)
        outtensors[0].update_dtype(intensors[0].dtype)


@NODE_REGISTRY.register()
class TransposeNode(Node):
    def __init__(self, n):
        super().__init__(n)
        self.add_default_value('perm', None)

    def shape_infer(self, intensors, outtensors):
        xshape = intensors[0].get_shape()
        if self.perm is None:
            yshape = xshape[::-1]
        else:
            # Handle case where perm has more dims than input shape
            # Pad xshape with 1s if needed (broadcasting case)
            if len(self.perm) > len(xshape):
                xshape = [1] * (len(self.perm) - len(xshape)) + list(xshape)
            yshape = [xshape[i] for i in self.perm]
        outtensors[0].update_shape(yshape)
        outtensors[0].update_dtype(intensors[0].dtype)


@NODE_REGISTRY.register()
class ConcatNode(Node):
    def __init__(self, n):
        super().__init__(n)
        self.add_default_value('axis', 0)

    def shape_infer(self, intensors, outtensors):
        outshape = list(intensors[0].get_shape())
        axis = self.axis if self.axis >= 0 else len(outshape) + self.axis

        # Handle axis out of range
        if axis >= len(outshape):
            outtensors[0].update_shape(outshape)
            outtensors[0].update_dtype(intensors[0].dtype)
            return

        for t in intensors[1:]:
            tshape = t.get_shape()
            if axis < len(tshape):
                outshape[axis] += tshape[axis]
        outtensors[0].update_shape(outshape)
        outtensors[0].update_dtype(intensors[0].dtype)


@NODE_REGISTRY.register()
class FlattenNode(Node):
    def __init__(self, n):
        super().__init__(n)
        self.add_default_value('axis', 1)

    def shape_infer(self, intensors, outtensors):
        x = intensors[0].get_shape()
        axis = self.axis if self.axis >= 0 else len(x) + self.axis
        outshape = [volume(x[:axis]), volume(x[axis:])]
        outtensors[0].update_shape(outshape)
        outtensors[0].update_dtype(intensors[0].dtype)


@NODE_REGISTRY.register()
class ReduceMeanNode(Node):
    def __init__(self, n):
        super().__init__(n)
        self.add_default_value('axes', None)
        self.add_default_value('keepdims', 1)

    def shape_infer(self, intensors, outtensors):
        xshape = intensors[0].get_shape()
        axes = self.axes if self.axes is not None else list(range(len(xshape)))

        yshape = []
        for i in range(len(xshape)):
            if i in axes:
                if self.keepdims:
                    yshape.append(1)
            else:
                yshape.append(xshape[i])

        outtensors[0].update_shape(yshape)
        outtensors[0].update_dtype(intensors[0].dtype)

    def profile(self, intensors, outtensors):
        return [volume(intensors[0].get_shape()) * ADD_MACS, 0]


@NODE_REGISTRY.register()
class ReduceSumNode(ReduceMeanNode): pass

@NODE_REGISTRY.register()
class ReduceMaxNode(ReduceMeanNode):
    def profile(self, intensors, outtensors):
        return [volume(intensors[0].get_shape()) * CMP_MACS, 0]


@NODE_REGISTRY.register()
class LSTMNode(Node):
    """LSTM node."""

    def __init__(self, n):
        super().__init__(n)
        self.add_default_value('hidden_size', None)
        self.add_default_value('direction', None)

    def shape_infer(self, intensors, outtensors):
        xshape = intensors[0].get_shape()
        wshape = intensors[1].get_shape()
        seq_len = xshape[0]
        batch = xshape[1]
        num_dir = wshape[0]
        h_len = wshape[1] // 4

        outtensors[0].update_shape([seq_len, num_dir, batch, h_len])
        outtensors[0].update_dtype(intensors[0].dtype)

        if len(outtensors) > 1:
            outtensors[1].update_shape([num_dir, batch, h_len])
            outtensors[1].update_dtype(intensors[0].dtype)

    def profile(self, intensors, outtensors):
        xshape = intensors[0].get_shape()
        wshape = intensors[1].get_shape()
        rshape = intensors[2].get_shape()

        batch = xshape[1]
        seq = xshape[0]
        h_len = wshape[1] // 4
        ht_size = volume([batch, seq, h_len])

        gemm_macs = (volume(wshape) + volume(rshape)) * batch * seq
        sig_macs = ht_size * EXP_MACS * 3
        tanh_macs = ht_size * TANH_MACS * 2
        blend_macs = ht_size * (ADD_MACS + MUL_MACS * 3)

        return [gemm_macs + sig_macs + tanh_macs + blend_macs, 0]


@NODE_REGISTRY.register()
class GRUNode(Node):
    """GRU node."""

    def shape_infer(self, intensors, outtensors):
        xshape = intensors[0].get_shape()
        wshape = intensors[1].get_shape()
        seq_len = xshape[0]
        batch = xshape[1]
        num_dir = wshape[0]
        h_len = wshape[1] // 3

        outtensors[0].update_shape([seq_len, num_dir, batch, h_len])
        outtensors[0].update_dtype(intensors[0].dtype)

    def profile(self, intensors, outtensors):
        xshape = intensors[0].get_shape()
        wshape = intensors[1].get_shape()
        rshape = intensors[2].get_shape()

        batch = xshape[1]
        seq = xshape[0]
        h_len = wshape[1] // 3
        ht_size = volume([batch, seq, h_len])

        gemm_macs = (volume(wshape) + volume(rshape)) * batch * seq
        sig_macs = ht_size * EXP_MACS * 2
        tanh_macs = ht_size * TANH_MACS
        blend_macs = ht_size * (ADD_MACS * 2 + MUL_MACS * 3)

        return [gemm_macs + sig_macs + tanh_macs + blend_macs, 0]


@NODE_REGISTRY.register()
class AttentionNode(Node):
    """Attention node (for transformer models)."""

    def profile(self, intensors, outtensors):
        # Q, K, V projections + attention + output projection
        qshape = intensors[0].get_shape()
        batch = qshape[0]
        seq = qshape[1]
        hidden = qshape[2]

        # Simplified: assume self-attention
        proj_macs = batch * seq * hidden * hidden * 3  # Q, K, V
        attn_macs = batch * seq * seq * hidden  # attention scores
        out_macs = batch * seq * hidden * hidden  # output projection

        return [proj_macs + attn_macs + out_macs, 0]


# Additional common nodes
@NODE_REGISTRY.register()
class DropoutNode(Node):
    def shape_infer(self, intensors, outtensors):
        outtensors[0].update_shape(intensors[0].get_shape())
        outtensors[0].update_dtype(intensors[0].dtype)


@NODE_REGISTRY.register()
class IdentityNode(Node):
    def shape_infer(self, intensors, outtensors):
        outtensors[0].update_shape(intensors[0].get_shape())
        outtensors[0].update_dtype(intensors[0].dtype)


@NODE_REGISTRY.register()
class ConstantNode(Node):
    def shape_infer(self, intensors, outtensors):
        if hasattr(self, 'value'):
            outtensors[0].update_shape(self.value.shape)
            outtensors[0].update_dtype(self.value.dtype.type)
            outtensors[0].update_tensor(self.value)


@NODE_REGISTRY.register()
class CastNode(Node):
    def shape_infer(self, intensors, outtensors):
        outtensors[0].update_shape(intensors[0].get_shape())
        outtensors[0].update_dtype(onnxdtype2npdtype(self.to))


@NODE_REGISTRY.register()
class SliceNode(Node):
    def __init__(self, n):
        super().__init__(n)
        self.add_default_value('axes', None)
        self.add_default_value('starts', None)
        self.add_default_value('ends', None)

    def shape_infer(self, intensors, outtensors):
        xshape = intensors[0].get_shape()
        outshape = list(xshape)

        # ONNX Slice inputs: data, starts, ends, axes, steps (last 4 optional)
        # If axes not provided, default to all axes [0, 1, ..., len(xshape)-1]
        # If steps not provided, default to 1 for each axis

        starts = None
        ends = None
        axes = None
        steps = None

        # Get starts from input or attribute
        if len(intensors) >= 2:
            starts_arr = intensors[1].get_numpy()
            starts = list(starts_arr.flatten()) if starts_arr.ndim > 0 else [starts_arr.item()]
        elif self.starts is not None:
            starts = list(self.starts)

        # Get ends from input or attribute
        if len(intensors) >= 3:
            ends_arr = intensors[2].get_numpy()
            ends = list(ends_arr.flatten()) if ends_arr.ndim > 0 else [ends_arr.item()]
        elif self.ends is not None:
            ends = list(self.ends)

        # Get axes from input or attribute
        if len(intensors) >= 4:
            axes_arr = intensors[3].get_numpy()
            axes = list(axes_arr.flatten()) if axes_arr.ndim > 0 else [axes_arr.item()]
        elif self.axes is not None:
            axes = list(self.axes)
        else:
            # Default: slice all axes
            axes = list(range(len(xshape)))

        # Get steps from input (optional)
        if len(intensors) >= 5:
            steps_arr = intensors[4].get_numpy()
            steps = list(steps_arr.flatten()) if steps_arr.ndim > 0 else [steps_arr.item()]
        else:
            steps = [1] * len(axes)

        # Compute output shape for each axis
        if starts and ends and axes and steps:
            for i, axis in enumerate(axes):
                axis = axis if axis >= 0 else len(xshape) + axis
                start = starts[i] if starts[i] >= 0 else xshape[axis] + starts[i]
                end = ends[i] if ends[i] >= 0 else xshape[axis] + ends[i]
                step = steps[i]

                # Clamp values
                start = max(0, min(start, xshape[axis]))
                end = max(0, min(end, xshape[axis]))

                # Compute output dimension
                if step > 0:
                    if end > start:
                        outshape[axis] = (end - start) // step
                    else:
                        outshape[axis] = 0
                else:
                    # Negative step (reverse slicing)
                    if start > end:
                        outshape[axis] = (start - end) // abs(step)
                    else:
                        outshape[axis] = 0

        outtensors[0].update_shape(outshape)
        outtensors[0].update_dtype(intensors[0].dtype)


@NODE_REGISTRY.register()
class GatherNode(Node):
    def __init__(self, n):
        super().__init__(n)
        self.add_default_value('axis', 0)

    def shape_infer(self, intensors, outtensors):
        xshape = intensors[0].get_shape()
        idxshape = intensors[1].get_shape()
        axis = self.axis if self.axis >= 0 else len(xshape) + self.axis

        yshape = []
        for i in range(len(xshape)):
            if i == axis:
                yshape.extend(idxshape)
            else:
                yshape.append(xshape[i])

        outtensors[0].update_shape(yshape)
        outtensors[0].update_dtype(intensors[0].dtype)


@NODE_REGISTRY.register()
class UnsqueezeNode(Node):
    def __init__(self, n):
        super().__init__(n)
        self.add_default_value('axes', [0])

    def shape_infer(self, intensors, outtensors):
        inshape = intensors[0].get_shape()
        # Handle scalar (0-d) array case for axes
        axes_arr = intensors[1].get_numpy() if len(intensors) > 1 else None
        if axes_arr is not None:
            axes = list(axes_arr.flatten()) if axes_arr.ndim > 0 else [axes_arr.item()]
        else:
            axes = self.axes
        axes = _axes_neg2pos(len(inshape) + len(axes), axes)

        newshape = []
        idx = 0
        for i in range(len(inshape) + len(axes)):
            if i in axes:
                newshape.append(1)
            else:
                newshape.append(inshape[idx])
                idx += 1

        outtensors[0].update_shape(newshape)
        outtensors[0].update_dtype(intensors[0].dtype)


@NODE_REGISTRY.register()
class SqueezeNode(Node):
    def __init__(self, n):
        super().__init__(n)
        self.add_default_value('axes', [0])

    def shape_infer(self, intensors, outtensors):
        inshape = intensors[0].get_shape()
        # Handle scalar (0-d) array case for axes
        axes_arr = intensors[1].get_numpy() if len(intensors) > 1 else None
        if axes_arr is not None:
            axes = list(axes_arr.flatten()) if axes_arr.ndim > 0 else [axes_arr.item()]
        else:
            axes = self.axes
        axes = _axes_neg2pos(len(inshape), axes)

        newshape = [inshape[i] for i in range(len(inshape)) if i not in axes]
        outtensors[0].update_shape(newshape)
        outtensors[0].update_dtype(intensors[0].dtype)


@NODE_REGISTRY.register()
class SplitNode(Node):
    def __init__(self, n):
        super().__init__(n)
        self.add_default_value('axis', 0)
        self.add_default_value('split', None)

    def shape_infer(self, intensors, outtensors):
        inshape = intensors[0].get_shape()
        axis = self.axis if self.axis >= 0 else len(inshape) + self.axis

        # Get split values from attribute or second input
        if self.split is not None:
            split = self.split
        elif len(intensors) > 1:
            split_arr = intensors[1].get_numpy()
            split = list(split_arr.flatten()) if split_arr.ndim > 0 else [split_arr.item()]
        else:
            split = [inshape[axis] // len(outtensors)]

        for i, out in enumerate(outtensors):
            shape = list(inshape)
            shape[axis] = split[i] if i < len(split) else split[-1]
            out.update_shape(shape)
            out.update_dtype(intensors[0].dtype)


@NODE_REGISTRY.register()
class PadNode(Node):
    def __init__(self, n):
        super().__init__(n)
        self.add_default_value('pads', None)

    def shape_infer(self, intensors, outtensors):
        inshape = intensors[0].get_shape()
        # Handle scalar (0-d) array case for pads
        if self.pads:
            pads = self.pads
        elif len(intensors) > 1:
            pads_arr = intensors[1].get_numpy()
            pads = list(pads_arr.flatten()) if pads_arr.ndim > 0 else [pads_arr.item()]
        else:
            pads = [0] * len(inshape) * 2

        newshape = []
        for i, v in enumerate(inshape):
            newshape.append(v + pads[i] + pads[i + len(inshape)])

        outtensors[0].update_shape(newshape)
        outtensors[0].update_dtype(intensors[0].dtype)


@NODE_REGISTRY.register()
class ExpandNode(Node):
    def shape_infer(self, intensors, outtensors):
        xshape = intensors[0].get_shape()
        # Handle scalar (0-d) array case for expandshape
        expand_arr = intensors[1].get_numpy()
        expandshape = list(expand_arr.flatten()) if expand_arr.ndim > 0 else [expand_arr.item()]

        if len(xshape) < len(expandshape):
            xshape = [1] * (len(expandshape) - len(xshape)) + xshape

        yshape = [max(x, e) for x, e in zip(xshape, expandshape)]
        outtensors[0].update_shape(yshape)
        outtensors[0].update_dtype(intensors[0].dtype)


@NODE_REGISTRY.register()
class ResizeNode(Node):
    def __init__(self, n):
        super().__init__(n)
        self.add_default_value('mode', b'nearest')

    def shape_infer(self, intensors, outtensors):
        xshape = intensors[0].get_shape()
        outshape = list(xshape)

        # ONNX Resize inputs: X, roi, scales, sizes (last 3 are optional)
        # Input order: X (required), roi (optional), scales (optional), sizes (optional)

        if len(intensors) >= 4 and intensors[3].get_numpy().size > 0:
            # sizes provided (input 3)
            sizes_arr = intensors[3].get_numpy()
            sizes = list(sizes_arr.flatten()) if sizes_arr.ndim > 0 else [sizes_arr.item()]
            if len(sizes) == len(xshape):
                outshape = sizes
            else:
                outshape = xshape[:2] + sizes[-2:]
        elif len(intensors) >= 3 and intensors[2].get_numpy().size > 0:
            # scales provided (input 2)
            scales_arr = intensors[2].get_numpy()
            scales = list(scales_arr.flatten()) if scales_arr.ndim > 0 else [scales_arr.item()]
            if len(scales) == len(xshape):
                outshape = [int(s * scale) for s, scale in zip(xshape, scales)]
            else:
                # scales might be partial (only for H, W)
                outshape = list(xshape[:2]) + [int(s * scale) for s, scale in zip(xshape[2:], scales[-2:])]

        outtensors[0].update_shape(outshape)
        outtensors[0].update_dtype(intensors[0].dtype)


@NODE_REGISTRY.register()
class UpsampleNode(ResizeNode): pass


def create_node(n: onnx.NodeProto):
    """Factory function to create appropriate node type."""
    node_class = NODE_REGISTRY.get(n.op_type + 'Node')
    if node_class:
        return node_class(n)

    # Default node for unknown types
    warnings.warn(f'Node type {n.op_type} not registered, using default (0 MACs)')
    return Node(n)


# ==============================================================================
# Graph Class
# ==============================================================================

_SHAPE_TENSORS = {
    "Reshape": ("1of2",),
    "Resize": ("2of3", "3of4"),
    "Slice": ("1,2of3", "1,2,3of4"),
}


def _contains_shape_tensor(n):
    """Check if node contains shape tensor inputs."""
    if n.op_type not in _SHAPE_TENSORS:
        return []

    shape_tensors = []
    for desc in _SHAPE_TENSORS[n.op_type]:
        indice, count = desc.split("of")
        if len(n.input) == int(count):
            for istr in indice.split(","):
                shape_tensors.append(n.input[int(istr)])
    return shape_tensors


class Graph:
    """ONNX computation graph for profiling."""

    def __init__(self, g: onnx.GraphProto, mcfg: ModelConfig):
        self.cfg = mcfg
        self.nodemap = {}
        self.tensormap = {}
        self.producedby = {}
        self.consumedby = {}
        self.initials = []
        self.dynamics = []
        self.input = []
        self.output = []
        self.valid_shape = False
        self.valid_profile = False
        self.sparse_model = False

        if g is not None:
            self._init_from_onnx(g)

    def log(self, msg):
        if self.cfg.verbose:
            print(msg)

    def _init_from_onnx(self, g):
        """Initialize graph from ONNX GraphProto."""
        self.node_count = 0
        tm = timer()

        # Create nodes
        for node_proto in g.node:
            node = create_node(node_proto)
            if not node.name:
                node.name = f"{node.op_type}_{self.node_count}"
            self.node_count += 1

            for tensor in node_proto.input:
                if tensor not in self.consumedby:
                    self.consumedby[tensor] = []
                self.consumedby[tensor].append(node.name)
                node.input.append(tensor)

            for tensor in node_proto.output:
                if tensor not in self.producedby:
                    self.producedby[tensor] = []
                self.producedby[tensor].append(node.name)
                node.output.append(tensor)

            self.nodemap[node.name] = node

        self.log(f"Node Init Time: {tm.stop():.4f}s")

        # Create tensors from initializers
        tm.start()
        for initial in g.initializer:
            self.tensormap[initial.name] = Tensor(initial)
            self.initials.append(initial.name)

        # Create tensors from inputs
        for inp in g.input:
            if inp.name not in self.tensormap:
                self.tensormap[inp.name] = Tensor(inp)

        # Create tensors from outputs
        for out in g.output:
            if out.name not in self.tensormap:
                self.tensormap[out.name] = Tensor(out)

        # Create tensors from value_info
        for valinfo in g.value_info:
            if valinfo.name not in self.tensormap:
                self.tensormap[valinfo.name] = Tensor(valinfo)

        self.log(f"Tensor Init Time: {tm.stop():.4f}s")

        # Initialize dynamic tensors and node connections
        for name, node in self.nodemap.items():
            for tensor in node.input:
                if tensor not in self.tensormap:
                    self.tensormap[tensor] = Tensor(tensor)
                if tensor not in self.initials and tensor not in self.dynamics:
                    self.dynamics.append(tensor)

            for tensor in node.output:
                if tensor not in self.tensormap:
                    self.tensormap[tensor] = Tensor(tensor)
                if tensor not in self.dynamics:
                    self.dynamics.append(tensor)

        # Find graph inputs and outputs
        self.input = [t for t in self.dynamics if t not in self.producedby]
        self.output = [t for t in self.dynamics if t not in self.consumedby or not self.consumedby[t]]

        # Mark shape calculation nodes
        self._find_shape_tensors()

    def _find_shape_tensors(self):
        """Identify nodes that compute shapes."""
        for name, node in self.nodemap.items():
            shape_tensors = _contains_shape_tensor(node.proto)
            if shape_tensors:
                node.shape_calc = True

    def graph_reorder_nodes(self):
        """Topologically sort nodes."""
        ordered = []
        deps = {name: 0 for name in self.nodemap}

        for name, node in self.nodemap.items():
            for t in node.input:
                if t in self.producedby:
                    deps[name] += 1

        queue = [n for n, d in deps.items() if d == 0]
        while queue:
            name = queue.pop(0)
            ordered.append(name)
            for t in self.nodemap[name].output:
                if t in self.consumedby:
                    for consumer in self.consumedby[t]:
                        deps[consumer] -= 1
                        if deps[consumer] == 0:
                            queue.append(consumer)

        self.nodemap = {n: self.nodemap[n] for n in ordered if n in self.nodemap}

    def shape_infer(self, inputs=None):
        """Infer shapes for all tensors."""
        self.valid_shape = False

        if inputs:
            for name, data in inputs.items():
                if name in self.tensormap:
                    self.tensormap[name].update_tensor(data)

        # Check inputs have valid shapes
        for name in self.input:
            shape = self.tensormap[name].get_shape()
            for val in shape:
                if isinstance(val, str) or (isinstance(val, int) and val < 0):
                    raise ValueError(f"Input '{name}' has invalid shape: {shape}")

        tm = timer()
        for name, node in self.nodemap.items():
            itensors = [self.tensormap[t] for t in node.input if t in self.tensormap]
            otensors = [self.tensormap[t] for t in node.output if t in self.tensormap]

            if otensors:
                node.shape_infer(itensors, otensors)

        self.log(f"Shape inference time: {tm.stop():.4f}s")
        self.valid_shape = True

    def profile(self):
        """Calculate MACs for all nodes."""
        self.valid_profile = False

        if not self.valid_shape:
            warnings.warn("Run shape_infer() before profile()")
            return

        self.macs = [0.0, 0.0]
        self.params = 0
        self.memory = 0
        params_counted = set()

        for name, node in self.nodemap.items():
            itensors = [self.tensormap[t] for t in node.input if t in self.tensormap]
            otensors = [self.tensormap[t] for t in node.output if t in self.tensormap]

            # Calculate parameters
            node_params = 0
            node_memory = 0
            for t in node.input:
                if t in self.initials and t not in params_counted:
                    vol = volume(self.tensormap[t].get_shape())
                    node_params += vol
                    node_memory += vol * self.tensormap[t].get_elementsize()
                    params_counted.add(t)

            # Calculate output memory
            for t in node.output:
                if node.op_type != "Constant":
                    node_memory += self.tensormap[t].get_memsize()

            # Calculate MACs
            macs = node.profile(itensors, otensors) if itensors and otensors else [0, 0]

            # Store results
            node.macs = macs
            node.params = node_params
            node.memory = node_memory
            node.inshape = itensors[0].get_shape() if itensors else (0,)
            node.outshape = otensors[0].get_shape() if otensors else (0,)

            self.macs[0] += macs[0]
            self.macs[1] += macs[1]
            self.params += node_params
            self.memory += node_memory

        self.valid_profile = True

    def print_node_map(self, f=None, metric="MACs", exclude_ops=None):
        """Print or save profiling results."""
        if not self.valid_profile:
            warnings.warn("Run profile() before print_node_map()")
            return 0

        from tabulate import tabulate

        factor = 2 if metric == "FLOPs" else 1
        forward_macs = int(round(self.macs[0]))
        ptable = []

        for name, node in self.nodemap.items():
            if exclude_ops and node.op_type in exclude_ops:
                continue

            row = [
                name, node.op_type,
                f"{int(node.macs[0] * factor):,}",
                f"{node.macs[0] / max(self.macs[0], 1):.2%}",
                f"{int(node.params):,}",
                tuple2str(node.inshape), tuple2str(node.outshape)
            ]
            ptable.append(row)

        # Total row
        ptable.append([
            "Total", "-",
            f"{int(forward_macs * factor):,}", "100%",
            f"{int(self.params):,}", "-", "-"
        ])

        header = ["Name", "Type", metric, "Percent", "Params", "InShape", "OutShape"]

        if f:
            with open(f, "w") as fp:
                if f.endswith(".csv"):
                    fp.write(",".join(header) + "\n")
                    for row in ptable:
                        fp.write(",".join(str(x) for x in row) + "\n")
                else:
                    fp.write(tabulate(ptable, headers=header))
        else:
            # Silent mode - just return the value
            pass

        return forward_macs


# ==============================================================================
# Model Class
# ==============================================================================

class Model:
    """ONNX Model wrapper."""

    def __init__(self, m, mcfg=None):
        if mcfg is None:
            mcfg = {}
        self.modelname = ''
        self.cfg = ModelConfig(mcfg)
        self.valid = False

        if isinstance(m, pathlib.Path):
            self.modelname = m.stem
            m = onnx.load_model(m)
        elif isinstance(m, str):
            self.modelname = os.path.splitext(os.path.basename(m))[0]
            m = onnx.load_model(m)

        if isinstance(m, onnx.ModelProto):
            self.valid = True
            self.mproto = m
            self.graph = Graph(m.graph, self.cfg)


# ==============================================================================
# Main API Function
# ==============================================================================

def model_profile(
    m,
    dynamic_shapes=None,
    hidden_ops=None,
    mcfg=None,
    save_profile=None,
    save_model=None,
):
    """
    Profile an ONNX model and calculate MACs (Multiply-Accumulate operations).

    Args:
        m: ONNX model file path or onnx.ModelProto
        dynamic_shapes: Dict of dynamic shape overrides, e.g. {'input': (1, 3, 224, 224)}
        hidden_ops: List of op types to exclude from profiling (default: NoMacsOps)
        mcfg: Model config dict, e.g. {"verbose": False}
        save_profile: Path to save profile result (txt or csv)
        save_model: Path to save the model after shape inference (not implemented)

    Returns:
        float: Total forward MACs count
    """
    if mcfg is None:
        mcfg = {"verbose": False}
    if hidden_ops is None:
        hidden_ops = NoMacsOps
    if dynamic_shapes is None:
        dynamic_shapes = {}

    model = Model(m, mcfg)
    if not model.valid:
        warnings.warn(f"Invalid ONNX model")
        return 0

    g = model.graph
    tm = timer()

    g.graph_reorder_nodes()
    g.shape_infer(dynamic_shapes)
    g.log(f"Shape inference: {tm.stop():.3f}s")

    tm.start()
    g.profile()
    g.log(f"Profiling: {tm.stop():.3f}s")

    macs = g.print_node_map(save_profile, exclude_ops=hidden_ops)
    return macs


# ==============================================================================
# Public API
# ==============================================================================

__all__ = [
    'model_profile',
    'NoMacsOps',
    'VERSION',
    'Model',
    'Graph',
    'Tensor',
    'Node',
]