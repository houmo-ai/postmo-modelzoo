# Copyright (c) 2026 HOUMO AI
#
# File: tcim_tensor_utils.py
# Description:
#   TCIM Tensor ROI validation and copy utilities.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# SPDX-License-Identifier: Apache-2.0

"""Utilities for validating and copying regions of TCIM tensors.

The main entry point is :func:`copy_tensor_roi`. It copies an N-dimensional
rectangular region from a source ``tcim.runtime.Tensor`` to a destination
``tcim.runtime.Tensor``. Both Host and Device tensors are supported by the
underlying TCIM ``Tensor.copy_to`` implementation.

ROI coordinates use ``start + shape`` semantics. ``start`` contains the first
element selected on every axis, and ``shape`` contains the number of selected
elements. The end position is exclusive::

    tensor axis:  0  1  2  3  4  5  6  7
                         |--------|
    start = 2            ^        ^
    shape = 4            2        2 + 4 = 6 (exclusive end)
    selected indices:    2, 3, 4, 5

For an N-dimensional ROI, the same rule is applied independently to every
axis::

    0 <= start[axis] < tensor_shape[axis]
    0 < shape[axis]
    start[axis] + shape[axis] <= tensor_shape[axis]

The helper functions in this module are public so callers can validate inputs
before scheduling a copy. They do not modify tensor data.
"""

from collections.abc import Sequence
from numbers import Integral

import tcim_lite as tcim


def normalize_int_sequence(values: Sequence[int], *, name: str) -> list[int]:
    """Validate an integer sequence and return it as ``list[int]``.

    This function accepts sequence types such as ``list`` and ``tuple``. It
    intentionally does not silently cast floating-point values. For example,
    ``1.5`` is rejected instead of being truncated to ``1``. Boolean values are
    also rejected because ``bool`` is a subclass of ``int`` in Python but does
    not represent a meaningful tensor coordinate.

    Parameters
    ----------
    values : Sequence[int]
        Sequence to validate and normalize.
    name : str, keyword-only
        Parameter name included in exception messages.

    Returns
    -------
    list[int]
        A new list containing the normalized Python integers.

    Raises
    ------
    TypeError
        If ``values`` is not iterable, or an element is not an integer, or an
        element is a boolean.

    Examples
    --------
    >>> normalize_int_sequence((0, 2, 4), name="start")
    [0, 2, 4]
    >>> normalize_int_sequence([0, 1.5], name="start")
    Traceback (most recent call last):
        ...
    TypeError: start must contain only integers
    """
    try:
        normalized = list(values)
    except TypeError as exc:
        raise TypeError(f"{name} must be a sequence of integers") from exc

    if not all(
        isinstance(value, Integral) and not isinstance(value, bool)
        for value in normalized
    ):
        raise TypeError(f"{name} must contain only integers")

    return [int(value) for value in normalized]


def validate_tensor(
    tensor: tcim.runtime.Tensor,
    *,
    name: str = "tensor",
) -> tcim.runtime.Tensor:
    """Validate that an object is a public TCIM runtime Tensor.

    Parameters
    ----------
    tensor : tcim.runtime.Tensor
        Object expected to be a ``tcim_lite.runtime.Tensor``. It may store data
        on Host or Device memory.
    name : str, keyword-only, default="tensor"
        Parameter name included in the exception message.

    Returns
    -------
    tcim.runtime.Tensor
        The original tensor object. No copy or allocation is performed.

    Raises
    ------
    TypeError
        If ``tensor`` is not an instance of ``tcim.runtime.Tensor``. NumPy
        arrays and internal ``pytcim.Tensor`` objects are not accepted.
    """
    if not isinstance(tensor, tcim.runtime.Tensor):
        raise TypeError(
            f"{name} must be a tcim_lite.runtime.Tensor, "
            f"got {type(tensor).__name__}"
        )
    return tensor


def get_tensor_shape(
    tensor: tcim.runtime.Tensor,
    *,
    name: str = "tensor",
) -> list[int]:
    """Return validated tensor dimensions as ``list[int]``.

    The function validates the tensor type and its basic shape metadata. A
    valid tensor must have at least one dimension and every dimension must be
    greater than zero.

    Parameters
    ----------
    tensor : tcim.runtime.Tensor
        Host or Device tensor whose shape is requested.
    name : str, keyword-only, default="tensor"
        Tensor name included in exception messages.

    Returns
    -------
    list[int]
        A new list containing ``tensor.info.shape``.

    Raises
    ------
    TypeError
        If ``tensor`` is not a public TCIM runtime Tensor, or shape metadata
        does not contain integers.
    ValueError
        If the shape is empty or contains a non-positive dimension.

    Notes
    -----
    This function validates logical dimensions only. It does not require the
    tensor to be contiguous and does not alter its stride.
    """
    tensor = validate_tensor(tensor, name=name)

    shape = normalize_int_sequence(tensor.info.shape, name=f"{name}.info.shape")
    if not shape:
        raise ValueError(f"{name} must have at least one dimension")
    if any(size <= 0 for size in shape):
        raise ValueError(f"{name} shape must contain only positive values: {shape}")
    return shape


def validate_tensor_start(
    tensor: tcim.runtime.Tensor,
    start: Sequence[int],
    *,
    name: str = "start",
) -> list[int]:
    """Validate that an N-dimensional start coordinate is inside a tensor.

    ``start`` must contain exactly one coordinate for every tensor axis. Each
    coordinate points to an existing tensor element::

        valid:   0 <= start[axis] < tensor_shape[axis]
        invalid: start[axis] == tensor_shape[axis]

    One-dimensional illustration for a tensor with shape ``[8]``::

        indices:       0  1  2  3  4  5  6  7
        valid starts:  ^  ^  ^  ^  ^  ^  ^  ^
        invalid:                              ^ index 8

    Parameters
    ----------
    tensor : tcim.runtime.Tensor
        Tensor used to validate coordinate bounds.
    start : Sequence[int]
        Start coordinate. Its length must equal the tensor rank.
    name : str, keyword-only, default="start"
        Coordinate name included in exception messages.

    Returns
    -------
    list[int]
        Validated start coordinate as a new list.

    Raises
    ------
    TypeError
        If ``tensor`` is invalid or ``start`` is not an integer sequence.
    ValueError
        If the coordinate rank differs from the tensor rank, a coordinate is
        negative, or a coordinate is at or beyond its axis boundary.

    Notes
    -----
    This function validates only the first selected element. Use
    :func:`validate_tensor_roi` to also verify ``start + shape``.
    """
    tensor_shape = get_tensor_shape(tensor)
    normalized_start = normalize_int_sequence(start, name=name)

    if len(normalized_start) != len(tensor_shape):
        raise ValueError(
            f"{name} has {len(normalized_start)} dimensions, "
            f"but tensor has {len(tensor_shape)}"
        )

    for axis, (index, limit) in enumerate(zip(normalized_start, tensor_shape)):
        if index < 0 or index >= limit:
            raise ValueError(
                f"{name}[{axis}] is outside the tensor: "
                f"index={index}, valid range=[0, {limit})"
            )

    return normalized_start


def validate_tensor_roi(
    tensor: tcim.runtime.Tensor,
    start: Sequence[int],
    shape: Sequence[int],
    *,
    name: str = "ROI",
) -> tuple[list[int], list[int]]:
    """Validate a complete N-dimensional rectangular tensor region.

    A region of interest (ROI) is represented by ``start`` and ``shape``.
    ``start`` is inclusive and ``start + shape`` is exclusive. All three
    sequences, tensor shape, ROI start, and ROI shape, must have the same rank.

    Two-dimensional illustration::

        tensor shape = [5, 8]
        start        = [1, 2]
        shape        = [3, 4]

             axis 1 -> 0 1 2 3 4 5 6 7
        axis 0          +---------------+
          0             | . . . . . . . |
          1             | . . X X X X . |  <- start [1, 2]
          2             | . . X X X X . |
          3             | . . X X X X . |
          4             | . . . . . . . |
                        +---------------+

        selected axis 0 indices: [1, 4) -> 1, 2, 3
        selected axis 1 indices: [2, 6) -> 2, 3, 4, 5

    A valid ROI satisfies these rules on every axis::

        len(start) == len(shape) == len(tensor.info.shape)
        0 <= start[axis] < tensor_shape[axis]
        shape[axis] > 0
        start[axis] + shape[axis] <= tensor_shape[axis]

    Parameters
    ----------
    tensor : tcim.runtime.Tensor
        Tensor containing the ROI.
    start : Sequence[int]
        Inclusive start coordinate for each tensor axis.
    shape : Sequence[int]
        Number of elements selected on each tensor axis. This is not an end
        coordinate.
    name : str, keyword-only, default="ROI"
        Region name included in exception messages.

    Returns
    -------
    tuple[list[int], list[int]]
        ``(normalized_start, normalized_shape)`` as new integer lists suitable
        for passing to ``Tensor.select_roi``.

    Raises
    ------
    TypeError
        If the tensor is invalid, or ``start``/``shape`` is not an integer
        sequence.
    ValueError
        If ranks differ, an extent is not positive, the start is outside the
        tensor, or ``start + shape`` exceeds a tensor boundary.

    Notes
    -----
    Tensor stride and memory contiguity do not change coordinate validation.
    ``Tensor.select_roi`` preserves the original tensor layout.
    """
    tensor_shape = get_tensor_shape(tensor)
    normalized_start = validate_tensor_start(
        tensor,
        start,
        name=f"{name} start",
    )
    normalized_shape = normalize_int_sequence(shape, name=f"{name} shape")

    if len(normalized_shape) != len(tensor_shape):
        raise ValueError(
            f"{name} shape has {len(normalized_shape)} dimensions, "
            f"but tensor has {len(tensor_shape)}"
        )

    for axis, (index, size, limit) in enumerate(
        zip(normalized_start, normalized_shape, tensor_shape)
    ):
        if size <= 0:
            raise ValueError(
                f"{name} shape[{axis}] must be positive, got {size}"
            )
        if index + size > limit:
            raise ValueError(
                f"{name} exceeds the tensor on axis {axis}: "
                f"start={index}, size={size}, tensor size={limit}"
            )

    return normalized_start, normalized_shape


def copy_tensor_roi(
    src_tensor: tcim.runtime.Tensor,
    dst_tensor: tcim.runtime.Tensor,
    *,
    src_start: Sequence[int],
    dst_start: Sequence[int],
    shape: Sequence[int],
) -> None:
    """Copy an N-dimensional ROI to a destination tensor position.

    The function selects two regions with the same ``shape`` and invokes TCIM
    ``Tensor.copy_to``::

        source = src_tensor.select_roi(src_start, shape)
        target = dst_tensor.select_roi(dst_start, shape)
        source.copy_to(target)

    Conceptual one-dimensional copy::

        source tensor
        indices:  0  1  2  3  4  5  6  7
        values:   a  b [c  d  e] f  g  h
                         |-----|
                   src_start=2, shape=3
                              |
                              v
        destination tensor
        indices:  0  1  2  3  4  5  6  7
        before:   .  .  .  . [.] .  .  .
        after:    .  .  .  . [c  d  e] .
                               dst_start=4

    The same coordinate rule applies independently to every tensor axis. For
    example, a 4-D copy may be written as::

        copy_tensor_roi(
            src_tensor,
            dst_tensor,
            src_start=[0, 0, 100, 0],
            dst_start=[0, 1, 200, 0],
            shape=[1, 1, 32, 256],
        )

    Parameters
    ----------
    src_tensor : tcim.runtime.Tensor
        Source Host or Device tensor.
    dst_tensor : tcim.runtime.Tensor
        Destination Host or Device tensor. Its dtype must match the source
        tensor dtype. Its full tensor shape may differ from the source shape,
        provided the destination ROI is valid.
    src_start : Sequence[int], keyword-only
        Inclusive source ROI start coordinate. Its length must equal the source
        tensor rank.
    dst_start : Sequence[int], keyword-only
        Inclusive destination ROI start coordinate. Its length must equal the
        destination tensor rank.
    shape : Sequence[int], keyword-only
        Shared ROI extent. Every value must be positive. ``shape`` is the
        number of copied elements per axis, not an end coordinate.

    Returns
    -------
    None
        The destination ROI is modified in place.

    Raises
    ------
    TypeError
        If either tensor is not ``tcim.runtime.Tensor``, coordinates are not
        integer sequences, or source and destination dtypes differ.
    ValueError
        If coordinate ranks are invalid, a start coordinate is outside its
        tensor, an ROI extent is non-positive, or either ROI exceeds a tensor
        boundary.
    RuntimeError
        If TCIM fails to select, clone, allocate, or copy a tensor ROI.

    Same-Tensor Copy
    ----------------
    If ``src_tensor is dst_tensor``, source and destination can overlap. TCIM
    ``Tensor.copy_to`` does not document memmove-like overlap guarantees, so
    this function first clones the source ROI::

        source ROI ---- clone ----> temporary tensor
                                        |
                                        +---- copy_to ----> target ROI

    Cloning guarantees stable source data but requires temporary Host or Device
    memory and one additional copy. The clone is performed for every same-object
    copy, even when the selected regions do not overlap, because the public API
    does not expose enough storage information for reliable overlap detection.

    Different-Tensor Copy
    ---------------------
    If ``src_tensor is not dst_tensor``, the source ROI is copied directly and
    no temporary clone is created. Supported directions depend on TCIM
    ``Tensor.copy_to`` and include the tested combinations D2D, D2H, H2D, and
    H2H.

    Warning
    -------
    Two distinct Python ``Tensor`` objects can still share an underlying buffer,
    for example when both were created as ROI views of one parent tensor. Object
    identity cannot detect that aliasing. Callers must not pass overlapping
    views as distinct source and destination tensors.

    Notes
    -----
    - The operation is rectangular ROI copying only. Negative indices, strides,
      ellipsis, axis insertion, and NumPy-style stepped slicing are unsupported.
    - ``Tensor.select_roi`` returns a view sharing the parent tensor storage.
      This function completes the copy before its local views go out of scope.
    - Source and destination strides may differ. TCIM ``Tensor.copy_to`` handles
      data alignment, subject to the runtime's supported layouts.

    Examples
    --------
    Copy between different tensors without a temporary clone::

        copy_tensor_roi(
            src_tensor,
            dst_tensor,
            src_start=[0, 0, 8190, 0],
            dst_start=[0, 1, 10, 0],
            shape=[1, 1, 2, 256],
        )

    Shift data left inside one tensor. The source ROI is cloned automatically::

        copy_tensor_roi(
            tensor,
            tensor,
            src_start=[0, 0, 1, 0],
            dst_start=[0, 0, 0, 0],
            shape=[1, 2, 8191, 256],
        )
    """
    validate_tensor(src_tensor, name="src_tensor")
    validate_tensor(dst_tensor, name="dst_tensor")

    src_start_list, shape_list = validate_tensor_roi(
        src_tensor,
        src_start,
        shape,
        name="Source ROI",
    )
    dst_start_list, dst_shape_list = validate_tensor_roi(
        dst_tensor,
        dst_start,
        shape_list,
        name="Destination ROI",
    )

    if src_tensor.info.dtype != dst_tensor.info.dtype:
        raise TypeError(
            f"Tensor dtype mismatch: source={src_tensor.info.dtype}, "
            f"destination={dst_tensor.info.dtype}"
        )

    source = src_tensor.select_roi(src_start_list, shape_list)
    target = dst_tensor.select_roi(dst_start_list, dst_shape_list)
    if src_tensor is dst_tensor:
        source = source.clone()
    source.copy_to(target)
