# Copyright 2025 HOUMO AI
#
# File: format_converter.py
# Description:
#  Format converter.
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

from typing import Any

import numpy as np
import torch
import torch.nn.functional as torch_nn_functional


# Conversion matrices and biases for supported color layouts.
_COLOR_PROFILES = {
    ("BT601", "RGB"): {
        "matrix": torch.tensor(
            [
                [0.299, 0.587, 0.114],
                [-0.169, -0.331, 0.500],
                [0.500, -0.419, -0.081],
            ],
            dtype=torch.float32,
        ),
        "bias": torch.tensor([0.0, 128.0, 128.0], dtype=torch.float32),
    },
    ("BT601", "BGR"): {
        "matrix": torch.tensor(
            [
                [0.114, 0.587, 0.299],
                [0.500, -0.331, -0.169],
                [-0.081, -0.419, 0.500],
            ],
            dtype=torch.float32,
        ),
        "bias": torch.tensor([0.0, 128.0, 128.0], dtype=torch.float32),
    },
}

# Horizontal and vertical subsampling factors for each packed YUV format.
_PACKING_RULES = {
    "444": (1, 1),
    "422": (2, 1),
    "420": (2, 2),
}


def _normalize_format(fmt: str) -> str:
    # Normalize aliases such as YUV420 and 420SP to the internal format key.
    compact = fmt.upper().replace("YUV", "")
    if compact.endswith("SP"):
        compact = compact[:-2]
    if compact not in _PACKING_RULES:
        raise ValueError(f"Unsupported format: {fmt}")
    return compact


def _ensure_chw_tensor(image: torch.Tensor) -> tuple[int, int]:
    # Accept only 3-channel CHW tensors and return spatial dimensions.
    if image.ndim != 3 or image.shape[0] != 3:
        raise ValueError(
            f"Expected CHW image tensor with 3 channels, got {tuple(image.shape)}"
        )
    return int(image.shape[1]), int(image.shape[2])


def _convert_channels(
    image: torch.Tensor, profile_key: tuple[str, str]
) -> torch.Tensor:
    # Apply the selected linear color transform and clamp the result to image range.
    profile = _COLOR_PROFILES.get(profile_key)
    if profile is None:
        raise ValueError(f"Unsupported conversion profile: {profile_key}")

    matrix = profile["matrix"].to(image.device)
    bias = profile["bias"].to(image.device).view(3, 1)
    flattened = image.to(dtype=torch.float32).reshape(3, -1)
    converted = matrix @ flattened
    converted += bias
    return converted.clamp_(0, 255).round_().reshape_as(image)


def _downsample_component(
    component: torch.Tensor,
    x_stride: int,
    y_stride: int,
    use_interpolation: bool,
) -> torch.Tensor:
    # Downsample chroma planes according to the target YUV packing rule.
    if x_stride == 1 and y_stride == 1:
        return component

    height, width = component.shape
    if height % y_stride != 0 or width % x_stride != 0:
        raise ValueError(
            f"Input size {height}x{width} is incompatible with the target YUV layout"
        )

    if not use_interpolation:
        # Use direct decimation when interpolation is not requested.
        return component[::y_stride, ::x_stride]

    # Use bilinear interpolation for smoother chroma reduction.
    reduced = torch_nn_functional.interpolate(
        component.unsqueeze(0).unsqueeze(0),
        size=(height // y_stride, width // x_stride),
        mode="bilinear",
        align_corners=False,
    )
    return reduced.squeeze(0).squeeze(0).clamp_(0, 255).round_()


def _serialize_yuv_planes(
    yuv_image: torch.Tensor, fmt: str, use_interpolation: bool
) -> torch.Tensor:
    # Pack Y, U, and V planes into the layout expected by the runtime input.
    image_height, image_width = _ensure_chw_tensor(yuv_image)
    x_stride, y_stride = _PACKING_RULES[fmt]

    y_plane = yuv_image[0]
    u_plane = _downsample_component(yuv_image[1], x_stride, y_stride, use_interpolation)
    v_plane = _downsample_component(yuv_image[2], x_stride, y_stride, use_interpolation)

    if fmt == "444":
        # Keep full-resolution Y, U, and V data for 4:4:4 output.
        return torch.stack((y_plane, u_plane, v_plane), dim=-1)

    # Store the Y plane first, then append interleaved UV data.
    output = torch.zeros(
        (3, image_height, image_width),
        dtype=yuv_image.dtype,
        device=yuv_image.device,
    )
    y_values = y_plane.reshape(-1)
    uv_values = torch.stack((u_plane, v_plane), dim=-1).reshape(-1)
    flat_output = output.reshape(-1)
    flat_output[: y_values.numel()] = y_values
    flat_output[y_values.numel() : y_values.numel() + uv_values.numel()] = uv_values
    return output


class _ColorConverter:
    # Shared converter implementation for different input channel orders.
    def __init__(
        self,
        source_layout: str,
        version: str = "BT601",
        fmt: str = "422",
        interpolation: bool = True,
    ) -> None:
        self._profile_key = (version, source_layout)
        self._format = _normalize_format(fmt)
        self._interpolation = interpolation

    def __call__(self, image: torch.Tensor) -> Any:
        # Convert the input tensor and pack it into the requested YUV layout.
        _ensure_chw_tensor(image)
        converted = _convert_channels(image, self._profile_key)
        return _serialize_yuv_planes(converted, self._format, self._interpolation)


class RGB2YUV(_ColorConverter):
    # Convert RGB CHW tensors to packed YUV tensors.
    def __init__(
        self, version: str = "BT601", fmt: str = "422", interpolation: bool = True
    ) -> None:
        super().__init__("RGB", version=version, fmt=fmt, interpolation=interpolation)


class BGR2YUV(_ColorConverter):
    # Convert BGR CHW tensors to packed YUV tensors.
    def __init__(
        self, version: str = "BT601", fmt: str = "422", interpolation: bool = True
    ) -> None:
        super().__init__("BGR", version=version, fmt=fmt, interpolation=interpolation)


def _is_supported_numpy_image(value: Any) -> bool:
    # Detect 2D or 3D NumPy image arrays.
    return isinstance(value, np.ndarray) and value.ndim in {2, 3}


class ToTensorNotNormal:
    def __call__(self, value: Any) -> torch.Tensor:
        # Convert tensor, NumPy array, or PIL-like image input to CHW tensor.
        default_dtype = torch.get_default_dtype()

        if isinstance(value, torch.Tensor):
            return (
                value.to(dtype=default_dtype) if value.dtype == torch.uint8 else value
            )

        if _is_supported_numpy_image(value):
            # Convert NumPy images from HWC to CHW layout.
            if value.ndim == 2:
                value = value[:, :, None]
            tensor = torch.from_numpy(np.ascontiguousarray(value.transpose((2, 0, 1))))
            return (
                tensor.to(dtype=default_dtype)
                if tensor.dtype == torch.uint8
                else tensor
            )

        image_mode = getattr(value, "mode", None)
        image_bands = getattr(value, "getbands", None)
        image_size = getattr(value, "size", None)
        if image_mode is None or image_bands is None or image_size is None:
            raise TypeError(f"Unsupported input type: {type(value)}")

        # Convert PIL-like images to NumPy first, then move to CHW tensor layout.
        mode_to_dtype = {"I": np.int32, "I;16": np.int16, "F": np.float32}
        array = np.array(
            value, dtype=mode_to_dtype.get(image_mode, np.uint8), copy=True
        )
        if image_mode == "1":
            array = array * 255
        if array.ndim == 2:
            array = array[:, :, None]

        tensor = torch.from_numpy(np.ascontiguousarray(array.transpose((2, 0, 1))))
        return tensor.to(dtype=default_dtype) if tensor.dtype == torch.uint8 else tensor

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}()"
