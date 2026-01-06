# Copyright 2025 HOUMO AI
#
# File: transform.py
# Description:
#     This file contains the transformation functions for the model.
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
import torch
import numpy as np
from typing import Any
from torchvision import transforms

# YUV = RGB * M[0:3] + M[3]
M_RGB2YUV = {
    # 'BT601': [
    #     [0.299, -0.168735892,  0.5],
    #     [0.587, -0.331264108, -0.418687589],
    #     [0.114,  0.5,         -0.081312411],
    #     [0, 128, 128] # bias
    # ],
    # Below is a lite version of above
    "BT601": [
        [0.299, -0.169, 0.5],
        [0.587, -0.331, -0.419],
        [0.114, 0.5, -0.081],
        [0, 128, 128],  # bias
    ],
}

# RBG = YUV * M[0:3] + M[3]
M_YUV2RGB = {
    # From Nvidia PVA
    "BT601": [
        [1, 1, 1],
        [0, -0.344, 1.772],
        [1.402, -0.714, 0],
        [-179.456, 135.459, -226.816],
    ],
}


M_BGR2YUV = {
    # Below is a lite version of above
    "BT601": [
        [0.114, 0.5, -0.081],
        [0.587, -0.331, -0.419],
        [0.299, -0.169, 0.5],
        [0, 128, 128],  # bias
    ],
}

# RBG = YUV * M[0:3] + M[3]
M_YUV2BGR = {
    # From Nvidia PVA
    "BT601": [
        [1, 1, 1],
        [1.772, -0.344, 0],
        [0, -0.714, 1.402],
        [-226.816, 135.459, -179.456],
    ],
}


class YUVFormat:
    """A class to format RGB image to YUV format with different sampling methods.

    Args:
        fmt (str): YUV format string (e.g., '420', '422', 'YUV420SP', etc.)
        interpolation (bool): Whether to use interpolation when downsampling
    """

    def __init__(self, fmt="422", interpolation=False) -> None:
        self.fmt = fmt
        self._MAP = {
            "420": (2, 2),
            "422": (2, 1),
            "YUV420": (2, 2),
            "YUV422": (2, 1),
            "420SP": (2, 2),
            "422SP": (2, 1),
            "YUV420SP": (2, 2),
            "YUV422SP": (2, 1),
        }
        self.interpolation = interpolation

    def __call__(self, img: torch.Tensor):
        """Convert an image tensor to YUV format.

        Args:
            img (torch.Tensor): Input image tensor in CHW format

        Returns:
            torch.Tensor: YUV formatted image tensor
        """
        _, img_h, img_w = img.size()
        # breakpoint()
        y, u, v = torch.split(img, 1, dim=0)
        if self.fmt in ["444", "444SP", "YUV444", "YUV444SP"]:
            uv = torch.stack([u, v], dim=-1)
            y = y.view(-1)
            uv = uv.view(-1)
            yuv = torch.cat((y, uv), 0)
            return yuv.view((img_h, img_w, 3))
        div_w, div_h = self._MAP[self.fmt]
        if self.interpolation:
            uv_resize = transforms.Resize((img_h // div_h, img_w // div_w))
            u = uv_resize(u)
            v = uv_resize(v)
            # Convert u and v to uint8 with clipping and rounding:
            u = u.clip(0, 255).round()
            v = v.clip(0, 255).round()
        else:
            u = u[:, 0::div_w]
            u = u[0::div_h, :]
            v = v[:, 0::div_w]
            v = v[0::div_h, :]
        uv = torch.stack([u, v], dim=-1)
        y = y.view(-1)
        uv = uv.view(-1)
        yuv = torch.cat((y, uv), 0)
        result = torch.zeros(img_h * img_w * 3)
        result[: yuv.shape[0]] = yuv
        return result.view((img_h, img_w, 3))


class RGB2YUV:
    """A transform to convert RGB image to YUV format.

    Args:
        version (str): Color space conversion standard (e.g., 'BT601')
        fmt (str): YUV format string (e.g., '422', '420')
        interpolation (bool): Whether to use interpolation when downsampling
    """

    def __init__(self, version="BT601", fmt="422", interpolation=True) -> None:
        """
        layout hwc or chw
        """
        Mb = M_RGB2YUV[version]

        self.M = torch.Tensor(Mb[0:3]).T
        self.b = torch.Tensor(Mb[3])
        self.b = self.b.view(3, 1, 1)
        self.formatter = YUVFormat(fmt, interpolation)

    def __call__(self, img: torch.Tensor) -> Any:
        """Convert RGB image to YUV format.

        Args:
            img (torch.Tensor): Input RGB image tensor in CHW format

        Returns:
            torch.Tensor: YUV formatted image tensor
        """
        # self.M.to(img.device)
        # self.b.to(img.device)
        result = torch.einsum("ij,jhw->ihw", [self.M, img])
        result = result + self.b
        result.clip_(0, 255).round_()
        # Change YUV store format
        result = self.formatter(result)
        return result


class BGR2YUV:
    """A transform to convert BGR image to YUV format.

    Args:
        version (str): Color space conversion standard (e.g., 'BT601')
        fmt (str): YUV format string (e.g., '422', '420')
        interpolation (bool): Whether to use interpolation when downsampling
    """

    def __init__(self, version="BT601", fmt="422", interpolation=True) -> None:
        """
        layout hwc or chw
        """
        Mb = M_BGR2YUV[version]
        self.M = torch.Tensor(Mb[0:3]).T
        self.b = torch.Tensor(Mb[3])
        self.b = self.b.view(3, 1, 1)
        self.formatter = YUVFormat(fmt, interpolation)

    def __call__(self, img: torch.Tensor) -> Any:
        """Convert BGR image to YUV format.

        Args:
            img (torch.Tensor): Input BGR image tensor in CHW format

        Returns:
            torch.Tensor: YUV formatted image tensor
        """
        # self.M.to(img.device)
        # self.b.to(img.device)
        result = torch.einsum("ij,jhw->ihw", [self.M, img])
        result = result + self.b
        result.clip_(0, 255).round_()
        # Change YUV store format
        result = self.formatter(result)
        return result


def _is_numpy(img: Any) -> bool:
    """Check if input is a numpy array.

    Args:
        img: Input to check

    Returns:
        bool: True if input is numpy array, False otherwise
    """
    return isinstance(img, np.ndarray)


def _is_numpy_image(img: Any) -> bool:
    """Check if input is a numpy image (2D or 3D array).

    Args:
        img: Input to check

    Returns:
        bool: True if input is 2D or 3D numpy array, False otherwise
    """
    return img.ndim in {2, 3}


class ToTensorNotNormal:
    """Convert a PIL image or numpy array to tensor without normalization.
    Similar to torchvision's ToTensor but without normalization to [0,1] range.
    """

    def __call__(self, pic):
        """Convert PIL image or numpy array to tensor.

        Args:
            pic: Input PIL image or numpy array

        Returns:
            torch.Tensor: Converted tensor
        """
        # TODO: The torchvision.transforms.functional_pil module is removed in 0.17**
        # if not(F_pil._is_pil_image(pic) or _is_numpy(pic)):
        #     raise TypeError('pic should be PIL Image or ndarray. Got {}'.format(type(pic)))

        if _is_numpy(pic) and not _is_numpy_image(pic):
            raise ValueError(
                "pic should be 2/3 dimensional. Got {} dimensions.".format(pic.ndim)
            )

        default_float_dtype = torch.get_default_dtype()

        if isinstance(pic, np.ndarray):
            # handle numpy array
            if pic.ndim == 2:
                pic = pic[:, :, None]

            img = torch.from_numpy(pic.transpose((2, 0, 1))).contiguous()
            # backward compatibility
            if isinstance(img, torch.ByteTensor):
                return img.to(dtype=default_float_dtype)
            else:
                return img

        try:
            import accimage
        except ImportError:
            accimage = None
        if accimage is not None and isinstance(pic, accimage.Image):
            nppic = np.zeros([pic.channels, pic.height, pic.width], dtype=np.float32)
            pic.copyto(nppic)
            return torch.from_numpy(nppic).to(dtype=default_float_dtype)

        # handle PIL Image
        mode_to_nptype = {"I": np.int32, "I;16": np.int16, "F": np.float32}
        img = torch.from_numpy(
            np.array(pic, mode_to_nptype.get(pic.mode, np.uint8), copy=True)
        )

        if pic.mode == "1":
            img = 255 * img
        img = img.view(pic.size[1], pic.size[0], len(pic.getbands()))
        # put it from HWC to CHW format
        img = img.permute((2, 0, 1)).contiguous()
        if isinstance(img, torch.ByteTensor):
            return img.to(dtype=default_float_dtype)
        else:
            return img

    def __repr__(self):
        return self.__class__.__name__ + "()"
