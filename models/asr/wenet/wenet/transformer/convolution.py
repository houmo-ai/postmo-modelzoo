# Copyright (c) 2020 Mobvoi Inc. (authors: Binbin Zhang, Di Wu)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# Modified from ESPnet(https://github.com/espnet/espnet)
"""ConvolutionModule definition."""

from typing import Tuple

import torch
from torch import nn

from wenet.utils.class_utils import WENET_NORM_CLASSES


class ConvolutionModule(nn.Module):
    """ConvolutionModule in Conformer model."""

    def __init__(
        self,
        channels: int,
        kernel_size: int = 15,
        activation: nn.Module = nn.ReLU(),
        norm: str = "batch_norm",
        causal: bool = False,
        bias: bool = True,
        norm_eps: float = 1e-5,
    ):
        """Construct an ConvolutionModule object.
        Args:
            channels (int): The number of channels of conv layers.
            kernel_size (int): Kernel size of conv layers.
            causal (int): Whether use causal convolution or not
        """
        super().__init__()

        self.pointwise_conv1 = nn.Conv1d(
            channels,
            2 * channels,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=bias,
        )
        # self.lorder is used to distinguish if it's a causal convolution,
        # if self.lorder > 0: it's a causal convolution, the input will be
        #    padded with self.lorder frames on the left in forward.
        # else: it's a symmetrical convolution
        if causal:
            padding = 0
            self.lorder = kernel_size - 1
        else:
            # kernel_size should be an odd number for none causal convolution
            assert (kernel_size - 1) % 2 == 0
            padding = (kernel_size - 1) // 2
            self.lorder = 0
        self.depthwise_conv = nn.Conv1d(
            channels,
            channels,
            kernel_size,
            stride=1,
            padding=padding,
            groups=channels,
            bias=bias,
        )

        assert norm in ['batch_norm', 'layer_norm', 'rms_norm']
        if norm == "batch_norm":
            self.use_layer_norm = False
            self.norm = WENET_NORM_CLASSES['batch_norm'](channels,
                                                         eps=norm_eps)
        else:
            self.use_layer_norm = True
            self.norm = WENET_NORM_CLASSES[norm](channels, eps=norm_eps)

        self.pointwise_conv2 = nn.Conv1d(
            channels,
            channels,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=bias,
        )
        self.activation = activation

    def forward(
        self,
        x: torch.Tensor,
        mask_pad: torch.Tensor = torch.ones((0, 0, 0), dtype=torch.bool),
        cache: torch.Tensor = torch.zeros((0, 0, 0)),
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute convolution module.
        Args:
            x (torch.Tensor): Input tensor (#batch, time, channels).
            mask_pad (torch.Tensor): used for batch padding (#batch, 1, time),
                (0, 0, 0) means fake mask.
            cache (torch.Tensor): left context cache, it is only
                used in causal convolution (#batch, channels, cache_t),
                (0, 0, 0) meas fake cache.
        Returns:
            torch.Tensor: Output tensor (#batch, time, channels).
        """
        # exchange the temporal dimension and the feature dimension
        x = x.transpose(1, 2)  # (#batch, channels, time)

        # mask batch padding
        if mask_pad.size(2) > 0:  # time > 0
            x.masked_fill_(~mask_pad, 0.0)

        if self.lorder > 0:
            if cache.size(2) == 0:  # cache_t == 0
                x = nn.functional.pad(x, (self.lorder, 0), 'constant', 0.0)
            else:
                assert cache.size(0) == x.size(0)  # equal batch
                assert cache.size(1) == x.size(1)  # equal channel
                x = torch.cat((cache, x), dim=2)
            assert (x.size(2) > self.lorder)
            new_cache = x[:, :, -self.lorder:]
        else:
            # It's better we just return None if no cache is required,
            # However, for JIT export, here we just fake one tensor instead of
            # None.
            new_cache = torch.zeros((0, 0, 0), dtype=x.dtype, device=x.device)

        # GLU mechanism
        x = self.pointwise_conv1(x)  # (batch, 2*channel, dim)
        x = nn.functional.glu(x, dim=1)  # (batch, channel, dim)

        # 1D Depthwise Conv
        x = self.depthwise_conv(x)
        if self.use_layer_norm:
            x = x.transpose(1, 2)
        x = self.activation(self.norm(x))
        if self.use_layer_norm:
            x = x.transpose(1, 2)
        x = self.pointwise_conv2(x)
        # mask batch padding
        if mask_pad.size(2) > 0:  # time > 0
            x.masked_fill_(~mask_pad, 0.0)
        return x.transpose(1, 2), new_cache
    
    def convert_conv1d_to_conv2d(self, conv1d: nn.Conv1d, aim_size=1) -> nn.Conv2d:
        """Convert Conv1d to Conv2d.
        Args:
            conv1d (nn.Conv1d): Conv1d module.
        Returns:
            nn.Conv2d: Converted Conv2d module.
        """
        conv2d = nn.Conv2d(
            conv1d.in_channels,
            conv1d.out_channels,
            kernel_size=(aim_size, conv1d.kernel_size[0]),
            stride=(1, conv1d.stride[0]),
            padding=(0, conv1d.padding[0]),
            bias=conv1d.bias is not None,
            groups=conv1d.groups,
        ).to(conv1d.weight.device)
        if aim_size == 1:
            conv2d.weight.data = conv1d.weight.data.unsqueeze(2)
        else:
            conv2d.weight.data = torch.zeros_like(conv2d.weight.data)
            conv2d.weight.data[:, :, aim_size // 2: aim_size // 2 + 1, :] = conv1d.weight.data.unsqueeze(2)
        if conv1d.bias is not None:
            conv2d.bias.data = conv1d.bias.data
        return conv2d
    
    def split_kernel15_to_kernel5(self, conv2d: nn.Conv2d) -> nn.Conv2d:
        """Split kernel size 15 to kernel size 5.
        Args:
            conv2d (nn.Conv2d): Conv2d module with kernel size 15.
        Returns:
            nn.Conv2d: Conv2d module with kernel size 5.
        """
        assert conv2d.kernel_size[1] == 15, "Only support kernel size 15"
        assert conv2d.kernel_size[0] == 5, "Only support kernel size 5"
        conv2d_new1 = nn.Conv2d(
            conv2d.in_channels,
            conv2d.out_channels,
            kernel_size=(5, 5),
            stride=conv2d.stride,
            padding=conv2d.padding,
            bias=conv2d.bias is not None,
            groups=conv2d.groups,
        ).to(conv2d.weight.device)
        conv2d_new1.weight.data = conv2d.weight.data[:, :, :, 0:5]
        if conv2d.bias is not None:
            conv2d_new1.bias.data = conv2d.bias.data / 3
            
        conv2d_new2 = nn.Conv2d(
            conv2d.in_channels,
            conv2d.out_channels,
            kernel_size=(5, 5),
            stride=conv2d.stride,
            padding=conv2d.padding,
            bias=conv2d.bias is not None,
            groups=conv2d.groups,
        ).to(conv2d.weight.device)
        conv2d_new2.weight.data = conv2d.weight.data[:, :, :, 5:10]
        if conv2d.bias is not None:
            conv2d_new2.bias.data = conv2d.bias.data / 3

        conv2d_new3 = nn.Conv2d(
            conv2d.in_channels,
            conv2d.out_channels,
            kernel_size=(5, 5),
            stride=conv2d.stride,
            padding=conv2d.padding,
            bias=conv2d.bias is not None,
            groups=conv2d.groups,
        ).to(conv2d.weight.device)
        conv2d_new3.weight.data = conv2d.weight.data[:, :, :, 10:]
        if conv2d.bias is not None:
            conv2d_new3.bias.data = conv2d.bias.data / 3
        return conv2d_new1, conv2d_new2, conv2d_new3
    
    def forward_export_onnx(
        self,
        x: torch.Tensor,
        mask_pad: torch.Tensor = torch.ones((0, 0, 0), dtype=torch.bool),
        cache: torch.Tensor = torch.zeros((0, 0, 0)),
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute convolution module.
        Args:
            x (torch.Tensor): Input tensor (#batch, time, channels).
            mask_pad (torch.Tensor): used for batch padding (#batch, 1, time),
                (0, 0, 0) means fake mask.
            cache (torch.Tensor): left context cache, it is only
                used in causal convolution (#batch, channels, cache_t),
                (0, 0, 0) meas fake cache.
        Returns:
            torch.Tensor: Output tensor (#batch, time, channels).
        """
        if isinstance(self.pointwise_conv1, nn.Conv1d):
            self.pointwise_conv1 = self.convert_conv1d_to_conv2d(self.pointwise_conv1)
        if isinstance(self.depthwise_conv, nn.Conv1d):
            self.depthwise_conv = self.convert_conv1d_to_conv2d(self.depthwise_conv, 5)
            self.depthwise_conv1, self.depthwise_conv2, self.depthwise_conv3 = self.split_kernel15_to_kernel5(self.depthwise_conv)
        if isinstance(self.pointwise_conv2, nn.Conv1d):
            self.pointwise_conv2 = self.convert_conv1d_to_conv2d(self.pointwise_conv2)
        # exchange the temporal dimension and the feature dimension
        x = x.transpose(1, 2)  # (#batch, channels, time)
 
        x = torch.mul(x, mask_pad)
        x = x.unsqueeze(-2)
  
        if self.lorder > 0:
            if cache.size(2) == 0:  # cache_t == 0
                x = nn.functional.pad(x, (self.lorder, 0, 2, 2), 'constant', 0.0)
            else:
                assert cache.size(0) == x.size(0)  # equal batch
                assert cache.size(1) == x.size(1)  # equal channel
                x = torch.cat((cache, x), dim=2)
            assert (x.size(3) > self.lorder)
            new_cache = x[:, :, -self.lorder:]
        else:
            # It's better we just return None if no cache is required,
            # However, for JIT export, here we just fake one tensor instead of
            # None.
            new_cache = torch.zeros((0, 0, 0), dtype=x.dtype, device=x.device)

        # GLU mechanism
        x = self.pointwise_conv1(x)  # (batch, 2*channel, dim)
        x = nn.functional.glu(x, dim=1)  # (batch, channel, dim)
        # x = nn.functional.pad(x, (0, 0, 2, 2), 'constant', 0.0)   
        # 1D Depthwise Conv 
        x1 = self.depthwise_conv1(x[:, :, :, :-10])
        x2 = self.depthwise_conv2(x[:, :, :,5:-5])
        x3 = self.depthwise_conv3(x[:, :, :, 10:])
        x = x1 + x2 + x3
        # x = self.depthwise_conv(x)
        x = x.squeeze(-2)
        if self.use_layer_norm:
            x = x.transpose(1, 2)
        x = self.activation(self.norm(x))
        if self.use_layer_norm:
            x = x.transpose(1, 2)
        x = x.unsqueeze(-2)
        x = self.pointwise_conv2(x)
        x = x.squeeze(-2)
        # mask batch padding
        x = torch.mul(x, mask_pad)
        return x.transpose(1, 2), new_cache
    
    def forward_chunk_export_onnx(
        self,
        x: torch.Tensor,
        mask_pad: torch.Tensor = torch.ones((0, 0, 0), dtype=torch.bool),
        cache: torch.Tensor = torch.zeros((0, 0, 0)),
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute convolution module.
        Args:
            x (torch.Tensor): Input tensor (#batch, time, channels).
            mask_pad (torch.Tensor): used for batch padding (#batch, 1, time),
                (0, 0, 0) means fake mask.
            cache (torch.Tensor): left context cache, it is only
                used in causal convolution (#batch, channels, cache_t),
                (0, 0, 0) meas fake cache.
        Returns:
            torch.Tensor: Output tensor (#batch, time, channels).
        """
        if isinstance(self.pointwise_conv1, nn.Conv1d):
            self.pointwise_conv1 = self.convert_conv1d_to_conv2d(self.pointwise_conv1)
        if isinstance(self.depthwise_conv, nn.Conv1d):
            self.depthwise_conv = self.convert_conv1d_to_conv2d(self.depthwise_conv, 5)
            self.depthwise_conv1, self.depthwise_conv2, self.depthwise_conv3 = self.split_kernel15_to_kernel5(self.depthwise_conv)
        if isinstance(self.pointwise_conv2, nn.Conv1d):
            self.pointwise_conv2 = self.convert_conv1d_to_conv2d(self.pointwise_conv2)
        # exchange the temporal dimension and the feature dimension
        x = x.transpose(1, 2)  # (#batch, channels, time)
 
        x = torch.mul(x, mask_pad)
        
        assert cache.size(2) != 0, "cache_t should not be 0"
        x = torch.cat((cache, x), dim=2)
        new_cache = x[:, :, -self.lorder:]
        x = x.unsqueeze(-2)
        x = nn.functional.pad(x, (0, 0, 2, 2), 'constant', 0.0)
        # GLU mechanism
        x = self.pointwise_conv1(x)  # (batch, 2*channel, dim)
        x = nn.functional.glu(x, dim=1)  # (batch, channel, dim)
        # x = nn.functional.pad(x, (0, 0, 2, 2), 'constant', 0.0)   
        # 1D Depthwise Conv 
        x1 = self.depthwise_conv1(x[:, :, :, :-10])
        x2 = self.depthwise_conv2(x[:, :, :,5:-5])
        x3 = self.depthwise_conv3(x[:, :, :, 10:])
        x = x1 + x2 + x3
        # x = self.depthwise_conv(x)
        x = x.squeeze(-2)
        if self.use_layer_norm:
            x = x.transpose(1, 2)
        x = self.activation(self.norm(x))
        if self.use_layer_norm:
            x = x.transpose(1, 2)
        x = x.unsqueeze(-2)
        x = self.pointwise_conv2(x)
        x = x.squeeze(-2)
        # mask batch padding
        x = torch.mul(x, mask_pad)
        return x.transpose(1, 2), new_cache
