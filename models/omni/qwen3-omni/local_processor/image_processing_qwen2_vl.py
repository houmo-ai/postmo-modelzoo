#!/usr/bin/python3
# coding=utf-8
# Copyright 2025 The Qwen Team, Alibaba Group and The HuggingFace Inc. team. All rights reserved.
# Copyright 2025 HOUMO AI. All rights reserved.
#
# Modifications:
# - Portions of this file have been modified by HOUMO AI.
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
#
# File: image_processing_qwen2_vl.py
# Description:
#   Qwen3-Omni image preprocessing adapted for the xh2 model zoo (xh2modelzoo).

from typing import Optional, Union
import torch
import numpy as np

from transformers.image_processing_utils import BatchFeature
from transformers.image_transforms import (
    convert_to_rgb,
    resize,
    to_channel_dimension_format,
)
from transformers.image_utils import (
    ChannelDimension,
    PILImageResampling,
    get_image_size,
    infer_channel_dimension_format,
    make_flat_list_of_images,
    make_list_of_images,
    to_numpy_array,
    valid_images,
    validate_preprocess_arguments,
)
from transformers.utils import TensorType, logging

from transformers.models.qwen2_vl.image_processing_qwen2_vl import Qwen2VLImageProcessor, smart_resize

logger = logging.get_logger(__name__)


class Qwen2_5_VLImageProcessor(Qwen2VLImageProcessor):
    # Local fork: pin the vision input to a fixed square resolution so the token
    # count is constant and matches the statically-exported HMONNX vision encoder
    # (448x448 -> 28x28 patches -> 28*28 // merge_size**2 = 196 image tokens).
    # Set FIXED_VISION_SIZE=None to fall back to the original dynamic smart_resize.
    FIXED_VISION_SIZE = 448

    def __init__(self, *args, **kwargs):
        min_pixels = kwargs.get("min_pixels")
        max_pixels = kwargs.get("max_pixels")
        fixed_size = kwargs.pop("fixed_vision_size", self.FIXED_VISION_SIZE)
        super().__init__(*args, **kwargs)
        if not hasattr(self, "min_pixels"):
            self.min_pixels = 65536 if min_pixels is None else min_pixels
        if not hasattr(self, "max_pixels"):
            self.max_pixels = 16777216 if max_pixels is None else max_pixels
        # Pin to a fixed square resolution. With min_pixels == max_pixels == size**2
        # and a pre-squared image, smart_resize is forced to return (size, size).
        self.fixed_vision_size = None
        self.set_fixed_vision_size(fixed_size)

    def set_fixed_vision_size(self, fixed_size):
        """Pin (or unpin) the square vision resolution at runtime. Pass an int to
        force every image to (fixed_size, fixed_size) -> a constant token count;
        pass None to restore the original dynamic smart_resize. Recomputes the
        derived min_pixels/max_pixels/size so a later resolution change (e.g. read
        back from the exported HMONNX vision graph) auto-aligns the processor."""
        self.fixed_vision_size = fixed_size
        if fixed_size is not None:
            self.min_pixels = fixed_size * fixed_size
            self.max_pixels = fixed_size * fixed_size
            self.size = {"shortest_edge": self.min_pixels, "longest_edge": self.max_pixels}

    def _square_images(self, images):
        """Force every image to a fixed square so the downstream smart_resize
        (factor=patch*merge) collapses to exactly (fixed_size, fixed_size).
        Returns channel-first (C, H, W) numpy arrays."""
        if self.fixed_vision_size is None:
            return images
        size = self.fixed_vision_size
        squared = []
        for image in images:
            arr = to_numpy_array(image)
            ch_fmt = infer_channel_dimension_format(arr)
            arr = resize(arr, size=(size, size), resample=self.resample, input_data_format=ch_fmt)
            arr = to_channel_dimension_format(arr, ChannelDimension.FIRST, input_channel_dim=ch_fmt)
            squared.append(arr)
        return squared

    @staticmethod
    def _to_torch_tensor(value):
        if isinstance(value, torch.Tensor):
            return value
        return torch.from_numpy(np.array(value))

    def _run_base_preprocess(
        self,
        images,
        do_resize,
        size,
        resample,
        do_rescale,
        rescale_factor,
        do_normalize,
        image_mean,
        image_std,
        patch_size,
        temporal_patch_size,
        merge_size,
        data_format,
        do_convert_rgb,
        input_data_format,
    ):
        # Pre-square the image so the parent's smart_resize collapses to the
        # fixed (fixed_vision_size, fixed_vision_size) grid.
        if self.fixed_vision_size is not None:
            images = self._square_images(make_list_of_images(images))
            input_data_format = ChannelDimension.FIRST
        processed = super().preprocess(
            images=images,
            do_resize=do_resize,
            size=size,
            resample=resample,
            do_rescale=do_rescale,
            rescale_factor=rescale_factor,
            do_normalize=do_normalize,
            image_mean=image_mean,
            image_std=image_std,
            patch_size=patch_size,
            temporal_patch_size=temporal_patch_size,
            merge_size=merge_size,
            do_convert_rgb=do_convert_rgb,
            return_tensors=None,
            data_format=data_format,
            input_data_format=input_data_format,
        )
        return processed["pixel_values"], processed["image_grid_thw"]

    def _hm_preprocess(
        self,
        images,
        do_resize = None,
        resample =  None,
        do_convert_rgb = None,
        data_format: Optional[ChannelDimension] = ChannelDimension.FIRST,
        input_data_format: Optional[Union[str, ChannelDimension]] = None,
    ):
        """
        Preprocess an image or batch of images. Copy of the `preprocess` method from `CLIPImageProcessor`.
        """
        images = make_list_of_images(images)

        if do_convert_rgb:
            images = [convert_to_rgb(image) for image in images]
        # All transformations expect numpy arrays.
        images = [to_numpy_array(image) for image in images]
        if input_data_format is None:
            # We assume that all images have the same channel dimension format.
            input_data_format = infer_channel_dimension_format(images[0])

        height, width = get_image_size(images[0], channel_dim=input_data_format)
        resized_height, resized_width = height, width
        processed_images = []
        for image in images:
            if do_resize:
                if self.fixed_vision_size is not None:
                    # Pinned square resolution -> constant token count matching
                    # the statically-exported HMONNX vision encoder.
                    resized_height = resized_width = self.fixed_vision_size
                else:
                    resized_height, resized_width = smart_resize(
                        height,
                        width,
                        factor=self.patch_size * self.merge_size,
                        min_pixels=self.min_pixels,
                        max_pixels=self.max_pixels,
                    )
                image = resize(
                    image,
                    size=(resized_height, resized_width),
                    resample=resample,
                    input_data_format=input_data_format,
                )
            image = to_channel_dimension_format(image, data_format, input_channel_dim=input_data_format)
            processed_images.append(image)

        patches = np.array(processed_images)
        if data_format == ChannelDimension.LAST:
            patches = patches.transpose(0, 3, 1, 2)
        return patches

    def preprocess(
        self,
        images,
        videos = None,
        do_resize: Optional[bool] = None,
        size: Optional[dict[str, int]] = None,
        min_pixels: Optional[int] = None,
        max_pixels: Optional[int] = None,
        resample: PILImageResampling = None,
        do_rescale: Optional[bool] = None,
        rescale_factor: Optional[float] = None,
        do_normalize: Optional[bool] = None,
        image_mean: Optional[Union[float, list[float]]] = None,
        image_std: Optional[Union[float, list[float]]] = None,
        patch_size: Optional[int] = None,
        temporal_patch_size: Optional[int] = None,
        merge_size: Optional[int] = None,
        do_convert_rgb: Optional[bool] = None,
        return_tensors: Optional[Union[str, TensorType]] = None,
        data_format: Optional[ChannelDimension] = ChannelDimension.FIRST,
        input_data_format: Optional[Union[str, ChannelDimension]] = None,
    ):
        min_pixels = min_pixels if min_pixels is not None else self.min_pixels
        max_pixels = max_pixels if max_pixels is not None else self.max_pixels

        if size is not None:
            if "shortest_edge" not in size or "longest_edge" not in size:
                raise ValueError("size must contain 'shortest_edge' and 'longest_edge' keys.")
            min_pixels = size["shortest_edge"]
        elif min_pixels is not None and max_pixels is not None:
            # backward compatibility: override size with min_pixels and max_pixels if they are provided
            size = {"shortest_edge": min_pixels, "longest_edge": max_pixels}
        else:
            size = {**self.size}

        do_resize = do_resize if do_resize is not None else self.do_resize

        resample = resample if resample is not None else self.resample
        do_rescale = do_rescale if do_rescale is not None else self.do_rescale
        rescale_factor = rescale_factor if rescale_factor is not None else self.rescale_factor
        do_normalize = do_normalize if do_normalize is not None else self.do_normalize
        image_mean = image_mean if image_mean is not None else self.image_mean
        image_std = image_std if image_std is not None else self.image_std
        patch_size = patch_size if patch_size is not None else self.patch_size
        temporal_patch_size = temporal_patch_size if temporal_patch_size is not None else self.temporal_patch_size
        merge_size = merge_size if merge_size is not None else self.merge_size
        do_convert_rgb = do_convert_rgb if do_convert_rgb is not None else self.do_convert_rgb

        if images is not None:
            images = self.fetch_images(images)
            images = make_flat_list_of_images(images)

        if images is not None and not valid_images(images):
            raise ValueError(
                "Invalid image type. Must be of type PIL.Image.Image, numpy.ndarray, "
                "torch.Tensor, tf.Tensor or jax.ndarray."
            )

        validate_preprocess_arguments(
            rescale_factor=rescale_factor,
            do_normalize=do_normalize,
            image_mean=image_mean,
            image_std=image_std,
            do_resize=do_resize,
            size=size,
            resample=resample,
        )

        data = {}
        if images is not None:
            pixel_values, vision_grid_thws, hm_pixel_values = [], [], []
            for image in images:
                patches, image_grid_thw = self._run_base_preprocess(
                    image,
                    do_resize=do_resize,
                    size=size,
                    resample=resample,
                    do_rescale=do_rescale,
                    rescale_factor=rescale_factor,
                    do_normalize=do_normalize,
                    image_mean=image_mean,
                    image_std=image_std,
                    patch_size=patch_size,
                    temporal_patch_size=temporal_patch_size,
                    merge_size=merge_size,
                    data_format=data_format,
                    do_convert_rgb=do_convert_rgb,
                    input_data_format=input_data_format,
                )
                hm_patches = self._hm_preprocess(
                    image,
                    do_resize=do_resize,
                    resample=resample,
                    do_convert_rgb=do_convert_rgb,
                    data_format=data_format,
                    input_data_format=input_data_format,
                )
                patches = self._to_torch_tensor(patches)
                image_grid_thw = self._to_torch_tensor(image_grid_thw)
                if image_grid_thw.ndim == 1:
                    image_grid_thw = image_grid_thw.unsqueeze(0)
                hm_pixel_values.append(torch.from_numpy(hm_patches).unsqueeze(2).repeat(1, 1, self.temporal_patch_size, 1, 1))
                pixel_values.append(patches)
                vision_grid_thws.append(image_grid_thw)
            pixel_values = torch.cat(pixel_values, dim=0)
            vision_grid_thws = torch.cat(vision_grid_thws, dim=0)
            hm_pixel_values = torch.cat(hm_pixel_values, dim=0)
            data.update({"pixel_values": pixel_values, "image_grid_thw": vision_grid_thws, "hm_pixel_values": hm_pixel_values})

        # kept for BC only and should be removed after v5.0
        if videos is not None:
            logger.warning(
                "`Qwen2VLImageProcessor` works only with image inputs and doesn't process videos anymore. "
                "This is a deprecated behavior and will be removed in v5.0. "
                "Your videos should be forwarded to `Qwen2VLVideoProcessor`. "
            )
            from transformers.video_utils import make_batched_videos
            videos = make_batched_videos(videos)
            pixel_values_videos, vision_grid_thws_videos = [], []
            for images in videos:
                patches, video_grid_thw = self._run_base_preprocess(
                    images,
                    do_resize=do_resize,
                    size=size,
                    resample=resample,
                    do_rescale=do_rescale,
                    rescale_factor=rescale_factor,
                    do_normalize=do_normalize,
                    image_mean=image_mean,
                    image_std=image_std,
                    patch_size=patch_size,
                    temporal_patch_size=temporal_patch_size,
                    merge_size=merge_size,
                    data_format=data_format,
                    do_convert_rgb=do_convert_rgb,
                    input_data_format=input_data_format,
                )
                patches = self._to_torch_tensor(patches)
                video_grid_thw = self._to_torch_tensor(video_grid_thw)
                if video_grid_thw.ndim == 1:
                    video_grid_thw = video_grid_thw.unsqueeze(0)
                pixel_values_videos.append(patches)
                vision_grid_thws_videos.append(video_grid_thw)
            data.update(
                {
                    "pixel_values_videos": torch.cat(pixel_values_videos, dim=0),
                    "video_grid_thw": torch.cat(vision_grid_thws_videos, dim=0),
                }
            )
        return data