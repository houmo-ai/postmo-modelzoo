# Copyright (c) 2025 HOUMO AI
#
# File: image_processing_qwen3.5.py
# Description:
#   Processing Qwen3.5 image data.
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
"""
Image processing for Qwen3.5 VL models.

This is a standalone copy from qwen3_vl/image_processing_qwen2_vl.py so that
Qwen3.5 export does NOT depend on the qwen3_vl package.
"""

from typing import Optional, Union

import numpy as np
import torch
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
from transformers.image_transforms import convert_to_rgb, resize, to_channel_dimension_format
from transformers.models.qwen2_vl.image_processing_qwen2_vl import Qwen2VLImageProcessor, smart_resize
from transformers.utils import TensorType, logging

logger = logging.get_logger(__name__)


def _plain_size_dict(value):
    if value.__class__.__name__ == "SizeDict":
        return {
            field: getattr(value, field)
            for field in getattr(value, "__dataclass_fields__", {})
            if getattr(value, field) is not None
        }
    return value


class Qwen3_5ImageProcessor(Qwen2VLImageProcessor):
    """Extended image processor that also produces hm_pixel_values."""

    def __init__(self, *args, **kwargs):
        min_pixels = kwargs.get("min_pixels")
        max_pixels = kwargs.get("max_pixels")
        if "size" in kwargs:
            kwargs["size"] = _plain_size_dict(kwargs["size"])
            if isinstance(kwargs["size"], dict):
                min_pixels = min_pixels if min_pixels is not None else kwargs["size"].get("shortest_edge")
                max_pixels = max_pixels if max_pixels is not None else kwargs["size"].get("longest_edge")
        super().__init__(*args, **kwargs)
        self.min_pixels = min_pixels if min_pixels is not None else 65536
        self.max_pixels = max_pixels if max_pixels is not None else 16777216

    def _hm_preprocess(
        self,
        images,
        do_resize=None,
        resample=None,
        do_convert_rgb=None,
        min_pixels=None,
        max_pixels=None,
        data_format: Optional[ChannelDimension] = ChannelDimension.FIRST,
        input_data_format: Optional[Union[str, ChannelDimension]] = None,
    ):
        min_pixels = min_pixels if min_pixels is not None else self.min_pixels
        max_pixels = max_pixels if max_pixels is not None else self.max_pixels
        images = make_list_of_images(images)

        if do_convert_rgb:
            images = [convert_to_rgb(image) for image in images]
        images = [to_numpy_array(image) for image in images]
        if input_data_format is None:
            input_data_format = infer_channel_dimension_format(images[0])

        height, width = get_image_size(images[0], channel_dim=input_data_format)
        processed_images = []
        for image in images:
            if do_resize:
                resized_height, resized_width = smart_resize(
                    height,
                    width,
                    factor=self.patch_size * self.merge_size,
                    min_pixels=min_pixels,
                    max_pixels=max_pixels,
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
        videos=None,
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
            max_pixels = size["longest_edge"]
        elif min_pixels is not None and max_pixels is not None:
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
            image_inputs = super().preprocess(
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
                return_tensors=return_tensors,
                data_format=data_format,
                input_data_format=input_data_format,
            )

            hm_pixel_values = []
            for image in images:
                hm_patches = self._hm_preprocess(
                    image,
                    do_resize=do_resize,
                    resample=resample,
                    do_convert_rgb=do_convert_rgb,
                    min_pixels=min_pixels,
                    max_pixels=max_pixels,
                    data_format=data_format,
                    input_data_format=input_data_format,
                )
                hm_pixel_values.append(
                    torch.from_numpy(hm_patches).unsqueeze(2).repeat(1, 1, self.temporal_patch_size, 1, 1)
                )
            data.update(image_inputs)
            data["hm_pixel_values"] = hm_pixel_values

        if videos is not None:
            logger.warning("`Qwen3_5ImageProcessor` works only with image inputs and doesn't process videos anymore.")
            from transformers.video_utils import make_batched_videos

            videos = make_batched_videos(videos)
            pixel_values_videos, vision_grid_thws_videos = [], []
            for vid_images in videos:
                patches, video_grid_thw = self._preprocess(
                    vid_images,
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
                pixel_values_videos.extend(patches)
                vision_grid_thws_videos.append(video_grid_thw)
            data.update(
                {
                    "pixel_values_videos": np.array(pixel_values_videos),
                    "video_grid_thw": np.array(vision_grid_thws_videos),
                }
            )
        return data
