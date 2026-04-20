# Copyright 2026 HOUMO AI. All rights reserved.
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
# ----------------------------------------------------------------              
# This file contains code modified from the Qwen3-VL project:
# Source: https://github.com/QwenLM/Qwen3-VL/blob/main/qwen-vl-utils/src/qwen_vl_utils/vision_process.py
# Original Author: Qwen Team, Alibaba Group.
# ----------------------------------------------------------------

"""
Vision processing utilities for CoPaw-Flash models.
Modified to support specific export logic for xh2 hardware.
"""

from __future__ import annotations
import base64
import copy
import logging
import math
import os
import sys
import warnings
from functools import lru_cache
from io import BytesIO
from typing import Optional
import requests
import torch
import torchvision
from packaging import version
from PIL import Image
from torchvision import io, transforms
from torchvision.transforms import InterpolationMode
logger = logging.getLogger(__name__)
IMAGE_FACTOR = 28
MIN_PIXELS = 4 * 28 * 28
MAX_PIXELS = 16384 * 28 * 28
MAX_RATIO = 200
VIDEO_MIN_PIXELS = 128 * 28 * 28
VIDEO_MAX_PIXELS = 768 * 28 * 28
FRAME_FACTOR = 2
FPS = 2.0
FPS_MIN_FRAMES = 4
FPS_MAX_FRAMES = 768
VIDEO_TOTAL_PIXELS = int(float(os.environ.get("VIDEO_MAX_PIXELS", 128000 * 28 * 28 * 0.9)))
def round_by_factor(number: int, factor: int) -> int:
    return round(number / factor) * factor
def ceil_by_factor(number: int, factor: int) -> int:
    return math.ceil(number / factor) * factor
def floor_by_factor(number: int, factor: int) -> int:
    return math.floor(number / factor) * factor
def smart_resize(
    height: int,
    width: int,
    factor: int = IMAGE_FACTOR,
    min_pixels: int = MIN_PIXELS,
    max_pixels: int = MAX_PIXELS,
) -> tuple[int, int]:
    if max(height, width) / min(height, width) > MAX_RATIO:
        raise ValueError(
            f"absolute aspect ratio must be smaller than {MAX_RATIO}, got {max(height, width) / min(height, width)}"
        )
    resized_height = max(factor, round_by_factor(height, factor))
    resized_width = max(factor, round_by_factor(width, factor))
    if resized_height * resized_width > max_pixels:
        beta = math.sqrt((height * width) / max_pixels)
        resized_height = floor_by_factor(height / beta, factor)
        resized_width = floor_by_factor(width / beta, factor)
    elif resized_height * resized_width < min_pixels:
        beta = math.sqrt(min_pixels / (height * width))
        resized_height = ceil_by_factor(height * beta, factor)
        resized_width = ceil_by_factor(width * beta, factor)
    return resized_height, resized_width
def to_rgb(pil_image: Image.Image) -> Image.Image:
    if pil_image.mode == "RGBA":
        white_background = Image.new("RGB", pil_image.size, (255, 255, 255))
        white_background.paste(pil_image, mask=pil_image.split()[3])
        return white_background
    return pil_image.convert("RGB")
def fetch_image(ele: dict[str, str | Image.Image], size_factor: int = IMAGE_FACTOR) -> Image.Image:
    image = ele.get("image", ele.get("image_url"))
    image_obj = None
    if isinstance(image, Image.Image):
        image_obj = image
    elif isinstance(image, str) and (image.startswith("http://") or image.startswith("https://")):
        with requests.get(image, stream=True) as response:
            response.raise_for_status()
            with BytesIO(response.content) as bio:
                image_obj = copy.deepcopy(Image.open(bio))
    elif isinstance(image, str) and image.startswith("file://"):
        image_obj = Image.open(image[7:])
    elif isinstance(image, str) and image.startswith("data:image") and "base64," in image:
        _, base64_data = image.split("base64,", 1)
        with BytesIO(base64.b64decode(base64_data)) as bio:
            image_obj = copy.deepcopy(Image.open(bio))
    elif isinstance(image, str):
        image_obj = Image.open(image)
    if image_obj is None:
        raise ValueError(f"Unrecognized image input: {image}")
    image_obj = to_rgb(image_obj)
    if "resized_height" in ele and "resized_width" in ele:
        resized_height, resized_width = smart_resize(
            ele["resized_height"],
            ele["resized_width"],
            factor=size_factor,
        )
    else:
        width, height = image_obj.size
        resized_height, resized_width = smart_resize(
            height,
            width,
            factor=size_factor,
            min_pixels=ele.get("min_pixels", MIN_PIXELS),
            max_pixels=ele.get("max_pixels", MAX_PIXELS),
        )
    return image_obj.resize((resized_width, resized_height))
def smart_nframes(ele: dict, total_frames: int, video_fps: int | float) -> int:
    assert not ("fps" in ele and "nframes" in ele), "Only accept either `fps` or `nframes`"
    if "nframes" in ele:
        nframes = round_by_factor(ele["nframes"], FRAME_FACTOR)
    else:
        fps = ele.get("fps", FPS)
        min_frames = ceil_by_factor(ele.get("min_frames", FPS_MIN_FRAMES), FRAME_FACTOR)
        max_frames = floor_by_factor(ele.get("max_frames", min(FPS_MAX_FRAMES, total_frames)), FRAME_FACTOR)
        nframes = total_frames / video_fps * fps
        nframes = min(min(max(nframes, min_frames), max_frames), total_frames)
        nframes = floor_by_factor(nframes, FRAME_FACTOR)
    if not (FRAME_FACTOR <= nframes <= total_frames):
        raise ValueError(f"nframes should in interval [{FRAME_FACTOR}, {total_frames}], but got {nframes}.")
    return nframes
def _read_video_torchvision(ele: dict) -> tuple[torch.Tensor, float]:
    video_path = ele["video"]
    if version.parse(torchvision.__version__) < version.parse("0.19.0"):
        if isinstance(video_path, str) and ("http://" in video_path or "https://" in video_path):
            warnings.warn("torchvision < 0.19.0 does not support http/https video path.")
        if isinstance(video_path, str) and video_path.startswith("file://"):
            video_path = video_path[7:]
    video, _, info = io.read_video(
        video_path,
        start_pts=ele.get("video_start", 0.0),
        end_pts=ele.get("video_end", None),
        pts_unit="sec",
        output_format="TCHW",
    )
    total_frames, video_fps = video.size(0), info["video_fps"]
    nframes = smart_nframes(ele, total_frames=total_frames, video_fps=video_fps)
    indices = torch.linspace(0, total_frames - 1, nframes).round().long()
    sample_fps = nframes / max(total_frames, 1e-6) * video_fps
    return video[indices], sample_fps
def is_decord_available() -> bool:
    import importlib.util
    return importlib.util.find_spec("decord") is not None
def calculate_video_frame_range(
    ele: dict,
    total_frames: int,
    video_fps: float,
) -> tuple[int, int, int]:
    if video_fps <= 0:
        raise ValueError("video_fps must be a positive number")
    if total_frames <= 0:
        raise ValueError("total_frames must be a positive integer")
    video_start = ele.get("video_start", None)
    video_end = ele.get("video_end", None)
    if video_start is None and video_end is None:
        return 0, total_frames - 1, total_frames
    max_duration = total_frames / video_fps
    start_frame = math.ceil(max(0.0, min(video_start, max_duration)) * video_fps) if video_start is not None else 0
    end_frame = (
        min(math.floor(max(0.0, min(video_end, max_duration)) * video_fps), total_frames - 1)
        if video_end is not None
        else total_frames - 1
    )
    if start_frame >= end_frame:
        raise ValueError(f"Invalid time range: {start_frame=} {end_frame=}")
    return start_frame, end_frame, end_frame - start_frame + 1
def _read_video_decord(ele: dict) -> tuple[torch.Tensor, float]:
    import decord
    video_path = ele["video"]
    reader = decord.VideoReader(video_path)
    total_frames, video_fps = len(reader), reader.get_avg_fps()
    start_frame, end_frame, total_frames = calculate_video_frame_range(ele, total_frames, video_fps)
    nframes = smart_nframes(ele, total_frames=total_frames, video_fps=video_fps)
    indices = torch.linspace(start_frame, end_frame, nframes).round().long().tolist()
    video = torch.tensor(reader.get_batch(indices).asnumpy()).permute(0, 3, 1, 2)
    sample_fps = nframes / max(total_frames, 1e-6) * video_fps
    return video, sample_fps
VIDEO_READER_BACKENDS = {
    "decord": _read_video_decord,
    "torchvision": _read_video_torchvision,
}
FORCE_QWENVL_VIDEO_READER = os.getenv("FORCE_QWENVL_VIDEO_READER", None)
@lru_cache(maxsize=1)
def get_video_reader_backend() -> str:
    if FORCE_QWENVL_VIDEO_READER is not None:
        backend = FORCE_QWENVL_VIDEO_READER
    elif is_decord_available():
        backend = "decord"
    else:
        backend = "torchvision"
    print(f"qwen-vl-utils using {backend} to read video.", file=sys.stderr)
    return backend
def fetch_video(
    ele: dict,
    image_factor: int = IMAGE_FACTOR,
    return_video_sample_fps: bool = False,
) -> torch.Tensor | list[Image.Image] | tuple[torch.Tensor | list[Image.Image], float]:
    if isinstance(ele["video"], str):
        backend = get_video_reader_backend()
        try:
            video, sample_fps = VIDEO_READER_BACKENDS[backend](ele)
        except Exception as exc:
            logger.warning(f"video_reader_backend {backend} error, fallback to torchvision: {exc}")
            video, sample_fps = VIDEO_READER_BACKENDS["torchvision"](ele)
        nframes, _, height, width = video.shape
        min_pixels = ele.get("min_pixels", VIDEO_MIN_PIXELS)
        total_pixels = ele.get("total_pixels", VIDEO_TOTAL_PIXELS)
        max_pixels = max(min(VIDEO_MAX_PIXELS, total_pixels / nframes * FRAME_FACTOR), int(min_pixels * 1.05))
        max_pixels = min(ele.get("max_pixels", max_pixels), max_pixels)
        if "resized_height" in ele and "resized_width" in ele:
            resized_height, resized_width = smart_resize(
                ele["resized_height"],
                ele["resized_width"],
                factor=image_factor,
            )
        else:
            resized_height, resized_width = smart_resize(
                height,
                width,
                factor=image_factor,
                min_pixels=min_pixels,
                max_pixels=max_pixels,
            )
        video = transforms.functional.resize(
            video,
            [resized_height, resized_width],
            interpolation=InterpolationMode.BICUBIC,
            antialias=True,
        ).float()
        if return_video_sample_fps:
            return video, sample_fps
        return video
    process_info = ele.copy()
    process_info.pop("type", None)
    process_info.pop("video", None)
    images = [fetch_image({"image": frame, **process_info}, size_factor=image_factor) for frame in ele["video"]]
    nframes = ceil_by_factor(len(images), FRAME_FACTOR)
    if len(images) < nframes:
        images.extend([images[-1]] * (nframes - len(images)))
    if return_video_sample_fps:
        return images, process_info.get("fps", 2.0)
    return images
def extract_vision_info(conversations: list[dict] | list[list[dict]]) -> list[dict]:
    vision_infos = []
    if isinstance(conversations[0], dict):
        conversations = [conversations]
    for conversation in conversations:
        for message in conversation:
            content = message.get("content")
            if not isinstance(content, list):
                continue
            for ele in content:
                if (
                    "image" in ele
                    or "image_url" in ele
                    or "video" in ele
                    or ele.get("type", "") in ("image", "image_url", "video")
                ):
                    vision_infos.append(ele)
    return vision_infos
def process_vision_info(
    conversations: list[dict] | list[list[dict]],
    return_video_kwargs: bool = False,
) -> tuple[list[Image.Image] | None, list[torch.Tensor | list[Image.Image]] | None, Optional[dict]]:
    vision_infos = extract_vision_info(conversations)
    image_inputs = []
    video_inputs = []
    video_sample_fps_list = []
    for vision_info in vision_infos:
        if "image" in vision_info or "image_url" in vision_info:
            image_inputs.append(fetch_image(vision_info))
        elif "video" in vision_info:
            video_input, video_sample_fps = fetch_video(vision_info, return_video_sample_fps=True)
            video_sample_fps_list.append(video_sample_fps)
            video_inputs.append(video_input)
        else:
            raise ValueError("image, image_url or video should in content.")
    if not image_inputs:
        image_inputs = None
    if not video_inputs:
        video_inputs = None
    if return_video_kwargs:
        return image_inputs, video_inputs, {"fps": video_sample_fps_list}
    return image_inputs, video_inputs

__all__ = ["extract_vision_info", "fetch_image", "fetch_video", "process_vision_info", "smart_resize"]