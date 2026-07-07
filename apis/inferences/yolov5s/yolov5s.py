# Copyright 2025 HOUMO AI
#
# File: yolov5s.py
# Description:
#   YOLOv5 Object Detection Python Example.
#   This file implements an object detection application using the YOLOv5 model.
#   It includes image preprocessing, model inference using the TCIM runtime, and postprocessing.
#   The implementation supports both native PyTorch postprocessing and ONNX Runtime-based
#   postprocessing for enhanced flexibility. The code handles multi-scale feature maps
#   and applies appropriate anchor boxes for detection.
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

import os
import sys
import numpy as np
import time
import argparse
from loguru import logger

import cv2
import torch
import torchvision

HOUMO_EXAMPLES_PATH = os.environ.get("HOUMO_EXAMPLES_PATH", "../../..")
sys.path.insert(0, f"{HOUMO_EXAMPLES_PATH}/apis/common/python")
sys.path.insert(0, f"{HOUMO_EXAMPLES_PATH}/hmatc")
from format_converter import BGR2YUV

import tcim_lite as tcim

HOUMO_TARGET = os.getenv("HOUMO_TARGET")
assert HOUMO_TARGET in ["xh2"], f"Unsupported HOUMO_TARGET: {HOUMO_TARGET}"

# COCO dataset class labels for YOLOv5 model
coco80_labels = [
    "person",
    "bicycle",
    "car",
    "motorbike",
    "aeroplane",
    "bus",
    "train",
    "truck",
    "boat",
    "traffic light",
    "fire hydrant",
    "stop sign",
    "parking meter",
    "bench",
    "bird",
    "cat",
    "dog",
    "horse",
    "sheep",
    "cow",
    "elephant",
    "bear",
    "zebra",
    "giraffe",
    "backpack",
    "umbrella",
    "handbag",
    "tie",
    "suitcase",
    "frisbee",
    "skis",
    "snowboard",
    "sports ball",
    "kite",
    "baseball bat",
    "baseball glove",
    "skateboard",
    "surfboard",
    "tennis racket",
    "bottle",
    "wine glass",
    "cup",
    "fork",
    "knife",
    "spoon",
    "bowl",
    "banana",
    "apple",
    "sandwich",
    "orange",
    "broccoli",
    "carrot",
    "hot dog",
    "pizza",
    "donut",
    "cake",
    "chair",
    "sofa",
    "pottedplant",
    "bed",
    "diningtable",
    "toilet",
    "tvmonitor",
    "laptop",
    "mouse",
    "remote",
    "keyboard",
    "cell phone",
    "microwave",
    "oven",
    "toaster",
    "sink",
    "refrigerator",
    "book",
    "clock",
    "vase",
    "scissors",
    "teddy bear",
    "hair drier",
    "toothbrush",
]


def get_args() -> argparse.Namespace:
    """Parse command-line arguments for the YOLOv5 example."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--enable_ort", action="store_true", help="use onnxruntime to post process."
    )
    args = parser.parse_args()
    return args


def infer_detect_onnx(onnx_path, inputs_list):
    """
    Perform post-processing using ONNX Runtime.

    Args:
        onnx_path (str): Path to the ONNX post-processing model
        inputs_list (list): List of input tensors for the post-processing model

    Returns:
        torch.Tensor: Processed output tensor
    """
    import onnxruntime as ort

    session = ort.InferenceSession(onnx_path)
    input_names = list()
    input_dict = dict()
    for idx, input in enumerate(session.get_inputs()):
        input_name = input.name
        input_dict[input_name] = inputs_list[idx]
        input_names.append(input_name)

    outputs = session.run(None, input_dict)
    logger.info(f"post-processing model output num: {len(outputs)}")
    tensor_res = torch.tensor(outputs[0])

    return tensor_res


def letterbox(
    im,
    new_shape=(640, 640),
    color=(114, 114, 114),
    auto=True,
    scale_fill=False,
    scaleup=True,
    stride=32,
):
    """
    Resize and pad image while meeting stride-multiple constraints.

    Args:
        im (numpy.ndarray): Input image
        new_shape (tuple): Target size for the image
        color (tuple): Padding color (BGR)
        auto (bool): Whether to use minimum rectangle padding
        scale_fill (bool): Whether to stretch the image
        scaleup (bool): Whether to scale up the image
        stride (int): Stride value for padding constraints

    Returns:
        tuple: Resized image, ratio, and padding values
    """
    shape = im.shape[:2]
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)

    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    if not scaleup:
        r = min(r, 1.0)

    ratio = r, r
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]
    if auto:
        dw, dh = np.mod(dw, stride), np.mod(dh, stride)
    elif scale_fill:
        dw, dh = 0.0, 0.0
        new_unpad = (new_shape[1], new_shape[0])
        ratio = new_shape[1] / shape[1], new_shape[0] / shape[0]

    dw /= 2
    dh /= 2

    # Resize image if dimensions changed
    if shape[::-1] != new_unpad:
        im = cv2.resize(im, new_unpad, interpolation=cv2.INTER_LINEAR)

    # Calculate padding amounts
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))

    # Apply padding with specified color
    im = cv2.copyMakeBorder(
        im,
        top,
        bottom,
        left,
        right,
        cv2.BORDER_CONSTANT,
        value=color,
    )
    return im, ratio, (dw, dh)


def xywh2xyxy(x):
    """
    Convert bounding boxes from [x, y, w, h] format to [x1, y1, x2, y2] format.

    Args:
        x (torch.Tensor or numpy.ndarray): Input boxes in [x, y, w, h] format

    Returns:
        torch.Tensor or numpy.ndarray: Converted boxes in [x1, y1, x2, y2] format
    """
    y = x.clone() if isinstance(x, torch.Tensor) else np.copy(x)
    y[:, 0] = x[:, 0] - x[:, 2] / 2  # top left x = center x - width / 2
    y[:, 1] = x[:, 1] - x[:, 3] / 2  # top left y = center y - height / 2
    y[:, 2] = x[:, 0] + x[:, 2] / 2  # bottom right x = center x + width / 2
    y[:, 3] = x[:, 1] + x[:, 3] / 2  # bottom right y = center y + height / 2
    return y


def box_area(box):
    """
    Calculate area of bounding boxes.

    Args:
        box (numpy.ndarray): Boxes in [x1, y1, x2, y2] format

    Returns:
        numpy.ndarray: Area of each box
    """
    return (box[2] - box[0]) * (box[3] - box[1])


def box_iou(box1, box2, eps=1e-7):
    """
    Calculate Intersection over Union (IoU) of two sets of bounding boxes.

    Args:
        box1 (torch.Tensor): First set of boxes in [x1, y1, x2, y2] format
        box2 (torch.Tensor): Second set of boxes in [x1, y1, x2, y2] format
        eps (float): Small value to prevent division by zero

    Returns:
        torch.Tensor: IoU matrix of shape [N, M] where N and M are the number of boxes in box1 and box2
    """
    (a1, a2), (b1, b2) = box1[:, None].chunk(2, 2), box2.chunk(2, 1)
    inter = (torch.min(a2, b2) - torch.max(a1, b1)).clamp(0).prod(2)

    return inter / (box_area(box1.T)[:, None] + box_area(box2.T) - inter + eps)


class YoloV5:
    """
    YOLOv5 object detection class with preprocessing and postprocessing methods.
    """

    def __init__(self, image_size=(640, 640), conf_threshold=0.25, iou_threshold=0.45):
        """
        Initialize the YOLOv5 detector.

        Args:
            image_size (tuple): Input image size (width, height)
            conf_threshold (float): Confidence threshold for detection
            iou_threshold (float): IoU threshold for NMS
        """
        self._image_size = image_size
        self._conf_threshold = conf_threshold
        self._iou_threshold = iou_threshold

    def preprocess(self, image):
        """
        Preprocess input image for YOLOv5 model.

        Args:
            image (numpy.ndarray): Input image

        Returns:
            numpy.ndarray: Preprocessed image in NCHW format
        """
        out, _, _ = letterbox(image, self._image_size, stride=64, auto=False)  # HWC
        out = np.transpose(out, (2, 0, 1))  # CHW
        out = np.expand_dims(out, axis=0)  # NCHW

        return out

    def yolo_detect(self, feats):
        """
        Perform YOLO detection on feature maps.

        Args:
            feats (list): List of feature maps from different scales

        Returns:
            torch.Tensor: Concatenated detection results
        """
        output = []

        # Process each feature map at different scales
        for i, feat in enumerate(feats):
            data = torch.tensor(feat)
            assert len(data.shape) == 5

            bs, channel, ny, nx, no = data.shape
            grid, anchor_grid = self._make_grid(nx, ny, i)

            # Apply YOLO transformation to center coordinates
            data[..., 0:2] = (data[..., 0:2] * 2 - 0.5 + grid) * self.stride[i]
            # Apply YOLO transformation to width and height
            data[..., 2:4] = (data[..., 2:4] * 2) ** 2 * anchor_grid

            output.append(data.reshape(bs, -1, no))

        # Concatenate results from all scales
        return torch.concat(output, dim=1)

    def _make_grid(self, nx=20, ny=20, i=0):
        """
        Generate grid for YOLO detection.

        Args:
            nx (int): Grid width
            ny (int): Grid height
            i (int): Feature level index

        Returns:
            tuple: Grid and anchor grid tensors
        """
        # Define anchor boxes for different scales
        anchors = torch.tensor(
            [
                [10, 13, 16, 30, 33, 23],  # Small objects (scale 0)
                [30, 61, 62, 45, 59, 119],  # Medium objects (scale 1)
                [116, 90, 156, 198, 373, 326],  # Large objects (scale 2)
            ]
        )

        self.stride = torch.tensor([8, 16, 32]).view(-1, 1, 1)
        anchors = anchors.view(3, 3, 2)
        anchors = anchors / self.stride

        yv, xv = torch.meshgrid(torch.arange(ny), torch.arange(nx), indexing="ij")
        grid = torch.stack((xv, yv), 2).expand((1, 3, ny, nx, 2)).float()
        anchor_grid = (
            (anchors[i] * self.stride[i])
            .view((1, 3, 1, 1, 2))
            .expand((1, 3, ny, nx, 2))
            .float()
        )

        return grid, anchor_grid

    def non_max_suppression(
        self,
        prediction,
        classes=None,
        agnostic=False,
        multi_label=False,
        labels=(),
        max_det=300,
        nm=0,
    ):
        """
        Perform Non-Maximum Suppression (NMS) on detection results.

        Args:
            prediction (torch.Tensor): Model predictions
            classes (list, optional): Filter by class indices
            agnostic (bool): Class-agnostic NMS
            multi_label (bool): Multiple labels per box
            labels (tuple): Ground truth labels
            max_det (int): Maximum number of detections
            nm (int): Number of masks

        Returns:
            list: List of detection results per image
        """
        conf_thres = self._conf_threshold
        iou_thres = self._iou_threshold
        bs = prediction.shape[0]  # batch size
        nc = prediction.shape[2] - nm - 5  # number of classes
        # candidate indices (confidence > threshold)
        xc = prediction[..., 4] > conf_thres

        assert (
            0 <= conf_thres <= 1
        ), f"Invalid Confidence threshold {conf_thres}, valid values are between 0.0 and 1.0"
        assert (
            0 <= iou_thres <= 1
        ), f"Invalid IoU {iou_thres}, valid values are between 0.0 and 1.0"

        # NMS parameters
        max_wh = 7680
        max_nms = 30000
        time_limit = 0.5 + 0.05 * bs
        redundant = True
        multi_label &= nc > 1
        merge = False

        t = time.time()
        mi = 5 + nc
        output = [torch.zeros((0, 6 + nm), device=prediction.device)] * bs

        # Process each image in the batch
        for xi, x in enumerate(prediction):
            x = x[xc[xi]]

            if labels and len(labels[xi]):
                lb = labels[xi]
                v = torch.zeros((len(lb), nc + 5), device=x.device)
                v[:, :4] = lb[:, 1:5]
                v[:, 4] = 1.0
                v[range(len(lb)), lb[:, 0].long() + 5] = 1.0
                x = torch.cat((x, v), 0)

            # Skip if no boxes remain
            if not x.shape[0]:
                continue

            # Compute final confidence by multiplying objectness with class probabilities
            x[:, 5:] *= x[:, 4:5]
            # Convert box format from center coordinates to corner coordinates
            box = xywh2xyxy(x[:, :4])
            mask = x[:, mi:]

            # Handle multi-label vs single-label detection
            if multi_label:
                i, j = (x[:, 5:mi] > conf_thres).nonzero(as_tuple=False).T
                x = torch.cat(
                    (box[i], x[i, j + 5, None], j[:, None].float(), mask[i]), dim=1
                )
            else:
                conf, j = x[:, 5:mi].max(1, keepdim=True)
                x = torch.cat((box, conf, j.float(), mask), dim=1)[
                    conf.view(-1) > conf_thres
                ]

            if classes is not None:
                x = x[(x[:, 5:6] == torch.tensor(classes, device=x.device)).any(1)]

            n = x.shape[0]
            if not n:
                continue

            # Sort by confidence and keep only top max_nms boxes
            x = x[x[:, 4].argsort(descending=True)[:max_nms]]
            # Apply class-agnostic or class-specific NMS
            c = x[:, 5:6] * (0 if agnostic else max_wh)
            boxes, scores = x[:, :4] + c, x[:, 4]
            i = torchvision.ops.nms(boxes, scores, iou_thres)
            i = i[:max_det]

            if merge and (1 < n < 3e3):
                iou = box_iou(boxes[i], boxes) > iou_thres
                weights = iou * scores[None]
                x[i, :4] = torch.mm(weights, x[:, :4]).float() / weights.sum(
                    1, keepdim=True
                )
                if redundant:
                    i = i[iou.sum(1) > 1]

            # Store results for this image
            output[xi] = x[i]

            if (time.time() - t) > time_limit:
                logger.warning(f"NMS time limit {time_limit:.3f}s exceeded")
                break

        return output

    def scale_coords(self, coords, img0_shape, ratio_pad=None):
        """
        Scale coordinates from prediction size to original image size.

        Args:
            coords (torch.Tensor): Coordinates to scale
            img0_shape (tuple): Original image shape (height, width)
            ratio_pad (tuple, optional): Precomputed ratio and padding

        Returns:
            torch.Tensor: Scaled coordinates
        """
        if ratio_pad is None:
            # Calculate scaling factor to maintain aspect ratio
            gain = min(
                self._image_size[0] / img0_shape[0], self._image_size[1] / img0_shape[1]
            )
            pad = (self._image_size[1] - img0_shape[1] * gain) / 2, (
                self._image_size[0] - img0_shape[0] * gain
            ) / 2
        else:
            gain = ratio_pad[0][0]
            pad = ratio_pad[1]

        # Remove padding from coordinates
        coords[:, [0, 2]] -= pad[0]
        coords[:, [1, 3]] -= pad[1]

        coords[:, :4] /= gain

        return coords


def draw_bbox(image, x1, y1, x2, y2, label, color, thickness=2):
    """
    Draw bounding box and label on the image.

    Args:
        image (numpy.ndarray): Input image to draw on
        x1, y1, x2, y2 (int): Bounding box coordinates
        label (str): Label text to display
        color (tuple): RGB color for the bounding box
        thickness (int): Thickness of the bounding box lines
    """
    # Draw bounding box
    cv2.rectangle(image, (x1, y1), (x2, y2), color[::-1], thickness)

    # Calculate text size and draw background
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.6
    text_size = cv2.getTextSize(label, font, font_scale, thickness)[0]
    text_x, text_y = x1, y1 - 5

    # Draw label background
    cv2.rectangle(
        image,
        (text_x, text_y - text_size[1] - 2),
        (text_x + text_size[0], text_y + 2),
        color[::-1],  # Convert RGB to BGR for OpenCV
        -1,  # Fill the rectangle
    )

    # Draw label text in white on the background
    cv2.putText(
        image,
        label,
        (text_x, text_y),
        font,
        font_scale,
        (255, 255, 255),  # White text
        thickness,
        cv2.LINE_AA,  # Anti-aliased lines
    )


def _get_color(cls_id):
    """
    Generate a color based on the class ID.

    Args:
        cls_id (int): Class ID

    Returns:
        tuple: RGB color tuple
    """
    np.random.seed(cls_id)
    color = tuple(np.random.randint(0, 255, size=3).tolist())
    return color


if __name__ == "__main__":
    args = get_args()
    logger.info("===> yolov5s python example start...")
    logger.info(
        f"houmo target: {HOUMO_TARGET}, enable ort: {args.enable_ort}, tcim runtime version: {tcim.runtime.get_version()}"
    )

    # Discover the local model files in the example directory.
    current_dir = os.path.dirname(os.path.abspath(__file__))
    hmm_files = [
        os.path.join(current_dir, name)
        for name in os.listdir(current_dir)
        if name.endswith(".hmm")
    ]
    assert hmm_files, f"No .hmm file found in {current_dir}"

    # Load model
    model_path = hmm_files[0]
    logger.info(f"Found model file: {model_path}")
    module = tcim.runtime.load(model_path)

    # Load the sample input image.
    yolov5 = YoloV5()
    img_path = "../../data/000000000139.jpg"
    image_data = cv2.imread(img_path)
    cv_image = image_data.copy()
    img_height, img_width = image_data.shape[:2]
    logger.info(f"input image shape: {image_data.shape}")

    resizer_crop_str = "resizer_crop"
    target_height, target_width = 640, 640
    max_img_height, max_img_width = 0, 0
    input_names = []
    input_num = module.get_num_inputs()
    logger.info("Model input info:")
    for idx in range(0, input_num):
        input_name = module.get_input_name(idx)
        input_info = module.get_input_info(input_name).ascontiguous()
        input_names.append(input_name)
        logger.info(
            f"  Input[{input_name}] shape = {input_info.shape}, dtype = {input_info.dtype}, mem_size = {input_info.mem_size}, stride = {input_info.stride}, format = {input_info.format.name}"
        )
        # The non-resizer input is the raw image canvas consumed by the hardware
        # resizer. Its H/W define the maximum image size we can upload.
        if resizer_crop_str not in input_name:
            max_img_height, max_img_width = input_info.shape[2], input_info.shape[3]

    # 2. Preprocess image
    # Keep the valid image region as crop info for the resizer input. When the
    # uploaded canvas is padded, only this crop region is resized by the model.
    crop_height, crop_width = max_img_height, max_img_width
    if img_height < max_img_height and img_width <= max_img_width:
        # Pad smaller images to the upload canvas. The gray value matches the
        # YOLOv5 letterbox padding color, while crop_height/crop_width preserve
        # the original valid image area.
        pad_bottom = max_img_height - img_height
        pad_right = max_img_width - img_width
        image_data = cv2.copyMakeBorder(
            image_data,
            0,
            pad_bottom,
            0,
            pad_right,
            cv2.BORDER_CONSTANT,
            value=(114, 114, 114),
        )
        crop_height = img_height
        crop_width = img_width
        logger.info(
            f"pad input image to {image_data.shape} height = {max_img_height}, width = {max_img_width}"
        )
    else:
        # If the image is larger than the upload canvas, resize it first and let
        # dyn_info describe the full resized canvas as the crop region.
        image_data = cv2.resize(image_data, (max_img_width, max_img_height))
        logger.info(
            f"resize input image to height = {max_img_height}, width = {max_img_width}"
        )

    # Convert HWC BGR input to CHW float32 before color space conversion.
    image_data = np.transpose(image_data, (2, 0, 1))  # CHW uint8
    input_data = torch.tensor(image_data.astype(np.float32))  # CHW float32

    # Convert BGR input to YUV420 as required by the runtime input format.
    bgr2yuv_func = BGR2YUV(fmt="YUV420")
    input_data = torch.unsqueeze(bgr2yuv_func(input_data), 0).numpy()  # NCHW float32
    input_data = input_data.astype(np.uint8)  # NCHW uint8

    # The hardware resizer consumes YUV420 input, so crop dimensions must be
    # even. Match the C++ helper's TO_EVEN behavior by rounding down.
    crop_height = crop_height - (crop_height % 2)
    crop_width = crop_width - (crop_width % 2)
    assert (
        crop_height % 2 == 0
        and crop_width % 2 == 0
        and crop_height > 0
        and crop_width > 2
    ), f"crop_height and crop_width must be even, got {crop_height} and {crop_width}"

    assert (
        target_height > 0
        and target_width > 0
        and target_height % 2 == 0
        and target_width % 2 == 0
    ), f"target_height and target_width must be positive even values, got {target_height} and {target_width}"

    # YOLOv5 uses letterbox preprocessing: resize the crop region with one
    # shared scale, then pad the remaining area to the fixed 640x640 canvas.
    scale = min(target_height / crop_height, target_width / crop_width)
    assert 1 / 32 <= scale <= 16, f"resize scale must be in [1/32, 16], got {scale}"

    # The resizer requires even output sizes. Rounding down after round() keeps
    # the size close to the ideal scaled value while satisfying that constraint.
    resizer_height = int(round(crop_height * scale)) & ~1
    resizer_width = int(round(crop_width * scale)) & ~1
    resizer_height = min(resizer_height, target_height)
    resizer_width = min(resizer_width, target_width)
    assert (
        resizer_height > 0 and resizer_width > 0
    ), f"resizer_height and resizer_width must be positive, got {resizer_height} and {resizer_width}"

    pad_height = target_height - resizer_height
    pad_width = target_width - resizer_width
    pad_h_top = (pad_height // 2) & ~1
    pad_w_left = (pad_width // 2) & ~1
    pad_h_bottom = pad_height - pad_h_top
    pad_w_right = pad_width - pad_w_left

    # dyn_info layout:
    # [crop_offset_h, crop_offset_w, crop_h, crop_w,
    #  resize_h, resize_w, pad_top, pad_left, pad_bottom, pad_right]
    # Nonzero padding tells the runtime to use proportional resize + padding
    # instead of stretching directly to target_height x target_width.
    dyn_info = np.array(
        [
            0,
            0,
            crop_height,
            crop_width,
            resizer_height,
            resizer_width,
            pad_h_top,
            pad_w_left,
            pad_h_bottom,
            pad_w_right,
        ],
        dtype=np.int32,
    )
    dyn_info = np.expand_dims(dyn_info, 0)
    logger.info(
        f"input_data shape: {input_data.shape}, dtype: {input_data.dtype}, dyn_info: {dyn_info}"
    )

    # 3. Feed image data and dynamic crop information into the model inputs.
    for input_name in input_names:
        # The compiled model has two input kinds: the YUV image canvas and the
        # int32 dynamic crop/resize/pad tensor used by the hardware resizer.
        if resizer_crop_str in input_name:
            module.set_input(input_name, dyn_info)
        else:
            module.set_input(input_name, input_data)

    # 4. Run inference and synchronize
    module.run()
    module.sync()

    # 5. Get output tensors from the model
    result_check = True
    outputs = []
    output_num = module.get_num_outputs()
    for idx in range(0, output_num):
        output_name = module.get_output_name(idx)
        output_info = (
            module.get_output_info(output_name).astype(np.float32).ascontiguous()
        )
        logger.info(
            f"output[{output_name}] shape = {output_info.shape}, dtype = {output_info.dtype}, format = {output_info.format.name}."
        )
        # Get the output tensor and convert to numpy array
        output_data = module.get_output(output_name).astype(np.float32).numpy()
        outputs.append(output_data)

    # 6. Postprocess outputs
    assert len(outputs) == 3
    if args.enable_ort:
        # 6.1 Use ONNX Runtime for post-processing
        onnx_path = "./yolov5s_640x640_postprocess.onnx"
        outputs = infer_detect_onnx(onnx_path, outputs)
    else:
        # 6.2 Use PyTorch functions for post-processing
        outputs = yolov5.yolo_detect(outputs)

    outputs = yolov5.non_max_suppression(outputs)
    outputs = outputs[0]
    image_size = (cv_image.shape[0], cv_image.shape[1])
    # Scale coordinates back to original image size
    outputs[:, :4] = yolov5.scale_coords(outputs[:, :4], image_size).round()
    boxes = outputs.numpy()

    # 7. Print and draw detection results
    logger.info(f"box num = {len(boxes)}")
    for det in boxes:
        (x1, y1, x2, y2), conf, cls_id = list(map(int, det[0:4])), det[4], int(det[5])
        color = _get_color(cls_id)
        # Create label with class name and confidence
        label = f"{coco80_labels[cls_id]} {conf:.2f}"
        # Draw bounding box and label on the image
        draw_bbox(cv_image, x1, y1, x2, y2, label, color, thickness=2)
        logger.info(
            "x1:{}, y1:{}, x2:{}, y2:{}, conf:{:.6f}, cls:{}".format(
                x1, y1, x2, y2, conf, int(cls_id)
            )
        )

    save_results = "demo_results/python"
    if not os.path.exists(save_results):
        os.makedirs(save_results)
    filename = os.path.basename(img_path)
    save_path = os.path.join(save_results, filename)
    cv2.imwrite(save_path, cv_image)
    logger.info(f"demo results saved to {save_path}")
    # Verify result count (modify when changing model or data)
    assert len(boxes) in [18, 19, 20, 21]

    logger.info("<=== yolov5s python example completed.")
