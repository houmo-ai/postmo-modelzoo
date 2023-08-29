#!/usr/bin/env python3
import argparse
import logging
import math
import multiprocessing as mp
import os
from typing import Any
from typing import Dict
from typing import Generator
from typing import List
from typing import Tuple

import cv2
import numpy as np

logging.basicConfig(
    format='[%(asctime)s %(name)s.%(funcName)s():%(lineno)s] %(process)d %(levelname)s  -  %(message)s',
    datefmt='%d-%b-%y %H:%M:%S',
)
logger = logging.getLogger(__file__)


def get_args() -> argparse.Namespace:
    """Parse commandline."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--coco-path',
        dest='coco_path',
        help='COCO dataset root path',
    )
    parser.add_argument(
        '--img-path',
        dest='img_path',
        help='Image path to be preprocessed',
    )
    parser.add_argument(
        '--output-path',
        default='preprocessed',
        dest='output_path',
        help='The path to store proprecessed files',
    )
    parser.add_argument(
        '--count',
        type=int,
        help='The count of images to be proprecessed',
    )
    parser.add_argument(
        '-n',
        default=1,
        type=int,
        help='The preprocess process count',
    )
    parser.add_argument(
        '--log-level',
        choices=['CRITCAL', 'ERROR', 'WARNING', 'INFO', 'DEBUG'],
        default='WARNING',
        help='level of messages to catch/display; level of messages to catch/display',
    )
    args = parser.parse_args()
    return args


def main(args: Any = None) -> None:
    """main function"""
    if args is None:
        args = get_args()
    logging.getLogger().setLevel(level=args.log_level)
    if args.coco_path:
        precess_dataset(args.output_path, args.coco_path, args.count, args.n)
    elif args.img_path:
        os.makedirs(args.output_path, exist_ok=True)
        _, img_file_name = os.path.split(args.img_path)
        sub_process_dataset(
            [(os.path.join(args.output_path, img_file_name), args.img_path)],
        )


def sub_process_dataset(image_path_list: List[Tuple[str, str]]):
    """sub process func"""
    proprecess_func = HmYuvInt8YoloV3(image_size=(416, 416))
    for output_file_path, src_file_path in image_path_list:
        yolov3_proprecess(proprecess_func, output_file_path, src_file_path)


def precess_dataset(output_path: str, coco_path: str, count: int, process_count: int) -> None:
    import json
    preprocessed_count = 0
    os.makedirs(output_path, exist_ok=True)
    annotation_file_path = os.path.join(
        coco_path, 'annotations', 'instances_val2017.json',
    )
    coco_image_path_root = os.path.join(coco_path, 'val2017')
    image_path_list = []
    with open(annotation_file_path) as f:
        coco = json.load(f)
        for img_info in coco['images']:
            if count is not None and preprocessed_count >= count:
                break
            src_file_path = os.path.join(
                coco_image_path_root, img_info['file_name'],
            )
            output_file_path = os.path.join(output_path, img_info['file_name'])
            print(src_file_path)
            image_path_list.append((output_file_path, src_file_path))
            preprocessed_count += 1
    print(f'preprocessed_count: {preprocessed_count}')
    sub_procs = []
    for idx in range(process_count):
        proc = mp.Process(
            target=sub_process_dataset, args=(
                image_path_list[idx::process_count],
            ),
        )
        proc.start()
        sub_procs.append(proc)
    for sub_proc in sub_procs:
        sub_proc.join()


def yolov3_proprecess(proprecess_func, output_path: str, image_path: str) -> None:
    """
    Prepare image for inputing to the neural network.
    """
    from PIL import Image
    img = Image.open(image_path).convert('RGB')
    img = proprecess_func(img)
    from hmquant.tools.dataset.preprocess.transform import RGB2YUV
    import torch
    rgb2yuv = RGB2YUV()
    img = img.astype(np.float32)
    img = torch.from_numpy(img)
    img = rgb2yuv(img).numpy()
    img = img.astype(np.uint8)
    img.tofile(output_path)


def letterbox(im, new_shape=(416, 416), color=(114, 114, 114), auto=True, scaleFill=False, scaleup=True, stride=32):
    # Resize and pad image while meeting stride-multiple constraints
    shape = im.shape[:2]  # current shape [height, width]
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)

    # Scale ratio (new / old)
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    if not scaleup:  # only scale down, do not scale up (for better val mAP)
        r = min(r, 1.0)

    # Compute padding
    ratio = r, r  # width, height ratios
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - \
        new_unpad[1]  # wh padding
    if auto:  # minimum rectangle
        dw, dh = np.mod(dw, stride), np.mod(dh, stride)  # wh padding
    elif scaleFill:  # stretch
        dw, dh = 0.0, 0.0
        new_unpad = (new_shape[1], new_shape[0])
        ratio = new_shape[1] / shape[1], new_shape[0] / \
            shape[0]  # width, height ratios

    dw /= 2  # divide padding into 2 sides
    dh /= 2

    if shape[::-1] != new_unpad:  # resize
        im = cv2.resize(im, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    im = cv2.copyMakeBorder(
        im, top, bottom, left, right,
        cv2.BORDER_CONSTANT, value=color,
    )  # add border
    return im, ratio, (dw, dh)


class HmYuvInt8YoloV3:

    def __init__(self, image_size: Tuple[int, int] = (416, 416)) -> None:
        self._image_size = image_size

    def __call__(self, img):
        """
        Prepare image for inputting to the neural network.

        Returns a Variable
        """
        img = np.array(img)
        self.img_shape = []
        imh, imw, imc = img.shape  # original shape

        r = self._image_size[0] / max(imh, imw)
        if r != 1:  # if sizes are not equal
            interp = cv2.INTER_LINEAR if (r > 1) else cv2.INTER_AREA
            img = cv2.resize(
                img, (math.ceil(imw * r), math.ceil(imh * r)), interpolation=interp,
            )

        h, w = img.shape[:2]

        # shape = self.batch_shapes[self.batch[index]] #if self.rect else self.input_shape
        # Padded resize
        img, ratio, pad = letterbox(
            img, self._image_size[0], stride=32, auto=False, scaleup=False,
        )
        img = img.transpose((2, 0, 1))
        img = np.ascontiguousarray(img)
        return img


if __name__ == '__main__':
    main()
