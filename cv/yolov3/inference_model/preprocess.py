#!/usr/bin/env python3
import argparse
import logging
import multiprocessing as mp
import os
from typing import Any
from typing import Dict
from typing import Generator
from typing import List
from typing import Tuple

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
    precess_dataset(args.output_path, args.coco_path, args.count, args.n)


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
    import cv2
    img = cv2.imread(image_path)
    img = proprecess_func(img)
    img = img.astype(np.uint8)
    img.tofile(output_path)


class HmYuvInt8YoloV3:

    def __init__(self, image_size: Tuple[int, int] = (416, 416)) -> None:
        self._image_size = image_size

    def _letterbox_image(self, img, inp_dim):
        '''resize image with unchanged aspect ratio using padding'''
        import cv2
        img_w, img_h = img.shape[1], img.shape[0]
        w, h = inp_dim
        new_w = int(img_w * min(w/img_w, h/img_h))
        new_h = int(img_h * min(w/img_w, h/img_h))
        resized_image = cv2.resize(
            img, (new_w, new_h), interpolation=cv2.INTER_CUBIC,
        )
        canvas = np.full((inp_dim[1], inp_dim[0], 3), 128)
        canvas[
            (h-new_h)//2:(h-new_h)//2 + new_h, (w-new_w) //
            2:(w-new_w)//2 + new_w, :
        ] = resized_image
        return canvas

    def __call__(self, img):
        """
        Prepare image for inputting to the neural network.

        Returns a Variable
        """
        img = (self._letterbox_image(img, self._image_size))
        img = img[:, :, ::-1].transpose((2, 0, 1)).copy()
        R = img[:1, :, :]
        GB = img[1:, :, :]
        GB = np.transpose(GB, (1, 2, 0))
        GB = np.resize(GB, (2, self._image_size[0], self._image_size[1]))
        img = np.concatenate((R, GB), axis=0)
        img = np.resize(img, (self._image_size[0], self._image_size[1], 3))
        img = img - 128
        return img


if __name__ == '__main__':
    main()
