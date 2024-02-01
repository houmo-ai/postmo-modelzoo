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
        '--imagenet-path',
        dest='imagenet_path',
        help='ImageNet2012 dataset root path',
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
    precess_dataset(args.output_path, args.imagenet_path, args.count, args.n)


def sub_process_dataset(image_path_list: List[Tuple[str, str]]):
    for output_file_path, src_file_path in image_path_list:
        efficientnet_proprecess(output_file_path, src_file_path)


def precess_dataset(output_path: str, imagenet_path: str, count: int, process_count: int) -> None:
    preprocessed_count = 0
    os.makedirs(output_path, exist_ok=True)
    image_path_list = []
    for root, dirs, files in os.walk(imagenet_path):
        files.sort()
        for file_name in files:
            if count is not None and preprocessed_count >= count:
                break
            src_file_path = os.path.join(root, file_name)
            output_file_path = os.path.join(output_path, file_name)
            image_path_list.append((output_file_path, src_file_path))
            preprocessed_count += 1
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


def efficientnet_proprecess(output_path: str, image_path: str) -> None:
    """
    Prepare image for inputing to the neural network.
    """
    import torchvision.transforms as transforms
    from torchvision.datasets.folder import pil_loader
    from hmquant.tools.dataset.preprocess.transform import RGB2YUV
    from hmquant.tools.dataset.preprocess.transform import ToTensorNotNormal
    calib_transform = transforms.Compose(
        [
            transforms.Resize(256), transforms.CenterCrop(224),
            ToTensorNotNormal(), RGB2YUV(),
        ],
    )

    img = pil_loader(image_path)
    img = calib_transform(img)
    img = img.numpy()
    img = img.astype(np.uint8)
    img.tofile(output_path)


if __name__ == '__main__':
    main()
