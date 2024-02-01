#!/usr/bin/env python3
import argparse
import logging
import os
from typing import Any

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
        help='COCO dataset root path',
    )
    parser.add_argument(
        '--predict-result',
        help='The result detection result json file path',
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
    calculate_meanap(args.predict_result, args.coco_path)


def calculate_meanap(predict_result_json_path: str, coco_path: str):
    """
    Calculate meanap of inference result
    """
    from pycocotools.cocoeval import COCOeval
    from pycocotools.coco import COCO
    import json

    detections = []
    image_indices = []
    with open(predict_result_json_path) as predict_file:
        output_det = json.load(predict_file)
    annotation_file_path = os.path.join(
        coco_path, 'annotations', 'instances_val2017.json',
    )
    label_map = {}
    with open(annotation_file_path) as fin:
        annotations = json.load(fin)
    for cnt, cat in enumerate(annotations['categories']):
        label_map[cat['id']] = cnt + 1
    inv_map = {v: k for k, v in label_map.items()}
    image_indices_set = set()
    for batch in range(0, len(output_det)):
        detection = output_det[batch]
        detection[6] = float(inv_map[int(detection[6])])
        image_indices_set.add(detection[0])
        detections.append(np.array(detection))
    detections = np.array(detections)
    image_indices = list(image_indices_set)
    image_indices.sort(reverse=True)

    # map indices to coco image id's
    cocoGt = COCO(annotation_file_path)
    cocoDt = cocoGt.loadRes(detections)
    cocoEval = COCOeval(cocoGt, cocoDt, iouType='bbox')
    cocoEval.params.imgIds = image_indices
    cocoEval.evaluate()
    cocoEval.accumulate()
    cocoEval.summarize()
    print(f'Accuracy: mAP: {cocoEval.stats[1]}')


if __name__ == '__main__':
    main()
