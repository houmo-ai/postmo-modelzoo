# Copyright 2025 HOUMO AI
#
# File: metrics.py
# Description:
#   Metrics for model evaluation
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

import json
import os
import cv2
import traceback
import numpy as np
from . import logger


def smooth(y, f=0.05):
    """
    Apply box filter smoothing to input array.

    Args:
        y (np.ndarray): Input array to be smoothed
        f (float): Fraction of the array length for filter size, default is 0.05

    Returns:
        np.ndarray: Smoothed array
    """
    # Box filter of fraction f
    nf = round(len(y) * f * 2) // 2 + 1  # number of filter elements (must be odd)
    p = np.ones(nf // 2)  # ones padding
    yp = np.concatenate((p * y[0], y, p * y[-1]), 0)  # y padded
    return np.convolve(yp, np.ones(nf) / nf, mode="valid")  # y-smoothed


def compute_ap(recall, precision):
    """
    Compute the average precision, given the recall and precision curves.

    Args:
        recall (list or np.ndarray): The recall curve
        precision (list or np.ndarray): The precision curve

    Returns:
        tuple: (Average precision, precision curve, recall curve)
    """
    # Append sentinel values to beginning and end
    mrec = np.concatenate(([0.0], recall, [1.0]))
    mpre = np.concatenate(([1.0], precision, [0.0]))

    # Compute the precision envelope
    mpre = np.flip(np.maximum.accumulate(np.flip(mpre)))

    # Integrate area under curve
    method = "interp"  # methods: "continuous", "interp"
    if method == "interp":
        x = np.linspace(0, 1, 101)  # 101-point interp (COCO)
        ap = np.trapz(np.interp(x, mrec, mpre), x)  # integrate
    else:  # "continuous"
        i = np.where(mrec[1:] != mrec[:-1])[0]  # points where x axis (recall) changes
        ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])  # area under curve

    return ap, mpre, mrec


def ap_per_class(tp, conf, pred_cls, target_cls, eps=1e-16):
    """
    Compute the average precision for each class, given true positives, confidence,
    predicted and target classes.

    Args:
        tp (np.ndarray): True positives (nx1 or nx10)
        conf (np.ndarray): Objectness value from 0-1
        pred_cls (np.ndarray): Predicted object classes
        target_cls (np.ndarray): True object classes
        eps (float): Small epsilon value to prevent division by zero, default is 1e-16

    Returns:
        tuple: (tp, fp, p, r, f1, ap, unique_classes)
            - tp: True positives per class
            - fp: False positives per class
            - p: Precision per class
            - r: Recall per class
            - f1: F1 score per class
            - ap: Average precision per class
            - unique_classes: Unique class indices
    """
    # Sort by objectness
    i = np.argsort(-conf)
    tp, conf, pred_cls = tp[i], conf[i], pred_cls[i]

    # Find unique classes
    unique_classes, nt = np.unique(target_cls, return_counts=True)
    nc = unique_classes.shape[0]  # number of classes, number of detections

    # Create Precision-Recall curve and compute AP for each class
    px, py = np.linspace(0, 1, 1000), []  # for plotting
    ap, p, r = np.zeros((nc, tp.shape[1])), np.zeros((nc, 1000)), np.zeros((nc, 1000))
    for ci, c in enumerate(unique_classes):
        i = pred_cls == c
        n_l = nt[ci]  # number of labels
        n_p = i.sum()  # number of predictions
        if n_p == 0 or n_l == 0:
            continue

        # Accumulate FPs and TPs
        fpc = (1 - tp[i]).cumsum(0)
        tpc = tp[i].cumsum(0)

        # Recall
        recall = tpc / (n_l + eps)  # recall curve
        r[ci] = np.interp(
            -px, -conf[i], recall[:, 0], left=0
        )  # negative x, xp because xp decreases

        # Precision
        precision = tpc / (tpc + fpc)  # precision curve
        p[ci] = np.interp(-px, -conf[i], precision[:, 0], left=1)  # p at pr_score

        # AP from recall-precision curve
        for j in range(tp.shape[1]):
            ap[ci, j], mpre, mrec = compute_ap(recall[:, j], precision[:, j])

    # Compute F1 (harmonic mean of precision and recall)
    f1 = 2 * p * r / (p + r + eps)
    i = smooth(f1.mean(0), 0.1).argmax()  # max F1 index
    p, r, f1 = p[:, i], r[:, i], f1[:, i]
    tp = (r * nt).round()  # true positives
    fp = (tp / (p + eps) - tp).round()  # false positives
    return tp, fp, p, r, f1, ap, unique_classes.astype(int)


def coco80_to_coco91_class():  # converts 80-index (val2014) to 91-index (paper)
    """
    Convert COCO 80-class index to COCO 91-class index.
    This mapping is used to convert the COCO dataset class indices from the 80-class format
    used in validation sets (val2014) to the 91-class format described in the original paper.

    Returns:
        list: Mapping from 80-class indices to 91-class indices
    """
    # https://tech.amikelive.com/node-718/what-object-categories-labels-are-in-coco-dataset/
    return [
        1,
        2,
        3,
        4,
        5,
        6,
        7,
        8,
        9,
        10,
        11,
        13,
        14,
        15,
        16,
        17,
        18,
        19,
        20,
        21,
        22,
        23,
        24,
        25,
        27,
        28,
        31,
        32,
        33,
        34,
        35,
        36,
        37,
        38,
        39,
        40,
        41,
        42,
        43,
        44,
        46,
        47,
        48,
        49,
        50,
        51,
        52,
        53,
        54,
        55,
        56,
        57,
        58,
        59,
        60,
        61,
        62,
        63,
        64,
        65,
        67,
        70,
        72,
        73,
        74,
        75,
        76,
        77,
        78,
        79,
        80,
        81,
        82,
        84,
        85,
        86,
        87,
        88,
        89,
        90,
    ]


def detections2txt(detections, filepath):
    """
    Write detection results to a text file.

    Args:
        detections (list): List of detections in format [x1, y1, x2, y2, conf, cls]
        filepath (str): Path to save the text file
    """
    with open(filepath, "w") as f:
        for det in detections:
            (x1, y1, x2, y2), conf, cls = det[0:4], det[4], det[5]
            text = "{} {} {} {} {} {}\n".format(conf, cls, x1, y1, x2, y2)
            f.write(text)


def detections_face2txt(detections, filepath):
    """
    Write face detection results to a text file in a specific format.

    Args:
        detections (list): List of face detections in format [x1, y1, x2, y2, conf]
        filepath (str): Path to save the text file
    """
    with open(filepath, "w") as f:
        file_name = os.path.basename(filepath)[:-4] + "\n"
        bboxs_num = str(len(detections)) + "\n"
        f.write(file_name)
        f.write(bboxs_num)
        for det in detections:
            f.write(
                "%d %d %d %d %.03f"
                % (det[0], det[1], det[2], det[3], det[4] if det[4] <= 1 else 1)
                + "\n"
            )


def detections_mask2json(detections, contours_lists: list, filepath):
    """
    Convert detection results with masks to COCO JSON format.

    Args:
        detections (list): List of detections in format [x1, y1, x2, y2, conf, cls]
        contours_lists (list): List of contours for segmentation masks
        filepath (str): Path to save the JSON file
    """
    with open(filepath, "w") as f:
        if not contours_lists:
            return
        filename = os.path.basename(filepath)
        name, ext = os.path.splitext(filename)
        image_id = int(name)
        pred_lists = list()
        for idx, det in enumerate(detections):
            (x1, y1, x2, y2), conf, cls = det[0:4], det[4], det[5]
            x1 = int(x1)
            y1 = int(y1)
            x2 = int(x2)
            y2 = int(y2)
            conf = float(conf)
            w = x2 - x1 + 1
            h = y2 - y1 + 1
            cls = int(cls)
            category_id = coco80_to_coco91_class()[cls]
            contours = contours_lists[idx]
            new_contours = list()
            area = 0
            for _, contour in enumerate(contours):
                if contour.shape[0] <= 2:
                    continue
                area += cv2.contourArea(contour)
                new_contour = contour.flatten().tolist()
                if len(new_contour) == 4:
                    new_contour.append(new_contour[-1])
                new_contours.append(new_contour)
            if len(new_contours) == 0:
                continue
            pred_lists.append(
                {
                    "image_id": image_id,
                    "category_id": category_id,
                    "bbox": [x1, y1, w, h],
                    "score": conf,
                    "segmentation": new_contours,
                    "area": area,
                    "iscrowd": 0,
                }
            )
        f.write(json.dumps(pred_lists))


def detections_kpt2json(outputs, filepath):
    """
    Convert keypoint detection results to COCO JSON format.

    Args:
        outputs (list): List of keypoint detections in format [x1, y1, x2, y2, conf, cls, kpt1_x, kpt1_y, kpt1_conf, ...]
        filepath (str): Path to save the JSON file
    """
    with open(filepath, "w") as f:
        filename = os.path.basename(filepath)
        name, ext = os.path.splitext(filename)
        image_id = int(name)
        pred_lists = list()
        for output in outputs:
            cls = int(output[5])
            conf = float(output[4])
            x1, y1, x2, y2 = output[0:4].tolist()
            w = x2 - x1 + 1
            h = y2 - y1 + 1
            kpts = output[6:].tolist()
            for k in range(len(kpts)):
                if (k + 1) % 3 == 0:
                    kpts[k] = 1
            category_id = coco80_to_coco91_class()[cls]
            pred_lists.append(
                {
                    "image_id": image_id,
                    "category_id": category_id,
                    "bbox": [x1, y1, w, h],
                    "score": conf,
                    "keypoints": kpts,
                    "iscrowd": 0,
                }
            )
        f.write(json.dumps(pred_lists))


def merge_json(save_results, pred_json):
    """
    Merge multiple JSON files containing detection results into a single JSON file.

    Args:
        save_results (str): Directory containing JSON result files
        pred_json (str): Path to save the merged JSON file
    """
    label_files = os.listdir(save_results)
    results = list()
    for filename in label_files:
        name, ext = os.path.splitext(filename)
        if ext != ".json":
            continue
        with open(os.path.join(save_results, filename), "r") as f:
            line = f.read().strip()
            if len(line) == 0:
                continue
            detections = json.loads(line)
            if not detections:
                continue
            results.extend(detections)
    with open(pred_json, "w") as f:
        json.dump(results, f)


def detection_txt2json(save_results, pred_json, to_coco91=True):
    """
    Convert detection results from text format to COCO JSON format.

    Args:
        save_results (str): Directory containing text result files
        pred_json (str): Path to save the JSON file
        to_coco91 (bool): Whether to convert to COCO 91-class format, default is True
    """
    label_files = os.listdir(save_results)
    pred_list = list()
    for filename in label_files:
        name, ext = os.path.splitext(filename)
        if ext != ".txt":
            continue
        image_id = int(name)
        with open(os.path.join(save_results, filename), "r") as f:
            lines = f.readlines()
            for line in lines:
                conf, cls, x1, y1, x2, y2 = line.strip().split()
                conf = float(conf)
                cls = int(float(cls))
                x1 = int(float(x1))
                y1 = int(float(y1))
                x2 = int(float(x2))
                y2 = int(float(y2))
                w = x2 - x1 + 1
                h = y2 - y1 + 1
                category_id = coco80_to_coco91_class()[cls] if to_coco91 else cls
                pred_list.append(
                    {
                        "image_id": image_id,
                        "category_id": category_id,
                        "bbox": [x1, y1, w, h],
                        "score": conf,
                    }
                )
    with open(pred_json, "w") as f:
        json.dump(pred_list, f)
    logger.info("Write pred results to json file -> {}".format(pred_json))


def coco_eval(pred_json, anno_json, image_ids, iou_type="bbox"):
    """
    Perform COCO evaluation using pycocotools.

    Args:
        pred_json (str): Path to prediction JSON file
        anno_json (str): Path to annotation JSON file
        image_ids (list): List of image IDs to evaluate
        iou_type (str): Type of IoU evaluation ('bbox', 'segm', 'keypoints'), default is 'bbox'

    Returns:
        tuple: (mAP@0.5:0.95, mAP@0.5)
    """
    logger.info("Evaluating pycocotools mAP... saving {}...".format(pred_json))
    try:  # https://github.com/cocodataset/cocoapi/blob/master/PythonAPI/pycocoEvalDemo.ipynb
        from pycocotools.coco import COCO
        from pycocotools.cocoeval import COCOeval

        cocoGt = COCO(anno_json)  # init annotations api
        pred = cocoGt.loadRes(pred_json)  # init predictions api
        eval = COCOeval(cocoGt, pred, iou_type)
        eval.params.imgIds = image_ids  # cocoGt.getImgIds()  # image IDs to evaluate
        eval.evaluate()
        eval.accumulate()
        eval.summarize()
        _map, map50 = eval.stats[:2]  # update results (mAP@0.5:0.95, mAP@0.5)
        return _map, map50
    except Exception as e:
        logger.error(
            "pycocotools unable to run: {}\n{}".format(e, traceback.format_exc())
        )
        exit(-1)


class StreamSegMetrics(object):
    """
    Stream Metrics for Semantic Segmentation Task.
    Provides evaluation metrics for semantic segmentation including overall accuracy,
    mean accuracy, mean IoU, and frequency weighted accuracy.
    """

    def __init__(self, n_classes):
        """
        Initialize the metrics calculator.

        Args:
            n_classes (int): Number of classes in the segmentation task
        """
        self.n_classes = n_classes
        self.confusion_matrix = np.zeros((n_classes, n_classes))

    def update(self, label_trues, label_preds):
        """
        Update confusion matrix with new batch of predictions.

        Args:
            label_trues (np.ndarray): Ground truth labels
            label_preds (np.ndarray): Predicted labels
        """
        for lt, lp in zip(label_trues, label_preds):
            self.confusion_matrix += self._fast_hist(lt.flatten(), lp.flatten())

    @staticmethod
    def to_str(results):
        """
        Convert results dictionary to string format.

        Args:
            results (dict): Dictionary containing evaluation metrics

        Returns:
            str: Formatted string representation of the results
        """
        string = "\n"
        for k, v in results.items():
            if k != "Class IoU":
                string += "%s: %f\n" % (k, v)

        # string+='Class IoU:\n'
        # for k, v in results['Class IoU'].items():
        #    string += "\tclass %d: %f\n"%(k, v)
        return string

    def _fast_hist(self, label_true, label_pred):
        """
        Calculate histogram for a single image.

        Args:
            label_true (np.ndarray): True labels
            label_pred (np.ndarray): Predicted labels

        Returns:
            np.ndarray: Confusion matrix histogram
        """
        mask = (label_true >= 0) & (label_true < self.n_classes)
        hist = np.bincount(
            self.n_classes * label_true[mask].astype(int) + label_pred[mask],
            minlength=self.n_classes**2,
        ).reshape(self.n_classes, self.n_classes)
        return hist

    def get_results(self):
        """
        Calculate and return evaluation metrics.

        Returns:
            dict: Dictionary containing Overall Acc, Mean Acc, FreqW Acc, Mean IoU, and Class IoU
        """
        hist = self.confusion_matrix
        acc = np.diag(hist).sum() / hist.sum()
        acc_cls = np.diag(hist) / hist.sum(axis=1)
        acc_cls = np.nanmean(acc_cls)
        iu = np.diag(hist) / (hist.sum(axis=1) + hist.sum(axis=0) - np.diag(hist))
        mean_iu = np.nanmean(iu)
        freq = hist.sum(axis=1) / hist.sum()
        fwavacc = (freq[freq > 0] * iu[freq > 0]).sum()
        cls_iu = dict(zip(range(self.n_classes), iu))

        return {
            "Overall Acc": acc,
            "Mean Acc": acc_cls,
            "FreqW Acc": fwavacc,
            "Mean IoU": mean_iu,
            "Class IoU": cls_iu,
        }

    def reset(self):
        """
        Reset the confusion matrix to zeros.
        """
        self.confusion_matrix = np.zeros((self.n_classes, self.n_classes))
