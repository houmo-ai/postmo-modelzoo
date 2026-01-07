# Copyright 2025 HOUMO AI
#
# File: model_impl.py
# Description:
#   YOLOv8 Classification model implementation.
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
import cv2
import numpy as np
from tqdm import tqdm
from typing import Dict, Any
from hmatc.utils import logger
from hmatc.utils.postprocess import softmax
from hmatc.base.base_model import BaseModel
from hmatc.datasets.imagenet import ILSVRC2012_LABELS


class YoloV8Cls(BaseModel):
    """
    YOLOv8 Classification model implementation.

    This class implements the YOLOv8 model for image classification tasks,
    inheriting from the BaseModel class. It provides functionality for
    postprocessing, model demonstration, and evaluation.
    """

    def __init__(self, **kwargs):
        """
        Initializes the YoloV8Cls model.

        Args:
            **kwargs: Arbitrary keyword arguments passed to the parent BaseModel class.
        """
        super().__init__(**kwargs)
        self.input_name = self.inputs_name[0]

    def postprocess(
        self, outs: Dict[str, np.ndarray], in_datas: Dict[str, np.ndarray]
    ) -> Any:
        """
        Postprocesses the model outputs to get classification results.

        Applies softmax to the output and returns the class index with the highest score
        along with its confidence score for each sample in the batch.

        Args:
            outs: Dictionary containing model outputs
            in_datas: Dictionary containing input data

        Returns:
            List of tuples containing (class_index, score) for each sample in the batch
        """
        output_name = list(outs.keys())[0]
        out = softmax(outs[output_name], axis=1, keepdims=True)
        max_idxes = np.argmax(out, axis=1, keepdims=True)
        batch = max_idxes.shape[0]
        res = list()
        for i in range(batch):
            max_idx = max_idxes[i][0]
            max_score = out[i][max_idx]
            res.append((max_idx, max_score))  # (cls_idx, score)
        return res

    def demo(self, filepaths: list):
        """
        Runs the model on a list of image files for demonstration purposes.

        For each image file, runs the model and prints the predicted class index,
        class name, and confidence score.

        Args:
            filepaths: List of file paths to images for demonstration
        """
        in_datas = dict()
        for idx, filepath in enumerate(filepaths):
            cv_image = cv2.imread(filepath)
            if cv_image is None:
                logger.warning(f"{filepath} not exists or decode failed")
                continue
            in_datas[self.input_name] = cv_image
            logger.info(f"[{idx}] {filepath}")
            outs = self.run(in_datas)
            # 只需取batch0
            out = outs[0]
            cls_idx = str(out[0])
            score = out[1]
            cls_name = ILSVRC2012_LABELS[cls_idx][0]
            logger.info(f"score: {score:.3f}, cls_idx: {cls_idx}, cls_name: {cls_name}")

    def evaluate(self, dataset, num=0):
        """
        Evaluates the model performance on a given dataset.

        Calculates the top-1 accuracy by comparing model predictions with ground truth labels.

        Args:
            dataset: Dataset object containing images and labels
            num: Number of samples to evaluate (0 means all samples)

        Returns:
            Dictionary containing evaluation metrics including input size, dataset name,
            number of samples, top-1 accuracy, and latency
        """
        img_paths, labels = dataset.get_datas(num)
        in_datas = dict()
        top1_acc = 0
        for idx, img_path in enumerate(tqdm(img_paths)):
            cv_image = cv2.imread(img_path)
            if cv_image is None:
                logger.warning(f"{img_path} not exists or decode failed")
                continue
            in_datas[self.input_name] = cv_image
            logger.debug(f"[{idx}] {img_path}")
            outs = self.run(in_datas)
            out = outs[0]
            cls_idx = str(out[0])
            gt_idx = str(labels[idx])
            if cls_idx == gt_idx:
                top1_acc += 1
        return {
            "input_size": self.inputs_cfg[self.input_name]["shape"],
            "dataset": dataset.dataset_name,
            "num": len(img_paths),
            "top1_acc": f"{top1_acc / len(img_paths):.6f}",
            "latency_ms": f"{self.ave_latency_ms:.6f}",
        }
