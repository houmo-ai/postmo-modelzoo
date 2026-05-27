# Copyright 2026 HOUMO AI
#
# File: model_impl.py
# Description:
#   DINOv3 base model implementation for image classification tasks.
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
import torch
import numpy as np
from tqdm import tqdm
from typing import Dict, Any
from hmatc.utils import logger
from hmatc.utils.postprocess import softmax
from hmatc.base.base_model import BaseModel
from hmatc.datasets.imagenet import ILSVRC2012_LABELS


class DinoV3(BaseModel):
    """
    DINOv3 base model implementation for image classification tasks.
    Inherits from BaseModel and provides postprocessing, demo and evaluation functionality.
    """

    def __init__(self, **kwargs):
        """
        Initialize DINOv3 model instance.

        Args:
            **kwargs: Additional keyword arguments passed to BaseModel constructor
        """
        super(DinoV3, self).__init__(**kwargs)
        self.input_name = self.inputs_name[0]
        # Load the linear classifier head for ImageNet classification
        self.classifier_head = torch.jit.load("./dinov3-vitb16-imagenet-linear-head.pt").eval()

    def postprocess(
        self, outs: Dict[str, np.ndarray], in_datas: Dict[str, np.ndarray]
    ) -> Any:
        """
        Postprocess the model outputs to get classification results.

        Applies softmax activation to the model outputs and finds the class with
        the highest probability for each sample in the batch.

        Args:
            outs: Dictionary containing model output tensors
            in_datas: Dictionary containing input data (not used in this implementation)

        Returns:
            List of tuples containing (class_index, confidence_score) for each sample in batch
        """
        output_name = list(outs.keys())[1]
        pooled_output = torch.from_numpy(outs[output_name]).cpu()

        # Run the pooled output through the linear classifier head to get logits
        logits = self.classifier_head(pooled_output).detach().numpy()

        out = softmax(logits, axis=1, keepdims=True)
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
        Run demo inference on a list of image file paths.

        Loads each image, runs inference through the model, and prints the predicted
        class index, name, and confidence score for each image.

        Args:
            filepaths: List of image file paths to run inference on
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
            # Only take batch 0
            out = outs[0]
            cls_idx = str(out[0])
            score = out[1]
            cls_name = ILSVRC2012_LABELS[cls_idx][0]
            logger.info(f"score: {score:.3f}, cls_idx: {cls_idx}, cls_name: {cls_name}")

    def evaluate(self, dataset, num=0):
        """
        Evaluate the model performance on a given dataset.

        Runs inference on all images in the dataset and calculates top-1 accuracy
        by comparing predictions with ground truth labels.

        Args:
            dataset: Dataset object containing images and labels for evaluation
            num: Number of samples to evaluate (0 means all samples)

        Returns:
            Dictionary containing evaluation metrics including input size, dataset name,
            number of samples, top-1 accuracy and latency
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
