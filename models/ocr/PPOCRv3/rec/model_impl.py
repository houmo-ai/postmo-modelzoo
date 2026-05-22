#!/usr/bin/python3
# -*- coding: utf-8 -*-
# Copyright 2025 HOUMO AI
#
# File: model.py
# Description:
#   ppocrv3 recognition model implementation on HOUMO AI device.
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
import cv2
import torch
import numpy as np
import re

from tqdm import tqdm
from rapidfuzz.distance import Levenshtein
from hmatc.utils import logger
from hmatc.base.base_model import BaseModel


class BaseRecLabelDecode(object):
    """Convert between text-label and text-index"""

    def __init__(self, character_dict_path=None, use_space_char=False):
        self.beg_str = "sos"
        self.end_str = "eos"
        self.reverse = False
        self.character_str = []

        if character_dict_path is None:
            self.character_str = "0123456789abcdefghijklmnopqrstuvwxyz"
            dict_character = list(self.character_str)
        else:
            with open(character_dict_path, "rb") as fin:
                lines = fin.readlines()
                for line in lines:
                    line = line.decode("utf-8").strip("\n").strip("\r\n")
                    self.character_str.append(line)
            if use_space_char:
                self.character_str.append(" ")
            dict_character = list(self.character_str)
            if "arabic" in character_dict_path:
                self.reverse = True

        dict_character = self.add_special_char(dict_character)
        self.dict = {}
        for i, char in enumerate(dict_character):
            self.dict[char] = i
        self.character = dict_character

    def pred_reverse(self, pred):
        pred_re = []
        c_current = ""
        for c in pred:
            if not bool(re.search("[a-zA-Z0-9 :*./%+-]", c)):
                if c_current != "":
                    pred_re.append(c_current)
                pred_re.append(c)
                c_current = ""
            else:
                c_current += c
        if c_current != "":
            pred_re.append(c_current)

        return "".join(pred_re[::-1])

    def add_special_char(self, dict_character):
        return dict_character

    def get_word_info(self, text, selection):
        state = None
        word_content = []
        word_col_content = []
        word_list = []
        word_col_list = []
        state_list = []
        valid_col = np.where(selection == True)[0]

        for c_i, char in enumerate(text):
            if "\u4e00" <= char <= "\u9fff":
                c_state = "cn"
            elif bool(re.search("[a-zA-Z0-9]", char)):
                c_state = "en&num"
            else:
                c_state = "splitter"

            if (
                char == "."
                and state == "en&num"
                and c_i + 1 < len(text)
                and bool(re.search("[0-9]", text[c_i + 1]))
            ):
                c_state = "en&num"
            if char == "-" and state == "en&num":
                c_state = "en&num"

            if state == None:
                state = c_state

            if state != c_state:
                if len(word_content) != 0:
                    word_list.append(word_content)
                    word_col_list.append(word_col_content)
                    state_list.append(state)
                    word_content = []
                    word_col_content = []
                state = c_state

            if state != "splitter":
                word_content.append(char)
                word_col_content.append(valid_col[c_i])

        if len(word_content) != 0:
            word_list.append(word_content)
            word_col_list.append(word_col_content)
            state_list.append(state)

        return word_list, word_col_list, state_list

    def decode(
        self,
        text_index,
        text_prob=None,
        is_remove_duplicate=False,
        return_word_box=False,
    ):
        """convert text-index into text-label."""
        result_list = []
        ignored_tokens = self.get_ignored_tokens()
        batch_size = len(text_index)
        for batch_idx in range(batch_size):
            selection = np.ones(len(text_index[batch_idx]), dtype=bool)
            if is_remove_duplicate:
                selection[1:] = text_index[batch_idx][1:] != text_index[batch_idx][:-1]
            for ignored_token in ignored_tokens:
                selection &= text_index[batch_idx] != ignored_token

            char_list = [
                self.character[text_id] for text_id in text_index[batch_idx][selection]
            ]
            if text_prob is not None:
                conf_list = text_prob[batch_idx][selection]
            else:
                conf_list = [1] * len(selection)
            if len(conf_list) == 0:
                conf_list = [0]

            text = "".join(char_list)

            if self.reverse:
                text = self.pred_reverse(text)

            if return_word_box:
                word_list, word_col_list, state_list = self.get_word_info(
                    text, selection
                )
                result_list.append(
                    (
                        text,
                        np.mean(conf_list).tolist(),
                        [
                            len(text_index[batch_idx]),
                            word_list,
                            word_col_list,
                            state_list,
                        ],
                    )
                )
            else:
                result_list.append((text, np.mean(conf_list).tolist()))
        return result_list

    def get_ignored_tokens(self):
        return [0]


class CTCLabelDecode(BaseRecLabelDecode):
    """Convert between text-label and text-index"""

    def __init__(self, character_dict_path=None, use_space_char=False, **kwargs):
        super(CTCLabelDecode, self).__init__(character_dict_path, use_space_char)

    def __call__(self, preds, return_word_box=False, **kwargs):
        if isinstance(preds, tuple) or isinstance(preds, list):
            preds = preds[-1]
        if isinstance(preds, torch.Tensor):
            preds = preds.numpy()
        preds_idx = preds.argmax(axis=2)
        preds_prob = preds.max(axis=2)
        text = self.decode(
            preds_idx,
            preds_prob,
            is_remove_duplicate=True,
            return_word_box=return_word_box,
        )
        if return_word_box:
            for rec_idx, rec in enumerate(text):
                wh_ratio = kwargs["wh_ratio_list"][rec_idx]
                max_wh_ratio = kwargs["max_wh_ratio"]
                rec[2][0] = rec[2][0] * (wh_ratio / max_wh_ratio)
        return text

    def add_special_char(self, dict_character):
        dict_character = ["blank"] + dict_character
        return dict_character


class BaseRecLabelEncode(object):
    """Convert between text-label and text-index"""

    def __init__(
        self,
        max_text_length,
        character_dict_path=None,
        use_space_char=False,
        lower=False,
    ):
        self.max_text_len = max_text_length
        self.beg_str = "sos"
        self.end_str = "eos"
        self.lower = lower

        if character_dict_path is None:
            logger.warning(
                "The character_dict_path is None, model can only recognize number and lower letters"
            )
            self.character_str = "0123456789abcdefghijklmnopqrstuvwxyz"
            dict_character = list(self.character_str)
            self.lower = True
        else:
            self.character_str = []
            with open(character_dict_path, "rb") as fin:
                lines = fin.readlines()
                for line in lines:
                    line = line.decode("utf-8").strip("\n").strip("\r\n")
                    self.character_str.append(line)
            if use_space_char:
                self.character_str.append(" ")
            dict_character = list(self.character_str)
        dict_character = self.add_special_char(dict_character)
        self.dict = {}
        for i, char in enumerate(dict_character):
            self.dict[char] = i
        self.character = dict_character

    def add_special_char(self, dict_character):
        return dict_character

    def encode(self, text):
        if len(text) == 0 or len(text) > self.max_text_len:
            return None
        if self.lower:
            text = text.lower()
        text_list = []
        for char in text:
            if char not in self.dict:
                continue
            text_list.append(self.dict[char])
        if len(text_list) == 0:
            return None
        return text_list


class CTCLabelEncode(BaseRecLabelEncode):
    """Convert between text-label and text-index"""

    def __init__(
        self, max_text_length, character_dict_path=None, use_space_char=False, **kwargs
    ):
        super(CTCLabelEncode, self).__init__(
            max_text_length, character_dict_path, use_space_char
        )

    def __call__(self, data):
        text = data["label"]
        text = self.encode(text)
        if text is None:
            return None
        data["length"] = np.array(len(text))
        text = text + [0] * (self.max_text_len - len(text))
        data["label"] = np.array(text)

        label = [0] * len(self.character)
        for x in text:
            label[x] += 1
        data["label_ace"] = np.array(label)
        return data

    def add_special_char(self, dict_character):
        dict_character = ["blank"] + dict_character
        return dict_character


class OCRRec(BaseModel):
    """PPOCRv3 recognition model implementation."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.input_name = self.inputs_name[0]
        _, C, H, W = self.inputs_cfg[self.input_name]["shape"]
        self.input_size = (H, W)

        # CTC decoder
        script_dir = os.path.dirname(os.path.abspath(__file__))
        char_dict_path = os.path.join(script_dir, "ppocr_keys_v1.txt")
        self.ctc_decode = CTCLabelDecode(char_dict_path, True)
        self.ctc_encode = CTCLabelEncode(25, char_dict_path, True)

        # Metric tracking
        self.correct_num = 0
        self.all_num = 0
        self.norm_edit_dis = 0
        self.eps = 1e-5

    def postprocess(self, outs, in_datas):
        """CTC decoding for recognition results."""
        net_out = outs[next(iter(outs))]
        if isinstance(net_out, torch.Tensor):
            net_out = net_out.numpy()
        pred_text = self.ctc_decode(net_out)
        return pred_text

    def process_label(self, label):
        """Encode label for evaluation."""
        ctc = self.ctc_encode.__call__({"label": label})
        if ctc is None:
            logger.warning(f"Label ctc is None!")
            return None
        return np.stack((ctc["label"],), axis=0)

    def rec_metric(self, preds, labels):
        """Calculate recognition metrics."""
        correct_num = 0
        all_num = 0
        norm_edit_dis = 0.0
        for (pred, _), (target, _) in zip(preds, labels):
            pred = pred.replace(" ", "")
            target = target.replace(" ", "")
            norm_edit_dis += Levenshtein.normalized_distance(pred, target)
            if pred == target:
                correct_num += 1
            all_num += 1
        self.correct_num += correct_num
        self.all_num += all_num
        self.norm_edit_dis += norm_edit_dis

    def get_metric(self):
        """Get accumulated metrics."""
        acc = 1.0 * self.correct_num / (self.all_num + self.eps)
        norm_edit_dis = 1 - self.norm_edit_dis / (self.all_num + self.eps)
        return {"acc": acc, "norm_edit_dis": norm_edit_dis}

    def demo(self, filepaths: list):
        """Run model demonstration."""
        save_path = f"./vis_{self.backend}"
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        in_datas = dict()
        for idx, filepath in enumerate(filepaths):
            file_name = os.path.basename(filepath)
            cv_image = cv2.imread(filepath)
            if cv_image is None:
                logger.warning(f"{filepath} not exists or decode failed")
                continue
            in_datas[self.input_name] = cv_image
            logger.info(f"Image[{idx}] {filepath}")

            preds = self.run(in_datas)
            plate_strs_list = [[pred, conf] for pred, conf in preds]
            logger.info(f"image => {file_name} recognition result: {plate_strs_list}")

            # Draw result on image
            text = preds[0][0] if preds else ""
            cv2.putText(
                cv_image, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2
            )
            save_dir = os.path.join(save_path, file_name)
            cv2.imwrite(save_dir, cv_image)
            logger.info(f"Save results to {save_dir}.")

    def evaluate(self, dataset, num=0):
        """Evaluate the model on dataset."""
        img_path_list = dataset.get_datas(num)
        # Use actual number of samples from data_lines, not glob scan result
        actual_num = len(dataset.data_lines)
        pbar = tqdm(total=actual_num, desc="eval:", position=0, leave=True)
        efficient_num = 0
        for data_line in dataset.data_lines:
            data_line = data_line.decode("utf-8")
            substr = data_line.strip("\n").split("\t")
            file_name = substr[0]
            label = substr[1]
            img_path = os.path.join(dataset.img_dir, file_name)
            if img_path not in img_path_list:
                continue
            cv_image = cv2.imread(img_path)
            if cv_image is None:
                logger.warning(f"{img_path} not exists or decode failed")
                continue
            in_datas = {self.input_name: cv_image}

            label_data = self.process_label(label)
            if label_data is None:
                continue

            preds = self.run(in_datas)
            gts = self.ctc_decode.decode(label_data)

            self.rec_metric(preds, gts)
            pbar.update(1)
            efficient_num += 1

        # ave_latency_ms is a computed property from time_span and total
        metric = self.get_metric()
        pbar.close()

        logger.info("metric eval ***********")
        for k, v in metric.items():
            logger.info(f"{k}:{v}")
        return {
            "input_size": [1, 3] + list(self.input_size),
            "dataset": dataset.dataset_name,
            "num": efficient_num,
            "acc": f"{metric['acc']:.6f}",
            "norm_edit_dis": f"{metric['norm_edit_dis']:.6f}",
            "latency": f"{self.ave_latency_ms:.6f}",
        }
