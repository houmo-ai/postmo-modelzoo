import os
import cv2
import json
import sys
import torch
from typing import Dict, Any
import numpy as np
import importlib.util

if not importlib.util.find_spec("rapidfuzz"):
    os.system("pip install rapidfuzz -i https://pypi.tuna.tsinghua.edu.cn/simple")
from rapidfuzz.distance import Levenshtein

import re

from tqdm import tqdm
from hmatc.utils import logger
from hmatc.infer import onnx_infer
from hmatc.utils.preprocess import convert_bgr_to_yuv

SCRIP_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(SCRIP_DIR)
from base_utils import *
from dataset import *

class BaseRecLabelDecode(object):
    """ Convert between text-label and text-index """

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
                    line = line.decode('utf-8').strip("\n").strip("\r\n")
                    self.character_str.append(line)
            if use_space_char:
                self.character_str.append(" ")
            dict_character = list(self.character_str)
            if 'arabic' in character_dict_path:
                self.reverse = True

        dict_character = self.add_special_char(dict_character)
        self.dict = {}
        for i, char in enumerate(dict_character):
            self.dict[char] = i
        self.character = dict_character

    def pred_reverse(self, pred):
        pred_re = []
        c_current = ''
        for c in pred:
            if not bool(re.search('[a-zA-Z0-9 :*./%+-]', c)):
                if c_current != '':
                    pred_re.append(c_current)
                pred_re.append(c)
                c_current = ''
            else:
                c_current += c
        if c_current != '':
            pred_re.append(c_current)

        return ''.join(pred_re[::-1])

    def add_special_char(self, dict_character):
        return dict_character

    def get_word_info(self, text, selection):
        state = None
        word_content = []
        word_col_content = []
        word_list = []
        word_col_list = []
        state_list = []
        valid_col = np.where(selection==True)[0]

        for c_i, char in enumerate(text):
            if '\u4e00' <= char <= '\u9fff':
                c_state = 'cn'
            elif bool(re.search('[a-zA-Z0-9]', char)):
                c_state = 'en&num'
            else:
                c_state = 'splitter'
            
            if char == '.' and state == 'en&num' and c_i + 1 < len(text) and bool(re.search('[0-9]', text[c_i+1])): # grouping floting number
                c_state = 'en&num'
            if char == '-' and state == "en&num": # grouping word with '-', such as 'state-of-the-art'
                c_state = 'en&num'
            
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

    def decode(self,
               text_index,
               text_prob=None,
               is_remove_duplicate=False,
               return_word_box=False):
        """ convert text-index into text-label. """
        result_list = []
        ignored_tokens = self.get_ignored_tokens()
        batch_size = len(text_index)
        for batch_idx in range(batch_size):
            selection = np.ones(len(text_index[batch_idx]), dtype=bool)
            if is_remove_duplicate:
                selection[1:] = text_index[batch_idx][1:] != text_index[
                    batch_idx][:-1]
            for ignored_token in ignored_tokens:
                selection &= text_index[batch_idx] != ignored_token

            char_list = [
                self.character[text_id]
                for text_id in text_index[batch_idx][selection]
            ]
            if text_prob is not None:
                conf_list = text_prob[batch_idx][selection]
            else:
                conf_list = [1] * len(selection)
            if len(conf_list) == 0:
                conf_list = [0]

            text = ''.join(char_list)

            if self.reverse:  # for arabic rec
                text = self.pred_reverse(text)

            if return_word_box:
                word_list, word_col_list, state_list = self.get_word_info(
                    text, selection)
                result_list.append((text, np.mean(conf_list).tolist(), [
                    len(text_index[batch_idx]), word_list, word_col_list,
                    state_list
                ]))
            else:
                result_list.append((text, np.mean(conf_list).tolist()))
        return result_list

    def get_ignored_tokens(self):
        return [0]  # for ctc blank


class CTCLabelDecode(BaseRecLabelDecode):
    """ Convert between text-label and text-index """

    def __init__(self, character_dict_path=None, use_space_char=False,
                 **kwargs):
        super(CTCLabelDecode, self).__init__(character_dict_path,
                                             use_space_char)

    def __call__(self,
                 preds,
                 return_word_box=False,
                 **kwargs):
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
            return_word_box=return_word_box)
        if return_word_box:
            for rec_idx, rec in enumerate(text):
                wh_ratio = kwargs['wh_ratio_list'][rec_idx]
                max_wh_ratio = kwargs['max_wh_ratio']
                rec[2][0] = rec[2][0] * (wh_ratio / max_wh_ratio)
        return text

    def add_special_char(self, dict_character):
        dict_character = ['blank'] + dict_character
        return dict_character

class BaseRecLabelEncode(object):
    """ Convert between text-label and text-index """

    def __init__(self,
                 max_text_length,
                 character_dict_path=None,
                 use_space_char=False,
                 lower=False):

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
                    line = line.decode('utf-8').strip("\n").strip("\r\n")
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
                # logger = get_logger()
                # logger.warning('{} is not in dict'.format(char))
                continue
            text_list.append(self.dict[char])
        if len(text_list) == 0:
            return None
        return text_list


class CTCLabelEncode(BaseRecLabelEncode):
    """ Convert between text-label and text-index """

    def __init__(self,
                 max_text_length,
                 character_dict_path=None,
                 use_space_char=False,
                 **kwargs):
        super(CTCLabelEncode, self).__init__(
            max_text_length, character_dict_path, use_space_char)

    def __call__(self, data):
        text = data['label']
        text = self.encode(text)
        if text is None:
            return None
        data['length'] = np.array(len(text))
        text = text + [0] * (self.max_text_len - len(text))
        data['label'] = np.array(text)

        label = [0] * len(self.character)
        for x in text:
            label[x] += 1
        data['label_ace'] = np.array(label)
        return data

    def add_special_char(self, dict_character):
        dict_character = ['blank'] + dict_character
        return dict_character

class OCRRec(object):
    def __init__(self, model_path):
        self.model_path = model_path
        self.model_ext = os.path.splitext(self.model_path)[1]
        if self.model_ext == ".onnx":
            self.module = onnx_infer.OnnxInfer()
            self.backend = "onnx"
        elif self.model_ext == ".hmm" and HOUMO_TARGET == "xh1":
            from hmatc.infer import xh1_infer
            self.module = xh1_infer.Xh1Infer()
            self.backend = "xh1"
        elif self.model_ext == ".hmm" and HOUMO_TARGET == "xh2":
            from hmatc.infer import xh2_infer
            self.module = xh2_infer.Xh2Infer()
            self.backend = "xh2"

        self.ctc_encode = CTCLabelEncode(25, os.path.join(SCRIP_DIR, "ppocr_keys_v1.txt"), True)
        self.ctc_decode = CTCLabelDecode(os.path.join(SCRIP_DIR, "ppocr_keys_v1.txt"), True)
        
        self.net_input_size = [48, 320]
        self.input_names = []
        self.output_names = []
        self.input_infos = {}
        self.output_infos = {}
        self.load()

        self.ave_latency_ms = 0.
        self.total_latency_time = 0.

        self.correct_num = 0
        self.all_num = 0
        self.norm_edit_dis = 0
        self.eps = 1e-5

    def load(self):
        self.module.load(self.model_path)
        if self.backend == "onnx":
            inputs_list, outputs_list = get_net_input_output_infos(self.model_path)
            for input in inputs_list:
                input_name = input['name']
                self.input_names.append(input_name)
                input.pop('name')
                self.input_infos[input_name] = input
            for output in outputs_list:
                output_name = output['name']
                self.output_names.append(output_name)
                output.pop('name')
                self.output_infos[output_name] = output
        else:
            input_num = self.module.engine.get_num_inputs()
            for idx in range(input_num):
                input_name = self.module.engine.get_input_name(idx)
                self.input_names.append(input_name)
                self.input_infos[input_name] = self.module.engine.get_input_info(input_name)
            output_num = self.module.engine.get_num_outputs()
            for idx in range(output_num):
                output_name = self.module.engine.get_output_name(idx)
                self.output_names.append(output_name)
                self.output_infos[output_name] = self.module.engine.get_output_info(output_name)


    def preprocess(self, in_datas: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        cv_image = in_datas[self.input_names[0]]
        dst_data = dict()
        if self.backend == "onnx":
            dst_data[self.input_names[0]] = onnx_preprocess(cv_image, self.net_input_size)
        elif self.backend == "xh1":
            data_fmt = self.input_infos[self.input_names[0]].format.name
            input_tensor, dyn_info = xh_preprocess(cv_image, self.net_input_size)
            yuv_pad_hwc = convert_bgr_to_yuv(input_tensor, fmt=data_fmt)

            yuv_pad = yuv_pad_hwc.detach().cpu().numpy().flatten()
            if data_fmt == "YUV420SP":
                valid_len = yuv_pad.size // 2
            elif data_fmt == "YUV422SP":
                valid_len = yuv_pad.size * 2 // 3
            elif data_fmt in ["YUV444SP", "YUV400"]:
                valid_len = yuv_pad.size
            else:
                logger.error(f"Unsupported format: {data_fmt}!")
                assert(0)
            yuv = yuv_pad[:valid_len]
            yuv = yuv.reshape(1, -1)

            dst_data[self.input_names[0]] = np.ascontiguousarray(yuv)  # np.ndarray
            dst_data[self.input_names[1]] = dyn_info.detach().cpu().numpy()
        elif self.backend == "xh2":
            input_data = onnx_preprocess(cv_image, self.net_input_size)
            input_data_fp16 = input_data.astype(np.float16)
            dst_data[self.input_names[0]] = input_data_fp16
        return dst_data
    
    def run(self, in_datas: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        """模型推理"""
        import time
        prerpcessed_in_datas = self.preprocess(in_datas)
        # 推理
        start_time = time.time()
        outs = self.module.run(prerpcessed_in_datas)
        self.total_latency_time += (time.time() - start_time)
        # xh1同时输出量化和反量化结果，只取反量化后的
        if isinstance(outs, tuple):
            outs = outs[1]
        outs = self.postprocess(outs, in_datas)
        return outs

    def process_label(self, label):
        ctc = self.ctc_encode.__call__({'label': label})
        if ctc is None:
            logger.warning(f"Label gtc or ctc is None!")
            return ctc
        return np.stack((ctc['label'],), axis=0)
    
    def postprocess(self, outs, in_datas):
        net_out = outs[next(iter(outs))]
        if isinstance(net_out, torch.Tensor):
            net_out = net_out.numpy()
        pred_text = self.ctc_decode(net_out)
        return pred_text
    
    def rec_metric(self, preds, labels):
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
        acc = 1.0 * self.correct_num / (self.all_num + self.eps)
        norm_edit_dis = 1 - self.norm_edit_dis / (self.all_num + self.eps)
        return {'acc': acc, 'norm_edit_dis': norm_edit_dis}
    
    def demo(self, filepaths: list):
        in_datas = dict()
        for idx, filepath in enumerate(filepaths):
            file_name = os.path.basename(filepath)
            cv_image = cv2.imread(filepath)
            if cv_image is None:
                logger.warning(f'{filepath} not exists or decode failed')
                continue
            in_datas[self.input_names[0]] = cv_image
            logger.info(f'Image[{idx}] {filepath}')
            
            preds = self.run(in_datas)

            plate_strs_list = [[pred, conf] for pred, conf in preds]
            logger.info(
                f"image => {file_name} have {len(plate_strs_list)} license plates, numbers: {plate_strs_list}"
            )        

    def evaluate(self, dataset: CCPD2020DataSet, num=0):
        img_path_list = dataset.get_datas(num)
        pbar = tqdm(total=len(img_path_list),
                    desc="eval:",
                    position=0,
                    leave=True)
        efficent_num = 0
        for data_line in dataset.data_lines:
            data_line = data_line.decode('utf-8')
            substr = data_line.strip("\n").split("\t")
            file_name = substr[0]
            label = substr[1]
            img_path = os.path.join(dataset.img_dir, file_name)
            if img_path not in img_path_list:
                continue
            cv_image = cv2.imread(img_path)
            if cv_image is None:
                logger.warning(f'{img_path} not exists or decode failed')
                continue
            in_datas = {self.input_names[0]: cv_image}

            label_data = self.process_label(label)
    
            preds = self.run(in_datas)
            gts = self.ctc_decode.decode(label_data)

            self.rec_metric(preds, gts)
            pbar.update(1)
            efficent_num += 1
        self.ave_latency_ms = self.total_latency_time / efficent_num
        
        metric = self.get_metric()
        pbar.close()

        logger.info("metric eval ***********")
        for k, v in metric.items():
            logger.info(f"{k}:{v}")
        return {
            "input_size": [1, 3] + self.net_input_size,
            "dataset": dataset.dataset_name,
            "num": efficent_num,
            "hmean": f"{metric['acc']:.6f}",
            "latency": f"{self.ave_latency_ms * 1000:.6f}",
        }


