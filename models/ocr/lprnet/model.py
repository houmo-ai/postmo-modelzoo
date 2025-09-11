import os
import cv2
import time
import numpy as np
from tqdm import tqdm
from hmatc.utils import logger
from hmatc.base.base_model import BaseModel


class LPRNet(BaseModel):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.input_name = self.inputs_name[0]
        _, C, H, W = self.inputs_cfg[self.input_name]["shape"]
        self.input_size = (H, W)
        self.CHARS = [
            "京",
            "沪",
            "津",
            "渝",
            "冀",
            "晋",
            "蒙",
            "辽",
            "吉",
            "黑",
            "苏",
            "浙",
            "皖",
            "闽",
            "赣",
            "鲁",
            "豫",
            "鄂",
            "湘",
            "粤",
            "桂",
            "琼",
            "川",
            "贵",
            "云",
            "藏",
            "陕",
            "甘",
            "青",
            "宁",
            "新",
            "0",
            "1",
            "2",
            "3",
            "4",
            "5",
            "6",
            "7",
            "8",
            "9",
            "A",
            "B",
            "C",
            "D",
            "E",
            "F",
            "G",
            "H",
            "J",
            "K",
            "L",
            "M",
            "N",
            "P",
            "Q",
            "R",
            "S",
            "T",
            "U",
            "V",
            "W",
            "X",
            "Y",
            "Z",
            "I",
            "O",
            "-",
        ]

        self.CHARS_DICT = {char: i for i, char in enumerate(self.CHARS)}

    def get_label(self, file_name):
        base_name = os.path.basename(file_name)
        imgname, _ = os.path.splitext(base_name)
        imgname = imgname.split("-")[0].split("_")[0]
        label = list()
        for c in imgname:
            label.append(self.CHARS_DICT[c])
        return label, [len(label)]

    def postprocess(self, outs, in_datas):
        prebs = outs[next(iter(outs))]
        preb_labels = list()
        for i in range(prebs.shape[0]):
            preb = prebs[i, :, :]
            preb_label = list()
            for j in range(preb.shape[1]):
                preb_label.append(np.argmax(preb[:, j], axis=0))
            no_repeat_blank_label = list()
            pre_c = preb_label[0]
            if pre_c != len(self.CHARS) - 1:
                no_repeat_blank_label.append(pre_c)
            for c in preb_label:
                if (pre_c == c) or (c == len(self.CHARS) - 1):
                    if c == len(self.CHARS) - 1:
                        pre_c = c
                    continue
                no_repeat_blank_label.append(c)
                pre_c = c
            preb_labels.append(no_repeat_blank_label)
        return preb_labels

    def demo(self, filepaths: list):
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

            plate_strs_list = []
            dump_str = ""
            for i, pred in enumerate(preds):
                plate_str = ""
                for p in pred:
                    plate_str += self.CHARS[p]
                plate_strs_list.append(plate_str)
                dump_str += f"{plate_str}, "
            logger.info(
                f"image => {file_name} have {len(plate_strs_list)} license plates, numbers: {dump_str}"
            )

    def evaluate(self, dataset, num=0):
        img_path_list = dataset.get_datas(num)
        Tp = 0
        Tn_1 = 0
        Tn_2 = 0
        t1 = time.time()
        pbar = tqdm(total=len(img_path_list), desc="eval:", position=0, leave=True)
        in_datas = dict()
        for idx, img_file in enumerate(img_path_list):
            labels, lengths = self.get_label(img_file)
            cv_image = cv2.imread(img_file)
            if cv_image is None:
                logger.warning(f"{img_file} not exists or decode failed")
                continue
            in_datas[self.input_name] = cv_image
            logger.debug(f"Image[{idx}] {img_file}")

            preb_labels = self.run(in_datas)

            start = 0
            targets = []
            for length in lengths:
                label = labels[start : start + length]
                targets.append(label)
                start += length
            targets = np.array([el for el in targets])

            for i, label in enumerate(preb_labels):
                if len(label) != len(targets[i]):
                    Tn_1 += 1
                    continue
                if (np.asarray(targets[i]) == np.asarray(label)).all():
                    Tp += 1
                else:
                    Tn_2 += 1
            pbar.update(1)
        pbar.close()
        Acc = Tp * 1.0 / (Tp + Tn_1 + Tn_2)
        logger.info(f"Test Accuracy: {Acc} [{Tp}:{Tn_1}:{Tn_2}:{Tp+Tn_1+Tn_2}]")
        t2 = time.time()
        logger.info(
            f"Test Speed: {(t2 - t1) / len(img_path_list)}s 1/{len(img_path_list)}"
        )
        return {
            "input_size": self.inputs_cfg[self.input_name]["shape"],
            "dataset": dataset.dataset_name,
            "num": Tp + Tn_1 + Tn_2,
            "acc": f"{Acc:.6f}",
            "latency": f"{self.ave_latency_ms:.6f}",
        }
