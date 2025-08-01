#!/usr/bin/env python  
# -*- coding:utf-8 _*-
import os
from ..base.base_dataset import BaseDataset
from ..utils import logger


class WiderFace(BaseDataset):
    def __init__(self, root_path):
        self._root_path = root_path
        if not os.path.exists(self._root_path):
            logger.error(f"root_path not exits -> {self._root_path}")
            exit(-1)

        self._list_file = os.path.join(self._root_path, "val", "wider_val.txt")
        if not os.path.exists(self._list_file):
            logger.error(f"wider_val.txt not exits -> {self._list_file}")
            exit(-1)

        self._dataset_val_path = os.path.join(self._root_path, "val", "images")
        if not os.path.exists(self._list_file):
            logger.error(f"image val not exits -> {self._dataset_val_path}")
            exit(-1)

        self._img_lists = list()
        self._img_relative_path = list()
        with open(self._list_file, "r") as f:
            lines = f.readlines()
            for line in lines:
                subpath = line.strip()
                img_path = self._dataset_val_path + subpath
                if not os.path.exists(img_path):
                    logger.warning(f"img_path not exist -> {img_path}")
                    continue
                self._img_lists.append(img_path)
                self._img_relative_path.append(subpath)
        self._total_num = len(self._img_lists)

        self._annotation_path = os.path.join(self._root_path, "ground_truth")
        if not os.path.exists(self._annotation_path):
            logger.error(f"annotation_path not exits -> {self._annotation_path}")
            exit(-1)

    def get_next_batch(self):
        """获取下一批数据
        """
        pass

    def get_datas(self, num: int):
        if num == 0:
            num = self._total_num
        elif num > self._total_num:
            num = self._total_num

        img_paths = self._img_lists[0:num]
        return img_paths

    def get_relative_path(self, idx):
        return self._img_relative_path[idx]

    @property
    def annotation_path(self):
        return self._annotation_path

    @property
    def dataset_name(self):
        return "widerface"
