#!/usr/bin/env python3

import os
from .base_dataset import BaseDataset
from ..utils import logger


class COCO2017Val(BaseDataset):
    """提供图片path和label
    """
    def __init__(self, root_path, batch_size=1):
        self._root_path = root_path
        self._batch_size = batch_size
        if not os.path.exists(self._root_path):
            logger.error("root_path not exits -> {}".format(self._root_path))
            exit(-1)

        self._annotations_file = os.path.join(self._root_path, "..", "annotations", "instances_val2017.json")
        self._annotations_kpt = os.path.join(self._root_path, "..", "annotations", "person_keypoints_val2017.json")
        if not os.path.exists(self._annotations_file):
            logger.error("annotations_file not exist -> {}".format(self._annotations_file))
            exit(-1)

        self._label_files = list()
        self._img_files = list()
        self._image_ids = list()

        self._filepath = os.path.join(self._root_path, "..", "val2017.txt")
        if os.path.exists(self._filepath):
            with open(self._filepath, "r") as f:
                lines = f.readlines()
                for line in lines:
                    sub_path = line.strip()
                    basename = os.path.basename(sub_path)
                    filename, ext = os.path.splitext(basename)
                    img_path = os.path.join(self._root_path, "..", sub_path)
                    if not os.path.exists(img_path):
                        logger.warning("img_path not exist -> {}".format(img_path))
                        continue
                    self._img_files.append(img_path)
                    self._image_ids.append(int(filename))
        elif os.path.isdir(self._root_path):
            logger.warning("filepath not exist -> {}, using {} files directly".format(self._filepath, self._root_path))
            for filepath in os.listdir(self._root_path):
                basename = os.path.basename(filepath)
                filename, ext = os.path.splitext(basename)
                if ext in [".jpg", ".JPEG", ".bmp", ".png", ".jpeg", ".BMP"]:
                    self._img_files.append(os.path.join(self._root_path, basename))
                    self._image_ids.append(int(filename))
        
        self._total_num = len(self._img_files)
        if (self._total_num < 5000):
            logger.warning("test number is less than 5000, using incomplete validation sets may result in inaccurate results.")

    @property
    def annotations_file(self):
        return self._annotations_file

    @property
    def annotations_kpt(self):
        return self._annotations_kpt

    @property
    def image_ids(self):
        return self._image_ids

    def get_next_batch(self):
        """获取下一批数据
        """
        pass

    def get_datas(self, num: int):
        """截取部分数据
        :param num: 0表示使用全部数据，否则按num截取，超出全部则按全部截取
        :return:
        """
        if num == 0:
            num = self._total_num
        elif num > self._total_num:
            num = self._total_num

        img_paths = self._img_files[0:num]
        # labels = self._labels[0:num]
        return img_paths

    @property
    def dataset_name(self):
        return "coco2017"


class COCO2014Val(BaseDataset):
    """提供图片path和label
    """
    # TODO
    pass
