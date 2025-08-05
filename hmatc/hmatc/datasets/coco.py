#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os
import json
from ..base.base_dataset import BaseDataset
from ..utils import logger


class COCO2017Val(BaseDataset):
    """提供图片path和label
    """
    def __init__(self, root_path):
        self.root_path = root_path
        if not os.path.exists(self.root_path):
            logger.error("root_path not exits -> {}".format(self.root_path))
            exit(-1)

        self.annotations_file = os.path.join(self.root_path, "annotations", "instances_val2017.json")
        self.annotations_kpt = os.path.join(self.root_path, "annotations", "person_keypoints_val2017.json")
        if not os.path.exists(self.annotations_file):
            logger.error(f"annotations_file not exist -> {self.annotations_file}")
            exit(-1)

        with open(self.annotations_file, "r") as f:
            annotations = json.load(f)
        images = annotations["images"]
        
        self.img_files = list()
        self.image_ids = list()
        self.image_ids_dict = dict()
        for image in images:
            filename = image["file_name"]
            image_id = int(image["id"])
            img_path = os.path.join(self.root_path, "val2017", filename)
            if not os.path.exists(img_path):
                # logger.warning(f"img_path not exist -> {img_path}")
                continue
            basename, _ = os.path.splitext(os.path.basename(img_path))
            self.image_ids_dict[basename] = image_id
            self.image_ids.append(image_id)
            self.img_files.append(img_path)

        self.total_num = len(self.img_files)

    def get_image_id(self, filename):
        return self.image_ids_dict[filename]
    
    def get_next_batch(self):
        pass

    def get_datas(self, num: int):
        if num == 0:
            num = self.total_num
        elif num > self.total_num:
            num = self.total_num
        img_paths = self.img_files[0:num]
        return img_paths

    @property
    def dataset_name(self):
        return "coco_2017Val"

