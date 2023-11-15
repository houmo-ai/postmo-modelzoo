#!/usr/bin/env python  

import os
import json
import xml.etree.ElementTree as ET
from ..utils import logger
from .base_dataset import BaseDataset


class VOC2007(BaseDataset):
    def __init__(self, root_path, batch_size=1):
        self._dataset_name = "voc2007"
        self._root_path = root_path
        self._batch_size = batch_size
        self._img_files = list()
        self._label_files = list()
        self._image_ids = list()

        if not os.path.exists(self._root_path):
            logger.error("VOC dataset not exist -> {}".format(self._root_path))
            exit(-1)

        test_file = os.path.join(self._root_path, "VOC2007", "ImageSets", "Main", "test.txt")
        if not os.path.exists(test_file):
            logger.error("test file not exist -> {}".format(test_file))
            exit(-1)
        with open(test_file, "r") as f:
            lines = f.readlines()
            for line in lines:
                basename = line.strip()
                img_path = os.path.join(self._root_path, "VOC2007", "JPEGImages", "{}.jpg".format(basename))
                xml_path = os.path.join(self._root_path, "VOC2007", "Annotations", "{}.xml".format(basename))
                if not os.path.exists(img_path):
                    logger.warning("img_path not exist -> {}".format(img_path))
                    continue
                if not os.path.exists(xml_path):
                    logger.warning("xml_path not exist -> {}".format(xml_path))
                    continue
                self._img_files.append(img_path)
                self._label_files.append(xml_path)
                self._image_ids.append(int(basename))

        # voc xml to coco json
        self._annotations_json = os.path.join(self._root_path, "voc2007_gt.json")

        categories = [{"supercategory": "none", "id": self.class_map[key], "name": key} for key in self.class_map]

        gt = {"annotations": list(), "images": list(), "categories": categories, "type": "instances"}
        cnt = 0
        for label_file in self._label_files:
            objects, image_info = self._parse_xml(label_file)
            gt["images"].append(image_info)
            for _, obj in enumerate(objects):
                cnt += 1
                obj["id"] = cnt
            gt["annotations"].extend(objects)

        with open(self._annotations_json, "w") as f:
            json.dump(gt, f)
        self._total_num = len(self._img_files)

    @property
    def annotations_file(self):
        return self._annotations_json

    @property
    def image_ids(self):
        return self._image_ids

    def get_next_batch(self):
        """获取下一批数据
        """
        pass

    def get_datas(self, num: int):
        """截取部分数据
        :param num: 0 表示使用全部数据，否则按num截取，超出全部则按全部截取
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
        return self._dataset_name

    @property
    def class_map(self):
        return {"background": 0, "aeroplane": 1, "bicycle": 2, "bird": 3, "boat": 4,
                "bottle": 5, "bus": 6, "car": 7, "cat": 8, "chair": 9, "cow": 10,
                "diningtable": 11, "dog": 12, "horse": 13, "motorbike": 14, "person": 15,
                "pottedplant": 16, "sheep": 17, "sofa": 18, "train": 19, "tvmonitor": 20}

    def _parse_xml(self, filename):
        """ Parse a PASCAL VOC xml file """
        name = os.path.basename(filename)
        basename, _ = os.path.splitext(name)
        tree = ET.parse(filename)
        size = tree.find("size")
        width = int(size.findtext("width"))
        height = int(size.findtext("height"))
        image_info = {
            "file_name": filename.replace(".txt", ".jpg"),
            "height": height,
            "width": width,
            "id": int(basename)
        }
        objects = list()
        for obj in tree.findall("object"):
            bbox = obj.find("bndbox")
            x1 = int(bbox.find("xmin").text) - 1
            y1 = int(bbox.find("ymin").text) - 1
            x2 = int(bbox.find("xmax").text) - 1
            y2 = int(bbox.find("ymax").text) - 1
            w = x2 - x1 + 1
            h = y2 - y1 + 1
            obj_struct = {
                "area": w * h,
                "iscrowd": 0,
                "image_id": int(basename),
                "category_id": self.class_map[obj.find("name").text],
                "bbox": [x1, y1, w, h],
                "ignore": 0,
                'segmentation': []  # This script is not for segmentation
            }
            objects.append(obj_struct)

        return objects, image_info


# m = VOC2007("/home/intellif/workspaces/dp2000/DEngine/tymodelzoo/data/VOCdevkit")
# pass
