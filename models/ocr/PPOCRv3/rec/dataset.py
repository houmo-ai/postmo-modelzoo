import os
import glob
from hmatc.utils import logger
from hmatc.base.base_dataset import BaseDataset

class CCPD2020DataSet(BaseDataset):
    def __init__(self, root_path):
        self.root_path = root_path
        if not os.path.exists(self.root_path):
            logger.error("root_path not exits -> {}".format(self.root_path))
            exit(-1)
        self.labels_file = os.path.join(self.root_path, "PPOCR/val/rec.txt")
        self.img_dir = os.path.join(self.root_path, "PPOCR")
        self.img_files = glob.glob(os.path.join(self.img_dir, "val/crop_imgs/*.jpg"))
        self.total_num = len(self.img_files)
        self.data_lines = self.get_image_info_list([self.labels_file])

    def get_datas(self, num: int):
        if num == 0:
            num = self.total_num
        elif num > self.total_num:
            num = self.total_num
        img_paths = self.img_files[0:num]
        return img_paths
    
    def get_image_info_list(self, file_list):
        if isinstance(file_list, str):
            file_list = [file_list]
        data_lines = []
        for idx, file in enumerate(file_list):
            with open(file, "rb") as f:
                lines = f.readlines()
                data_lines.extend(lines)
        return data_lines
    
    @property
    def dataset_name(self):
        return "CCPD2020Val"
    
    def get_next_batch(self):
        pass