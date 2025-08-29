import os
import glob
from hmatc.utils import logger
from hmatc.base.base_dataset import BaseDataset


class CCPD2019SubDataSet(BaseDataset):
    def __init__(self, root_path):
        self.root_path = root_path
        if not os.path.exists(self.root_path):
            logger.error("root_path not exits -> {}".format(self.root_path))
            exit(-1)
        self.img_files = glob.glob(os.path.join(self.root_path, "*.jpg"))
        self.total_num = len(self.img_files)

    def get_datas(self, num: int):
        if num == 0:
            num = self.total_num
        elif num > self.total_num:
            num = self.total_num
        img_paths = self.img_files[0:num]
        return img_paths

    @property
    def dataset_name(self):
        return "CCPD2019Sub"

    def get_next_batch(self):
        pass
