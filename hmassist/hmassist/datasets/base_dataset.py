#!/usr/bin/env python3

import abc


class BaseDataset(object, metaclass=abc.ABCMeta):
    """提供图片path和label
    """
    @abc.abstractmethod
    def __init__(self, root_path, batch_size=1):
        self._dataset_name = ""

    @abc.abstractmethod
    def get_next_batch(self):
        """获取下一批数据
        """
        pass

    @abc.abstractmethod
    def get_datas(self, num: int):
        """截取部分数据
        :param num: 0表示使用全部数据，否则按num截取，超出全部则按全部截取
        :return:
        """
        pass

    @property
    @abc.abstractmethod
    def dataset_name(self):
        return self._dataset_name
