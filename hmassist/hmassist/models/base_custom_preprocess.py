#!/usr/bin/env python3

import abc


class BaseCustomPreprocess(object, metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def __init__(self, input_infos: list, calib_num: int, data_dir: str):
        """
        :param input_infos: 模型输入信息
        :param calib_num:  校准数据量
        :param data_dir:  校准数据目录
        """
        pass

    @abc.abstractmethod
    def get_single_data(self, filepath, idx):
        """预处理指定路径图片
        :param filepath:  图片路径
        :param idx:  不支持输入的索引
        :return:
        """

    @abc.abstractmethod
    def get_data(self):
        """工具链内部调用
        不需要norm归一化，工具链会根据配置文件，内部进行norm
        """
        pass

