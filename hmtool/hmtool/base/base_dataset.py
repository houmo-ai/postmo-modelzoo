#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import abc


class BaseDataset(object, metaclass=abc.ABCMeta):
    """提供图片path和label
    """
    def __init__(self, **kwargs):
        """传入数据集目录"""
        pass

    @abc.abstractmethod
    def get_next_batch(self):
        """获取下一批数据"""
        pass

    @abc.abstractmethod
    def get_datas(self, num: int):
        """截取部分数据"""
        pass

    @property
    @abc.abstractmethod
    def dataset_name(self):
        return "BaseDataset"
