"""

Author: Nan.xu
Maintainer: Nan.xu
Date: 2025/07/30
Company: Houmo

"""
from ..onnxUtils.generalClassUtils import Singleton
from ..onnxUtils.commonUtils import *


class OnnxCfg(metaclass=Singleton):
    cfg={}

    @staticmethod
    def set_cfg(cfg):
        OnnxCfg.cfg=cfg

    @staticmethod
    def get_cfg():
        return OnnxCfg.cfg

    @staticmethod
    def get_val(key, default_val=None):
        if key in OnnxCfg.cfg.keys():
            return OnnxCfg.cfg[key]
        else:
            OnnxCfg.cfg[key] = default_val
            return default_val

    @staticmethod
    def get_sub_val(first_key,second_key,default_val=None):
        if default_val is not None:
            val_tmp=get_dict_default(OnnxCfg.cfg[first_key],second_key,default_val)
            if val_tmp==default_val:
                OnnxCfg.cfg[first_key]=val_tmp
            return val_tmp
        else:
            return OnnxCfg.cfg[first_key][second_key]

    @staticmethod
    def set_val(key,val):
        OnnxCfg.cfg[key]=val

    @staticmethod
    def check_exist(key):
        return key in OnnxCfg.cfg.keys() and OnnxCfg.cfg[key]