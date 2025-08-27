#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""

Author: Nan Xu
Maintainer: Nan Xu
Date: 2025/07/15
Company: houmo

"""

class Singleton(type):
    _instances = {}

    def __call__(cls, *args, **kwargs):
        if cls not in cls._instances:
            cls._instances[cls] = super(Singleton, cls).__call__(*args, **kwargs)
        return cls._instances[cls]