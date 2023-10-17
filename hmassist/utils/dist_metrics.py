#!/usr/bin/env python3

import numpy as np


def cosine_distance(data1, data2):
    """余弦距离
    :param data1:
    :param data2:
    :return:
    """
    v1_d = data1.flatten().astype("float64")
    v2_d = data2.flatten().astype("float64")
    assert len(v1_d) == len(v2_d), "v1 dim must be == v2 dim"
    v1_d[v1_d == np.inf] = np.finfo(np.float16).max
    v2_d[v2_d == np.inf] = np.finfo(np.float16).max
    v1_d[v1_d == -np.inf] = np.finfo(np.float16).min
    v2_d[v2_d == -np.inf] = np.finfo(np.float16).min
    v1_norm = v1_d / np.linalg.norm(v1_d)
    v2_norm = v2_d / np.linalg.norm(v2_d)
    return np.dot(v1_norm, v2_norm)