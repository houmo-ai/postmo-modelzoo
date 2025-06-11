#!/usr/bin/env python3

import numpy as np
from .utils import logger

def cosine_distance(data1, data2, check_shape=True):
    """
    余弦距离
    """
    if check_shape:
        if data1.shape != data2.shape:
            logger.error("shape not equal {} vs {}".format(data1.shape, data2.shape))
            return -1
    v1_d = data1.flatten().astype("float64")
    v2_d = data2.flatten().astype("float64")
    if len(v1_d) != len(v2_d):
        logger.error("v1 dim {} != v2 dim {}".format(len(v1_d), len(v2_d)))
        return -1
    v1_d[v1_d == np.inf] = np.finfo(np.float16).max
    v2_d[v2_d == np.inf] = np.finfo(np.float16).max
    v1_d[v1_d == -np.inf] = np.finfo(np.float16).min
    v2_d[v2_d == -np.inf] = np.finfo(np.float16).min
    v1_norm = v1_d / np.linalg.norm(v1_d)
    v2_norm = v2_d / np.linalg.norm(v2_d)
    cosine_dist = np.dot(v1_norm, v2_norm)
    if np.isnan(cosine_dist):
        return -1
    return float(cosine_dist)


def euclid_distance(data1, data2, check_shape=True):
    """
    欧氏距离
    """
    if check_shape:
        if data1.shape != data2.shape:
            logger.error("shape not equal {} vs {}".format(data1.shape, data2.shape))
            return -1
    return np.linalg.norm(data1 - data2)


def relative_euclidean_distance(x, y):
    return float(abs((np.sum(x**2) / np.sum(y**2) + np.finfo(np.float64).eps) - 1.0))