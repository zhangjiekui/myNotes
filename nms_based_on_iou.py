# -*- coding: utf-8 -*-
# @Time : 2021/10/12 11:30
# @Author : zjk
# @File : nms_based_on_iou.py
# @Project : pythonProject
# @IDE :PyCharm

# import numpy as np
# np.set_printoptions(suppress=True) # turn off scientific notation
# import os
# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
# import tensorflow as tf
# import torch
import numpy as np


def cpu_nms(dets, thresh):
    x1 = np.ascontiguousarray(dets[:, 0])
    y1 = np.ascontiguousarray(dets[:, 1])
    x2 = np.ascontiguousarray(dets[:, 2])
    y2 = np.ascontiguousarray(dets[:, 3])

    areas = (x2 - x1) * (y2 - y1)
    order = dets[:, 4].argsort()[::-1]
    keep = list()

    while order.size > 0:
        pick_ind = order[0]
        keep.append(pick_ind)

        xx1 = np.maximum(x1[pick_ind], x1[order[1:]])
        yy1 = np.maximum(y1[pick_ind], y1[order[1:]])
        xx2 = np.minimum(x2[pick_ind], x2[order[1:]])
        yy2 = np.minimum(y2[pick_ind], y2[order[1:]])

        inter = np.maximum(0.0, xx2 - xx1) * np.maximum(0.0, yy2 - yy1)
        iou = inter / (areas[pick_ind] + areas[order[1:]] - inter)

        order = order[np.where(iou <= thresh)[0] + 1]

    return keep

dets = np.array([
    [11.5, 12.0, 311.4, 410.6, 0.85],
    [0.5, 1.0, 300.4, 400.5, 0.97],
    [200.5, 300.0, 700.4, 1000.6, 0.65],
    [250.5, 310.0, 700.4, 1000.6, 0.72],
])
np.set_printoptions(suppress=True)
print("before nms:\n", dets)
keep = cpu_nms(dets, 0.5)
print("after nms:\n", dets[keep])
