# -*- encoding: utf-8 -*-

"""
@File    : read_ckpt_tensor.py
@Time    : 2019/10/22 16:28
@Author  : sunyihuan
"""
import numpy as np
from tensorflow.python import pywrap_tensorflow

checkpoint_path = "E:/ckpt_dirs/Food_detection/multi_food/20190923/yolov3_train_loss=4.9217.ckpt-220"

# read data from checkpoint file
reader = pywrap_tensorflow.NewCheckpointReader(checkpoint_path)
var_to_shape_map = reader.get_variable_to_shape_map()

data_print = np.array([])
for key in var_to_shape_map:
    print('tensor_name', key)
#     ckpt_data = np.array(reader.get_tensor(key))  # cast list to np arrary
#     ckpt_data = ckpt_data.flatten()  # flatten list
#     data_print = np.append(data_print, ckpt_data, axis=0)
#
# print(data_print, data_print.shape, np.max(data_print), np.min(data_print), np.mean(data_print))
