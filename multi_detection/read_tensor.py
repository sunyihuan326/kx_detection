# -*- encoding: utf-8 -*-

"""
读取模型文件中tensor的name
pb、ckpt文件均可

@File    : read_tensor.py
@Time    : 2019/10/22 16:28
@Author  : sunyihuan
"""
import numpy as np
from tensorflow.python import pywrap_tensorflow
import tensorflow as tf


def read_tensor_name(model_path, typ):
    '''
    读取tensor的名字
    :param model_path: 模型文件路径
    :param typ: 类型，pb或ckpt
    :return:
    '''
    assert typ in ["pb", "ckpt"]
    key_name = []
    if typ == "ckpt":
        reader = pywrap_tensorflow.NewCheckpointReader(model_path)
        var_to_shape_map = reader.get_variable_to_shape_map()
        for key in var_to_shape_map:
            if "darknet/conv0/weight" ==key:
                print(key)
                w=reader.get_tensor(key)
                print(w.shape)
                # print(sum(w))
                print(reader.get_tensor(key))
            if "darknet/conv1/weight" ==key:
                print(key)
                w=reader.get_tensor(key)
                print(w.shape)
                # print(sum(w))
                print(reader.get_tensor(key))
            key_name.append(key)
    else:
        with tf.Session() as sess:
            tf.global_variables_initializer().run()
            output_graph_def = tf.GraphDef()
            with open(model_path, "rb") as f:
                output_graph_def.ParseFromString(f.read())
                tf.import_graph_def(output_graph_def, name="")
            constant_ops = [op for op in sess.graph.get_operations() if op.type == "Const"]
            for constant_op in constant_ops:
                key_name.append(constant_op.name)
    return key_name


ckpt_path =  "E:/ckpt_dirs/Food_detection/multi_food3/20200604_22class/yolov3_train_loss=4.9799.ckpt-158"
pb_path = "E:/multi_yolov3_predict-20191220/checkpoint/yolov3_1220.pb"
key_name = read_tensor_name(ckpt_path, "ckpt")
# print(key_name)
