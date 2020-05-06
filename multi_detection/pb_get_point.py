# -*- coding: utf-8 -*-
# @Time    : 2020/4/30
# @Author  : sunyihuan
# @File    : pb_get_point.py


import tensorflow as tf
import os

model_name = 'E:/kx_detection/multi_detection/model/yolo_model.pb'


# 读取并创建一个图graph来存放Google训练好的模型（函数）
def create_graph():
    with tf.gfile.FastGFile(model_name, 'rb') as f:
        # 使用tf.GraphDef()定义一个空的Graph
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())
        # Imports the graph from graph_def into the current default Graph.
        tf.import_graph_def(graph_def, name='')


# 创建graph
# create_graph()
with tf.Session() as sess:
    with tf.gfile.FastGFile(model_name, 'rb') as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())
        # Imports the graph from graph_def into the current default Graph.
        tf.import_graph_def(graph_def, name='')
# print(tf.get_default_graph().as_graph_def().node)
print(tf.get_default_graph().as_graph_def().node)
tensor_name_list = [tensor for tensor in tf.get_default_graph().as_graph_def().node]
result_file = 'result0.txt'
with open(result_file, 'w+') as f:
    for tensor_name in tensor_name_list:
        print(tensor_name)
        f.write(str(tensor_name) + '\n')
