# -*- encoding: utf-8 -*-

"""
pb文件转为pbtxt

@File    : pb2pbtxt.py
@Time    : 202003/1/9 13:08
@Author  : sunyihuan
"""
import tensorflow as tf

graph_filename = "E:/kx_detection/multi_detection/model/yolo_model.pb"

graph = tf.Graph()
with graph.as_default():
    output_graph_def = tf.GraphDef()
    with open(graph_filename, "rb") as f:
        output_graph_def.ParseFromString(f.read())
        _ = tf.import_graph_def(output_graph_def, name="")

    sess = tf.Session()
    tf.train.write_graph(sess.graph_def, 'E:/kx_detection/multi_detection/model', 'yolo_model.pbtxt', True)
# with tf.gfile.GFile(graph_filename, "rb") as f:
#     graph_def = tf.GraphDef()
#     graph_def.ParseFromString(f.read())
#     tf.train.write_graph(graph_def, 'E:/kx_detection/multi_detection/model', 'yolo_model_r.pbtxt', True)
