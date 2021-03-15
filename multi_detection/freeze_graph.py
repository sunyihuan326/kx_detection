# -*- encoding: utf-8 -*-

"""
ckpt文件转成pb文件

@File    : freeze_graph.py
@Time    : 2019/12/21 11:43
@Author  : sunyihuan
"""

import tensorflow as tf
from multi_detection.core.yolov3 import YOLOV3

pb_file = "E:/kx_detection/multi_detection/model/yolov3.pb"
ckpt_file = "E:/kx_detection/multi_detection/checkpoint/yolov3_train_loss=19.8911.ckpt-13"
output = ["define_loss/pred_sbbox/concat_2", "define_loss/pred_mbbox/concat_2",
          "define_loss/pred_lbbox/concat_2", "define_loss/layer_classes"]

with tf.name_scope('define_input'):  # 输出
    input_data = tf.placeholder(dtype=tf.float32, shape=(None, 320, 320, 3), name='input_data')
    # trainable = tf.placeholder(dtype=tf.bool, name='training')

with tf.name_scope("define_loss"):  # 输出
    model = YOLOV3(input_data, trainable=False)

sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True))
saver = tf.train.Saver()
saver.restore(sess, ckpt_file)
# 生成pb文件
converted_graph_def = tf.graph_util.convert_variables_to_constants(sess,
                                                                   input_graph_def=sess.graph.as_graph_def(),
                                                                   output_node_names=output)

with tf.gfile.GFile(pb_file, "wb") as f:
    f.write(converted_graph_def.SerializeToString())
    print("%d ops in the final graph." % len(converted_graph_def.node))  # 得到当前图有几个操作节点



