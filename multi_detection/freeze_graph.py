#! /usr/bin/env python
# coding=utf-8



import tensorflow as tf
from multi_detection.core.yolov3 import YOLOV3

pb_file = "./yolov3_coco.pb"
ckpt_file = "E:/ckpt_dirs/Food_detection/tf_yolov3/20190819/yolov3_train_loss=7.1672.ckpt-148"
output_node_names = ["input/input_data", "pred_sbbox/concat_2", "pred_mbbox/concat_2", "pred_lbbox/concat_2"]

with tf.name_scope('input'):
    input_data = tf.placeholder(dtype=tf.float32, name='input_data')

model = YOLOV3(input_data, trainable=False)
print(model.pred_sbbox, model.pred_mbbox, model.pred_lbbox)

sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True))
saver = tf.train.Saver()
saver.restore(sess, ckpt_file)

converted_graph_def = tf.graph_util.convert_variables_to_constants(sess,
                                                                   input_graph_def=sess.graph.as_graph_def(),
                                                                   output_node_names=output_node_names)

with tf.gfile.GFile(pb_file, "wb") as f:
    f.write(converted_graph_def.SerializeToString())
    print("%d ops in the final graph." % len(converted_graph_def.node))  # 得到当前图有几个操作节点
