# -*- encoding: utf-8 -*-

"""
量化压缩，未完成！！！！！！！

@File    : tflite_quantization.py
@Time    : 202003/1/9 9:32
@Author  : sunyihuan
"""
import matplotlib
from multi_detection.core.yolov3 import YOLOV3
import tensorflow as tf

pb_file = "E:/kx_detection/multi_detection/model/yolov3.pb"
input_array = ["define_input/input_data"]
output = ["define_loss/pred_sbbox/concat_2", "define_loss/pred_mbbox/concat_2",
          "define_loss/pred_lbbox/concat_2", "define_loss/layer_classes"]
with tf.name_scope('define_input'):
    input_data = tf.placeholder(dtype=tf.float32, shape=(None, 320, 320, 3), name='input_data')
    # trainable = tf.placeholder(dtype=tf.bool, name='training')

with tf.name_scope("define_loss"):
    model = YOLOV3(input_data, trainable=False)

sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True))

converter = tf.lite.TFLiteConverter.from_session(sess, [input_data],
                                                 [model.pred_sbbox, model.pred_mbbox, model.pred_lbbox,
                                                  model.predict_op])
converter.post_training_quantize = True
convert_model = converter.convert()
open("yolo3.tflite", "wb").write(convert_model)
