# -*- coding: utf-8 -*-
# @Time    : 2021/2/19
# @Author  : sunyihuan
# @File    : ckpt2lite.py
import os
import tensorflow as tf
MODEL_SAVE_PATH="E:/ckpt_dirs/Food_detection/multi_food5/20210218"

def pb_to_tflite(input_name, output_name):
    graph_def_file = os.path.join(MODEL_SAVE_PATH, 'yolo_model (7).pb')
    input_arrays = [input_name]
    output_arrays = [output_name]

    converter = tf.lite.TFLiteConverter.from_frozen_graph(graph_def_file, input_arrays, output_arrays)
    tflite_model = converter.convert()
    tflite_file = os.path.join(MODEL_SAVE_PATH, 'tflite_model', 'converted_model.tflite')
    open(tflite_file, "wb").write(tflite_model)
output = ["define_loss/pred_sbbox/concat_2", "define_loss/pred_mbbox/concat_2",
          "define_loss/pred_lbbox/concat_2", "define_loss/layer_classes"]
pb_to_tflite(output)