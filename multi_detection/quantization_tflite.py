# -*- encoding: utf-8 -*-

"""
@File    : tflite_quantization.py
@Time    : 2020/1/9 9:32
@Author  : sunyihuan
"""
import matplotlib

matplotlib.use('TkAgg')

import tensorflow as tf


def main():
    # 指定要使用的模型的路径  包含图结构，以及参数

    graph_def_file = 'E:/kx_detection/multi_detection/model'

    # 重新定义一个图

    converter = tf.lite.TFLiteConverter.from_saved_model(graph_def_file)
    converter.optimizations = [tf.lite.Optimize.OPTIMIZE_FOR_SIZE]
    tflite_quant_model = converter.convert()
    open('E:/kx_detection/multi_detection/model', "wb").write(tflite_quant_model)

    # output_graph_def = tf.GraphDef()
    #
    # with tf.gfile.GFile(graph_def_file, 'rb')as fid:
    #     # 将*.pb文件读入serialized_graph
    #
    #     serialized_graph = fid.read()
    #
    #     # 将serialized_graph的内容恢复到图中
    #
    #     output_graph_def.ParseFromString(serialized_graph)
    #
    #     # print(output_graph_def)
    #
    #     # 将output_graph_def导入当前默认图中(加载模型)
    #
    #     tf.import_graph_def(output_graph_def, name='')
    #
    # print('模型加载完成')
    #
    # # 使用默认图，此时已经加载了模型
    #
    # detection_graph = tf.get_default_graph()
    # output = ["define_loss/pred_sbbox/concat_2", "define_loss/pred_mbbox/concat_2",
    #           "define_loss/pred_lbbox/concat_2", "define_loss/layer_classes"]
    #
    # with tf.Session(graph=detection_graph) as sess:
    #     '''
    #
    #     获取模型中的tensor
    #
    #     '''
    #
    #     image_tensor = detection_graph.get_tensor_by_name('define_input/input_data:0')  # pb模型输入的名字
    #     trainable = detection_graph.get_tensor_by_name('define_input/training:0')  # pb模型输入的名字
    #
    #     converter = tf.lite.TFLiteConverter.from_frozen_graph(graph_def_file,
    #                                                           ['define_input/input_data', 'define_input/training'],
    #                                                           output, input_shapes={
    #             'define_input/input_data': [1, 416, 416, 3],
    #             'define_input/training': [1]})  # pb模型输入、输出的名字以及输入的大小
    #
    #     # converter.post_training_quantize = True
    #
    #     tflite_model = converter.convert()
    #
    #     converter.optimizations = [tf.lite.Optimize.DEFAULT]

        # open('E:/kx_detection/multi_detection/model', "wb").write(tflite_quant_model)


if __name__ == '__main__':
    main()
