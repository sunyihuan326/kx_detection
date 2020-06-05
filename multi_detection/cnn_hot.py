# -*- coding: utf-8 -*-
# @Time    : 2020/5/21
# @Author  : sunyihuan
# @File    : cnn_hot.py


"""
卷积热力图

未完成！！！！

"""

import tensorflow as tf
import tensorflow.keras.backend as K
import numpy as np
import cv2
import multi_detection.core.utils as utils
import matplotlib.pyplot as plt
class YoloPredict(object):
    '''
    预测结果
    '''

    def __init__(self):
        self.input_size = 416  # 输入图片尺寸（默认正方形）
        self.num_classes = 30  # 种类数
        self.score_threshold = 0.45
        self.iou_threshold = 0.5
        self.pb_file = "E:/multi_yolov3_predict-20191220/checkpoint/yolov3_1220.pb"  # ckpt文件地址
        self.write_image = True  # 是否画图
        self.show_label = True  # 是否显示标签

        graph = tf.Graph()
        with graph.as_default():
            output_graph_def = tf.GraphDef()
            with open(self.pb_file, "rb") as f:
                output_graph_def.ParseFromString(f.read())
                _ = tf.import_graph_def(output_graph_def, name="")

            self.sess = tf.Session()

            # 输入
            self.input = self.sess.graph.get_tensor_by_name("define_input/input_data:0")
            self.trainable = self.sess.graph.get_tensor_by_name("define_input/training:0")

            #输出的前一层
            self.pred_pre_sbbox = graph.get_tensor_by_name("define_loss/conv_sobj_branch/Relu:0")
            self.pred_pre_mbbox = graph.get_tensor_by_name("define_loss/conv_mobj_branch/Relu:0")
            self.pred_pre_lbbox = graph.get_tensor_by_name("define_loss/conv_lobj_branch/Relu:0")

            # 输出
            self.pred_sbbox = self.sess.graph.get_tensor_by_name("define_loss/pred_sbbox/concat_2:0")
            self.pred_mbbox = self.sess.graph.get_tensor_by_name("define_loss/pred_mbbox/concat_2:0")
            self.pred_lbbox = self.sess.graph.get_tensor_by_name("define_loss/pred_lbbox/concat_2:0")

            # 烤层
            self.layer_num = graph.get_tensor_by_name("define_loss/layer_classes:0")

img_path = "C:/Users/sunyihuan/Desktop/0518/20/20200520_023341673.jpg"
Y=YoloPredict()
image = cv2.imread(img_path)  # 图片读取
org_image = np.copy(image)
org_h, org_w, _ = org_image.shape

image_data = utils.image_preporcess(image, [416, 416])
image_data = image_data[np.newaxis, ...]

import tensorflow.keras.backend as K
with tf.GradientTape() as gtape:
    pred_sbbox = Y.pred_sbbox[:, 35]  # 预测向量
    pred_pre_sbbox = Y.pred_pre_sbbox  #分类前的一个卷积层
    grads = gtape.gradient(pred_sbbox, pred_pre_sbbox)  # 类别与卷积层的梯度 (1,14,14,512)
    pooled_grads = K.mean(grads, axis=(0,1,2)) # 特征层梯度的全局平均代表每个特征层权重
heatmap = tf.reduce_mean(tf.multiply(pooled_grads, pred_pre_sbbox), axis=-1) #权重与特征层相乘，512层求和平均

heatmap = np.maximum(heatmap, 0)
max_heat = np.max(heatmap)
if max_heat == 0:
    max_heat = 1e-10
heatmap /= max_heat
plt.matshow(heatmap[0], cmap='viridis')
