# -*- coding: utf-8 -*-
# @Time    : 2020/5/20
# @Author  : sunyihuan
# @File    : get_hot.py

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

pred_sbbox = Y.pred_sbbox[:, 35]   # 预测向量

pred_pre_sbbox = Y.pred_pre_sbbox # block5_conv3层的输出特征图，它是VGG16的最后一个卷积层

grads = K.gradients(pred_sbbox, pred_pre_sbbox)[0]   # 非洲象类别相对于block5_conv3输出特征图的梯度

pooled_grads = K.mean(grads, axis=(0, 1, 2))   # 形状是（512， ）的向量，每个元素是特定特征图通道的梯度平均大小

iterate = K.function([Y.input], [pooled_grads, pred_pre_sbbox])  # 这个函数允许我们获取刚刚定义量的值：对于给定样本图像，pooled_grads和block5_conv3层的输出特征图

pooled_grads_value, conv_layer_output_value = iterate([image_data])  # 给我们两个大象样本图像，这两个量都是Numpy数组

for i in range(512):
    conv_layer_output_value[:, :, i] *= pooled_grads_value[i]  # 将特征图数组的每个通道乘以这个通道对大象类别重要程度

heatmap = np.mean(conv_layer_output_value, axis=-1)
heatmap = np.maximum(heatmap, 0)
heatmap /= np.max(heatmap)
plt.matshow(heatmap)
plt.show()
