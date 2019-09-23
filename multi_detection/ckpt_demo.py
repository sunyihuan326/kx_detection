#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2019/7/24 
# @Author  : sunyihuan

'''
ckpt文件预测某一张图片结果
'''

import cv2
import numpy as np
import tensorflow as tf
import multi_detection.core.utils as utils
from multi_detection.core.config import cfg
from multi_detection.core.yolov3 import YOLOV3
import os


class YoloTest(object):
    def __init__(self):
        self.input_size = 416  # 输入图片尺寸（默认正方形）
        self.num_classes = 26  # 种类数
        self.score_threshold = 0.45
        self.iou_threshold = 0.5
        self.weight_file = "E:/ckpt_dirs/Food_detection/multi_food/20190904/yolov3_train_loss=5.2960.ckpt-62"  # ckpt文件地址
        self.write_image = True  # 是否画图
        self.show_label = True  # 是否显示标签

        graph = tf.Graph()
        with graph.as_default():
            self.saver = tf.train.import_meta_graph("{}.meta".format(self.weight_file))
            self.sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True))
            self.saver.restore(self.sess, self.weight_file)

            self.input = graph.get_tensor_by_name("define_input/input_data:0")
            self.trainable = graph.get_tensor_by_name("define_input/training:0")

            self.pred_sbbox = graph.get_tensor_by_name("define_loss/pred_sbbox/concat_2:0")
            self.pred_mbbox = graph.get_tensor_by_name("define_loss/pred_mbbox/concat_2:0")
            self.pred_lbbox = graph.get_tensor_by_name("define_loss/pred_lbbox/concat_2:0")

            self.layer_num = graph.get_tensor_by_name("define_loss/layer_classes:0")

    def predict(self, image):
        org_image = np.copy(image)
        org_h, org_w, _ = org_image.shape

        image_data = utils.image_preporcess(image, [self.input_size, self.input_size])
        image_data = image_data[np.newaxis, ...]

        pred_sbbox, pred_mbbox, pred_lbbox, layer_n = self.sess.run(
            [self.pred_sbbox, self.pred_mbbox, self.pred_lbbox, self.layer_num],
            feed_dict={
                self.input: image_data,
                self.trainable: False
            }
        )

        pred_bbox = np.concatenate([np.reshape(pred_sbbox, (-1, 5 + self.num_classes)),
                                    np.reshape(pred_mbbox, (-1, 5 + self.num_classes)),
                                    np.reshape(pred_lbbox, (-1, 5 + self.num_classes))], axis=0)

        bboxes = utils.postprocess_boxes(pred_bbox, (org_h, org_w), self.input_size, self.score_threshold)
        bboxes = utils.nms(bboxes, self.iou_threshold)

        return bboxes, layer_n[0]

    def result(self, image_path, save_dir):
        image = cv2.imread(image_path)
        bboxes_pr, layer_n = self.predict(image)  # 预测结果
        print(bboxes_pr)
        print(layer_n)

        if self.write_image:
            image = utils.draw_bbox(image, bboxes_pr, show_label=self.show_label)
            drawed_img_save_to_path = str(image_path).split("/")[-1]
            drawed_img_save_to_path = str(drawed_img_save_to_path).split(".")[0] + "_" + str(layer_n) + ".jpg"
            # cv2.imshow('Detection result', image)
            cv2.imwrite(save_dir + "/" + drawed_img_save_to_path, image)


if __name__ == '__main__':
    img_dir = "C:/Users/sunyihuan/Desktop/st_before/kaohuasheng/kaopan"
    save_dir = "C:/Users/sunyihuan/Desktop/st_before/kaohuasheng/kaopan_layer_detection"
    # img_path = "E:/DataSets/KX_FOODSets_model_data/23classes_0808_test/JPGImages/8_Toast.jpg"
    # YoloTest().result(img_path, save_dir)
    Y = YoloTest()
    # Y.result(img_path, "E:/Joyoung_WLS_github/tf_yolov3")
    for file in os.listdir(img_dir):
        if file.endswith("jpg"):
            image_path = img_dir + "/" + file
            print(image_path)
            Y.result(image_path, save_dir)
