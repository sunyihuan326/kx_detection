# -*- coding: utf-8 -*-
# @Time    : 2020/7/15
# @Author  : sunyihuan
# @File    : cnn_hot_dir.py
'''
画出文件夹下所有图片的热力图
'''

import tensorflow as tf
from tensorflow.python import pywrap_tensorflow

import numpy as np
import cv2
import multi_detection.core.utils as utils
import os
from tqdm import tqdm


class YoloPredict(object):
    '''
    预测结果
    '''

    def __init__(self):
        self.input_size = 416  # 输入图片尺寸（默认正方形）
        self.num_classes = 40  # 种类数
        self.top_n = 3
        self.score_cls_threshold = 0.001
        self.score_threshold = 0.45
        self.iou_threshold = 0.5
        self.weight_file = self.weight_file = "E:/ckpt_dirs/Food_detection/multi_food5/20200914/yolov3_train_loss=6.9178.ckpt-95"   # ckpt文件地址
        self.write_image = True  # 是否画图
        self.show_label = True  # 是否显示标签

        self.reader = pywrap_tensorflow.NewCheckpointReader(self.weight_file)

        graph = tf.Graph()
        with graph.as_default():
            self.saver = tf.train.import_meta_graph("{}.meta".format(self.weight_file))
            self.sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True))
            self.saver.restore(self.sess, self.weight_file)

            # 输入
            self.input = self.sess.graph.get_tensor_by_name("define_input/input_data:0")
            self.trainable = self.sess.graph.get_tensor_by_name("define_input/training:0")

            # 输出的前一层
            self.pred_pre_sbbox = graph.get_tensor_by_name("define_loss/conv_sobj_branch/Relu:0")
            self.pred_pre_mbbox = graph.get_tensor_by_name("define_loss/conv_mobj_branch/Relu:0")
            self.pred_pre_lbbox = graph.get_tensor_by_name("define_loss/conv_lobj_branch/Relu:0")

            # 输出
            self.pred_sbbox = self.sess.graph.get_tensor_by_name("define_loss/pred_sbbox/concat_2:0")
            self.pred_mbbox = self.sess.graph.get_tensor_by_name("define_loss/pred_mbbox/concat_2:0")
            self.pred_lbbox = self.sess.graph.get_tensor_by_name("define_loss/pred_lbbox/concat_2:0")

            # 烤层
            self.layer_num_pre = graph.get_tensor_by_name("define_loss/darknet/residual20/add:0")
            self.layer_num = graph.get_tensor_by_name("define_loss/layer_classes:0")


class get_cam(object):
    def __init__(self, img_path, Y):
        self.img_path = img_path
        self.Y = Y

    def get_top_cls(self, pred_bbox, org_h, org_w, top_n):
        '''
        获取top_n，类别和得分
        :param pred_bbox:所有框
        :param org_h:高
        :param org_w:宽
        :param top_n:top数
        :return:按置信度前top_n个，输出类别、置信度，
        例如
        [(18, 0.9916), (19, 0.0105), (15, 0.0038), (1, 0.0018), (5, 0.0016), (13, 0.0011)]
        '''
        bboxes = utils.postprocess_boxes(pred_bbox, (org_h, org_w), Y.input_size, Y.score_cls_threshold)
        classes_in_img = list(set(bboxes[:, 5]))
        best_bboxes = {}
        for cls in classes_in_img:
            cls_mask = (bboxes[:, 5] == cls)
            cls_bboxes = bboxes[cls_mask]
            best_score = 0
            for i in range(len(cls_bboxes)):
                if cls_bboxes[i][-2] > best_score:
                    best_score = cls_bboxes[i][-2]
            if int(cls) not in best_bboxes.keys():
                best_bboxes[int(cls)] = round(best_score, 4)
        best_bboxes = sorted(best_bboxes.items(), key=lambda best_bboxes: best_bboxes[1], reverse=True)
        return best_bboxes[:top_n]

    def cam(self, sml_type):

        image = cv2.imread(self.img_path)  # 图片读取
        org_image = np.copy(image)
        org_h, org_w, _ = org_image.shape

        image_data = utils.image_preporcess(image, [Y.input_size, Y.input_size])
        image_data = image_data[np.newaxis, ...]

        pred_sbbox, pred_pre_sbbox, pred_mbbox, pred_pre_mbbox, pred_lbbox, pred_pre_lbbox, layer_num_pre, layer_n = Y.sess.run(
            [Y.pred_sbbox, Y.pred_pre_sbbox, Y.pred_mbbox, Y.pred_pre_mbbox, Y.pred_lbbox, Y.pred_pre_lbbox,
             Y.layer_num_pre,
             Y.layer_num],
            feed_dict={
                Y.input: image_data,
                Y.trainable: False
            }
        )

        typ_dict = {"l": "conv_lbbox", "m": "conv_mbbox", "s": "conv_sbbox"}
        pre_dict = {"l": pred_pre_lbbox[0], "m": pred_pre_mbbox[0], "s": pred_pre_sbbox[0]}
        size_dict = {"l": 1024, "m": 512, "s": 256}

        pred_bbox = np.concatenate([np.reshape(pred_sbbox, (-1, 5 + Y.num_classes)),
                                    np.reshape(pred_mbbox, (-1, 5 + Y.num_classes)),
                                    np.reshape(pred_lbbox, (-1, 5 + Y.num_classes))], axis=0)

        best_bboxes = self.get_top_cls(pred_bbox, org_h, org_w, 2)  # 获取top_n类别和置信度
        print(best_bboxes)

        classes = list(dict(best_bboxes).keys())[0]
        print("classes::::::::::", classes)
        reader = Y.reader
        lbbox_w = reader.get_tensor("{}/weight".format(typ_dict[sml_type]))
        lbbox_w = lbbox_w.reshape((int(size_dict[sml_type]), 3, 5 + Y.num_classes))
        lbbox_w = lbbox_w[:, :, 5 + classes]
        lbbox_w = np.sum(lbbox_w, 1)
        heatmap_lbbox = np.multiply(pre_dict[sml_type], lbbox_w)

        return heatmap_lbbox


if __name__ == "__main__":
    Y = YoloPredict()

    img_dir = "F:/chiffon10_0929/chiffoncake8"
    cam_dir ="F:/chiffon10_0929/chiffoncake8_cam"
    for img_name in os.listdir(img_dir):
        if img_name.endswith(".jpg"):
            img_path = img_dir + "/" + img_name
            heatmap = get_cam(img_path, Y).cam("s")

            heatmap = np.mean(heatmap, axis=-1)
            heatmap = np.maximum(heatmap, 0)
            try:
                if np.max(heatmap) == 0:
                    print("heatmap values all zero!!!!!!!!!!!!!!!!!!!!!!!!")
                heatmap /= np.max(heatmap)

                img = cv2.imread(img_path)

                heatmap = cv2.resize(heatmap, (img.shape[1], img.shape[0]))

                heatmap = np.uint8(255 * heatmap)

                heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

                superimposed_img = heatmap * 0.4 + img
                cv2.imwrite('{0}/{1}_cam_s.jpg'.format(cam_dir, img_name.split(".jpg")[0]), superimposed_img)
            except:
                print(img_path)
