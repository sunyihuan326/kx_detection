# -*- coding: utf-8 -*-
# @Time    : 2021/4/19
# @Author  : sunyihuan
# @File    : from_predict_copy_high_score_data.py

'''
直接从预测结果中，分类置信度高于0.8的数据
'''

import cv2
import numpy as np
import tensorflow as tf
import multi_detection.core.utils as utils
import os
import shutil
from tqdm import tqdm
import xlwt
import time
from multi_detection.core.config import cfg
from multi_detection.food_correct_utils import correct_bboxes

# gpu限制
gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.8)
config = tf.ConfigProto(gpu_options=gpu_options)


class YoloPredic(object):
    '''
    预测结果
    '''

    def __init__(self):
        self.input_size = 416  # 输入图片尺寸（默认正方形）
        self.num_classes = 40  # 种类数
        self.score_cls_threshold = 0.001
        self.score_threshold = 0.3
        self.iou_threshold = 0.5
        self.top_n = 5
        self.pb_file = "E:/模型交付版本/multi_yolov3_predict-20210129/checkpoint/yolov3_0129.pb"  # pb文件地址

        graph = tf.Graph()
        with graph.as_default():
            output_graph_def = tf.GraphDef()
            with open(self.pb_file, "rb") as f:
                output_graph_def.ParseFromString(f.read())
                _ = tf.import_graph_def(output_graph_def, name="")

            self.sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True))

            self.input = self.sess.graph.get_tensor_by_name("define_input/input_data:0")
            self.trainable = self.sess.graph.get_tensor_by_name("define_input/training:0")

            self.pred_sbbox = self.sess.graph.get_tensor_by_name("define_loss/pred_sbbox/concat_2:0")
            self.pred_mbbox = self.sess.graph.get_tensor_by_name("define_loss/pred_mbbox/concat_2:0")
            self.pred_lbbox = self.sess.graph.get_tensor_by_name("define_loss/pred_lbbox/concat_2:0")

            self.layer_num = self.sess.graph.get_tensor_by_name("define_loss/layer_classes:0")

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
        bboxes = utils.postprocess_boxes(pred_bbox, (org_h, org_w), self.input_size, self.score_cls_threshold)
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

    def predict(self, image):
        org_image = np.copy(image)
        org_h, org_w, _ = org_image.shape

        image_data = utils.image_preporcess(image, [self.input_size, self.input_size])
        image_data = image_data[np.newaxis, ...]

        pred_sbbox, pred_mbbox, pred_lbbox, layer_ = self.sess.run(
            [self.pred_sbbox, self.pred_mbbox, self.pred_lbbox, self.layer_num],
            feed_dict={
                self.input: image_data,
                self.trainable: False
            }
        )
        pred_bbox = np.concatenate([np.reshape(pred_sbbox, (-1, 5 + self.num_classes)),
                                    np.reshape(pred_mbbox, (-1, 5 + self.num_classes)),
                                    np.reshape(pred_lbbox, (-1, 5 + self.num_classes))], axis=0)

        best_bboxes = self.get_top_cls(pred_bbox, org_h, org_w, self.top_n)  # 获取top_n类别和置信度
        bboxes = utils.postprocess_boxes(pred_bbox, (org_h, org_w), self.input_size, self.score_threshold)
        bboxes = utils.nms(bboxes, self.iou_threshold)
        layer_n = layer_[0]  # 烤层结果

        return bboxes, layer_n, best_bboxes


if __name__ == '__main__':
    img_root = "F:/serve_data/202101-04/covert_jpg"  # 图片地址
    img_save = "F:/serve_data/202101-04/classes"  # 图片地址

    import time

    classes = utils.read_class_names(cfg.YOLO.CLASSES)
    classes[40] = "potatom"
    classes[41] = "sweetpotatom"
    classes[101] = "chiffon_size4"

    start_time = time.time()
    Y = YoloPredic()
    end_time0 = time.time()
    print("加载时间：", end_time0 - start_time)
    for img in tqdm(os.listdir(img_root)):
        img_path = img_root + "/" + img
        image = cv2.imread(img_path)  # 图片读取
        bboxes, layer_n, best_bboxes = Y.predict(image)
        bboxes, layer_n, best_bboxes = correct_bboxes(bboxes, layer_n, best_bboxes)  # 矫正输出结果

        if len(bboxes) > 0:
            if bboxes[0][-2] > 0.8:
                p_name = classes[int(bboxes[0][-1])]
                save_dir = img_save + "/" + p_name
                if not os.path.exists(save_dir): os.mkdir(save_dir)
                shutil.copy(img_path, save_dir + "/" + img)
