#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2019/7/24 
# @Author  : sunyihuan

'''
ckpt文件预测某一文件夹下所有图片结果
并输出食材类别准确率结果
'''

import cv2
import numpy as np
import tensorflow as tf
import multi_detection.core.utils as utils
import os
import shutil


def correct_result(bboxes_pr):
    '''
    将目标检测结果转为识别结果，类别为1类
    :param bboxes_pr:
    :return:
    '''
    pre_list = []
    score_list = []
    for bbox in bboxes_pr:
        coor = np.array(bbox[:4], dtype=np.int32)
        score = bbox[4]
        class_ind = int(bbox[5])
        score_list.append(score)
        pre_list.append(class_ind)
    pre_c = {}
    # 对类别结果个数统计
    for p in pre_list:
        if p not in pre_c.keys():
            pre_c[p] = 1
        else:
            pre_c[p] += 1
    pre_cc = sorted(pre_c.items(), key=lambda x: x[1], reverse=True)  # 类别结果按个数排序
    if len(pre_cc) == 1:
        # 若输出种类只有1类，直接取
        pre = pre_cc[0][0]
    else:
        if pre_cc[0][1] != pre_cc[1][1]:  # 若输出种类排序最多的仅有1个，取第一个
            pre = pre_cc[0][0]
        else:
            # 如果输种类排序最多的大于1个，返回score得分最高的类别
            pre = pre_list[score_list.index(max(score_list))]
    return pre


class YoloTest(object):
    def __init__(self):
        self.input_size = 416  # 输入图片尺寸（默认正方形）
        self.num_classes = 27  # 种类数
        self.score_threshold = 0.45
        self.iou_threshold = 0.5
        self.weight_file = "E:/ckpt_dirs/Food_detection/multi_food/20191106/yolov3_train_loss=8.5058.ckpt-28"  # ckpt文件地址
        self.write_image = True  # 是否画图
        self.show_label = True  # 是否显示标签

        graph = tf.Graph()
        with graph.as_default():
            # 模型加载
            self.saver = tf.train.import_meta_graph("{}.meta".format(self.weight_file))
            self.sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True))
            self.saver.restore(self.sess, self.weight_file)

            # 输入
            self.input = graph.get_tensor_by_name("define_input/input_data:0")
            self.trainable = graph.get_tensor_by_name("define_input/training:0")

            # 输出检测结果
            self.pred_sbbox = graph.get_tensor_by_name("define_loss/pred_sbbox/concat_2:0")
            self.pred_mbbox = graph.get_tensor_by_name("define_loss/pred_mbbox/concat_2:0")
            self.pred_lbbox = graph.get_tensor_by_name("define_loss/pred_lbbox/concat_2:0")

            # 输出烤层结果
            self.layer_num = graph.get_tensor_by_name("define_loss/layer_classes:0")

    def predict(self, image):
        '''
        预测结果
        :param image: 图片数据，shape为[800,600,3]
        :return:
            bboxes：食材检测预测框结果，格式为：[x_min, y_min, x_max, y_max, probability, cls_id],
            layer_n[0]：烤层检测结果，0：最下层、1：中间层、2：最上层、3：其他
        '''
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
        '''
        得出预测结果并保存
        :param image_path: 图片地址
        :param save_dir: 预测结果原图标注框，保存地址
        :return:
        '''
        image = cv2.imread(image_path)  # 图片读取
        bboxes_pr, layer_n = self.predict(image)  # 预测结果
        print(bboxes_pr)
        print(layer_n)

        if self.write_image:
            image = utils.draw_bbox(image, bboxes_pr, show_label=self.show_label)
            drawed_img_save_to_path = str(image_path).split("/")[-1]
            drawed_img_save_to_path = str(drawed_img_save_to_path).split(".")[0] + "_" + str(
                layer_n) + ".jpg"  # 图片保存地址，烤层结果在命名中
            # cv2.imshow('Detection result', image)
            cv2.imwrite(save_dir + "/" + drawed_img_save_to_path, image)  # 保存图片
        return bboxes_pr, layer_n


if __name__ == '__main__':
    img_dir = "C:/Users/sunyihuan/Desktop/tttttt"  # 文件夹地址
    save_dir = "C:/Users/sunyihuan/Desktop/tttttt/detection"  # 预测结果标出保存地址
    Y = YoloTest()  # 加载模型
    # Y.result(img_path, "E:/Joyoung_WLS_github/tf_yolov3")
    # for file in os.listdir(img_dir):
    #     if file.endswith("jpg"):
    #         image_path = img_dir + "/" + file
    #         print(image_path)
    #         Y.result(image_path, save_dir)  # 预测每一张结果并保存
    classes = ["co"]

    classes_id = 1

    error_noresults = 0  # 无任何结果统计
    food_acc = 0  # 食材准确数统计
    all_jpgs = 0  # 图片总数统计
    for c in classes:
        img_dirs = img_dir + "/" + c
        save_dirs = save_dir + "/" + c
        if os.path.exists(save_dirs): shutil.rmtree(save_dirs)
        os.mkdir(save_dirs)
        for file in os.listdir(img_dirs):
            if file.endswith("jpg"):
                all_jpgs += 1  # 统计总jpg图片数量
                image_path = img_dirs + "/" + file
                print(image_path)
                bboxes_pr, layer_n = Y.result(image_path, save_dirs)  # 预测每一张结果并保存
                if len(bboxes_pr) == 0:  # 无任何结果返回，输出并统计+1
                    print("no_result")
                    error_noresults += 1
                else:
                    pre = correct_result(bboxes_pr)  # 矫正输出结果
                    print("predictions::::::::::::::::::::")
                    print(pre)
                    if pre == classes_id:  # 若结果正确，食材正确数+1
                        food_acc += 1
    print("food accuracy:", round((food_acc / all_jpgs) * 100, 2))  # 输出食材正确数
    print("no result:", error_noresults)  # 输出无任何结果总数
