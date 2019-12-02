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
from tqdm import tqdm


def correct_bboxes(bboxes_pr, layer_n):
    '''
    bboxes_pr结果矫正
    :param bboxes_pr: 模型预测结果，格式为[x_min, y_min, x_max, y_max, probability, cls_id]
    :param layer_n:
    :return:
    '''
    num_label = len(bboxes_pr)
    # 未检测食材
    if num_label == 0:
        return bboxes_pr, layer_n

    # 检测到一个食材
    elif num_label == 1:
        if bboxes_pr[0][4] < 0.9 and bboxes_pr[0][4] >= 0.45:
            bboxes_pr[0][4] = 0.9
        return bboxes_pr, layer_n

    # 检测到多个食材
    else:
        same_label = True
        for i in range(num_label):
            if i == (num_label - 1):
                break
            if bboxes_pr[i][5] == bboxes_pr[i + 1][5]:
                continue
            else:
                same_label = False

        sumProb = 0.
        # 多个食材，同一标签
        if same_label:
            # for i in range(num_label):
            #    sumProb += bboxes_pr[i][4]
            # avrProb = sumProb/num_label
            # bboxes_pr[0][4] = avrProb
            bboxes_pr[0][4] = 0.98
            return bboxes_pr, layer_n
        # 多个食材，非同一标签
        else:
            problist = list(map(lambda x: x[4], bboxes_pr))
            labellist = list(map(lambda x: x[5], bboxes_pr))

            labeldict = {}
            for key in labellist:
                labeldict[key] = labeldict.get(key, 0) + 1
                # 按同种食材label数量降序排列
            s_labeldict = sorted(labeldict.items(), key=lambda x: x[1], reverse=True)

            n_name = len(s_labeldict)
            name1 = s_labeldict[0][0]
            num_name1 = s_labeldict[0][1]

            # 数量最多label对应的食材占比0.7以上
            if num_name1 / num_label > 0.7:
                num_label0 = []
                for i in range(num_label):
                    if name1 == bboxes_pr[i][5]:
                        num_label0.append(bboxes_pr[i])
                num_label0[0][4] = 0.95
                return num_label0, layer_n

            # 按各个label的probability降序排序
            else:
                # 计数
                bboxes_pr = sorted(bboxes_pr, key=lambda x: x[4], reverse=True)
                for i in range(len(bboxes_pr)):
                    bboxes_pr[i][4] = bboxes_pr[i][4] * 0.9
                return bboxes_pr, layer_n


class YoloTest(object):
    def __init__(self):
        self.input_size = 416  # 输入图片尺寸（默认正方形）
        self.num_classes = 27  # 种类数
        self.score_threshold = 0.45
        self.iou_threshold = 0.5
        self.weight_file = "E:/ckpt_dirs/Food_detection/multi_food7/20191202/yolov3_train_loss=8.8523.ckpt-10"  # ckpt文件地址
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
        # print(bboxes_pr)
        # print(layer_n)

        if self.write_image:
            image = utils.draw_bbox(image, bboxes_pr, show_label=self.show_label)
            drawed_img_save_to_path = str(image_path).split("/")[-1]
            drawed_img_save_to_path = str(drawed_img_save_to_path).split(".")[0] + "_" + str(
                layer_n) + ".jpg"  # 图片保存地址，烤层结果在命名中
            # cv2.imshow('Detection result', image)
            cv2.imwrite(save_dir + "/" + drawed_img_save_to_path, image)  # 保存图片
        return bboxes_pr, layer_n


if __name__ == '__main__':
    img_dir = "C:/Users/sunyihuan/Desktop/test_results_jpg"  # 文件夹地址
    save_dir = "C:/Users/sunyihuan/Desktop/test_results_jpg/detection"  # 预测结果标出保存地址
    Y = YoloTest()  # 加载模型
    # Y.result(img_path, "E:/Joyoung_WLS_github/tf_yolov3")
    # for file in os.listdir(img_dir):
    #     if file.endswith("jpg"):
    #         image_path = img_dir + "/" + file
    #         print(image_path)
    #         Y.result(image_path, save_dir)  # 预测每一张结果并保存
    # classes = ["Beefsteak", "CartoonCookies", "Cookies", "CupCake", "Pizzafour",
    #            "Pizzaone", "Pizzasix", "ChickenWings", "ChiffonCake6", "ChiffonCake8",
    #            "CranberryCookies", "EggTart", "EggTartBig", "nofood", "Peanuts",
    #            "PorkChops", "PotatoCut", "Potatol", "Potatom", "Potatos",
    #            "RoastedChicken", "SweetPotatoCut", "SweetPotatol", "SweetPotatom",
    #            "Pizzatwo", "SweetPotatoS", "Toast"]
    # classes = ["potatol", "potatom", "sweetpotatom", "sweetpotatol"]
    # classes = ["potatol", "potatom", "sweetpotatom"]
    classes = ["peanuts"]
    classes_id = {"peanuts": 11}

    # classes_id = {"CartoonCookies": 1, "Cookies": 5, "CupCake": 7, "Beefsteak": 0, "ChickenWings": 2,
    #               "ChiffonCake6": 3, "ChiffonCake8": 4, "CranberryCookies": 6, "EggTart": 8, "EggTartBig": 9,
    #               "nofood": 10, "Peanuts": 11, "porkchops": 16, "PotatoCut": 17, "potatol": 18,
    #               "potatom": 19, "Potatos": 20, "SweetPotatoCut": 21, "sweetpotatol": 22, "sweetpotatom": 23,
    #               "Pizzafour": 12, "Pizzaone": 13, "Pizzasix": 14, "RoastedChicken": 25,
    #               "Pizzatwo": 15, "SweetPotatoS": 24, "Toast": 26,"jpgs":19}

    for c in classes:
        error_noresults = 0  # 无任何结果统计
        food_acc = 0  # 食材准确数统计
        all_jpgs = 0  # 图片总数统计
        img_dirs = img_dir + "/" + c
        save_dirs = save_dir + "/" + c
        if os.path.exists(save_dirs): shutil.rmtree(save_dirs)
        os.mkdir(save_dirs)
        for file in tqdm(os.listdir(img_dirs)):
            if file.endswith("jpg"):
                all_jpgs += 1  # 统计总jpg图片数量
                image_path = img_dirs + "/" + file
                bboxes_pr, layer_n = Y.result(image_path, save_dirs)  # 预测每一张结果并保存
                # try:
                #     bboxes_pr, layer_n = Y.result(image_path, save_dirs)  # 预测每一张结果并保存
                # except:
                #     pass
                if len(bboxes_pr) == 0:  # 无任何结果返回，输出并统计+1
                    error_noresults += 1
                else:
                    bboxes_pr, layer_n = correct_bboxes(bboxes_pr, layer_n)  # 矫正输出结果
                    pre = bboxes_pr[0][-1]
                    if pre == classes_id[c]:  # 若结果正确，食材正确数+1
                        food_acc += 1
                    # else:
                    #     print(pre)
        print("food name:", c)
        print("food accuracy:", round((food_acc / all_jpgs) * 100, 2))  # 输出食材正确数
        print("no result:", error_noresults)  # 输出无任何结果总数
