# -*- encoding: utf-8 -*-

"""
@File    : ckpt_layer_demo.py
@Time    : 2019/12/11 11:39
@Author  : sunyihuan
"""

'''
ckpt文件预测某一文件夹下所有图片烤层结果
并输出烤层准确率
'''

import cv2
import numpy as np
import tensorflow as tf
import multi_detection.core.utils as utils
import os
import shutil
from tqdm import tqdm
import xlwt
from sklearn.metrics import confusion_matrix


class YoloTest(object):
    def __init__(self):
        self.input_size = 416  # 输入图片尺寸（默认正方形）
        self.num_classes = 30  # 种类数
        self.score_threshold = 0.3
        self.iou_threshold = 0.5
        self.weight_file = "E:/ckpt_dirs/Food_detection/multi_food7/20191211/yolov3_train_loss=4.9269.ckpt-50"  # ckpt文件地址
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

    def result(self, image_path):
        '''
        得出预测结果并保存
        :param image_path: 图片地址
        :return:
        '''
        image = cv2.imread(image_path)  # 图片读取
        bboxes_pr, layer_n = self.predict(image)  # 预测结果
        return bboxes_pr, layer_n


if __name__ == '__main__':
    img_dir = "C:/Users/sunyihuan/Desktop/test_jpg_check20191208/orignal"  # 文件夹地址
    layer_error_dir = "C:/Users/sunyihuan/Desktop/test_jpg_check20191208/orignal/layer_error7"  # 预测结果错误保存地址
    if not os.path.exists(layer_error_dir): os.mkdir(layer_error_dir)
    Y = YoloTest()  # 加载模型

    classes = ["Beefsteak", "CartoonCookies", "Cookies", "CupCake", "Pizzafour",
               "Pizzatwo", "Pizzaone", "Pizzasix", "ChickenWings", "ChiffonCake6",
               "ChiffonCake8", "CranberryCookies", "eggtarts", "eggtartl", "nofood",
               "Peanuts", "PorkChops", "PotatoCut", "Potatol", "Potatom",
               "Potatos", "RoastedChicken", "SweetPotatoCut", "SweetPotatol", "SweetPotatom",
               "SweetPotatoS", "Toast"]
    # ab_classes = ["Pizzafour", "Pizzatwo", "Pizzaone", "Pizzasix",
    #               "PotatoCut", "Potatol", "Potatom",
    #               "RoastedChicken",
    #               "SweetPotatoCut", "SweetPotatol", "SweetPotatom", "SweetPotatoS",
    #               "Toast"]

    classes_id = {"CartoonCookies": 1, "Cookies": 5, "CupCake": 7, "Beefsteak": 0, "ChickenWings": 2,
                  "ChiffonCake6": 3, "ChiffonCake8": 4, "CranberryCookies": 6, "eggtarts": 8, "eggtartl": 9,
                  "nofood": 10, "Peanuts": 11, "PorkChops": 16, "PotatoCut": 17, "Potatol": 18,
                  "Potatom": 19, "Potatos": 20, "SweetPotatoCut": 21, "SweetPotatol": 22, "SweetPotatom": 23,
                  "Pizzafour": 12, "Pizzaone": 13, "Pizzasix": 14, "RoastedChicken": 25,
                  "Pizzatwo": 15, "SweetPotatoS": 24, "Toast": 26}
    jpgs_count_all = 0
    jpgs_acc = 0

    workbook = xlwt.Workbook(encoding='utf-8')
    sheet1 = workbook.add_sheet("multi_food")
    sheet1.write(0, 0, "classes")
    sheet1.write(0, 1, "layer_acc")
    sheet1.write(0, 2, "jpgs_all")
    sheet1.write(0, 3, "acc")

    img_true = []
    img_pre = []
    for i in range(len(classes)):
        c = classes[i].lower()
        layer_acc = 0  # 烤层准确数统计
        all_jpgs = 0  # 图片总数统计

        img_dirs = img_dir + "/" + c
        layer_error_c_dirs = layer_error_dir + "/" + c
        if os.path.exists(layer_error_c_dirs): shutil.rmtree(layer_error_c_dirs)
        os.mkdir(layer_error_c_dirs)
        for file in tqdm(os.listdir(img_dirs + "/bottom")):  # 底层
            if file.endswith("jpg"):
                all_jpgs += 1  # 统计总jpg图片数量
                image_path = img_dirs + "/bottom" + "/" + file
                bboxes_pr, layer_n = Y.result(image_path)  # 预测每一张结果并保存
                img_true.append(0)
                img_pre.append(layer_n)
                if layer_n != 0:
                    shutil.copy(image_path,
                                layer_error_c_dirs + "/" + file.split(".jpg")[0] + "_" + str(layer_n) + ".jpg")
                else:
                    layer_acc += 1
        for file in tqdm(os.listdir(img_dirs + "/middle")):  # 中层
            if file.endswith("jpg"):
                all_jpgs += 1  # 统计总jpg图片数量
                image_path = img_dirs + "/middle" + "/" + file
                bboxes_pr, layer_n = Y.result(image_path)  # 预测每一张结果并保存
                img_true.append(1)
                img_pre.append(layer_n)
                if layer_n != 1:
                    shutil.copy(image_path,
                                layer_error_c_dirs + "/" + file.split(".jpg")[0] + "_" + str(layer_n) + ".jpg")
                else:
                    layer_acc += 1
        for file in tqdm(os.listdir(img_dirs + "/top")):
            if file.endswith("jpg"):
                all_jpgs += 1  # 统计总jpg图片数量
                image_path = img_dirs + "/top" + "/" + file
                bboxes_pr, layer_n = Y.result(image_path)  # 预测每一张结果并保存
                img_true.append(2)
                img_pre.append(layer_n)
                if layer_n != 2:
                    shutil.copy(image_path,
                                layer_error_c_dirs + "/" + file.split(".jpg")[0] + "_" + str(layer_n) + ".jpg")
                else:
                    layer_acc += 1
        for file in tqdm(os.listdir(img_dirs + "/others")):
            if file.endswith("jpg"):
                all_jpgs += 1  # 统计总jpg图片数量
                image_path = img_dirs + "/others" + "/" + file
                bboxes_pr, layer_n = Y.result(image_path)  # 预测每一张结果并保存
                img_true.append(3)
                img_pre.append(layer_n)
                if layer_n != 3:
                    shutil.copy(image_path,
                                layer_error_c_dirs + "/" + file.split(".jpg")[0] + "_" + str(layer_n) + ".jpg")
                else:
                    layer_acc += 1

        sheet1.write(i + 1, 0, c)

        sheet1.write(i + 1, 1, layer_acc)
        sheet1.write(i + 1, 2, all_jpgs)
        sheet1.write(i + 1, 3, round((layer_acc / all_jpgs) * 100, 2))

        print("food name:", c)
        print("layer accuracy:", round((layer_acc / all_jpgs) * 100, 2))  # 输出烤层正确数
        jpgs_count_all += all_jpgs
        jpgs_acc += layer_acc
    print("all layer accuracy:", round((jpgs_acc / jpgs_count_all) * 100, 2))  # 输出烤层正确数

    conf = confusion_matrix(y_pred=img_pre, y_true=img_true)

    print(conf)
    print(sum(sum(conf)))
    sheet1.write(35, 1, jpgs_acc)
    sheet1.write(35, 2, jpgs_count_all)
    sheet1.write(35, 3, round((jpgs_acc / jpgs_count_all) * 100, 2))

    workbook.save("C:/Users/sunyihuan/Desktop/test_jpg_check20191208/orignal/layer_multi7.xls")
