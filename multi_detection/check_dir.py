# -*- encoding: utf-8 -*-

"""
@File    : check_dir.py
@Time    : 2019/11/20 8:42
@Author  : sunyihuan
"""

'''

ckpt文件预测文件夹下所有图片结果
并输出食材类别准确率结果
文件名为类别名称

'''

import cv2
import numpy as np
import tensorflow as tf
import multi_detection.core.utils as utils
import os
import shutil
from tqdm import tqdm
from sklearn.metrics import confusion_matrix, accuracy_score
from multi_detection.food_correct_utils import *


class YoloTest(object):
    def __init__(self):
        self.input_size = 416  # 输入图片尺寸（默认正方形）
        self.num_classes = 27  # 种类数
        self.score_threshold = 0.1
        self.iou_threshold = 0.5
        # self.weight_file = "E:/ckpt_dirs/Food_detection/multi_food3/20191118/yolov3_train_loss=5.0565.ckpt-100"  # ckpt文件地址
        self.pb_file = "E:/ckpt_dirs/Food_detection/local/20191216/yolo_model.pb"
        self.write_image = True  # 是否画图
        self.show_label = True  # 是否显示标签
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
    classes = ["roastedchicken"]
    # classes = ["porkchops", "beefsteak", "cartooncookies", "chickenwings", "chiffoncake6",
    #            "chiffoncake8", "cookies", "cranberrycookies", "cupcake", "eggtartl", "eggtarts",
    #            "nofood", "peanuts", "roastedchicken", "toast",
    #            "pizzaone", "pizzatwo", "pizzafour", "pizzasix",
    #            "potatol", "potatom", "potatos", "potatocut",
    #            "sweetpotatom", "sweetpotatol", "sweetpotatos", "sweetpotatocut"]

    classes_id = {"cartooncookies": 1, "cookies": 5, "cupcake": 7, "beefsteak": 0, "chickenwings": 2,
                  "chiffoncake6": 3, "chiffoncake8": 4, "cranberrycookies": 6, "eggtarts": 8, "eggtartl": 9,
                  "nofood": 10, "peanuts": 11, "porkchops": 16, "potatocut": 17, "potatol": 18,
                  "potatom": 19, "potatos": 20, "sweetpotatocut": 21, "sweetpotatol": 22, "sweetpotatom": 23,
                  "pizzafour": 12, "pizzaone": 13, "pizzasix": 14, "roastedchicken": 25,
                  "pizzatwo": 15, "sweetpotatos": 24, "toast": 26}
    food_name_pre = []
    food_name_true = []

    food_name_dir = "E:/kx_detection/error_roastedchicken_results"
    no_results_dir = "E:/kx_detection/no_results"

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
                    shutil.copy(image_path,
                                no_results_dir + "/" + file)
                else:
                    bboxes_pr, layer_n = correct_bboxes(bboxes_pr, layer_n)  # 矫正输出结果
                    pre = bboxes_pr[0][-1]
                    food_name_true.append(classes_id[c])
                    food_name_pre.append(pre)
                    if pre == classes_id[c]:  # 若结果正确，食材正确数+1
                        food_acc += 1
                    else:
                        drawed_img_save_to_path = str(file).split(".")[0] + "_" + str(
                            layer_n) + ".jpg"
                        shutil.copy(save_dirs + "/" + drawed_img_save_to_path,
                                    food_name_dir + "/" + file)
        if all_jpgs == 0:
            all_jpgs = 20000000000
        print("food name:", c)
        print("food accuracy:", round((food_acc / all_jpgs) * 100, 2))  # 输出食材正确数
        print("no result:", error_noresults)  # 输出无任何结果总数

    food_name_matrix = confusion_matrix(food_name_true, food_name_pre,
                                        labels=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19,
                                                20, 21, 22, 23, 24, 25, 26])
    print("食材检测混淆矩阵：")
    print(food_name_matrix)

    acc = accuracy_score(y_true=food_name_true, y_pred=food_name_pre)
    print("准确率：", acc)
