# -*- encoding: utf-8 -*-

"""
@File    : ckpt_test_xishi.py
@Time    : 2020/06/22 18:13
@Author  : sunyihuan
"""

'''
ckpt文件预测某一文件夹下各类所有图片食材结果
并输出各准确率至excel表格中

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
from sklearn.metrics import confusion_matrix
from multi_detection.food_correct_utils import correct_bboxes, get_potatoml

# gpu限制
gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.8)
config = tf.ConfigProto(gpu_options=gpu_options)


def he_foods(pre):
    '''
    针对合并的类别判断输出是否在合并类别内
    :param pre:
    :return:
    '''
    # if pre in [8, 9] and classes_id[classes[i]] in [8, 9]:  # 合并蛋挞
    #     rigth_label = True
    # elif pre in [12, 14] and classes_id[classes[i]] in [12, 14]:  # 合并四分之一披萨、六分之一披萨
    #     rigth_label = True
    # elif pre in [18, 19] and classes_id[classes[i]] in [18, 19]:  # 合并中土豆、大土豆
    #     rigth_label = True
    # elif pre in [22, 23] and classes_id[classes[i]] in [22, 23]:  # 合并中红薯、大红薯
    #     rigth_label = True
    # else:
    #     rigth_label = False
    rigth_label = False
    return rigth_label


class YoloTest(object):
    def __init__(self):
        self.input_size = 320  # 输入图片尺寸（默认正方形）
        self.num_classes = 22  # 种类数
        self.score_threshold = 0.3
        self.iou_threshold = 0.5
        self.weight_file = "E:/ckpt_dirs/Food_detection/multi_food3/checkpoint/yolov3_train_loss=4.9086.ckpt-161"  # ckpt文件地址
        # self.weight_file = "./checkpoint/yolov3_train_loss=4.7681.ckpt-80"
        self.write_image = True  # 是否画图
        self.show_label = True  # 是否显示标签

        graph = tf.Graph()
        with graph.as_default():
            # 模型加载
            self.saver = tf.train.import_meta_graph("{}.meta".format(self.weight_file))
            self.sess = tf.Session(config=config)
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
        # image = utils.white_balance(image)  # 图片白平衡处理
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

    classes_label22 = ["beefsteak", "cartooncookies", "chickenwings", "chiffoncake6",
                       "cookies", "cranberrycookies", "cupcake", "eggtart",
                       "peanuts", "pizzaone", "pizzatwo", "pizzacut",
                       "porkchops", "potatocut", "potatol",
                       "potatos", "sweetpotatocut", "sweetpotatol", "sweetpotatos",
                       "roastedchicken", "toast", "chiffoncake8", ]

    classes_id22 = {"cartooncookies": 1, "cookies": 4, "cupcake": 6, "beefsteak": 0, "chickenwings": 2,
                    "chiffoncake6": 3, "cranberrycookies": 5, "eggtart": 7,
                    "nofood": 8, "peanuts": 9, "porkchops": 13, "potatocut": 14, "potatol": 15,
                    "potatos": 16, "sweetpotatocut": 17, "sweetpotatol": 18,
                    "pizzacut": 10, "pizzaone": 11, "roastedchicken": 20,
                    "pizzatwo": 12, "sweetpotatos": 19, "toast": 21, "chiffoncake8": 24}
    # 需要修改
    classes_id = classes_id22  #######
    classes = classes_label22  #######
    mode = "multi3_158"  #######
    tag = ""
    img_root = "F:/test_from_xishi/orignal_data/X6_all"
    save_root = "F:/test_from_xishi/orignal_data/X6_all/detection"  # 图片保存地址
    if not os.path.exists(save_root): os.mkdir(save_root)
    Y = YoloTest()  # 加载模型
    new_classes = {v: k for k, v in classes_id.items()}

    jpgs_count_all = 0
    layer_jpgs_acc = 0
    food_jpgs_acc = 0

    workbook = xlwt.Workbook(encoding='utf-8')
    sheet1 = workbook.add_sheet("all_jpg")

    sheet1.write(0, 0, "jpg_name")
    sheet1.write(0, 1, "true_clas_name")
    sheet1.write(0, 2, "pre_cls_name")
    sheet1.write(0, 3, "cls_id")
    sheet1.write(0, 4, "cls_score")

    all_jpg = 0

    food_img_true = []
    food_img_pre = []
    for k in classes:
        sheet1_i = workbook.add_sheet(k)
        sheet1_i.write(0, 0, "jpg_name")
        sheet1_i.write(0, 1, "true_clas_name")
        sheet1_i.write(0, 2, "pre_cls_name")
        sheet1_i.write(0, 3, "cls_id")
        sheet1_i.write(0, 4, "cls_score")
        img_dir = img_root + "/" + k
        save_dir = save_root + "/" + k
        if not os.path.exists(save_dir): os.mkdir(save_dir)
        for i, img in enumerate(os.listdir(img_dir)):
            if img.endswith(".jpg"):
                img_name = img_dir + "/" + img
                bboxes_pr, layer_n = Y.result(img_name, save_dir)  # 预测每一张结果并保存
                bboxes_pr, layer_n = correct_bboxes(bboxes_pr, layer_n)  # 矫正输出结果
                if len(bboxes_pr) == 0:
                    sheet1_i.write(i + 1, 0, img)
                    sheet1_i.write(i + 1, 1, k)
                    sheet1_i.write(i + 1, 2, "noreslut")
                    sheet1_i.write(i + 1, 3, "noreslut")
                    sheet1_i.write(i + 1, 4, "noreslut")
                    all_jpg += 1
                    sheet1.write(all_jpg, 0, img)
                    sheet1.write(all_jpg, 1, k)
                    sheet1.write(all_jpg, 2, "noreslut")
                    sheet1.write(all_jpg, 3, "noreslut")
                    sheet1.write(all_jpg, 4, "noreslut")
                else:
                    pre = bboxes_pr[0]
                    if int(pre[-1]) == 3:
                        bboxes_pr, layer_n = get_potatoml(bboxes_pr, layer_n)
                    pre = bboxes_pr[0]
                    print(pre)
                    sheet1_i.write(i + 1, 0, img)
                    sheet1_i.write(i + 1, 1, k)
                    sheet1_i.write(i + 1, 2, new_classes[int(pre[-1])])
                    sheet1_i.write(i + 1, 3, pre[-1])
                    sheet1_i.write(i + 1, 4, pre[-2])

                    all_jpg += 1
                    sheet1.write(all_jpg, 0, img)
                    sheet1.write(all_jpg, 1, k)
                    sheet1.write(all_jpg, 2, new_classes[int(pre[-1])])
                    sheet1.write(all_jpg, 3, pre[-1])
                    sheet1.write(all_jpg, 4, pre[-2])

                    if k == new_classes[int(pre[-1])]:
                        sheet1.write(all_jpg, 7, 1)
                    else:
                        sheet1.write(all_jpg, 7, 0)
    workbook.save(img_root + "/all_{0}{1}.xls".format(mode, tag))
