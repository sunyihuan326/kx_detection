# -*- coding: utf-8 -*-
# @Time    : 2020/7/27
# @Author  : sunyihuan
# @File    : major_acc_print.py
'''
针对22分类模型，输出大类结果，及大类topn结果
'''

import cv2
import numpy as np
import tensorflow as tf
import multi_detection.core.utils as utils
import os
import time
from multi_detection.food_correct_utils import correct_bboxes, cls_major_result
import shutil
from tqdm import tqdm
import xlwt

# gpu限制
gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.2)
config = tf.ConfigProto(gpu_options=gpu_options)


class YoloPredict(object):
    def __init__(self):
        self.input_size = 320  # 输入图片尺寸（默认正方形）
        self.num_classes = 22  # 种类数
        self.top_n = 10
        self.score_cls_threshold = 0.0000001
        self.score_threshold = 0.45
        self.iou_threshold = 0.5
        self.weight_file = "E:/ckpt_dirs/Food_detection/multi_food3/20200717/yolov3_train_loss=4.9898.ckpt-197"  # ckpt文件地址
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

    def get_top_cls(self, pred_bbox, org_h, org_w, top_n):
        '''
        12大类输出

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
        best_major_bboxes = {}
        for bc in best_bboxes.keys():
            major_c_score = best_bboxes[bc]
            major_c = cls_major_result(bc)
            if major_c not in best_major_bboxes.keys():  # 如果best_major_bboxes中无该类别，直接加入
                best_major_bboxes[major_c] = major_c_score
            else:
                if best_major_bboxes[major_c] < major_c_score:  # 如果best_major_bboxes中有该类别，加入大的那个
                    best_major_bboxes[major_c] = major_c_score
        best_major_bboxes = sorted(best_major_bboxes.items(), key=lambda best_major_bboxes: best_major_bboxes[1],
                                   reverse=True)
        return best_major_bboxes[:top_n]

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

        return org_h, org_w, pred_bbox, bboxes, layer_n

    def result(self, image_path):
        '''
        得出预测结果并保存
        :param image_path: 图片地址
        :param save_dir: 预测结果原图标注框，保存地址
        :return:
        '''
        image = cv2.imread(image_path)  # 图片读取
        # image = utils.white_balance(image)  # 图片白平衡处理
        org_h, org_w, pred_bbox, bboxes, layer_n = self.predict(image)  # 预测结果
        # print(bboxes_pr)
        # print(layer_n)

        # if self.write_image:
        #     image = utils.draw_bbox(image, bboxes_pr, show_label=self.show_label)
        #     drawed_img_save_to_path = str(image_path).split("/")[-1]
        #     drawed_img_save_to_path = str(drawed_img_save_to_path).split(".")[0] + "_" + str(
        #         layer_n) + ".jpg"  # 图片保存地址，烤层结果在命名中
        #     # cv2.imshow('Detection result', image)
        #     cv2.imwrite(save_dir + "/" + drawed_img_save_to_path, image)  # 保存图片
        return org_h, org_w, pred_bbox, bboxes, layer_n


if __name__ == '__main__':
    start_time = time.time()
    img_root =  "E:/check_2_phase/JPGImages__0"  # 图片文件地址
    Y = YoloPredict()
    end_time0 = time.time()
    print("model loading time:", end_time0 - start_time)
    clses = ["beefsteak", "cartooncookies", "chickenwings", "chiffoncake6",
             "cookies", "cranberrycookies", "cupcake", "eggtart", "nofood", "peanuts",
             "pizzacut", "pizzaone", "pizzatwo", "porkchops", "potatocut",
             "potatol", "potatos", "sweetpotatocut", "sweetpotatol", "sweetpotatos",
             "roastedchicken", "toast"]

    # clses = ["sweetpotatol"]
    classes_id22 = {"beefsteak": 0, "cartooncookies": 1, "chickenwings": 2, "chiffoncake6": 3, "chiffoncake8": 3,
                    "cookies": 4, "cranberrycookies": 5, "cupcake": 6, "eggtart": 7, "nofood": 8, "peanuts": 9,
                    "pizzacut": 10, "pizzaone": 11, "pizzatwo": 12, "porkchops": 13, "potatocut": 14,
                    "potatol": 15, "potatos": 16, "sweetpotatocut": 17, "sweetpotatol": 18, "sweetpotatos": 19,
                    "roastedchicken": 20, "toast": 21, "potatom": 15, "sweetpotatom": 18, }

    all_jpg = 0
    acc_jpg = 0
    noresults = 0
    food_topn_acc_nums = 0

    workbook = xlwt.Workbook(encoding='utf-8')
    sheet1 = workbook.add_sheet("multi_food")
    sheet1.write(0, 0, "classes")

    sheet1.write(0, 1, "acc")
    sheet1.write(0, 2, "major_acc")
    sheet1.write(0, 3, "major_top3_acc")
    sheet1.write(0, 4, "major_top5_acc")
    sheet1.write(0, 5, "major_top8_acc")
    sheet1.write(0, 6, "major_top10_acc")
    for c in clses:
        all_c_jpg = 0
        acc_c_jpg = 0
        major_c_jpg = 0
        food_top3_cls_acc_nums = 0
        food_top5_cls_acc_nums = 0
        food_top8_cls_acc_nums = 0
        food_top10_cls_acc_nums = 0

        if c not in ["nofood", "sweetpotatom", "potatom"]:
            img_dir = img_root + "/" + c

            for img in tqdm(os.listdir(img_dir)):
                if img.endswith("jpg"):
                    all_c_jpg += 1
                    all_jpg += 1
                    img_path = img_dir + "/" + img
                    end_time1 = time.time()

                    org_h, org_w, pred_bbox, bboxes, layer_n = Y.result(img_path)
                    best_bboxes_3 = Y.get_top_cls(pred_bbox, org_h, org_w, 3)  # 获取top_n类别和置信度
                    best_bboxes_5 = Y.get_top_cls(pred_bbox, org_h, org_w, 5)  # 获取top_n类别和置信度
                    best_bboxes_8 = Y.get_top_cls(pred_bbox, org_h, org_w, 8)  # 获取top_n类别和置信度
                    best_bboxes_10 = Y.get_top_cls(pred_bbox, org_h, org_w, 10)  # 获取top_n类别和置信度

                    if len(bboxes) == 0:
                        noresults += 1
                    else:
                        bboxes_pr, layer_n = correct_bboxes(bboxes, bboxes)  # 矫正输出结果
                        # bboxes_pr, layer_n = get_potatoml(bboxes_pr, layer_n)  # 根据输出结果对中大红薯，中大土豆做输出
                        if len(bboxes_pr) == 0:
                            noresults += 1
                            # if not os.path.exists(img_dir + "/noresult"): os.mkdir(img_dir + "/noresult")
                            # shutil.move(img_path, img_dir + "/noresult" + "/" + img)
                        else:
                            pre = int(bboxes_pr[0][-1])
                            major_pre = cls_major_result(pre)
                            major_true = cls_major_result(int(classes_id22[c]))
                            # print(pre, int(classes_id22[c]))
                            # print(major_pre, major_true)

                            if major_true in dict(best_bboxes_3).keys():
                                # food_topn_acc_b+=1
                                food_top3_cls_acc_nums += 1
                            if major_true in dict(best_bboxes_5).keys():
                                # food_topn_acc_b+=1
                                food_top5_cls_acc_nums += 1
                            if major_true in dict(best_bboxes_8).keys():
                                # food_topn_acc_b+=1
                                food_top8_cls_acc_nums += 1
                            if major_true in dict(best_bboxes_10).keys():
                                # food_topn_acc_b+=1
                                food_top10_cls_acc_nums += 1
                            else:
                                print(major_true, dict(best_bboxes_10).keys())

                            if pre == int(classes_id22[c]):
                                acc_jpg += 1
                                acc_c_jpg += 1
                            if int(major_true) == int(major_pre):
                                major_c_jpg += 1
                            # if not os.path.exists(img_dir + "/" + str(clses[pre])): os.mkdir(
                            #     img_dir + "/" + str(clses[pre]))
                            # shutil.move(img_path, img_dir + "/" + str(clses[pre]) + "/" + img)
            print(acc_c_jpg, major_c_jpg,food_top3_cls_acc_nums)
            sheet1.write(clses.index(c) + 1, 0, c)
            sheet1.write(clses.index(c) + 1, 1, acc_c_jpg / all_c_jpg)
            sheet1.write(clses.index(c) + 1, 2, major_c_jpg / all_c_jpg)
            sheet1.write(clses.index(c) + 1, 3, food_top3_cls_acc_nums / all_c_jpg)
            sheet1.write(clses.index(c) + 1, 4, food_top5_cls_acc_nums / all_c_jpg)
            sheet1.write(clses.index(c) + 1, 5, food_top8_cls_acc_nums / all_c_jpg)
            sheet1.write(clses.index(c) + 1, 6, food_top10_cls_acc_nums / all_c_jpg)
    workbook.save("E:/check_2_phase/JPGImages__0/major_top_n_m3_0717.xls")
    print("正确数：", acc_jpg)
    print("无任何结果数：", noresults)
    # print("top3正确数结果数：", food_top3_cls_acc_nums)
    print("总数：", all_jpg)
    print("正确率：：：", acc_jpg / all_jpg)
    # print("topn正确率：：：：", food_topn_acc_nums / all_jpg)
    print("无任何结果占比：：：：", noresults / all_jpg)
