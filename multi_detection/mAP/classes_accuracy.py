#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2019/7/31 
# @Author  : sunyihuan
'''
从ground-truth文件、predicted文件中查看标签的准确率

并将标签判断错误的图片拷贝到error文件中

'''
import os
import shutil
from sklearn.metrics import confusion_matrix

gt_txt_root = "E:/multi_food_detection/mAP/ground-truth"
pre_txt_root = "E:/multi_food_detection/mAP/predicted"

CLASSES = ["beefsteak", "cartooncookies", "chickenwings", "chiffoncake", "cookies",
           "cranberrycookies", "cupcake", "eggtart", "nofood", "peanuts",
           "pizza", "porkchops", "purplesweetpotato", "roastedchicken", "toast",
           "potatos", "potatom", "potatol", "sweetpotatos", "sweetpotatom", "sweetpotatol",
           "potatocut", "sweetpotatocut", "pizzaone", "pizzatwo", "pizzafour", "pizzasix"]


def get_accuracy(error_write=True):
    '''
    由txt文件，查看classes准确率
    :param error_write: 是否将错误图片数据写入到error文件中，True/False
    :return: error_,
            error_c,
            acc
    '''

    error_dir = "E:/Joyoung_WLS_github/tf_yolov3/mAP/error/"
    if os.path.exists(error_dir): shutil.rmtree(error_dir)
    os.mkdir(error_dir)

    detection_dir = "E:/multi_food_detection/data/detection/"

    pre_txt_list = os.listdir(pre_txt_root)

    class_true = []
    class_pre = []
    no_result = {}

    error_ = 0  # 输出标签种类不等于1的nums
    error_c = 0  # 输出标签种类错误的nums
    acc = 0  # 输出标签种类正确的nums
    error_noresults = 0  # 输出无结果的nums
    for pre in pre_txt_list:
        if pre.endswith("txt"):
            with open(os.path.join(pre_txt_root, pre), "r") as f:
                pre_c = {}
                for line in f.readlines():
                    c = line.split(" ")[0]
                    if c not in pre_c.keys():
                        pre_c[c] = 1
                    else:
                        pre_c[c] += 1
                pre_cc = sorted(pre_c.items(), key=lambda x: x[1], reverse=True)
                with open(os.path.join(gt_txt_root, pre), "r") as fg:
                    for lin in fg.readlines():
                        cc = lin.split(" ")[0]

                if len(pre_cc) > 0:  # 将结果写入到class_true、class_pre
                    class_true.append(CLASSES.index(cc))
                    class_pre.append(CLASSES.index(pre_cc[0][0]))

                if len(pre_cc) == 1:
                    if pre_cc[0][0] != cc:
                        print(pre)
                        error_c += 1
                        if error_write:
                            shutil.copy(detection_dir + pre.split(".")[0] + ".jpg",
                                        error_dir + pre.split(".")[0] + ".jpg")  # 拷贝错误图片到error

                            shutil.copy(os.path.join(gt_txt_root, pre),
                                        error_dir + pre.split(".")[0] + "_gt.txt")  # 拷贝ground_truth文件
                            shutil.copy(os.path.join(pre_txt_root, pre),
                                        error_dir + pre.split(".")[0] + "_pre.txt")  # 拷贝predicted文件
                    else:
                        acc += 1
                else:
                    error_ += 1
                    print(pre)
                    if error_write:
                        shutil.copy(detection_dir + pre.split(".")[0] + ".jpg",
                                    error_dir + pre.split(".")[0] + ".jpg")  # 拷贝错误图片到error

                        shutil.copy(os.path.join(gt_txt_root, pre),
                                    error_dir + pre.split(".")[0] + "_gt.txt")  # 拷贝ground_truth文件
                        shutil.copy(os.path.join(pre_txt_root, pre),
                                    error_dir + pre.split(".")[0] + "_pre.txt")  # 拷贝predicted文件
                    if len(pre_cc) == 0:
                        error_noresults += 1

                        if cc not in no_result.keys():  # no result写入到字典no_result中
                            no_result[cc] = 1
                        else:
                            no_result[cc] += 1
    matrix = confusion_matrix(y_pred=class_pre, y_true=class_true)
    print(matrix)
    print("no_result:", no_result)
    return error_, error_c, acc, round(100 * acc / (error_ + error_c + acc), 2), error_noresults


if __name__ == "__main__":
    error_, error_c, acc, acc_percent, error_noresults = get_accuracy()
    print("输出标签不唯一的总数：", error_)
    print("标签错误数量：", error_c)
    print("标签正确数量：", acc)
    print("class 标签准确率：  ", str(acc_percent) + "%")
    print("无任何结果输出数量：", error_noresults)

