# -*- encoding: utf-8 -*-

"""
@File    : check_local.py
@Time    : 2019/9/11 9:07
@Author  : sunyihuan
"""

import os
from tqdm import tqdm
import shutil
from PIL import Image
import numpy as np

from multi_detection.service.service import *
import time


def check_file(file_path):
    '''
    检测本地图片的分类情况
    :param file_path: 待检测文件路径
    :return: model_res,expert_res 模型预测,专家过滤后预测
    '''
    image = cv2.imread(file_path)  # 图片读取
    bboxes_pr, layer_n = Y.predict(image)  # 预测结果
    return layer_n


def check_dir(file_dir):
    '''
    :param file_dir:待测试文件夹路径
    :param classes:int 分类数
    :return:
    '''
    file_list = os.listdir(file_dir)
    divide_dirs = ["bottom", "middle", "top", "others"]

    for dir in divide_dirs:
        fdir = file_dir + '/' + dir
        if not os.path.exists(fdir):  # 路径是否存在
            os.makedirs(fdir)

    for file in tqdm(file_list, ncols=70, leave=False, unit='b'):
        if file != ".DS_Store" and file not in divide_dirs:
            try:
                print(file)
                # file_path = os.path.join(file_dir, file)
                file_path = file_dir + '/' + file
                print(file_path)
                model_res = check_file(file_path)
                print(model_res[0])
                div_path = os.path.join(file_dir, divide_dirs[int(model_res[0])], file)
                shutil.move(file_path, div_path)
            except:
                print("error:", file)


if __name__ == '__main__':
    Y = YoloPredict()
    for k in ["test", "val"]:
        file_dir = "C:/Users/sunyihuan/Desktop/WLS/kaopankaojia_model_results0911/old_sets_test_val/{}".format(k)
        check_dir(file_dir)
