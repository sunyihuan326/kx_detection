# -*- coding: utf-8 -*-
# @Time    : 2020/10/21
# @Author  : sunyihuan
# @File    : data_process_aug.py
'''
数据脚本：原图做增强、并生成对应的txt脚本供模型使用，同时合并数据至统一txt文件

'''

import cv2
import os
from tqdm import tqdm
import shutil
import numpy as np
from PIL import Image, ImageEnhance
import matplotlib.pyplot as plt
import random


class aug_mist(object):
    '''
    针对原图，做增强
    增强方法：泛绿、泛红、泛紫、泛黄
    '''

    def __init__(self, img_dir, img_save_dir):
        self.img_dir = img_dir
        self.img_save_dir = img_save_dir

    def img_mist(self, typ):
        '''
        图片中加入水雾效果
        :param typ: 增强类型，如：lv、hot、huang、zi
        :return:
        '''
        for img_name in tqdm(os.listdir(self.img_dir)):
            if "_hot.jpg" not in img_name and "_lv.jpg" not in img_name and "_zi.jpg" not in img_name and "_huang.jpg" not in img_name:
                try:
                    self.img_path = self.img_dir + "/" + img_name
                    self.img_save_name = str(img_name).split(".")[0] + "_{}".format(typ) + ".jpg"  # 图片名称
                    self.img_save_path = self.img_save_dir + "/" + self.img_save_name  # 图片保存地址

                    img1 = cv2.imread(self.img_path)  # 目标图片
                    img2 = cv2.imread('C:/Users/sunyihuan/Desktop/material/{}.jpg'.format(typ))  # 增强融合图片
                    img2 = cv2.resize(img2, (img1.shape[1], img1.shape[0]))  # 统一图片大小
                    dst = cv2.addWeighted(img1, 0.8, img2, 0.3, 0)  # 图片融合
                    blank = np.zeros(img1.shape, img1.dtype)
                    dst = cv2.addWeighted(dst, 0.8, blank, 0.1, 0)  # 图片融合

                    # 图片增强
                    cv2.imwrite(self.img_save_path, dst)
                    img = Image.open(self.img_save_path)
                    tmp = ImageEnhance.Brightness(img)
                    img = tmp.enhance(1.1)
                    enh_col = ImageEnhance.Contrast(img)
                    img = enh_col.enhance(1.2)
                    img = np.array(img)
                    plt.imsave(self.img_save_path, img.astype(np.uint8))  # 保存图片
                except:
                    print(img_name)


class generate_txt(object):
    '''
    生成增强txt文件
    '''

    def change_txt_jpgname(self, original_txt, save_txt, file_path, typ):
        '''
        生成对应的txt文件
        可生成独立的增强文件和服务端对应数据的txt文件
        :param save_txt:
        :param file_path:
        :param typ: 类型，可用：lv、zi、hot、huang、serve
        :return:
        '''
        self.original_txt = original_txt
        self.save_txt = save_txt
        txt_file = open(self.original_txt, "r")
        txt_files = txt_file.readlines()

        train_all_list = []
        for txt_file_one in txt_files:
            img_path_name = txt_file_one
            txt_file_name = file_path

            if typ == "serve":
                print(img_path_name)
                txt_file_name += img_path_name.split("JPGImages")[1]
                train_all_list.append(txt_file_name)  # 读取一个插入一个
            else:  # .jpg前的字段需要更改
                jpg_name = str(img_path_name.split("JPGImages")[1]).split(".jpg")[0] + "_{}.jpg".format(typ) + \
                           str(img_path_name.split("JPGImages")[1]).split(".jpg")[1]
                txt_file_name += jpg_name
                train_all_list.append(txt_file_name)

        file = open(self.save_txt, "w")
        for i in train_all_list:
            file.write(i)


class append_txt(object):
    '''
    合并所有txt至一个文件夹
    '''

    def __init__(self, txt_list):
        self.txt_list = txt_list

    def append_txt2all(self, save_txt_path, n):
        '''
        将txt_list中所有数据拷贝到一起并保存
        规则为：txt_list[0]+txt_list[1:]/n
        :param save_txt_path:
        :param n:
        :return:
        '''
        self.save_txt_path = save_txt_path
        assert len(self.txt_list) > 1
        txt0 = self.txt_list[0]

        train_txt_file = open(txt0, "r")
        train_txt_files = train_txt_file.readlines()
        all_txt_list = train_txt_files

        for txt_i in self.txt_list[1:]:
            test_txt_file = open(txt_i, "r")
            test_txt_files = test_txt_file.readlines()
            random.shuffle(test_txt_files)  # 打乱
            all_txt_list = list(set(all_txt_list) | set(test_txt_files[:int(len(test_txt_files) / n)]))

        new_txt_file = open(self.save_txt_path, "w")
        for i in all_txt_list:
            new_txt_file.write(i)


if __name__ == "__main__":
    root_dir = "F:/serve_data/for_model/202101_03"
    img_dir = "{}/JPGImages".format(root_dir)
    img_save_dir = "{}/JPGImages".format(root_dir)
    original_txt = "{}/train42.txt".format(root_dir)
    if not os.path.exists(img_save_dir): os.mkdir(img_save_dir)
    aug = aug_mist(img_dir, img_save_dir)  # 图片增强处理

    g_t = generate_txt()
    # 生成对应的增强txt文件
    for typ in ["lv", "zi", "hot", "huang"]:
        aug.img_mist(typ)  # 图片增强处理
        file_path = "F:/serve_data/for_model/202101_03/JPGImages"
        save_txt_path = "F:/serve_data/for_model/202101_03/train42_{}.txt".format(typ)
        g_t.change_txt_jpgname(original_txt, save_txt_path, file_path, typ)  # 生成增强单独txt文件
    # 合并原图txt和增强txt，且增强数据仅取部分
    txt_list = [
        "{}/train42.txt".format(root_dir),
        "{}/train42_lv.txt".format(root_dir),
        "{}/train42_zi.txt".format(root_dir),
        "{}/train42_hot.txt".format(root_dir),
        "{}/train42_huang.txt".format(root_dir),
    ]
    a_t = append_txt(txt_list)
    a_t.append_txt2all("{}/train42_huang_hot_lv_zi.txt".format(root_dir), 2)

    # 生成serve端数据
    serve_file_path = "/home/sunyihuan/sunyihuan_algorithm/data/KX_data/3660_202008/bu/serve_data/202101_03/JPGImages"
    save_serve_txt_path = "{}/serve_3660train42_huang_hot_lv_zi.txt".format(root_dir)
    g_t.change_txt_jpgname("{}/train42_huang_hot_lv_zi.txt".format(root_dir), save_serve_txt_path, serve_file_path,
                           "serve")
