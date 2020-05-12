#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2019/7/24 
# @Author  : sunyihuan
'''
将JPGImages、Annotations下的文件，按ImageSets/Main下test.txt、train.txt目录拷贝到对应的test或者train文件夹中。
即：生成所有的train、test、val文件夹
'''

import shutil
from tqdm import tqdm
import os


def copy_img_to_only_dir(tpye, root_path, save_to_path):
    '''
    拷贝test、train图片、标注文件到独立的文件夹中
    :param tpye: 集合类型。train/test
    :param root_path: 原数据保存根目录
                        目录中含有的文件如下：JPGImages
                                              Annotations
                                              ImageSets
                                                  Main
                                                     test_all.txt
                                                     train_all.txt
                          其中，JPGImages目录下为所有jpg图片文件
                                Annotations目录下为所有xml标注文件
                                test.txt为所有的test集图片name（不含.jpg）
                                train.txt为所有的train集图片name（不含.jpg）
    :param save_to_path: 要保存的文件目录，如："./15classes_0722_train"
    :return:
    '''
    txt_name = root_path + "/" + "ImageSets/Main/" + "{}.txt".format(tpye)
    txt_file = open(txt_name, "r")
    txt_files = txt_file.readlines()
    print(len(txt_files))

    img_dir = save_to_path + "/JPGImages"
    xml_dir = save_to_path + "/Annotations"
    if not os.path.exists(xml_dir): os.mkdir(xml_dir)
    if not os.path.exists(img_dir): os.mkdir(img_dir)
    for txt_file_one in tqdm(txt_files):
        txt_file_one = txt_file_one.strip()
        img_name = txt_file_one + ".jpg"
        xml_name = txt_file_one + ".xml"
        img_file = root_path + "/JPGImages/" + img_name
        xml_file = root_path + "/Annotations/" + xml_name

        shutil.copy(img_file, img_dir + "/" + img_name)
        shutil.copy(xml_file, xml_dir + "/" + xml_name)


if __name__ == "__main__":
    tpye = "val"
    root_path = "E:/DataSets/KX_FOODSets_model_data/21classes_0807"
    save_to_path = "E:/DataSets/KX_FOODSets_model_data/21classes_0807_val"
    copy_img_to_only_dir(tpye, root_path, save_to_path)
