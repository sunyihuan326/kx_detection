# -*- coding: utf-8 -*-
# @Time    : 2021/4/2
# @Author  : sunyihuan
# @File    : extract_file_from_jpg_xml.py
'''
从各类jpg文件中抽取x张，同时抽取对应的xml文件

'''

import os
import random
import shutil
from tqdm import tqdm

def random_jpg(jpg_root, nums, save_root):
    '''
    从jpg_root中，按类别，大于nums的随机抽取nums张，小于直接拷贝
    文件转存至save_root中
    :param jpg_root:
    :param xml_root:
    :param nums:
    :param save_root:
    :return:
    '''
    class_list = os.listdir(jpg_root)
    save_jpg_dir = save_root + "/JPGImages"
    if not os.path.exists(save_jpg_dir): os.mkdir(save_jpg_dir)

    for c in tqdm(class_list):
        save_c_jpg_dir = save_jpg_dir + "/" + c
        j_list = os.listdir(jpg_root + "/" + c)
        jpg_n = len(j_list)
        if jpg_n > nums:
            if not os.path.exists(save_c_jpg_dir): os.mkdir(save_c_jpg_dir)
            samples = random.sample(j_list, nums)
            for s in samples:
                shutil.copy(jpg_root + "/" + c + "/" + s, save_c_jpg_dir + "/" + s)  # 拷贝图片
        else:
            shutil.copytree(jpg_root + "/" + c, save_c_jpg_dir)  # 拷贝整个jpg文件夹

def random_jpg_xml(jpg_root, xml_root, nums, save_root):
    '''
    从jpg_root中，按类别，大于nums的随机抽取nums张，小于直接拷贝
    文件转存至save_root中
    :param jpg_root:
    :param xml_root:
    :param nums:
    :param save_root:
    :return:
    '''
    class_list = os.listdir(jpg_root)
    save_jpg_dir = save_root + "/JPGImages"
    save_xml_dir = save_root + "/Annotations"
    if not os.path.exists(save_jpg_dir): os.mkdir(save_jpg_dir)
    if not os.path.exists(save_xml_dir): os.mkdir(save_xml_dir)

    for c in tqdm(class_list):
        save_c_jpg_dir = save_jpg_dir + "/" + c
        save_c_xml_dir = save_xml_dir + "/" + c

        j_list = os.listdir(jpg_root + "/" + c)
        jpg_n = len(j_list)
        if jpg_n > nums:
            if not os.path.exists(save_c_jpg_dir): os.mkdir(save_c_jpg_dir)
            if not os.path.exists(save_c_xml_dir): os.mkdir(save_c_xml_dir)
            samples = random.sample(j_list, nums)
            for s in samples:
                xml_name = s.split(".jpg")[0] + ".xml"
                shutil.copy(jpg_root + "/" + c + "/" + s, save_c_jpg_dir + "/" + s)  # 拷贝图片
                if os.path.exists(xml_root + "/" + c + "/" + xml_name):
                    shutil.copy(xml_root + "/" + c + "/" + xml_name, save_c_xml_dir + "/" + xml_name)  # 拷贝xml文件
        else:
            shutil.copytree(jpg_root + "/" + c, save_c_jpg_dir)  # 拷贝整个jpg文件夹
            shutil.copytree(xml_root + "/" + c, save_c_xml_dir)  # 拷贝整个xml文件夹


if __name__ == "__main__":
    jpg_root = "F:/serve_data/ZG_data/20210129/biaozhu_20210428/yuantu"
    xml_root = "F:/serve_data/202101-03formodel/Annotations"
    save_root = "F:/serve_data/ZG_data/20210129/biaozhu_20210428/exrtact_file"
    nums = 700
    random_jpg(jpg_root, nums, save_root)
