# -*- coding: utf-8 -*-
# @Time    : 2020/7/30
# @Author  : sunyihuan
# @File    : from_txt_generate_classification_data.py
'''
从目标检测标签文件txt中生成分类文件夹
'''

import os
import shutil
from multi_detection.core.config import cfg
from tqdm import tqdm


def read_class_names(class_file_name):
    '''loads class name from a file'''
    names = []
    with open(class_file_name, 'r') as data:
        for name in data:
            names.append(name.strip('\n'))
    return names


def generate_data(classes, txt_path, data_root):
    txt_list = open(txt_path, "r").readlines()
    print(len(txt_list))
    for t in tqdm(txt_list):
        t = t.strip()
        if len(t.split(" ")) > 2:
            if "_hot.jpg" not in t:
                cls = classes[int(t.split(" ")[-1].split(",")[-1])]
                data_dir = data_root + "/" + cls
                if not os.path.exists(data_dir): os.mkdir(data_dir)
                if os.path.exists(t.split(" ")[0]):
                    dst_img = t.split(" ")[0].split("/")[-1]
                    shutil.copy(t.split(" ")[0], data_dir + "/" + dst_img)


if __name__ == "__main__":
    txt_path = "E:/kx_detection/multi_detection/data/dataset/202007/test39_new.txt"
    classes = read_class_names(cfg.YOLO.CLASSES)
    data_root = "E:/DataSets/classification_data/202007data"
    generate_data(classes, txt_path, data_root)
