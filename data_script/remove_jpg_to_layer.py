# -*- coding: utf-8 -*-
# @Time    : 2020/7/29
# @Author  : sunyihuan
# @File    : remove_jpg_to_layer.py
'''
按图片名称中bottom、middle、top等将图片分至对应文件夹
'''
import os
import shutil
from tqdm import tqdm


def split_layer(img_dir):
    t_list = ["bottom", "middle", "top", "others"]
    for t in t_list:
        if not os.path.exists(img_dir + "/{}".format(t)):
            os.mkdir(img_dir + "/{}".format(t))
        for img in os.listdir(img_dir):
            if img.endswith(".jpg"):
                if t in img:
                    shutil.move(img_dir + "/" + img, img_dir + "/{}".format(t) + "/" + img)


if __name__ == "__main__":
    img_root = "E:/DataSets/2020_two_phase_KXData/202005bu/layer_data"
    for c in tqdm(["chestnut", "container", "cornone", "corntwo", "drumsticks",
                   "duck", "eggplant", "eggplant_cut_sauce", "fish",
                   "hotdog", "redshrimp", "strand", "taro", ]):
        img_dir = img_root + "/" + c
        split_layer(img_dir)
