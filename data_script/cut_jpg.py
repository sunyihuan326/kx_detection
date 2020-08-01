# -*- coding: utf-8 -*-
# @Time    : 2020/7/16
# @Author  : sunyihuan
# @File    : cut_jpg.py
'''
文件夹中删除部分数据
随机删除
'''
import os
import random
from tqdm import tqdm


def cut_img_from_nums(img_dir, save_nums):
    '''
    从数量上删除部分图片
    :param img_dir:
    :param save_nums:
    :return:
    '''
    all_nums = len(os.listdir(img_dir))
    img_list = os.listdir(img_dir)
    random.shuffle(img_list)
    for i in range(all_nums - save_nums):
        os.remove(img_dir + "/" + img_list[i])
        print("remove:::",img_dir + "/" + img_list[i])


def cut_img_from_str(img_dir, org_str):
    '''
    删除图片名中带有org_str的所有图片
    :param img_dir: 图片地址
    :param org_str: 特殊字符
    :return:
    '''
    img_list = os.listdir(img_dir)
    random.shuffle(img_list)
    for i in range(len(img_list)):
        if org_str in img_list[i]:
            os.remove(img_dir + "/" + img_list[i])


if __name__ == "__main__":
    img_root = "E:/check_2_phase/JPGImages_2"
    # save_nums = 900
    for c in tqdm(["bread", "chestnut", "container", "cornone", "corntwo", "drumsticks", "duck",
                   "eggplant", "eggplant_cut_sauce", "fish", "hotdog", "redshrimp", "strand", "taro"]):
        img_dir__ = img_root + "/" + c
        for kk in ["bottom", "middle", "top", "others"]:
            try:
                img_dir = img_dir__ + "/" + kk
                cut_img_from_str(img_dir, "X5")
            except:
                print(img_dir__, kk)
