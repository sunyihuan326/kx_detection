# -*- coding: utf-8 -*-
# @Time    : 202003/3/6
# @Author  : sunyihuan
# @File    : from_jpgname_split_layer.py

'''
将img文件中图片按图片名称中的烤层信息分类到对应的烤层文件夹
'''

import os
import shutil
from tqdm import tqdm


def copy_layer_jpgs(img_dir, layer_dir):
    '''
    将img文件中图片按图片名称中的烤层标签分类到对应的烤层文件夹
    :param img_dir: 图片文件夹
    :param layer_dir: 烤层分类文件夹
    :return:
    '''
    bottom_dir = layer_dir + "/bottom"  # 底层
    if not os.path.exists(bottom_dir): os.mkdir(bottom_dir)

    middle_dir = layer_dir + "/middle"  # 中层
    if not os.path.exists(middle_dir): os.mkdir(middle_dir)

    top_dir = layer_dir + "/top"  # 上层
    if not os.path.exists(top_dir): os.mkdir(top_dir)

    others_dir = layer_dir + "/others"  # 其他
    if not os.path.exists(others_dir): os.mkdir(others_dir)

    for jpgname in os.listdir(img_dir):
        if jpgname.endswith(".jpg"):
            if "top" in jpgname:  # “shang” 上层标签
                shutil.copy(img_dir + "/" + jpgname, top_dir + "/" + jpgname)
            elif "middle" in jpgname:  # “zhong” 中层标签
                shutil.copy(img_dir + "/" + jpgname, middle_dir + "/" + jpgname)
            elif "bottom" in jpgname:  # “xia” 下层标签
                shutil.copy(img_dir + "/" + jpgname, bottom_dir + "/" + jpgname)
            elif "chazi" in jpgname:  # “chazi” 下层标签
                shutil.copy(img_dir + "/" + jpgname, others_dir + "/" + jpgname)
            else:
                print("error")
                print(jpgname)


if __name__ == "__main__":
    img_root = "E:/DataSets/X_3660_data/bu/20200902/JPGImages"
    layer_root = "E:/DataSets/X_3660_data/bu/20200902/layer_data"
    # cls_list = ["beefsteak", "bread", "cartooncookies", "chestnut", "chickenwings",
    #             "chiffoncake6", "chiffoncake8", "container", "container_nonhigh", "cookies",
    #             "cornone", "corntwo", "cranberrycookies", "cupcake", "drumsticks",
    #             "eggplant", "eggplant_cut_sauce", "eggtart", "fish", "hotdog",
    #             "peanuts", "pizzacut", "pizzaone", "pizzatwo", "porkchops",
    #             "potatocut", "potatol", "potatos", "redshrimp", "roastedchicken",
    #             "shrimp", "steamedbread", "strand", "sweetpotatocut", "sweetpotatol",
    #             "sweetpotatos", "taro", "toast", "duck"]
    cls_list= ["cranberrycookies"]
    for c in tqdm(cls_list):
        img_dir = img_root + "/" + c
        layer_dir = layer_root + "/" + c
        if not os.path.exists(layer_dir): os.mkdir(layer_dir)
        copy_layer_jpgs(img_dir, layer_dir)
    # img_dir = "E:/已标数据备份/二期数据/第一批/X5_20200310/chestnut"
    # layer_dir = "E:/已标数据备份/二期数据/第一批/X5_20200310/layer_data/chestnut"
    #
    # copy_layer_jpgs(img_dir, layer_dir)
