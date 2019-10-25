# -*- encoding: utf-8 -*-

"""
将分类文件夹中的图片移至根目录下
@File    : move_jpgs.py
@Time    : 2019/10/25 9:20
@Author  : sunyihuan
"""
import shutil
import os
from tqdm import tqdm

clasees = ["BeefSteak", "CartoonCookies", "ChickenWings", "ChiffonCake6", "ChiffonCake8",
           "Cookies", "CranberryCookies", "CupCake", "EggTart", "EggTartBig",
           "nofood", "Peanuts", "Pizzafour", "Pizzaone", "Pizzasix",
           "Pizzatwo", "PorkChops", "PotatoCut", "Potatol", "Potatom",
           "Potatos", "SweetPotatoCut", "SweetPotatol", "SweetPotatom", "SweetPotatos",
           "RoastedChicken", "Toast"]  # 类别


def move_JPGImages(img_root_path):
    '''
    拷贝数据
    :param img_root_path: 根目录地址
    :return:
    '''
    for c in clasees:
        img_dir = img_root_path + "/" + c
        for img in os.listdir(img_dir):
            img_path = img_dir + "/" + img
            shutil.copy(img_path, img_root_path + "/" + img)


def move_layer_images(layer_root_path):
    '''
    将各类layer数据拷贝至统一layer中，如：将所有类别中的bottom下数据，拷贝至根目录下的bottom文件中
    :param layer_root_path: layer数据根目录地址
    :return:
    '''
    # 根目录下创建bottom文件夹
    bottom_dir = layer_root_path + "/" + "bottom"
    if os.path.exists(bottom_dir): shutil.rmtree(bottom_dir)
    os.mkdir(bottom_dir)
    # 根目录下创建middle文件夹
    middle_dir = layer_root_path + "/" + "middle"
    if os.path.exists(middle_dir): shutil.rmtree(middle_dir)
    os.mkdir(middle_dir)
    # 根目录下创建top文件夹
    top_dir = layer_root_path + "/" + "top"
    if os.path.exists(top_dir): shutil.rmtree(top_dir)
    os.mkdir(top_dir)
    # 根目录下创建others文件夹
    others_dir = layer_root_path + "/" + "others"
    if os.path.exists(others_dir): shutil.rmtree(others_dir)
    os.mkdir(others_dir)

    for c in clasees:
        bottom_img_dir = layer_root_path + "/" + c + "/bottom"  # 单一类别下bottom地址
        for img in tqdm(os.listdir(bottom_img_dir)):
            img_path = bottom_img_dir + "/" + img  # 单一类别下bottom中图片路径
            shutil.copy(img_path, bottom_dir + "/" + img)  # 拷贝图片至根目录下的bottom中

        middle_img_dir = layer_root_path + "/" + c + "/middle"  # 单一类别下middle地址
        for img in tqdm(os.listdir(middle_img_dir)):
            img_path = middle_img_dir + "/" + img  # 单一类别下middle中图片路径
            shutil.copy(img_path, middle_dir + "/" + img)  # 拷贝图片至根目录下的middle中

        top_img_dir = layer_root_path + "/" + c + "/top"  # 单一类别下top地址
        for img in tqdm(os.listdir(top_img_dir)):
            img_path = top_img_dir + "/" + img  # 单一类别下top中图片路径
            shutil.copy(img_path, top_dir + "/" + img)  # 拷贝图片至根目录下的top中

        others_img_dir = layer_root_path + "/" + c + "/others"  # 单一类别下others地址
        if os.path.exists(others_img_dir):
            for img in tqdm(os.listdir(others_img_dir)):
                img_path = others_img_dir + "/" + img  # 单一类别下others中图片路径
                shutil.copy(img_path, others_dir + "/" + img)  # 拷贝图片至根目录下的others中


if __name__ == "__main__":
    # img_root_path = "E:/DataSets/KX_FOODSets_model_data/X_27classes_1025/JPGImages"  #图片根目录地址
    # move_JPGImages(img_root_path)
    layer_root_path = "E:/layer_data/X_KX_data"  # layer数据根目录地址
    move_layer_images(layer_root_path)
