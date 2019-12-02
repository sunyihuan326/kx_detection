# -*- encoding: utf-8 -*-

"""
将layer数据中的bottom、middle、top、others按train、test、val分为对应的数据集

@File    : layer_train_test_split.py
@Time    : 2019/10/25 9:49
@Author  : sunyihuan
"""
from tqdm import tqdm
import os
import shutil

# 创建layer_train、layer_test、layer_val文件夹
layer_train_dir = "E:/已标数据备份/peanuts_layer/train"
layer_test_dir = "E:/已标数据备份/peanuts_layer/test"
layer_val_dir = "E:/已标数据备份/peanuts_layer/val"
if os.path.exists(layer_train_dir): shutil.rmtree(layer_train_dir)  # 判断是否存在，存在则删除
os.mkdir(layer_train_dir)  # 创建文件夹
if os.path.exists(layer_test_dir): shutil.rmtree(layer_test_dir)
os.mkdir(layer_test_dir)
if os.path.exists(layer_val_dir): shutil.rmtree(layer_val_dir)
os.mkdir(layer_val_dir)


def layer_data_split(layer_root, txt_root):
    '''
    分layer数据，到对应的train/bottom、train/middle、train/others、train/top
                        test/bottom、test/middle、test/others、test/top
                        val/bottom、val/middle、val/others、val/top 中
    :param layer_root: layer数据根目录
    :param txt_root: train、test、val 对应txt数据目录
    :return:
    '''
    # 获取train中的所有文件名称
    train_txt = txt_root + "/" + "train.txt"
    txt_file = open(train_txt, "r")
    train_txt_files = txt_file.readlines()
    train_txt_files = [v.strip() for v in train_txt_files]
    # 获取test中的所有文件名称
    test_txt = txt_root + "/" + "test.txt"
    txt_file = open(test_txt, "r")
    test_txt_files = txt_file.readlines()
    test_txt_files = [v.strip() for v in test_txt_files]
    # 获取val中的所有文件名称
    val_txt = txt_root + "/" + "val.txt"
    txt_file = open(val_txt, "r")
    val_txt_files = txt_file.readlines()
    val_txt_files = [v.strip() for v in val_txt_files]

    for c in ["bottom", "middle", "top", "others"]:
        c_dir = layer_root + "/" + c
        os.mkdir(layer_train_dir + "/" + c)  # train下创建bottom、middle、top、others
        os.mkdir(layer_test_dir + "/" + c)  # test下创建bottom、middle、top、others
        os.mkdir(layer_val_dir + "/" + c)  # val下创建bottom、middle、top、others
        for img in tqdm(os.listdir(c_dir)):
            if img.split(".")[0] in val_txt_files:  # 判断若img名称在val中，拷贝图片至val中
                shutil.copy(c_dir + "/" + img, layer_val_dir + "/" + c + "/" + img)
            elif img.split(".")[0] in test_txt_files:  # 判断若img名称在test中，拷贝图片至test中
                shutil.copy(c_dir + "/" + img, layer_test_dir + "/" + c + "/" + img)
            else:  # 其他的拷贝图片至train中
                shutil.copy(c_dir + "/" + img, layer_train_dir + "/" + c + "/" + img)


if __name__ == "__main__":
    layer_root = "E:/已标数据备份/peanuts_layer/JPGImages"  # layer数据根目录
    txt_root = "E:/已标数据备份/peanuts_layer/ImageSets/Main"  # txt数据文件夹地址

    layer_data_split(layer_root, txt_root)
