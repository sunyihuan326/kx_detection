# -*- encoding: utf-8 -*-

"""
@File    : layer_copy_from_all_data.py
@Time    : 2019/12/3 9:07
@Author  : sunyihuan
"""
import os
import shutil


def copy_img_to_layer_dir(img_root, img_dir, save_root):
    bottom_dir = img_root + "/bottom"
    middle_dir = img_root + "/middle"
    top_dir = img_root + "/top"
    bottom_list = os.listdir(bottom_dir)
    middle_list = os.listdir(middle_dir)
    top_list = os.listdir(top_dir)

    save_bottom = save_root + "/bottom"
    save_middle = save_root + "/middle"
    save_top = save_root + "/top"

    if os.path.exists(save_bottom): shutil.rmtree(save_bottom)
    os.mkdir(save_bottom)
    if os.path.exists(save_middle): shutil.rmtree(save_middle)
    os.mkdir(save_middle)
    if os.path.exists(save_top): shutil.rmtree(save_top)
    os.mkdir(save_top)
    for img_name in os.listdir(img_dir):
        if img_name.endswith(".jpg"):
            if img_name in bottom_list:
                shutil.copy(img_dir + "/" + img_name, save_bottom + "/" + img_name)
            elif img_name in middle_list:
                shutil.copy(img_dir + "/" + img_name, save_middle + "/" + img_name)
            elif img_name in top_list:
                shutil.copy(img_dir + "/" + img_name, save_top + "/" + img_name)
            else:
                print(img_name)


if __name__ == "__main__":
    img_root = "E:/已标数据备份/X_KX_data/SweetPotatoCut"
    img_dir = "C:/Users/sunyihuan/Desktop/20191203新数据/数据清洗 by lzp 20191128/JPGImages/SweetPotatoCut"
    save_root = "C:/Users/sunyihuan/Desktop/20191203新数据/数据清洗 by lzp 20191128/JPGImages/layer_data/SweetPotatoCut"

    copy_img_to_layer_dir(img_root, img_dir, save_root)
