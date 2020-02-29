# -*- coding: utf-8 -*-
# @Time    : 2020/2/29
# @Author  : sunyihuan
# @File    : copy_imge.py


import shutil
import os

img_dir = "E:/layer_data/26classes_0920/val"

dst_img_dir = "E:/DataSets/KX_FOODSets_model_data/XandOld20200229/layer_data/X_KX_data_27_1127/val"
for l in ["bottom", "middle", "others", "top"]:
    img_dir_name = img_dir + "/" + l
    for jpg_ in os.listdir(img_dir_name):
        jpg_name = img_dir_name + "/" + jpg_

        dst_jpg_name = dst_img_dir + "/" + l + "/" + jpg_

        print(jpg_name)
        print(dst_jpg_name)
        shutil.copy(jpg_name, dst_jpg_name)
