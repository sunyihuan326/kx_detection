# -*- encoding: utf-8 -*-

"""
@File    : tt.py
@Time    : 2019/12/6 15:51
@Author  : sunyihuan
"""
import os
import shutil

txt_dir = "E:/DataSets/2020_two_phase_KXData/20200318/ImageSets/Main"
JPGImages = "E:/DataSets/2020_two_phase_KXData/20200318/JPGImages"
JPGImages_save = "E:/DataSets/2020_two_phase_KXData/20200318/JPGImages_save"
if not os.path.exists(JPGImages_save): os.mkdir(JPGImages_save)

for t in os.listdir(txt_dir):
    if "_val" in t:
        cls = t.split("_")[0]
        if not os.path.exists(JPGImages_save + "/" + cls): os.mkdir(JPGImages_save + "/" + cls)
        file_list = open(txt_dir + "/" + t, "r").readlines()
        for f in file_list:
            jpg_name = f.strip() + ".jpg"
            shutil.copy(JPGImages + "/" + jpg_name, JPGImages_save + "/" + cls + "/" + jpg_name)

# txt_path_1203 = "E:/DataSets/KX_FOODSets_model_data/20191206data/ImageSets/Main/train_all.txt"
# txt_file_1203 = open(txt_path_1203, "r")
# txt_files_1203 = txt_file_1203.readlines()
# txt_files_1203 = [a.strip() for a in txt_files_1203]
# print(len(txt_files_1203))
# print(len(set(txt_files_1203)))
#
# txt_path_ = "E:/DataSets/KX_FOODSets_model_data/20191206data/ImageSets/Main/test_all.txt"
# txt_path_ = open(txt_path_, "r")
# txt_path_ = txt_path_.readlines()
# print(len(txt_path_))
#
# txt_path = "E:/DataSets/KX_FOODSets_model_data/20191206data/ImageSets/Main/val.txt"
# txt_path_0 = open(txt_path, "r")
# txt_path_0 = txt_path_0.readlines()
# print(len(txt_path_0))
#
#
# u = set(txt_files_1203) & set(txt_path_) & set(txt_path_0)
# print(u)
#
# img_path = "E:/DataSets/KX_FOODSets_model_data/20191206data/JPGImages"
# img_list = [img_name.split(".jpg")[0] for img_name in os.listdir(img_path)]
# c = 0
# for t in txt_files_1203:
#     if t not in img_list:
#         c += 1
#
# print(c)
