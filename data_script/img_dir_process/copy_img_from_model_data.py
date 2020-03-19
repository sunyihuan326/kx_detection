# -*- encoding: utf-8 -*-

"""
从模型训练数据中拷贝部分数据至新文件夹

@File    : copy_img_from_model_data.py
@Time    : 2019/8/12 11:06
@Author  : sunyihuan
"""
import os
import shutil

model_data_dir = "E:/DataSets/KX_FOODSets_model_data/23classes_0808/Annotations"
save_dir = "E:/已标数据备份/KX38I95FOODSETS_Annotation_0806/Anotations/PotatoS"

for xmlfile in os.listdir(model_data_dir):
    if xmlfile.endswith("xml"):
        if "tudou_xiao" in xmlfile and "xxxxxtudou_xiao" not in xmlfile:
            print(1)
            shutil.copy(model_data_dir + "/" + xmlfile, save_dir + "/" + xmlfile)
