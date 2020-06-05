# -*- coding: utf-8 -*-
# @Time    : 2020/5/22
# @Author  : sunyihuan
# @File    : copy_xml_from_jpg.py
import shutil
import os
from tqdm import tqdm

xml_root = "E:/DataSets/X_data_27classes/Xdata_he/Annotations"
xml_save_root = "E:/DataSets/X_data_27classes/Xdata_he/cut_data/Annotations"
xml_cut_root = "E:/DataSets/X_data_27classes/Xdata_he/cut_data/Annotations_cut"
jpg_root = "E:/DataSets/X_data_27classes/Xdata_he/cut_data/JPGImages"
jpg_cut_root = "E:/DataSets/X_data_27classes/Xdata_he/cut_data/JPGImages_cut"

if not os.path.exists(xml_save_root): os.mkdir(xml_save_root)
if not os.path.exists(xml_cut_root): os.mkdir(xml_cut_root)

# for i in tqdm(os.listdir(jpg_root)):
#     if i.endswith(".jpg"):
#         xml_name = i.split(".jpg")[0] + ".xml"
#         xmlpath = xml_root + "/" + xml_name
#         shutil.copy(xmlpath, xml_save_root + "/" + xml_name)
for i in tqdm(os.listdir(jpg_cut_root)):
    if i.endswith(".jpg"):
        xml_name = i.split(".jpg")[0] + ".xml"
        xmlpath = xml_root + "/" + xml_name
        shutil.copy(xmlpath, xml_cut_root + "/" + xml_name)