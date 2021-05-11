# -*- coding: utf-8 -*-
# @Time    : 2021/4/1
# @Author  : sunyihuan
# @File    : rename_from_dir.py

'''
根据文件夹名称的变化，更改文件名，并同时修改对应anntation标注文件名称
例：   JPJImages
            beefsteak
                Tgu
                   2021_beefstaek.jpg    修改为：2021_Tgu_beefstaek.jpg
    同时  Annotations
            beefsteak
                   2021_beefstaek.xml    修改为：2021_Tgu_beefstaek.xml
'''
import os
from tqdm import tqdm


def rename(JPG_dir, ann_dir):
    noxml_nums = 0
    for classes_name in tqdm(os.listdir(JPG_dir)):
        class_dir = JPG_dir + "/" + classes_name  # 对应类别jpg文件夹
        xml_dir = ann_dir + "/" + classes_name  # 对应类别xml文件夹
        # 读取类别下文件
        all_j = os.listdir(class_dir)
        # 判断类别下是否有子类别
        if ".jpg" in all_j[0]:  # 无子类别
            continue
        else:  # 有子类别
            for file_ in all_j:
                jpg_dir = class_dir + "/" + file_
                for jpg in os.listdir(jpg_dir):
                    name = str(jpg.split(".jpg")[0])
                    jpg_name = name.split("_")[0] + "_{}".format(file_) + "_{}".format(
                        name.split("_")[-1]) + ".jpg"  # 图片名称
                    xml_name = name.split("_")[0] + "_{}".format(file_) + "_{}".format(
                        name.split("_")[-1]) + ".xml"  # xml文件名称
                    try:
                        os.rename(jpg_dir + "/" + jpg, class_dir + "/" + jpg_name)
                        os.rename(xml_dir + "/" + name + ".xml", xml_dir + "/" + xml_name)
                    except:
                        print("no xml", name)
                        noxml_nums += 1
    return noxml_nums


if __name__ == "__main__":
    jpg_dir = "F:/serve_data/202101-03formodel/JPGImages"
    ann_dir = "F:/serve_data/202101-03formodel/Annotations"
    print(rename(jpg_dir, ann_dir))
