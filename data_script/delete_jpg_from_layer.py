# -*- coding: utf-8 -*-
# @Time    : 2020/7/29
# @Author  : sunyihuan
# @File    : delete_jpg_from_xml.py

'''
若图片未做烤层分类，则单独拷出该图片
若layer_data中已有该图，但原图片已删除，则拷出该layer图片
'''

import os
from tqdm import tqdm
import shutil


def delete__jpg_diff(img_dir, layer_dir, cut_save_dir):
    '''
    删除多余的xml文件和jpg文件
    :param img_dir: 图片地址
    :param xml_dir: xml文件标注地址
    :return:
    '''
    xml_name_list = []
    for b in os.listdir(layer_dir):
        xml_name_list += os.listdir(layer_dir + "/" + b)

    img_name_list = os.listdir(img_dir)

    xml_cut_save_dir = cut_save_dir + "/layer_data"
    jpg_cut_save_dir = cut_save_dir + "/JPGImages"
    if not os.path.exists(xml_cut_save_dir): os.mkdir(xml_cut_save_dir)
    if not os.path.exists(jpg_cut_save_dir): os.mkdir(jpg_cut_save_dir)

    print("图片总数：", len(img_name_list))
    print("未分类图片名称：")
    for i in img_name_list:
        try:
            if i not in xml_name_list:
                print(img_dir + "/" + i)
                shutil.move(img_dir + "/" + i, jpg_cut_save_dir + "/" + i)
        except:
            print(img_dir + "/" + i)
    #
    # # layer中有，jpg中没有的
    # print("已标注总数：", len(xml_name_list))
    # print("已标注，但图片已删除名称：")
    # for i in xml_name_list:
    #     if str(i.split(".xml")[0] + ".jpg") not in img_name_list:
    #             print(xml_dir + "/" + i)
    #             shutil.move(xml_dir + "/" + i, xml_cut_save_dir + "/" + i)



if __name__ == "__main__":
    img_root = "E:/已标数据备份/二期数据(无serve_data)2020/JPGImages"
    xml_root = "E:/已标数据备份/二期数据(无serve_data)2020/layer_data"
    cut_save_root = "E:/已标数据备份/二期数据(无serve_data)2020/cut"
    if not os.path.exists(cut_save_root): os.mkdir(cut_save_root)

    cls_list = os.listdir(img_root)
    for c in tqdm(cls_list):
        if not c.endswith("xls"):
            img_dir = img_root + "/" + c
            xml_dir = xml_root + "/" + c
            cut_save_dir = cut_save_root + "/" + c
            if not os.path.exists(cut_save_dir): os.mkdir(cut_save_dir)
            delete__jpg_diff(img_dir, xml_dir, cut_save_dir)
