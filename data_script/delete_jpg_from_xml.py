# -*- coding: utf-8 -*-
# @Time    : 2020/7/29
# @Author  : sunyihuan
# @File    : delete_jpg_from_xml.py

'''
若图片不存在xml文件，则删除该图片
若xml中jpg图片已不存在，则删除该图片
'''

import os
from tqdm import tqdm
import shutil


def delete_xmljpg_diff(img_dir, xml_dir, cut_save_dir):
    '''
    删除多余的xml文件和jpg文件
    :param img_dir: 图片地址
    :param xml_dir: xml文件标注地址
    :return:
    '''
    xml_name_list = os.listdir(xml_dir)
    img_name_list = os.listdir(img_dir)

    xml_cut_save_dir = cut_save_dir + "/Annotations"
    jpg_cut_save_dir = cut_save_dir + "/JPGImages"
    if not os.path.exists(xml_cut_save_dir): os.mkdir(xml_cut_save_dir)
    if not os.path.exists(jpg_cut_save_dir): os.mkdir(jpg_cut_save_dir)

    # jpg中有,xml中没有
    print("图片总数：", len(img_name_list))
    print("未标注图片名称：")
    for i in img_name_list:
        try:
            if not i.endswith(".jpg"):
                os.remove(img_dir + "/" + i)
            if str(i.split(".jpg")[0] + ".xml") not in xml_name_list:
                print(img_dir + "/" + i)
                shutil.move(img_dir + "/" + i, jpg_cut_save_dir + "/" + i)
        except:
            print(img_dir + "/" + i)

    # xml中有，jpg中没有的
    print("已标注总数：", len(xml_name_list))
    print("已标注，但图片已删除名称：")
    for i in xml_name_list:
        # print(i)
        if not i.endswith(".xml"):
            os.remove(xml_dir + "/" + i)
        else:
            if str(i.split(".xml")[0] + ".jpg") not in img_name_list:
                print(xml_dir + "/" + i)
                shutil.move(xml_dir + "/" + i, xml_cut_save_dir + "/" + i)
        # try:
        #     if not i.endswith(".xml"):
        #         os.remove(img_dir + "/" + i)
        #     if str(i.split(".xml")[0] + ".jpg") not in img_name_list:
        #         print(xml_dir + "/" + i)
        #         shutil.move(img_dir + "/" + i, xml_cut_save_dir + "/" + i)
        # except:
        #     print(xml_dir + "/" + i)


if __name__ == "__main__":
    img_root = "F:/serve_data/202101-03formodel/exrtact_file/JPGImages"
    xml_root = "F:/serve_data/202101-03formodel/exrtact_file/Annotations"
    cut_save_root = "F:/serve_data/202101-03formodel/exrtact_file/cut"
    if not os.path.exists(cut_save_root): os.mkdir(cut_save_root)
    # cls_list = ["beefsteak", "bread", "cartooncookies", "chestnut", "chickenwings",
    #             "chiffoncake6", "chiffoncake8", "container", "container_nonhigh", "cookies",
    #             "cornone", "corntwo", "cranberrycookies", "cupcake", "drumsticks",
    #             "eggplant", "eggplant_cut_sauce", "eggtart", "fish", "hotdog",
    #             "peanuts", "pizzacut", "pizzaone", "pizzatwo", "porkchops",
    #             "potatocut", "potatol", "potatos", "redshrimp", "roastedchicken",
    #             "shrimp", "steamedbread", "strand", "sweetpotatocut", "sweetpotatol",
    #             "sweetpotatos", "taro", "toast", "duck"]
    cls_list = os.listdir(img_root)
    for c in tqdm(cls_list):
        if not c.endswith("xls"):
            img_dir = img_root + "/" + c
            xml_dir = xml_root + "/" + c
            cut_save_dir = cut_save_root + "/" + c
            if not os.path.exists(cut_save_dir): os.mkdir(cut_save_dir)
            delete_xmljpg_diff(img_dir, xml_dir, cut_save_dir)
