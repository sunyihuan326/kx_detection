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

def delete_xmljpg_diff(img_dir, xml_dir):
    '''
    删除多余的xml文件和jpg文件
    :param img_dir: 图片地址
    :param xml_dir: xml文件标注地址
    :return:
    '''
    xml_name_list = os.listdir(xml_dir)
    img_name_list = os.listdir(img_dir)

    # jpg中有,xml中没有
    print("图片总数：", len(img_name_list))
    print("未标注图片名称：")
    for i in img_name_list:
        try:
            if not i.endswith(".jpg"):
                os.remove(img_dir + "/" + i)
            if str(i.split(".jpg")[0] + ".xml") not in xml_name_list:
                print(img_dir + "/" + i)
                os.remove(img_dir + "/" + i)
        except:
            print(img_dir + "/" + i)

    # xml中有，jpg中没有的
    print("已标注总数：", len(xml_name_list))
    print("已标注，但图片已删除名称：")
    for i in xml_name_list:
        try:
            if not i.endswith(".xml"):
                os.remove(img_dir + "/" + i)
            if str(i.split(".xml")[0] + ".jpg") not in img_name_list:
                print(xml_dir + "/" + i)
                os.remove(xml_dir + "/")
        except:
            print(xml_dir + "/" + i)


if __name__ == "__main__":
    xml_root = "/Volumes/SYH/Joyoung/3660摄像头补图202007/Annotations"
    img_root = "/Volumes/SYH/Joyoung/3660摄像头补图202007/JPGImages/已标"
    for c in tqdm(["beefsteak", "cartooncookies", "chestnut", "chickenwings", "chiffoncake6", "cookies",
              "cornone","corntwo", "cranberrycookies", "cupcake", "eggtart", "peanuts", "pizzacut", "pizzaone",
              "pizzatwo","porkchops","potatocut", "potatol", "potatos", "roastedchicken",
              "steamedbread", "sweetpotatol", "sweetpotatos", "taro", "toast"]):
        print(c)
        img_dir = img_root + "/" + c
        xml_dir = xml_root + "/" + c
        delete_xmljpg_diff(img_dir, xml_dir)
