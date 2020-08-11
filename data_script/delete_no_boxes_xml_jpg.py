# coding:utf-8 
'''
删除没有bboxes标注框的xml文件和jpg文件

created on 2020-08-04

@author:sunyihuan
'''

import os
import xml.etree.ElementTree as ET
from tqdm import tqdm
import shutil


def delete_xml_jpg(xml_dir, jpg_dir):
    '''
    删除无标签框图片和xml文件
    :param inputpath: xml文件夹地址
    :return:
    '''
    listdir = os.listdir(xml_dir)
    for file in tqdm(listdir):
        if file.endswith('xml'):
            file_path = os.path.join(xml_dir, file)
            tree = ET.parse(file_path)
            root = tree.getroot()
            bboxes_nums = len(root.findall('object'))
            if bboxes_nums < 1:
                print(file)
                os.remove(file_path)
                if os.path.exists(jpg_dir + "/" + file.split(".")[0] + ".jpg"):
                    os.remove(jpg_dir + "/" + file.split(".")[0] + ".jpg")


if __name__ == "__main__":
    xml_root = "/Volumes/SYH/Joyoung/3660摄像头补图202007/Annotations"
    img_root = "/Volumes/SYH/Joyoung/3660摄像头补图202007/JPGImages/已标"
    # for c in tqdm(["beefsteak", "cartooncookies", "chestnut", "chickenwings", "chiffoncake6", "cookies",
    #                "cornone", "corntwo", "cranberrycookies", "cupcake", "eggtart", "peanuts", "pizzacut", "pizzaone",
    #                "pizzatwo", "porkchops", "potatocut", "potatol", "potatos", "roastedchicken",
    #                "steamedbread", "sweetpotatol", "sweetpotatos", "taro", "toast"]):
    for c in tqdm(["porkchops", "potatocut", "potatol", "potatos", "roastedchicken",
                   "steamedbread", "sweetpotatol", "sweetpotatos", "taro", "toast"]):
        img_dir = img_root + "/" + c
        xml_dir = xml_root + "/" + c
        delete_xml_jpg(xml_dir, img_dir)
