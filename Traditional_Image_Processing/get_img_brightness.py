# -*- coding: utf-8 -*-
# @Time    : 2021/4/13
# @Author  : sunyihuan
# @File    : get_img_brightness.py
'''
获取图片亮度信息
'''
from PIL import Image, ImageStat
import numpy as np
import xml.etree.ElementTree as ET
import os
import xlwt, xlrd


def get_crop_size(xml_path):
    '''
    获取标签框区域
    :param xml_path:
    :return:
    '''
    crop_size = []
    if xml_path.endswith('xml'):
        tree = ET.parse(xml_path)
        root = tree.getroot()
        for object1 in root.findall('object'):
            box = object1.find("bndbox")
            box.find("xmin")
            crop_size.append(
                [int(box.find("xmin").text), int(box.find("ymin").text), int(box.find("xmax").text),
                 int(box.find("ymax").text)])
    return crop_size


def get_bright(img, crop_size):
    '''
    获取该区域亮度
    :param img:
    :param crop_size:
    :return:
    '''
    if len(crop_size) == 0:
        iw, ih = img.size
        crop_size = [0, 0, iw, ih]
    img = img.crop(crop_size)

    img = img.convert("YCbCr")
    start = ImageStat.Stat(img)
    # print(start.mean[0])
    return start.mean[0]


def get_b(img_path, xml_path):
    crop_size_list = get_crop_size(xml_path)
    img = Image.open(img_path)
    b = 0
    for i in range(len(crop_size_list)):
        crop_size = crop_size_list[i]
        b_s = get_bright(img, crop_size)
        b = max(b, b_s)
    return b


def writ_b_excel(img_dir, xml_dir, excel_save):
    w = xlwt.Workbook()
    sheet = w.add_sheet("jpg_b")
    sheet.write(0, 0, "jpg_name")
    sheet.write(0, 1, "brightness")
    for i, img_p in enumerate(os.listdir(img_dir)):
        print(i, img_p)
        img_path = img_dir + "/" + img_p
        sheet.write(i + 1, 0, img_path)
        try:
            xml_path = xml_dir + "/" + img_p.split(".")[0] + ".xml"
            b = get_b(img_path, xml_path)
            sheet.write(i + 1, 1, b)
        except:
            print("error::::::")
    w.save(excel_save)


if __name__ == "__main__":
    img_dir = "F:/model_data/ZG/Li/vocleddata-food38-20210118/train/JPGImages"
    xml_dir = "F:/model_data/ZG/Li/vocleddata-food38-20210118/train/Annotations"
    excel_save = "F:/model_data/ZG/Li/vocleddata-food38-20210118/train/brightness.xls"

    writ_b_excel(img_dir, xml_dir, excel_save)
