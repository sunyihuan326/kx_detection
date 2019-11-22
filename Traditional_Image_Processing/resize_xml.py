# -*- encoding: utf-8 -*-

"""
将图片resize并矫正xml文件

@File    : resize_xml.py
@Time    : 2019/11/8 16:01
@Author  : sunyihuan
"""
from PIL import Image
import numpy as np
import xml.etree.ElementTree as ET
import os
from tqdm import tqdm
import matplotlib.pyplot as plt


def pic_resize(img_path, resize_wh, crop_wh):
    '''
    图片拉伸后裁剪
    :param img_path: 图片地址
    :return:
    '''
    img = Image.open(img_path)
    img_new = img.resize(resize_wh)
    # img_new = img_new.crop(crop_wh)
    img_new = np.array(img_new)
    return img_new, img.size


def xml_padding(xml_path, img_size, resize_wh, crop_wh, xml_save_path):
    '''
    xml文件中坐标位置修改
    :param xml_path: xml文件位置
    :param img_size:图片原始尺寸，格式为(w,h)
    :param resize_wh:图片resize后尺寸，格式为(w,h)
    :param crop_wh:图片裁剪区域，格式为(xmin,ymin,xmax,ymax)
    :param xml_save_path:xml文件保存路径
    :return:
    '''
    tree = ET.parse(xml_path)
    root = tree.getroot()
    w = img_size[0]
    h = img_size[1]
    resize_w = resize_wh[0]
    resize_h = resize_wh[1]
    for object1 in root.findall('object'):
        for sku in object1.findall('bndbox'):
            xmin = sku.find("xmin")
            xmax = sku.find("xmax")
            ymin = sku.find("ymin")
            ymax = sku.find("ymax")
            # xmin.text = str(int(xmin.text) * resize_w / w )
            # ymin.text = str(int(ymin.text) * resize_h / h )
            # xmax.text = str(int(xmax.text) * resize_w / w)
            # ymax.text = str(int(ymax.text) * resize_h / h)
            xmin.text = str(int(xmin.text) * resize_w / w - crop_wh[0])
            ymin.text = str(int(ymin.text) * resize_h / h - crop_wh[1])
            xmax.text = str(int(xmax.text) * resize_w / w - crop_wh[0])
            ymax.text = str(int(ymax.text) * resize_h / h - crop_wh[1])
    tree.write(xml_save_path, encoding='utf-8')


def img_dir_aug(img_dir, xml_dir, img_save_dir, xml_save_dir):
    '''
    文件夹padding和保存
    :param img_dir:图片文件夹地址
    :param xml_dir: xml文件夹地址
    :param img_save_dir:图片保存文件夹地址
    :param xml_save_dir: xml保存文件夹地址
    :return:
    '''
    resize_wh = (800, 800)
    crop_wh = (0, 0, 800, 600)
    for img_file in tqdm(os.listdir(img_dir)):
        img_path = img_dir + "/" + img_file
        img, img_size = pic_resize(img_path, resize_wh, crop_wh)  # 图片padding处理

        img_name = str(img_file).split(".")[0] + "_cropx" + ".jpg"  # 图片名称
        plt.imsave(img_save_dir + "/" + img_name, img.astype(np.uint8))  # 保存图片

        xml_name = xml_dir + "/" + str(img_file).split(".")[0] + ".xml"  # xml文件名称
        xml_save_name = xml_save_dir + "/" + str(img_name).split(".")[0] + ".xml"  # xml文件保存名称
        xml_padding(xml_name, img_size,resize_wh, crop_wh, xml_save_name)  # xml文件标注更改、保存


if __name__ == "__main__":
    img_dir = "E:/DataSets/KX_FOODSets_model_data/26classes_0920/JPGImages"
    xml_dir = "E:/DataSets/KX_FOODSets_model_data/26classes_0920/Annotations"
    img_save_dir = "E:/DataSets/KX_FOODSets_model_data/26classes_0920_cropy/JPGImages"
    xml_save_dir = "E:/DataSets/KX_FOODSets_model_data/26classes_0920_cropy/Annotations"

    if not os.path.exists(img_save_dir): os.mkdir(img_save_dir)
    if not os.path.exists(xml_save_dir): os.mkdir(xml_save_dir)
    img_dir_aug(img_dir, xml_dir, img_save_dir, xml_save_dir)
