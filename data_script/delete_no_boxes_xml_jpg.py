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


def delete_xml_jpg(jpg_dir, xml_dir, cut_save_dir):
    '''
    删除无标签框图片和xml文件
    :param inputpath: xml文件夹地址
    :return:
    '''
    xml_cut_save_dir = cut_save_dir + "/Annotations"
    jpg_cut_save_dir = cut_save_dir + "/JPGImages"
    if not os.path.exists(xml_cut_save_dir): os.mkdir(xml_cut_save_dir)
    if not os.path.exists(jpg_cut_save_dir): os.mkdir(jpg_cut_save_dir)

    listdir = os.listdir(xml_dir)
    for file in tqdm(listdir):
        if file.endswith('xml'):
            file_path = os.path.join(xml_dir, file)
            tree = ET.parse(file_path)
            root = tree.getroot()
            bboxes_nums = len(root.findall('object'))
            if bboxes_nums < 1:
                print(file)
                shutil.move(file_path, xml_cut_save_dir + "/" + file)
                if os.path.exists(jpg_dir + "/" + file.split(".")[0] + ".jpg"):
                    shutil.move(jpg_dir + "/" + file.split(".")[0] + ".jpg",
                                jpg_cut_save_dir + "/" + + file.split(".")[0] + ".jpg")
            # try:
            #
            # except:
            #     print("error:::::::", file)


if __name__ == "__main__":
    xml_root = "F:/serve_data/202101-03formodel/exrtact_file/Annotations"
    img_root = "F:/serve_data/202101-03formodel/exrtact_file/JPGImages"
    cut_save_dir = "F:/serve_data/202101-03formodel/exrtact_file/cut"
    if not os.path.exists(cut_save_dir): os.mkdir(cut_save_dir)

    cls_list = os.listdir(img_root)
    for c in tqdm(cls_list):
        img_dir = img_root + "/" + c
        xml_dir = xml_root + "/" + c
        delete_xml_jpg(img_dir, xml_dir, cut_save_dir)
