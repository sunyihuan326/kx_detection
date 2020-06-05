# -*- coding: utf-8 -*-
# @Time    : 2020/5/22
# @Author  : sunyihuan
# @File    : deletexml_from_label.py
'''
通过判断标签名称，删除xml数据和jpg数据

created on 2020/5/19

@author:sunyihuan
'''
import os
import xml.etree.ElementTree as ET
from tqdm import tqdm
import shutil


def changesku(inputpath):
    '''
    更改标签名称
    :param inputpath: xml文件夹地址
    :return:
    '''
    listdir = os.listdir(inputpath)
    for file in tqdm(listdir):
        if file.endswith('xml'):
            file_path = os.path.join(inputpath, file)
            tree = ET.parse(file_path)
            root = tree.getroot()
            for object1 in root.findall('object'):
                for sku in object1.findall('name'):
                    if "_others" in sku.text.lower():
                        try:
                            shutil.move(inputpath + "/" + file, save_dir+"/"+file)
                            shutil.move(inputpath.split("/An")[0]+"/JPGImages/"+file.split(".xml")[0]+".jpg",
                                        save_dir+"/"+file.split(".xml")[0]+".jpg")
                        except:
                            print(file)



if __name__ == '__main__':
    inputpath = "E:/DataSets/X_data_27classes/26classes_0920_he/Annotations"  # 这是xml文件的文件夹的绝对地址
    save_dir = "E:/DataSets/X_data_27classes/26classes_0920_he/_others"
    if not os.path.exists(save_dir): os.mkdir(save_dir)
    changesku(inputpath)
    # check_labelname(inputpath)
