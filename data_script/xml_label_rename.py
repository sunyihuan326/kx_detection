# coding:utf-8 
'''
统一修改xml文件中label标签

created on 2019/7/18

@author:sunyihuan
'''
import os
import xml.etree.ElementTree as ET


def changesku(inputpath, label_name='Potatom'):
    '''

    :param inputpath: xml文件夹地址
    :param label_name: 标签名称
    :return:
    '''
    listdir = os.listdir(inputpath)
    for file in listdir:
        if file.endswith('xml'):
            file = os.path.join(inputpath, file)
            tree = ET.parse(file)
            root = tree.getroot()
            for object1 in root.findall('object'):
                for sku in object1.findall('name'):
                    sku.text = label_name
                    tree.write(file, encoding='utf-8')


if __name__ == '__main__':
    inputpath = "C:/Users/sunyihuan/Desktop/mantou"  # 这是xml文件的文件夹的绝对地址
    label_name = "steamedbread"
    changesku(inputpath,label_name)
