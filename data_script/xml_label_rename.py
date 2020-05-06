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
                    # if sku.text=="corn_others":
                    #     print(file)
                    sku.text = label_name
                    tree.write(file, encoding='utf-8')


def check_labelname(inputpath, label_name='Potatom'):
    '''
    检查标签是否有更改完成，标签名不正确的输出
    :param inputpath: xml文件路径
    :param label_name: 标签名
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
                    # if sku.text=="corn_others":
                    #     print(file)
                    if sku.text != label_name:
                        print(file)


if __name__ == '__main__':
    inputpath = "E:/WLS_originalData/二期数据/20200423/Annotations/shrimp_red0"  # 这是xml文件的文件夹的绝对地址
    label_name = "redshrimp"
    changesku(inputpath, label_name)

    check_labelname(inputpath, label_name)
