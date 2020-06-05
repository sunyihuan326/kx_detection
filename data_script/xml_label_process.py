# coding:utf-8 
'''
修改xml文件中label标签
主要是合并蛋挞大、蛋挞小；戚风8寸、戚风6寸；土豆大、土豆中；红薯大、红薯中；披萨四分之一、披萨六分之一

created on 2020/5/19

@author:sunyihuan
'''
import os
import xml.etree.ElementTree as ET
from tqdm import tqdm


def changesku(inputpath):
    '''
    更改标签名称
    :param inputpath: xml文件夹地址
    :return:
    '''
    listdir = os.listdir(inputpath)
    for file in tqdm(listdir):
        if file.endswith('xml'):
            file = os.path.join(inputpath, file)
            tree = ET.parse(file)
            root = tree.getroot()
            for object1 in root.findall('object'):
                for sku in object1.findall('name'):
                    if sku.text.lower() == "eggtartbig" or sku.text.lower() == "eggtart":  # 合并蛋挞
                        sku.text = "eggtart"
                    elif sku.text.lower() == "chiffoncake6" or sku.text.lower() == "chiffoncake8":  # 合并戚风
                        sku.text = "chiffoncake"
                    elif sku.text.lower() == "potatol" or sku.text.lower() == "potatom":  # 合并中大土豆
                        sku.text = "potatol"
                    elif sku.text.lower() == "sweetpotatol" or sku.text.lower() == "sweetpotatom":  # 合并中大红薯
                        sku.text = "sweetpotatol"
                    elif sku.text.lower() == "pizzasix" or sku.text.lower() == "pizzafour":  # 合并1/4、1/6披萨
                        sku.text = "pizzacut"
                    else:
                        sku.text = sku.text
                    tree.write(file, encoding='utf-8')


def check_labelname(inputpath):
    '''
    检查标签是否有更改完成，标签名不正确的输出
    :param inputpath: xml文件路径
    :param label_name: 标签名
    :return:
    '''
    listdir = os.listdir(inputpath)
    for file in tqdm(listdir):
        if file.endswith('xml'):
            file = os.path.join(inputpath, file)
            tree = ET.parse(file)
            root = tree.getroot()
            for object1 in root.findall('object'):
                for sku in object1.findall('name'):
                    if sku.text.lower()=="eggtartbig":
                        print(file)



if __name__ == '__main__':
    inputpath = "E:/DataSets/X_data_27classes/Xdata_he/Annotations"  # 这是xml文件的文件夹的绝对地址
    changesku(inputpath)
    # check_labelname(inputpath)


