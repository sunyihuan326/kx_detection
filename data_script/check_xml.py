# coding:utf-8 
'''
检查xml文件是否漏标注
created on 2019/7/22

@author:sunyihuan
'''
import os


def check_xml(xml_dir, img_dir):
    '''
    检查xml文件中漏标注，并print
    :param xml_dir: xml文件地址（全路径）
    :param img_dir: jpg文件地址（全路径）
    :return:
    '''
    xml_files = os.listdir(xml_dir)
    print(len(xml_files))
    print(len(os.listdir(img_dir)))
    for img_file in os.listdir(img_dir):
        file_name = img_file.split(".")[0]
        file_name = file_name + ".xml"
        if file_name not in xml_files:
            print(file_name)


if __name__ == "__main__":
    img_dir = "E:/已标数据备份/二期数据/第一批/X4_20200311/taro"
    xml_dir = "E:/已标数据备份/二期数据/第一批/X4_Annotations/taro"
    check_xml(xml_dir, img_dir)
