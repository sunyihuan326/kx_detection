# -*- coding: utf-8 -*-
# @Time    : 2020/5/21
# @Author  : sunyihuan
# @File    : cut_imgandtxt.py
'''
去除txt文件中部分数据，并同时去除JPGImages中对应的图片和xml文件中的数据
'''
import random
import shutil
from tqdm import tqdm


def cut_txt_data(cut_name, cut_len, new_txt_name, jpg_src_root, jpg_dst_root, xml_src_root, xml_dst_root):
    '''
    去除掉txt中含cut_name多余的数据，仅保留cut_name中的cut_len项
    :param cut_name:去掉的名字
    :param cut_len:要保留的cut_name数量
    :return:
    '''
    txt_file = open(txt_path, "r")
    txt_files = txt_file.readlines()
    print(len(txt_files))

    cut_all_list = []
    train_all_list = []
    save_cut_count = 0
    for txt_file_one in txt_files:
        if cut_name in txt_file_one:
            if save_cut_count <= cut_len:
                train_all_list.append(txt_file_one)
                save_cut_count += 1
            else:
                cut_all_list.append(txt_file_one)

        else:
            train_all_list.append(txt_file_one)

    random.shuffle(train_all_list)
    print(len(train_all_list))
    # 移动图片至cut文件夹
    for i in tqdm(cut_all_list):
        try:
            shutil.move(jpg_src_root + "/" + i.strip() + ".jpg", jpg_dst_root + "/" + i.strip() + ".jpg")
        except:
            print(i)
    # 移动xml至cut文件夹
    for i in tqdm(cut_all_list):
        try:
            shutil.move(xml_src_root + "/" + i.strip() + ".xml", xml_dst_root + "/" + i.strip() + ".xml")
        except:
            print(i)
    file = open(new_txt_name, "w")
    for i in train_all_list:
        file.write(i)


if __name__ == "__main__":
    txt_path = "E:/DataSets/X_data_27classes/Xdata/cut_data/ImageSets/Main/train.txt"
    new_txt_name = "E:/DataSets/X_data_27classes/Xdata/cut_data/ImageSets/Main/train_new.txt"
    jpg_src_root = "E:/DataSets/X_data_27classes/Xdata/cut_data/JPGImages"
    jpg_dst_root = "E:/DataSets/X_data_27classes/Xdata/cut_data/JPGImages_cut"
    xml_src_root = "E:/DataSets/X_data_27classes/Xdata/cut_data/Annotations"
    xml_dst_root = "E:/DataSets/X_data_27classes/Xdata/cut_data/Annotations_cut"
    cut_txt_data("Pizzafour", 240, new_txt_name, jpg_src_root, jpg_dst_root, xml_src_root, xml_dst_root)
