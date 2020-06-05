# -*- encoding: utf-8 -*-

"""
去掉txt文件中部分数据，
仅保留部分含有SweetPotatos的数据

@File    : cut_img_txt.py
@Time    : 2019/11/23 10:03
@Author  : sunyihuan
"""
import random


def cut_txt_data(cut_name, cut_len):
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
    for txt_file_one in txt_files:
        if cut_name in txt_file_one:
            if len(txt_file_one) <= cut_len:
                cut_all_list.append(txt_file_one)
        else:
            train_all_list.append(txt_file_one)

    random.shuffle(train_all_list)
    print(len(train_all_list))

    file = open(new_txt_name, "w")
    for i in train_all_list:
        file.write(i)


if __name__ == "__main__":
    txt_path = "E:/DataSets/X_data_27classes/serve_train23.txt"
    new_txt_name =  "E:/DataSets/X_data_27classes/serve_train23_new.txt"
    cut_txt_data("20200522data", 0)
