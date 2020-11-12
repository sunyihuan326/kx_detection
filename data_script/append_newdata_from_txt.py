# -*- coding: utf-8 -*-
# @Time    : 2020/5/19
# @Author  : sunyihuan
# @File    : append_newdata_from_txt.py
'''
将txt2中，新的数据添加到txt1中，即，set(txt1+txt2)
'''
import random


def append_data_from_txt(train_txt, test_txt, new_txt):
    '''
    by：孙义环
    修改日期：2020年8月18日

    将两个txt文件中的数据合并，并去重
    仅合并test_txt中的一半

    :param train_txt: train.txt文件地址
    :param test_txt: test.txt文件地址
    :param new_txt:合并后txt文件地址

    :return:
    '''
    train_txt_file = open(train_txt, "r")
    train_txt_files = train_txt_file.readlines()
    print(len(train_txt_files))
    test_txt_file = open(test_txt, "r")
    test_txt_files = test_txt_file.readlines()
    random.shuffle(test_txt_files)  # 打乱
    print(len(test_txt_files))

    print(len(train_txt_files) + len(test_txt_files))

    all_txt_list = list(set(train_txt_files) | set(test_txt_files[:int(len(test_txt_files) / 2)]))

    print(len(all_txt_list))

    new_txt_file = open(new_txt, "w")
    for i in all_txt_list:
        new_txt_file.write(i)


if __name__ == "__main__":
    root = "E:/DataSets/X_3660_data/bu/20201020"
    train_txt = "{}/train41_huang_hong_zi_lv.txt".format(root)
    test_txt = "{}/train41_lv.txt".format(root)
    # test_txt = "E:/ckpt_dirs/Food_detection/multi_food5/serve_train39_new.txt"
    new_txt = "{}/train41_huang_hong_zi_lv.txt".format(root)
    append_data_from_txt(train_txt, test_txt, new_txt)
