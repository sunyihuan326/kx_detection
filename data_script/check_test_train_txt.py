# -*- coding: utf-8 -*-
# @Time    : 2020/3/20
# @Author  : sunyihuan
# @File    : check_test_train_txt.py
'''
检查两个txt文件是否有重复
'''

def check_2_txt(train_txt, test_txt):
    '''
    检查两个txt文件是否有重复，
    如果有重复，去除test.txt中数据
    :param train_txt: train.txt文件地址
    :param test_txt: test.txt文件地址
    :return:
    '''
    train_txt_file = open(train_txt, "r")
    train_txt_files = train_txt_file.readlines()
    print(len(train_txt_files))
    test_txt_file = open(test_txt, "r")
    test_txt_files = test_txt_file.readlines()
    print(len(test_txt_files))
    if len(list(set(train_txt_files) & set(test_txt_files))) > 0:  # 判断是否有重复数据
        t_txt_list = []
        for t_txt in test_txt_files:
            if t_txt in train_txt_files:  # test中的数据在train中剔除
                continue
            else:
                t_txt_list.append(t_txt)
        print(len(t_txt_list))
        test_txt_file = open(test_txt, "w")
        for i in t_txt_list:
            test_txt_file.write(i)


if __name__ == "__main__":
    train_txt = "F:/model_data/ZG/SF1_202011/train38.txt"
    test_txt = "F:/model_data/ZG/SF1_202011/test38.txt"
    check_2_txt(train_txt, test_txt)
