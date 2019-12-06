# -*- encoding: utf-8 -*-

"""
去掉txt文件中部分数据，如：去掉文件名中带有：SweetPotatos and 1024的数据
@File    : cut_img_txt.py
@Time    : 2019/11/23 10:03
@Author  : sunyihuan
"""
import random

txt_path = "E:/kx_detection/multi_detection/data/dataset/20191205/base_line_train1205.txt"
txt_file = open(txt_path, "r")
txt_files = txt_file.readlines()
print(len(txt_files))

Cookies_all_list = []
BeefSteak_all_list = []
CartoonCookies_all_list = []
CupCake_all_list = []
train_all_list = []
for txt_file_one in txt_files:
    # if "Pizza" in txt_file_one:
    #     continue
    # if "SweetPotato" in txt_file_one:
    #     continue
    # if "Potato" in txt_file_one:
    #     continue
    # if "Toast" in txt_file_one:
    #     continue
    if "Cookies" in txt_file_one:
        if "CartoonCookies" in txt_file_one:
            CartoonCookies_all_list.append(txt_file_one)
        else:
            Cookies_all_list.append(txt_file_one)

    elif "Cup" in txt_file_one:
        CupCake_all_list.append(txt_file_one)
    elif "Beefsteak" in txt_file_one:
        BeefSteak_all_list.append(txt_file_one)
    else:
        train_all_list.append(txt_file_one)
    # elif "Pizza" in txt_file_one:
    #     continue
#     # elif "Potato" in txt_file_one:
#     #     continue
#     # else:
#     #     train_all_list.append(txt_file_one)  # 读取一个插入一个
print(len(CupCake_all_list))
print(len(train_all_list))
random.shuffle(Cookies_all_list)
random.shuffle(CartoonCookies_all_list)
random.shuffle(BeefSteak_all_list)
random.shuffle(CupCake_all_list)
for i in range(480):
    train_all_list.append(Cookies_all_list[i])
for i in range(480):
    train_all_list.append(CartoonCookies_all_list[i])
for i in range(480):
    train_all_list.append(BeefSteak_all_list[i])
for i in range(480):
    train_all_list.append(CupCake_all_list[i])


print(len(train_all_list))

new_txt_name = "E:/kx_detection/multi_detection/data/dataset/20191205/base_line_480_train1205.txt"
file = open(new_txt_name, "w")
for i in train_all_list:
    file.write(i)
