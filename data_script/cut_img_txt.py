# -*- encoding: utf-8 -*-

"""
去掉txt文件中部分数据，如：去掉文件名中带有：SweetPotatos and 1024的数据
@File    : cut_img_txt.py
@Time    : 2019/11/23 10:03
@Author  : sunyihuan
"""
txt_path = "E:/ckpt_dirs/Food_detection/multi_food/foodSets1111_aug1128train27.txt"
txt_file = open(txt_path, "r")
txt_files = txt_file.readlines()
print(len(txt_files))

train_all_list = []
for txt_file_one in txt_files:
    # if "_y.jpg" in txt_file_one:
    #     continue
    if "SweetPotatos" in txt_file_one:
        if "1024" in txt_file_one:
            continue
        elif "1023" in txt_file_one:
            continue
        else:
            train_all_list.append(txt_file_one)
    else:
        train_all_list.append(txt_file_one)  # 读取一个插入一个

new_txt_name = "E:/ckpt_dirs/Food_detection/multi_food/foodSets1111_aug1128train27_new.txt"
file = open(new_txt_name, "w")
for i in train_all_list:
    file.write(i)
