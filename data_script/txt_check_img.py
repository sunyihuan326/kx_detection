# -*- coding: utf-8 -*-
# @Time    : 2020/7/14
# @Author  : sunyihuan
# @File    : txt_check_img.py
'''
按文件夹中，若图片已经删除，则在txt中删除该行数据

'''
import os


def cut_txt(train_txt, new_txt_name, typ):
    txt_file = open(train_txt, "r")
    txt_files = txt_file.readlines()
    train_all_list = []
    if typ == "voc":
        for txt_file_one in txt_files:
            img_path = txt_file_one.split(" ")[0]
            if "/home/sunyihuan/sunyihuan_algorithm/data/KX_data/Xdata_he/cut_data/JPGImages" not in img_path:
                train_all_list.append(txt_file_one)
            else:
                print(img_path)
        # for txt_file_one in txt_files:
        #     img_path = txt_file_one.split(" ")[0]
        #     if os.path.exists(img_path):
        #         train_all_list.append(txt_file_one)
        #     else:
        #         print(img_path)
    else:
        for txt_file_one in txt_files:
            img_path = img_root + "/" + txt_file_one.strip() + ".jpg"
            if os.path.exists(img_path):
                train_all_list.append(txt_file_one)
            else:
                print(img_path)
    file = open(new_txt_name, "w")
    for i in train_all_list:
        file.write(i)


if __name__ == "__main__":
    img_root = "E:/DataSets/X_data_27classes/Xdata_he/cut_data/JPGImages"
    typ = "voc"
    train_txt = "E:/ckpt_dirs/Food_detection/multi_food5/serve_train39.txt"
    new_txt_name = "E:/ckpt_dirs/Food_detection/multi_food5/serve_train39_new.txt"
    cut_txt(train_txt, new_txt_name, typ)
