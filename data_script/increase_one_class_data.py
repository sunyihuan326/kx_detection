# -*- coding: utf-8 -*-
# @Time    : 2021/1/28
# @Author  : sunyihuan
# @File    : increase_one_class_data.py

'''
针对txt训练文件中某一类别，单独拿出该类别数据，并按比例加入到整体txt训练文件中
'''
import random


def one_class_list(txt_files, target_id):
    '''
    读出某一类别所有数据
    :param txt_files:
    :param target_id:
    :return:
    '''
    txt_file_new_list = []
    for txt_file_one in txt_files:
        if len(txt_file_one.split(" ")) > 2:
            class_id = int(txt_file_one.split(" ")[-1].strip().split(",")[-1])
            if class_id == target_id:
                txt_file_new_list.append(txt_file_one)
    return txt_file_new_list


def append_data(original_txt, target_id, ratio, save_txt):
    '''
    将目标txt训练文件中某一类，按1/ratio比例加入到原数据中，并生成新的txt训练文件
    :param original_txt:
    :param target_id:
    :param ratio:
    :param save_txt:
    :return:
    '''
    txt_file = open(original_txt, "r")
    txt_files = txt_file.readlines()
    txt_file_new_list = one_class_list(txt_files, target_id)
    random.shuffle(txt_file_new_list)
    all_txt_list = txt_files + txt_file_new_list[:int(len(txt_file_new_list) / ratio)]
    new_txt_file = open(save_txt, "w")
    for i in all_txt_list:
        new_txt_file.write(i)


if __name__ == "__main__":
    original_txt = "E:/ckpt_dirs/Food_detection/multi_food5/serve_3660train39_hot_zi_lv_strand20210115.txt"
    target_id = 10
    save_txt = "E:/ckpt_dirs/Food_detection/multi_food5/serve_3660train39_hot_zi_lv_strand20210128.txt"
    append_data(original_txt, target_id, 3, save_txt)
