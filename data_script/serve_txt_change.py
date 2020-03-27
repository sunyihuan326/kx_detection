# -*- encoding: utf-8 -*-

"""
直接更改train.txt文件中的图片地址
@File    : peanuts_txt_change.py
@Time    : 2019/11/28 14:08
@Author  : sunyihuan
"""


def change_txt(txt_path, src_txtpath, file_path, typ):
    '''
    更改txt文件中的图片地址
    修改日期：2020/03/23   孙义环

    :param txt_path: 原txt文件路径
    :param src_txtpath: 更改后txt保存路径
    :param file_path: 图片新地址
    :param typ: serve或者其他，str类型
    :return:
    '''
    txt_file = open(txt_path, "r")
    txt_files = txt_file.readlines()
    print(len(txt_files))

    train_all_list = []
    for txt_file_one in txt_files:
        img_path_name = txt_file_one
        print(img_path_name.split("JPGImages")[1])
        txt_file_name = file_path
        if typ == "serve":
            txt_file_name += img_path_name.split("JPGImages")[1]
            train_all_list.append(txt_file_name)  # 读取一个插入一个
        else:  # .jpg前的字段需要更改
            jpg_name = str(img_path_name.split("JPGImages")[1]).split(".jpg")[0] + "_resize.jpg" + \
                       str(img_path_name.split("JPGImages")[1]).split(".jpg")[1]
            txt_file_name += jpg_name

    file = open(src_txtpath, "w")
    for i in train_all_list:
        file.write(i)


if __name__ == "__main__":
    txt_path = "E:/kx_detection/multi_detection/data/dataset/202003_aug/test_all0318_resize.txt"
    new_txt_name = "E:/kx_detection/multi_detection/data/dataset/202003_aug_serve/serve_test_all0318_resize.txt"
    file_path = "/home/sunyihuan/sunyihuan_algorithm/data/2020_two_phase_KXData/all_data36classes" + "/JPGImages"

    change_txt(txt_path, new_txt_name, file_path, "serve")
