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
                serve:将图片路径改为serve端路径
                others：替换.jpg前字段，如直接图片名字_resize
    :return:
    '''
    txt_file = open(txt_path, "r")
    txt_files = txt_file.readlines()
    print(len(txt_files))

    train_all_list = []
    for txt_file_one in txt_files:
        img_path_name = txt_file_one
        # print(img_path_name.split("JPGImages")[1])
        txt_file_name = file_path
        if typ == "serve":
            txt_file_name += img_path_name.split("JPGImages")[1]
            train_all_list.append(txt_file_name)  # 读取一个插入一个
        else:  # .jpg前的字段需要更改
            jpg_name = str(img_path_name.split("JPGImages")[1]).split(".jpg")[0] + "_{}.jpg".format(typ) + \
                       str(img_path_name.split("JPGImages")[1]).split(".jpg")[1]
            print(jpg_name)
            txt_file_name += jpg_name
            train_all_list.append(txt_file_name)

    file = open(src_txtpath, "w")
    for i in train_all_list:
        file.write(i)


def replace_txt_path(txt_path, src_txtpath, file_path, target_path):
    '''
    替换txt文件中特别字段
    :param txt_path: 原txt文件地址
    :param src_txtpath: 保存地址
    :param file_path: 被替换字段
    :param target_path: 目标字段
    :return:
    '''
    txt_file = open(txt_path, "r")
    txt_files = txt_file.readlines()
    print(len(txt_files))

    train_all_list = []
    for txt_file_one in txt_files:
        img_path_name = txt_file_one
        print(img_path_name)
        img_path_name = img_path_name.replace(file_path, target_path)
        train_all_list.append(img_path_name)

    file = open(src_txtpath, "w")
    for i in train_all_list:
        file.write(i)


if __name__ == "__main__":
    txt_path = "E:/DataSets/X_3660_data/bu/20201020/train41_huang_hong_zi_lv.txt"
    new_txt_name = "E:/DataSets/X_3660_data/bu/20201020/serve_3660train41_huang_hong_zi_lv.txt"
    file_path = "E:/DataSets/X_3660_data/bu/20201020/JPGImages"

    change_txt(txt_path, new_txt_name, file_path, "lv")
    target_path = "/home/sunyihuan/sunyihuan_algorithm/data/KX_data/3660_202008/bu/20201020/JPGImages"
    # #
    replace_txt_path(txt_path, new_txt_name, file_path, target_path)
