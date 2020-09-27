# -*- coding: utf-8 -*-
# @Time    : 2020/9/2
# @Author  : sunyihuan
# @File    : write_classes_nums_from_txt.py
'''
从生成的txt文件中，输出各类的数量
'''


def nums(txt_path):
    nums_dict = {}
    txt_file = open(txt_path, "r")
    txt_files = txt_file.readlines()
    print(len(txt_files))
    for tt in txt_files:
        tt = tt.strip()
        cls = tt.split(" ")[-1].split(",")[-1]
        if cls not in nums_dict.keys():
            nums_dict[cls] = 1
        else:
            nums_dict[cls] += 1
    return nums_dict


if __name__ == "__main__":
    txt_path = "E:/ckpt_dirs/Food_detection/multi_food5/serve_3660train39_hot_zi_lv_strand.txt"
    nums_dict = nums(txt_path)
    all_c = 0
    for nn in nums_dict.keys():
        all_c += nums_dict[nn]
    print(all_c)
    print(nums_dict)
