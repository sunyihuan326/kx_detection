# -*- coding: utf-8 -*-
# @Time    : 2020/8/27
# @Author  : sunyihuan
# @File    : change_txt_bboxes.py
'''
修改txt文件中某一类数据的标签框
'''
import os

txt_path = "E:/DataSets/X_3660_data/train39_zi_hot_and_old_strand.txt"

txt_lists = open(txt_path, "r").readlines()


def bboxes_generate(bboxes):
    '''
    针对标签框，生成一个大框
    :param bboxes:
    :return:
    '''
    x_min = []
    y_min = []
    x_max = []
    y_max = []
    for k in bboxes:
        x_min.append(int(k.split(",")[0]))
        y_min.append(int(k.split(",")[1]))
        x_max.append(int(k.split(",")[2]))
        y_max.append(int(k.split(",")[3]))

    xmin = min(x_min)
    ymin = min(y_min)
    xmax = max(x_max)
    ymax = max(y_max)
    return str(xmin), str(ymin), str(xmax), str(ymax), bboxes[0].split(",")[-1]
    # return int(xmin), int(ymin), int(xmax), int(ymax), int(bboxes[0].split(",")[-1])


new_txt_path = "E:/DataSets/X_3660_data/train39_zi_hot_and_old_strand0.txt"
new_txt_lists = []
for i, t in enumerate(txt_lists):
    t = t.strip()
    bboxes = t.split(" ")[2:]
    if int(bboxes[0].split(",")[-1]) == 37:  # 针对某一类单独处理成大框（38：strand   37：shrimp）
        xmin, ymin, xmax, ymax, cls = bboxes_generate(bboxes)
        new_t = t.split(" ")[0] + " " + t.split(" ")[1] + " " + ','.join([xmin, ymin, xmax, ymax, str(cls)]) + "\n"
        new_txt_lists.append(new_t)
    else:
        new_txt_lists.append(t + "\n")
# for i, t in enumerate(txt_lists):
#     t = t.strip()
#     if "_XZ" in t:  # 直接修改类别标签
#         bboxes = t.split(" ")[2:]
#         if len(bboxes)>1:
#             break
#         for bb in bboxes:
#             xmin=bb.split(",")[0]
#             ymin = bb.split(",")[1]
#             xmax = bb.split(",")[2]
#             ymax = bb.split(",")[3]
#             new_t = t.split(" ")[0] + " " + t.split(" ")[1] + " " + ','.join([xmin, ymin, xmax, ymax, str(39)]) + "\n"
#             new_txt_lists.append(new_t)
#     else:
#         new_txt_lists.append(t + "\n")
file = open(new_txt_path, "w")
for i in new_txt_lists:
    file.write(i)
