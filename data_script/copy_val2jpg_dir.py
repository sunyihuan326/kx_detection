# -*- coding: utf-8 -*-
# @Time    : 2020/4/7
# @Author  : sunyihuan
# @File    : copy_val2jpg_dir.py

'''
将xxx_val.txt图片拷贝到对应的xxx_val文件中
并对其进行分层
'''

import os
import shutil


def copy_val2jpg_dir(txt_root, all_jpg_dir, save_dir):
    '''
    将val中的数据拷贝到单独的文件夹中，并且烤层也分类

    :param txt_root: val数据根目录，如：ImageSets/Main
    :param all_jpg_dir: 所有图片地址
    :param save_dir: 要保存图片地址
    :return:
    '''
    layer_data = "E:/DataSets/2020_two_phase_KXData/only2phase_data/layer_data/val"
    val_b_list = os.listdir(layer_data + "/bottom")
    val_m_list = os.listdir(layer_data + "/middle")
    val_t_list = os.listdir(layer_data + "/top")
    val_o_list = os.listdir(layer_data + "/others")

    for txt in os.listdir(txt_root):
        if "_val" in txt:
            save_name = save_dir + "/" + txt.split("_val")[0]
            os.mkdir(save_name)
            save_b_name = save_name + "/bottom"
            save_m_name = save_name + "/middle"
            save_t_name = save_name + "/top"
            save_o_name = save_name + "/others"
            os.mkdir(save_b_name), os.mkdir(save_m_name), os.mkdir(save_t_name), os.mkdir(save_o_name)  # 创建烤层对应文件夹

            val_jpg_names = open(txt_root + "/" + txt).readlines()
            for jpg_name in val_jpg_names:
                jpg_name = jpg_name.strip("\n")
                jpg_name = jpg_name + ".jpg"
                if jpg_name in val_b_list:
                    shutil.copy(all_jpg_dir + "/" + jpg_name, save_b_name + "/" + jpg_name)
                elif jpg_name in val_m_list:
                    shutil.copy(all_jpg_dir + "/" + jpg_name, save_m_name+ "/" + jpg_name)
                elif jpg_name in val_t_list:
                    shutil.copy(all_jpg_dir + "/" + jpg_name, save_t_name + "/" + jpg_name)
                elif jpg_name in val_o_list:
                    shutil.copy(all_jpg_dir + "/" + jpg_name, save_o_name + "/" + jpg_name)
                else:
                    print(jpg_name)


txt_root = "E:/DataSets/2020_two_phase_KXData/only2phase_data/ImageSets/Main"
all_jpg_dir = "E:/DataSets/2020_two_phase_KXData/only2phase_data/JPGImages"
save_dir = "E:/DataSets/2020_two_phase_KXData/only2phase_data/JPGImages46"
if os.path.exists(save_dir):shutil.rmtree(save_dir),os.mkdir(save_dir)
copy_val2jpg_dir(txt_root, all_jpg_dir, save_dir)
