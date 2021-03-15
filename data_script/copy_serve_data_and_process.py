# -*- coding: utf-8 -*-
# @Time    : 2021/3/5
# @Author  : sunyihuan
# @File    : copy_serve_data_and_process.py

'''
拷贝所有serve_data至备份文件
并删除aug数据
将文件按单独文件夹分类

'''
import os
import shutil
from tqdm import tqdm


def copy2all(serve_root, t_list, src_root):
    '''
    拷贝文件至所有
    :param serve_root:
    :param t_list:
    :param src_root:
    :return:
    '''
    for t in tqdm(t_list):
        data_root = serve_root + "/" + t  # 20201109

        # for t_p in ["JPGImages", "Annotations"]:
        #     t_dir = data_root + "/" + t_p  # 原数据路径
        #     src_dir = src_root + "/" + t_p  # 保存文件路径
        #     if not os.path.exists(src_dir): os.mkdir(src_dir)
        #     for file in os.listdir(t_dir):
        #         file_name = t_dir + "/" + file
        #         try:
        #             shutil.copy(file_name, src_dir + "/" + file)
        #         except:
        #             print(file_name)
        t_p = "layer_data"
        t_dir = data_root + "/" + t_p  # 原数据路径
        src_dir = src_root + "/" + t_p  # 保存文件路径
        if not os.path.exists(src_dir): os.mkdir(src_dir)
        for c in os.listdir(t_dir):
            src_c_dir = src_dir + "/" + c
            if not os.path.exists(src_c_dir): os.mkdir(src_c_dir)
            for file in os.listdir(t_dir + "/" + c):
                file_name = t_dir + "/" + c + "/" + file
                try:
                    shutil.copytree(file_name, src_c_dir + "/" + file)
                except:
                    print(file_name)


if __name__ == "__main__":
    serve_root = "E:/DataSets/X_3660_data/bu/serve_data"
    t_list = ["20201109", "20210115", "202011041030", "202011120900", "202011181630", "202012030843"]
    src_root = "E:/已标数据备份/serve数据/2020年补充"
    # copy2all(serve_root, t_list, src_root)

    from data_script.all2classses import split_jpg_xml

    # split_jpg_xml(src_root)

    from data_script.cut_aug_data_from_dir import cut_aug_data
    cut_aug_data(src_root)
