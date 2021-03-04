# -*- coding: utf-8 -*-
# @Time    : 2021/2/5
# @Author  : sunyihuan
# @File    : copy_file.py
'''
将文件夹A中含有xxx字符的文件，拷贝至B
'''
import os
import shutil
from tqdm import tqdm


def copy_file2dir(src, dst, target_str):
    '''

    :param src:
    :param dst:
    :param target_str:
    :return:
    '''
    for f in tqdm(os.listdir(src)):
        if target_str in f:
            shutil.copy(src + "/" + f, dst + "/" + f)


if __name__ == "__main__":
    src = "E:/DataSets/X_3660_data/bu/serve_data/20210115/Annotations"
    dst = "F:/serve_data/for_model/20210115/Annotations(按文件夹单独分类)/chestnut"
    copy_file2dir(src, dst, "chestnut")
