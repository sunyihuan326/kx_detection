# -*- coding: utf-8 -*-
# @Time    : 2021/4/20
# @Author  : sunyihuan
# @File    : shutil_img_from_img.py
'''
判断A文件夹下的图片是否在B文件夹下，若在，从A中移至C文件夹

'''
import os
import shutil
from tqdm import tqdm


def shutil_img_file(a_dir, b_dir, c_dir):
    '''
    拷贝数据
    :param a_dir:
    :param b_dir:
    :param c_dir:
    :return:
    '''
    # a_list = os.listdir(a_dir)  # a文件夹下无层级目录
    for c in tqdm(os.listdir(b_dir)):
        try:
            a_list = os.listdir(a_dir + "/" + c)
            for img in os.listdir(b_dir + "/" + c):
                # img = img.split(".jpg")[0].split("_")[0]
                for aa in a_list:
                    if img in aa:
                        if not os.path.exists(c_dir + "/" + c): os.mkdir(c_dir + "/" + c)
                        shutil.move(a_dir + "/" + c + "/" + aa, c_dir + "/" + c + "/" + aa)
                        # shutil.move(a_dir + "/" + aa, c_dir + "/" + c + "/" + aa)
        except:
            print(c)


if __name__ == "__main__":
    a_dir = "F:/serve_data/ZG_data/20210129/biaozhu_20210428/yuantu"
    b_dir = "F:/serve_data/ZG_data/20210129/biaozhu_20210428/exrtact_file/JPGImages"
    c_dir = "F:/serve_data/ZG_data/20210129/biaozhu_20210428/nouse"
    shutil_img_file(a_dir, b_dir, c_dir)
