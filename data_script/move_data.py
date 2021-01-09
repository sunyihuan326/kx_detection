# -*- coding: utf-8 -*-
# @Time    : 2021/1/9
# @Author  : sunyihuan
# @File    : move_data.py

import os
import shutil
from tqdm import tqdm


def move2dir(jpg_dir, save_dir):
    for f in tqdm(os.listdir(jpg_dir)):
        if f[-4:] == ".jpg":
            data_file_path = os.path.join(jpg_dir, f)
            shutil.move(data_file_path, save_dir + "/" + f)


for kk in os.listdir("F:/serve_data/OVEN"):
    jpg_dir = "F:/serve_data/OVEN/{}".format(kk)
    save_dir = "F:/serve_data/OVEN/{}/base_jpg".format(kk)
    if not os.path.exists(save_dir): os.mkdir(save_dir)

    move2dir(jpg_dir, save_dir)
