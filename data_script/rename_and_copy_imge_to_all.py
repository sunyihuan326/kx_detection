# -*- coding: utf-8 -*-
# @Time    : 202003/2/29
# @Author  : sunyihuan
# @File    : rename_and_copy_imge_to_all.py
'''
将图片重命名，并从子文件夹中拷贝到根目录下

'''

import shutil
import os
from tqdm import tqdm


def rename_copy_img(c, img_dirs, dst_img_dir, target):
    i = 0
    for l in ["bottom", "middle", "top"]:
        img_dir_name = img_dirs + "/" + l
        print(img_dir_name)
        for jpg_ in tqdm(os.listdir(img_dir_name)):
            if jpg_.endswith(".jpg"):
                i += 1
                name = str(i) + "_20200801_X5_" + l + "_{}".format(c) + "_{}.jpg".format(target)
                os.rename(img_dir_name + "/" + jpg_, img_dir_name + "/" + name)  # 重命名
                shutil.copy(img_dir_name + "/" + name, dst_img_dir + "/" + name)  # 拷贝


def change_jpg_name(root_path, org_str, dst_str):
    for i, item in enumerate(os.listdir(root_path)):
        if item.endswith('.jpg'):
            src = os.path.join(os.path.abspath(root_path), item)
            try:
                # 修改命名，规则为：i_日期_炸锅型号/其他说明_类别名.jpg
                # tj:托架、xz:锡纸、gyz:硅油纸、jk:净空、cp:瓷盘
                # qh:浅红、zh:正红、ch:橙红、sh:深红
                if org_str in src:
                    dst = src.replace(org_str, dst_str)
                    os.rename(src, dst)
                    print('converting %s to %s ...' % (src, dst))
                else:
                    pass

            except:
                pass


if __name__ == "__main__":
    classes_label22 = ["buxiugangcanju", "taocicanju"]
    img_root = "G:/Joyoung/3660摄像头补图202007/原数据/X5/zaliangxiaomantou"
    dst_root = "E:/WLS_originalData/3660bucai(annotation)/JPGImages/X5/steamedbread"
    if not os.path.exists(dst_root): os.mkdir(dst_root)
    # for c in classes_label22:
    #     img_dirs = img_root + "/" + c
    #     dst_img_dir = dst_root + "/" + c
    # if not os.path.exists(dst_img_dir): os.mkdir(dst_img_dir)
    # rename_copy_img("{}".format(c), img_dirs, dst_root, "container")
    for ty in ["kaojia", "xizhi"]:
        img_dirs_ = img_root + "/" + ty
        rename_copy_img("{}".format(ty), img_dirs_, dst_root, "steamedbread")
    # change_jpg_name(dst_root, "X1_", "X5_")
