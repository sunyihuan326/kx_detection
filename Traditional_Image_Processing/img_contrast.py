# -*- coding: utf-8 -*-
# @Time    : 2020/9/2
# @Author  : sunyihuan
# @File    : img_contrast.py
from PIL import Image, ImageEnhance
import os


def aug(img_path, save_dir):
    img = Image.open(img_path)
    enh_con = ImageEnhance.Contrast(img)
    contrast = 1.5
    img_contrasted = enh_con.enhance(contrast)
    enh_sha = ImageEnhance.Sharpness(img_contrasted)
    img_contrasted = enh_sha.enhance(1.9)
    img_contrasted.save(save_dir + "/" + img_path.split("/")[-1])


if __name__ == "__main__":
    img_dir = "E:/DataSets/all_shrimp/JPGImages"
    save_dir = "E:/DataSets/all_shrimp/JPGImages_aug"
    for j in os.listdir(img_dir):
        if j.endswith(".jpg"):
            img_path = img_dir + "/" + j
            aug(img_path, save_dir)
