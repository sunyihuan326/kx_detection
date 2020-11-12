# -*- coding: utf-8 -*-
# @Time    : 2020/10/27
# @Author  : sunyihuan
# @File    : img_dir_blur.py
'''
测试处采集验证集，按文件夹目录，在图片中随机区域模糊
'''
from PIL import Image, ImageFilter
import random
import os
from tqdm import tqdm
import logging
logging.basicConfig(level=logging.INFO)
class MyGaussianBlur(ImageFilter.Filter):
    name = "GaussianBlur"

    def __init__(self, radius=1, bounds=None):
        self.radius = radius
        self.bounds = bounds

    def filter(self, image):
        if self.bounds:
            clips = image.crop(self.bounds).gaussian_blur(self.radius)
            image.paste(clips, self.bounds)
            return image
        else:
            return image.gaussian_blur(self.radius)


if __name__ == "__main__":
    img_root = "F:/test_from_yejing_202010/TXKX_all_20201019_rename"
    img_save_root = "F:/test_from_yejing_202010/TXKX_all_20201019_rename_blur"
    if not os.path.exists(img_save_root): os.mkdir(img_save_root)
    for ss in os.listdir(img_root):  # 类别
        if not ss.endswith(".xls"):
            assert "bottom" in os.listdir(img_root + "/" + ss)
            if not os.path.exists(img_save_root + "/" + ss): os.mkdir(img_save_root + "/" + ss)
            for kk in os.listdir(img_root + "/" + ss):  # 烤层
                logging.info('n=%s' % ss)
                img_save_dir = img_save_root + "/" + ss + "/" + kk  # 图片保存文件夹
                if not os.path.exists(img_save_dir): os.mkdir(img_save_dir)
                img_dir = img_root + "/" + ss + "/" + kk
                for im in tqdm(os.listdir(img_dir)):  # 图片
                    img_path = img_dir + "/" + im
                    image = Image.open(img_path)
                    for k in range(random.randint(2, 7)):
                        b_x_min = random.randint(0, int(image.size[0] / 2))
                        b_y_min = random.randint(0, int(image.size[1] / 2))
                        x_ = random.randint(100, 300)
                        y_ = random.randint(50, 200)
                        bounds = (b_x_min, b_y_min, min(b_x_min + x_, image.size[0]), min(b_y_min + y_, image.size[1]))
                        image = image.filter(MyGaussianBlur(radius=random.randint(5, 20), bounds=bounds))
                    image.save(img_save_dir + "/" + im)  # 保存图片
