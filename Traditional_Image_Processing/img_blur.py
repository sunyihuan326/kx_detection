# -*- coding: utf-8 -*-
# @Time    : 2020/10/26
# @Author  : sunyihuan
# @File    : img_blur.py
'''
图片随机区域模糊

'''
from PIL import Image, ImageFilter
import random
import cv2
import numpy as np


# class MyGaussianBlur(ImageFilter.Filter):
#
#     def __init__(self, radius=1, bounds=None):
#         self.radius = radius
#         self.bounds = bounds
#
#     def filter(self, image):
#         if self.bounds:
#             clips = image.crop(self.bounds).gaussian_blur(self.radius)
#             image.paste(clips, self.bounds)
#             return image
#         else:
#             return image.gaussian_blur(self.radius)
#
#
# if __name__ == "__main__":
#     img_path = 'C:/Users/sunyihuan/Desktop/test_img/1_bottom_test2020_gaodianya_banli.jpg'
#     image = Image.open(img_path)
#     for k in range(random.randint(2, 7)):
#         b_x_min = random.randint(0, int(image.size[0] / 2))
#         b_y_min = random.randint(0, int(image.size[1] / 2))
#         x_ = random.randint(0, random.randint(100, 300))
#         y_ = random.randint(0, random.randint(100, 200))
#         bounds = (b_x_min, b_y_min, min(b_x_min + x_, image.size[0]), min(b_y_min + y_, image.size[1]))
#         image = image.filter(MyGaussianBlur(radius=random.randint(5, 20), bounds=bounds))
#     image.show()
#     image.save('C:/Users/sunyihuan/Desktop/test_img/1_bottom_test2020_gaodianya_banli_blur.jpg')
def cv2_blur(img):
    img = cv2.imread(img)
    image = np.array(img)
    h, w, _ = image.shape
    b_x_min = random.randint(100, int(3*w / 4))
    b_y_min = random.randint(100, int(3*h / 4))
    x_ = random.randint(50, 600)
    y_ = random.randint(50, 600)
    img2 = image[b_x_min:min(b_x_min + x_, w), b_y_min:min(b_y_min + y_, h), :]
    img2 = cv2.GaussianBlur(img2, (9, 9), 50)
    image[b_x_min:min(b_x_min + x_, w), b_y_min:min(b_y_min + y_, h), :] = img2

    return image


if __name__ == "__main__":
    img_path = 'C:/Users/sunyihuan/Desktop/test_img/1_bottom_test2020_gaodianya_banli.jpg'
    #
    # cv2.imshow('img', img)
    gaosi = cv2_blur(img_path)
    cv2.imshow('gaosi', gaosi)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
