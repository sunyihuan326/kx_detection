# coding:utf-8 
'''
created on 2019-11-16

@author:sunyihuan
'''
#
# print("Hello world!")
# for i in range(100):
#     print(pow(i, 2))

import numpy as np
import cv2
image_path="E:/DataSets/2020_two_phase_KXData/all_data36classes/JPGImages/train/14_191106X3_zl_kaopan_Toast" \
           ".jpg"
image = np.array(cv2.imread(image_path))

print(image)
print(image.shape)