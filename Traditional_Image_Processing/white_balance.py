# -*- coding: utf-8 -*-
# @Time    : 2020/7/14
# @Author  : sunyihuan
# @File    : white_balance.py

import cv2
import numpy as np
import os

bf = np.ones([3], dtype=np.float)

img = "E:/WLS_originalData/3660camera_data202007/X3_original/20200630102015.jpg"
# Load the image.
cvImg = cv2.imread(img, cv2.IMREAD_UNCHANGED)
vcMask = np.ones([cvImg.shape[0], cvImg.shape[1]], dtype=np.float)
# Balance the input image
for i in range(cvImg.shape[2]):
    cvImg[:, :, i] = cv2.scaleAdd(cvImg[:, :, i], bf[i], np.zeros_like(cvImg[:, :, i]))
    cvImg[:, :, i] = cv2.multiply(cvImg[:, :, i], vcMask, dtype=cv2.CV_8UC1)

# Get the name components of the file name.
fn = os.path.split(img)[1]
# ext = os.path.splitext( fn )[1]
fn = os.path.splitext(fn)[0]

# Only supports PNG currently.
ext = "white_b.jpg"

# Save the balanced image.
# cv2.imwrite( args.output_dir + "/" + fn + "_Balanced" + ext, cvImg )
cv2.imwrite(ext, cvImg, [cv2.IMWRITE_PNG_COMPRESSION, 0])
