# coding:utf-8 
'''

不均匀光照补偿

created on 2019/7/8

@author:sunyihuan
'''
import cv2
import numpy as np
import os
import matplotlib.pyplot as plt
from tqdm import tqdm
import shutil


def warm_pic(img, delta):
    '''
    图片暖色调处理，
    暂未完成！！！！！！！！！！！！！！！！！！！！！！！！！
    :param img:
    :param delta:
    :return:
    '''
    # b = img[:, :, 0]
    img[:, :, 1] = img[:, :, 1] + delta
    img[:, :, 2] = img[:, :, 2] + delta
    # b, g, r = cv2.split(img)
    # g = g + delta
    # r = r + delta
    img = img * (img <= 255) + 255 * (img > 255)
    img = img.astype(np.uint8)
    print(img.shape)
    return img


def unevenLightCompensate(img, blockSize=100, tpy="Gaussian"):
    '''
    光照补偿
    :param img:
    :param blockSize:
    :return:
    '''
    # gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = img
    average = np.mean(gray)
    rows_new = int(np.ceil(gray.shape[0] / blockSize))
    cols_new = int(np.ceil(gray.shape[1] / blockSize))
    blockImage = np.zeros((rows_new, cols_new, gray.shape[2]), dtype=np.float32)
    for r in range(rows_new):
        for c in range(cols_new):
            rowmin = r * blockSize
            rowmax = (r + 1) * blockSize
            if (rowmax > gray.shape[0]):
                rowmax = gray.shape[0]
            colmin = c * blockSize
            colmax = (c + 1) * blockSize
            if (colmax > gray.shape[1]):
                colmax = gray.shape[1]

            imageROI = gray[rowmin:rowmax, colmin:colmax, :]
            temaver = np.mean(imageROI)
            blockImage[r, c, :] = temaver
    blockImage = blockImage - average
    blockImage2 = cv2.resize(blockImage, (gray.shape[1], gray.shape[0]), interpolation=cv2.INTER_CUBIC)
    gray2 = gray.astype(np.float32)
    dst = gray2 - blockImage2
    dst = dst.astype(np.uint8)
    if tpy == "Gaussian":
        dst = cv2.GaussianBlur(dst, (3, 3), 0)  # 高斯滤波
    else:
        dst = cv2.medianBlur(dst, 5)  # 中值滤波
    # dst = cv2.cvtColor(dst, cv2.COLOR_GRAY2BGR)
    return dst


if __name__ == '__main__':
    data_dir = "/Users/sunyihuan/Desktop/WLS/testData/originalData"
    save_dir = "/Users/sunyihuan/Desktop/WLS/testData/unevenLightData"
    if os.path.exists(save_dir): shutil.rmtree(save_dir)
    os.mkdir(save_dir)

    for file in tqdm(os.listdir(data_dir)):
        filename = os.path.join(data_dir, file)
        savefile = os.path.join(save_dir, file)

        blockSize = 200
        img = cv2.imread(str(filename))
        # print(img)
        # cv2.imshow("Image", img)

        result = unevenLightCompensate(img, blockSize, "Gaussian")

        result = np.concatenate([img, result], axis=1)

        cv2.imwrite(savefile, result)
