# -*- encoding: utf-8 -*-

"""
将一个文件夹中所有图片的平均r、g、b值写入到excel中

@File    : get_r_g_b.py
@Time    : 2019/12/5 9:34
@Author  : sunyihuan
"""
import cv2
import xlwt
import os


def get_rgb(img_path):
    '''
    获取图片中r、g、b的平均值
    :param img_path: 图片地址
    :return:
    '''
    img = cv2.imread(img_path)

    # 1.计算三通道灰度平均值
    imgB, imgG, imgR = cv2.split(img)

    bAve = cv2.mean(imgB)[0]
    gAve = cv2.mean(imgG)[0]
    rAve = cv2.mean(imgR)[0]

    return rAve, gAve, bAve


def write2excel(img_dir, excel_path):
    '''
    将文件夹中所有图片的平均rgb值写入到excel中
    :param img_dir: 图片地址
    :param excel_path: excel保存地址
    :return:
    '''
    img_list = os.listdir(img_dir)
    workbook = xlwt.Workbook(encoding='utf-8')
    sheet1 = workbook.add_sheet("rgb")
    sheet1.write(0, 0, "img_name")
    sheet1.write(0, 1, "rAve")
    sheet1.write(0, 2, "gAve")
    sheet1.write(0, 3, "bAve")
    for i in range(len(img_list)):
        img_path = img_dir + "/" + img_list[i]
        rAve, gAve, bAve = get_rgb(img_path)

        sheet1.write(i + 1, 0, img_list[i])
        sheet1.write(i + 1, 1, rAve)
        sheet1.write(i + 1, 2, gAve)
        sheet1.write(i + 1, 3, bAve)
    workbook.save(excel_path)


if __name__ == "__main__":
    img_dir = "C:/Users/sunyihuan/Desktop/rgb_test/PorkChops_new/top"
    excel_dir = "C:/Users/sunyihuan/Desktop/rgb_test/PorkChops_newtop.xls"
    write2excel(img_dir, excel_dir)
