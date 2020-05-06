# coding:utf-8 
'''

图片名称写入到txt文件中，与voc数据集中ImageSets/Main/trainval.txt类似


created on 2019/7/18

@author:sunyihuan
'''
import os


def filename_txt(imgdir, txt_path):
    '''
    将图片名称写入到txt文件中
    :param imgdir: 图片文件夹地址
    :param txt_path: txt文件地址
    :return:
    '''
    f = open(txt_path, 'w')
    for i, path in enumerate(os.listdir(imgdir)):
        img_names = str(path).split(".")[0]
        f.write(img_names + "\n")
    f.close()


if __name__ == "__main__":
    imgdir = "/Users/sunyihuan/Desktop/WLS/KX38I95FOODSETS/train_2classes/JPGImages"
    txt_path = "/Users/sunyihuan/Desktop/WLS/KX38I95FOODSETS/train_2classes/ImageSets/train_all.txt"
    filename_txt(imgdir, txt_path)
