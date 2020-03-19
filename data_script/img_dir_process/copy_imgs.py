# coding:utf-8 
'''

按xml文件中的文件名称，从所有图片中拷贝部分至另一个文件夹

created on 2019/7/17

@author:sunyihuan
'''

import os
import shutil

from tqdm import tqdm


class copy_img(object):
    '''
    按xml文件中的文件名称，从所有图片中拷贝部分至另一个文件夹
    '''
    def __init__(self, xml_dir, img_orginal_dir, img_copy_dir):
        '''
        :param xml_dir: xml文件地址（全路径）
        :param img_orginal_dir: 原jpg图片地址，含即所有图片的文件夹
        :param img_copy_dir: xml文件对应jpg图片要保存的文件夹（全路径）
        '''
        self.xml_dir = xml_dir
        self.img_orginal_dir = img_orginal_dir
        self.img_copy_dir = img_copy_dir

    def copy_imgs(self):
        for file in tqdm(os.listdir(self.xml_dir)):
            if str(file).endswith("xml"):
                file = str(file).split(".")[0]
                try:
                    img_orginal_file = os.path.join(self.img_orginal_dir, file + ".jpg")
                    img_copy_file = os.path.join(self.img_copy_dir, file + ".jpg")
                    shutil.copy(img_orginal_file, img_copy_file)
                except:
                    print(file)


if __name__ == "__main__":
    xml_dir = "H:/Joyoung/WLS/KX_FOODSets_model_data/Annotations/EggTart"
    img_orginal_dir = "H:/Joyoung/WLS/KX_FOODSets/JPGImages/EggTart"
    img_copy_dir = "H:/Joyoung/WLS/KX_FOODSets_model_data/JPGImages/ee"
    ci = copy_img(xml_dir, img_orginal_dir, img_copy_dir)
    ci.copy_imgs()
