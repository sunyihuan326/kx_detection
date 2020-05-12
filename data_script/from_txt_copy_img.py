# -*- encoding: utf-8 -*-

"""
从txt文件中读取图片地址，并将图片保存至统一文件夹

@File    : from_txt_copy_img.py
@Time    : 2019/12/5 16:52
@Author  : sunyihuan
"""
import shutil
from tqdm import tqdm
import os


def from_txt_copy_data2all(txt_path, save_dir, jpg_typ, layer_tpy=True):
    '''
    2020年3月20日修改
    :param txt_path:txt文件路径，全路径
    :param save_dir:保存地址
    :param jpg_typ:jpg或者xml，str格式
    :param layer_tpy:是否保存layer数据，与jpg_typ=jpg同时使用
    :return:
    '''
    txt_file = open(txt_path, "r")
    txt_files = txt_file.readlines()
    print(len(txt_files))

    assert jpg_typ in ["jpg", "xml"]
    if layer_tpy:
        layer_dir = save_dir + "/layer_data"
        if os.path.exists(layer_dir): shutil.rmtree(layer_dir)
        os.mkdir(layer_dir)
        # 创建各层文件夹
        os.mkdir(layer_dir + "/bottom")
        os.mkdir(layer_dir + "/middle")
        os.mkdir(layer_dir + "/top")
        os.mkdir(layer_dir + "/others")
    if jpg_typ == "jpg":  # 拷贝jpg数据
        for file in tqdm(txt_files):
            img_name = file.split(" ")[0]
            jpg_name = str(img_name).split("/")[-1]
            shutil.copy(img_name, save_dir + "/" + jpg_name)
            if layer_tpy:
                if file.split(" ")[1] == "0":
                    print(layer_dir + "/bottom" + "/" + jpg_name)
                    shutil.copy(img_name, layer_dir + "/bottom" + "/" + jpg_name)
                elif file.split(" ")[1] == "1":
                    shutil.copy(img_name, layer_dir + "/middle" + "/" + jpg_name)
                elif file.split(" ")[1] == "2":
                    shutil.copy(img_name, layer_dir + "/top" + "/" + jpg_name)
                else:
                    shutil.copy(img_name, layer_dir + "/others" + "/" + jpg_name)
    else:  # 拷贝xml数据
        for file in tqdm(txt_files):
            img_name = file.split(" ")[0]
            jpg_name = str(img_name).split("/")[-1]
            na = str(jpg_name.split(".jpg")[0]) + ".xml"
            xml_path = img_name.split("JPGImages")[0] + "Annotations/" + na

            shutil.copy(xml_path, save_dir + na)


if __name__ == "__main__":
    txt_path = "E:/kx_detection/multi_detection/data/dataset/202003/train_all_0318.txt"
    save_dir = "E:/DataSets/2020_two_phase_KXData/all_data36classes/Annotations/train/"
    if not os.path.exists(save_dir): os.mkdir(save_dir)
    from_txt_copy_data2all(txt_path, save_dir, "xml", False)
