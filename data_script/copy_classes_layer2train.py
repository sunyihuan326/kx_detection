# -*- encoding: utf-8 -*-

"""
将每类layer数据按train.txt、test.txt、val.txt分到完整的train、test、val中
@File    : copy_classes_layer2train.py
@Time    : 2019/12/3 19:35
@Author  : sunyihuan
"""
import shutil
import os

# 创建layer_train、layer_test、layer_val文件夹中的bottom
layer_train_bottom_dir = "E:/DataSets/KX_FOODSets_model_data/20191205toast_nofood/layer_data/train/top"
layer_test_bottom_dir = "E:/DataSets/KX_FOODSets_model_data/20191205toast_nofood/layer_data/test/top"
layer_val_bottom_dir = "E:/DataSets/KX_FOODSets_model_data/20191205toast_nofood/layer_data/val/top"
if os.path.exists(layer_train_bottom_dir): shutil.rmtree(layer_train_bottom_dir)  # 判断是否存在，存在则删除
os.mkdir(layer_train_bottom_dir)  # 创建文件夹
if os.path.exists(layer_test_bottom_dir): shutil.rmtree(layer_test_bottom_dir)
os.mkdir(layer_test_bottom_dir)
if os.path.exists(layer_val_bottom_dir): shutil.rmtree(layer_val_bottom_dir)
os.mkdir(layer_val_bottom_dir)


def copy_img2train(layer_root, txt_root):
    # 获取train中的所有文件名称
    train_txt = txt_root + "/" + "train.txt"
    txt_file = open(train_txt, "r")
    train_txt_files = txt_file.readlines()
    train_txt_files = [v.strip() for v in train_txt_files]
    # 获取test中的所有文件名称
    test_txt = txt_root + "/" + "test.txt"
    txt_file = open(test_txt, "r")
    test_txt_files = txt_file.readlines()
    test_txt_files = [v.strip() for v in test_txt_files]
    # 获取val中的所有文件名称
    val_txt = txt_root + "/" + "val.txt"
    txt_file = open(val_txt, "r")
    val_txt_files = txt_file.readlines()
    val_txt_files = [v.strip() for v in val_txt_files]

    # clasees = ["nofood", "pizza_four", "pizza_one", "pizza_six", "pizza_two",
    #            "porkchops", "PotatoCut", "potatol", "potatom", "Potatos",
    #            "SweetPotatoCut", "sweetpotatol", "sweetpotatom", "sweetpotatos", "toast", ]
    clasees = ["nofood", "toast"]
    for c in clasees:
        cl_dir = layer_root + "/" + c + "/top"
        try:
            for img_file in os.listdir(cl_dir):
                if img_file.split(".")[0] in train_txt_files:
                    shutil.copy(cl_dir + "/" + img_file, layer_train_bottom_dir + "/" + img_file)
                elif img_file.split(".")[0] in test_txt_files:
                    shutil.copy(cl_dir + "/" + img_file, layer_test_bottom_dir + "/" + img_file)
                elif img_file.split(".")[0] in val_txt_files:
                    shutil.copy(cl_dir + "/" + img_file, layer_val_bottom_dir + "/" + img_file)
                else:
                    print(img_file)
        except:
            pass


if __name__ == "__main__":
    layer_root = "E:/DataSets/KX_FOODSets_model_data/20191205toast_nofood/layer_data"
    txt_root = "E:/DataSets/KX_FOODSets_model_data/20191205toast_nofood/ImageSets/Main"
    copy_img2train(layer_root, txt_root)
