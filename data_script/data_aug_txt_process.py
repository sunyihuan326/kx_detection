# -*- coding: utf-8 -*-
# @Time    : 2020/5/29
# @Author  : sunyihuan
# @File    : data_aug_txt_process.py
'''
处理oven数据中txt文件
'''


def process_txt(txt_path, new_txt_file):
    txt_file = open(txt_path, "r")
    txt_files = txt_file.readlines()
    new_txt_files = []
    for txt_f in txt_files:
        txt_f = txt_f.strip()
        img_name = txt_f.split(".jpg")[0] + ".jpg"
        img_name = img_name.split(" ")[-1]  # 图片地址，全路径
        txt_new_f = img_name
        all_data = txt_f.split(".jpg")[1]

        all_data = all_data.split(" ")
        layer_nums = int(all_data[-5]) - 32  # 烤层数据
        txt_new_f += " " + str(layer_nums)
        bb_cls = ""
        cls_ = []
        for i in range(int((len(all_data) - 7) / 5)):
            cls = all_data[i * 5 + 3]
            cls_.append(cls)
            bboxes = ""
            for c in all_data[i * 5 + 4:i * 5 + 8]:
                bboxes += c + " "
            b_cls = bboxes + " " + cls + " "
            bb_cls = bb_cls + b_cls
        txt_new_f += " " + bb_cls + "\n"
        if len(cls_) == 0:
            print(txt_f)
        else:
            if int(cls_[0]) <= 22:
                new_txt_files.append(txt_new_f)
            else:
                print(txt_f)
    file = open(new_txt_file, "w")
    for i in new_txt_files:
        file.write(i)


if __name__ == "__main__":
    txt_path = "E:/OVEN/trainval_xml_new_add_random.txt"
    new_txt_file = "E:/OVEN/trainval_xml_new_add_random_new.txt"
    process_txt(txt_path, new_txt_file)
