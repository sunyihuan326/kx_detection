# -*- coding: utf-8 -*-
# @Time    : 202003/3/18
# @Author  : sunyihuan
# @File    : data_process.py

'''
原数据格式说明：
             JPGImages             Annotations             ImageSets/Main         layer_data
                 xxxx                    xxxx                      *.txt               xxxx
                    *.jpg                   *.xml                                         bottom
                                                                                             *.jpg
                                                                                           middle
                                                                                             *.jpg
                                                                                           top
                                                                                             *.jpg
将原始数据处理成./multi_detection/scripts/voc_annotation.py可用数据
step1:各类别数据单独分xx_train.txt、xx_test.txt、xx_val.txt
step2:将各类别数据合成完整train.txt、test_all.txt、val.txt
step3:拷贝所有单类别图片至JPGImages中、拷贝所有xml文件至Annotations中
step4:layer数据按train.txt、test_all.txt、val.txt分到相应文件夹
'''
import os
import random
import shutil
from tqdm import tqdm


class process(object):
    '''
    数据处理，生成标准格式
    '''

    def __init__(self, data_root):
        self.data_root = data_root
        self.image_root = self.data_root + '/JPGImages'
        self.xml_root = self.data_root + "/Annotations"
        self.txt_root = self.data_root + '/ImageSets/Main'

        self.layer_root = self.data_root + "/layer_data"

    def split_data(self, classes, test_percent, val_percent):
        '''
        按类别名称将每一类分为test、train
        :param clasees: 类别名称，格式为list
        :param test_percent: test集的占比，一般为0.1-0.2
        :param val_percent: val集的占比，一般为0.1-0.2
        :return:
        '''
        if not os.path.exists(self.data_root):
            print("cannot find such directory: " + self.data_root)
            exit()
        for c in classes:
            # xmlfilepath = self.xml_root + "/{}".format(classes)
            imgfilepath = self.image_root + "/{}".format(c)
            txtsavepath = self.txt_root
            if not os.path.exists(txtsavepath):
                os.makedirs(txtsavepath)

            total_xml = []
            for a in os.listdir(imgfilepath):
                if a.endswith(".jpg"):
                    total_xml.append(a)

            random.shuffle(total_xml)  # 打乱total_xml
            num = len(total_xml)

            te = int(num * test_percent)  # test集数量
            test = total_xml[:te]  # test集列表数据内容
            val = int(num * val_percent)  # val集数量
            val_list = total_xml[:te + val]  # val集列表数据内容
            tr = num - val - te  # val集数量
            print(c)
            print("train size:", tr)
            print("test size:", te)
            print("val size:", val)
            ftest = open(txtsavepath + '/{}_test.txt'.format(str(c).lower()), 'w')
            ftrain = open(txtsavepath + '/{}_train.txt'.format(str(c).lower()), 'w')
            fval = open(txtsavepath + '/{}_val.txt'.format(str(c).lower()), 'w')

            for x in total_xml:
                if str(x).endswith("jpg"):
                    name = x[:-4] + '\n'
                    if x in test:
                        ftest.write(name)
                    elif x in val_list:
                        fval.write(name)
                    else:
                        ftrain.write(name)

            ftrain.close()
            ftest.close()
            fval.close()

    def train_all_txt(self, txt_names):
        '''
        读取所有的行
        :param txt_name: txt文件名称["train"、"test"]
        :return:
        '''
        for txt_name in txt_names:  # train、 test、 val
            train_all_list = []  # 创建train_list列表
            for t in os.listdir(self.txt_root):  # 单类别文件循环
                if "_{}".format(txt_name) in t:  # 找到所有xx_train.txt文件
                    txt_all = self.txt_root + "/" + t
                    txt_file = open(txt_all, "r")
                    txt_files = txt_file.readlines()  # xx_train.txt 文件中数据
                    for txt_file_one in txt_files:
                        train_all_list.append(txt_file_one)  # xx_train.txt 中数据插入到train_all_list中
            random.shuffle(train_all_list)  # 打乱train_all_list列表
            print(len(train_all_list))
            all_txt_name = self.txt_root + "/" + txt_name + ".txt"  # 写入train.txt文件
            file = open(all_txt_name, "w")
            for i in train_all_list:
                file.write(i)

    def copy2dir(self, classes, typ):
        '''
        拷贝所有jpg（或xml）文件至目录下
        :param classes: 需要拷贝的类别
        :param typ: jpg或者xml，str格式
        :return:
        '''
        assert typ in ["xml", "jpg"]  # 判断是否为xml或者jpg拷贝
        if typ == "xml":
            file_dir = self.xml_root
        else:
            file_dir = self.image_root
        for c in tqdm(classes):
            for file in os.listdir(file_dir + "/{}".format(c)):
                file_name = file_dir + "/{}".format(c) + "/" + file
                shutil.copy(file_name, file_dir + "/" + file)

    def copy_layer2split_dir(self, classes):
        '''
        按train、test、val将单类别下的烤层数据分到对应的train、test、val下
        原格式为：
                 layer_data
                       xxx
                         bottom
                         middle
                         top
                         others
        分类后：
                 layer_data
                      train                       test                     val
                         bottom                      bottom                   bottom
                         middle                      middle                   middle
                         top                         top                      top
                         others                      others                   others
        :param classes: 要处理的类别，list格式
        :return:
        '''
        # layer下创建对应train、test、val文件夹
        if os.path.exists(self.layer_root + "/train"): shutil.rmtree(self.layer_root + "/train")
        os.mkdir(self.layer_root + "/train")
        if os.path.exists(self.layer_root + "/test"): shutil.rmtree(self.layer_root + "/test")
        os.mkdir(self.layer_root + "/test")
        if os.path.exists(self.layer_root + "/val"): shutil.rmtree(self.layer_root + "/val")
        os.mkdir(self.layer_root + "/val")
        for l in ["bottom", "middle", "top", "others"]:
            layer_train_bottom_dir = self.layer_root + "/train/{}".format(l)
            layer_test_bottom_dir = self.layer_root + "/test/{}".format(l)
            layer_val_bottom_dir = self.layer_root + "/val/{}".format(l)
            if os.path.exists(layer_train_bottom_dir): shutil.rmtree(layer_train_bottom_dir)  # 判断是否存在，存在则删除
            os.mkdir(layer_train_bottom_dir)  # 创建文件夹
            if os.path.exists(layer_test_bottom_dir): shutil.rmtree(layer_test_bottom_dir)
            os.mkdir(layer_test_bottom_dir)
            if os.path.exists(layer_val_bottom_dir): shutil.rmtree(layer_val_bottom_dir)
            os.mkdir(layer_val_bottom_dir)

        # 获取train中的所有文件名称
        train_txt = self.txt_root + "/" + "train.txt"
        txt_file = open(train_txt, "r")
        train_txt_files = txt_file.readlines()
        train_txt_files = [v.strip() for v in train_txt_files]
        # 获取test中的所有文件名称
        test_txt = self.txt_root + "/" + "test.txt"
        txt_file = open(test_txt, "r")
        test_txt_files = txt_file.readlines()
        test_txt_files = [v.strip() for v in test_txt_files]
        # 获取val中的所有文件名称
        val_txt = self.txt_root + "/" + "val.txt"
        txt_file = open(val_txt, "r")
        val_txt_files = txt_file.readlines()
        val_txt_files = [v.strip() for v in val_txt_files]

        for c in classes:
            for l in ["bottom", "middle", "top", "others"]:
                img_layer_dir = self.layer_root + "/" + c + "/" + l
                try:
                    for img in tqdm(os.listdir(img_layer_dir)):
                        if img.split(".")[0] in val_txt_files:  # 判断若img名称在val中，拷贝图片至val中
                            shutil.copy(img_layer_dir + "/" + img, self.layer_root + "/val/" + l + "/" + img)
                        elif img.split(".")[0] in test_txt_files:  # 判断若img名称在test中，拷贝图片至test中
                            shutil.copy(img_layer_dir + "/" + img, self.layer_root + "/test/" + l + "/" + img)
                        elif img.split(".")[0] in train_txt_files:  # 其他的拷贝图片至train中
                            shutil.copy(img_layer_dir + "/" + img, self.layer_root + "/train/" + l + "/" + img)
                        else:
                            print("*****************:", img)
                except:
                    print(img_layer_dir)


if __name__ == "__main__":
    data_root = "F:/serve_data/for_model/202101_03"
    dprocess = process(data_root)

    classes =os.listdir(data_root+"/JPGImages")
    val_percent = 0
    test_percent = 0.1
    dprocess.split_data(classes, test_percent, val_percent)
    dprocess.train_all_txt(["train", "test", "val"])
    dprocess.copy2dir(classes, "xml")
    dprocess.copy2dir(classes, "jpg")
    dprocess.copy_layer2split_dir(classes)
