# -*- coding: utf-8 -*-
# @Time    : 2020/3/20
# @Author  : sunyihuan
# @File    : voc_annotation_aug.py

'''
voc标准数据处理增强后的数据
'''
import os
import argparse
import xml.etree.ElementTree as ET
import random
from tqdm import tqdm


def get_layer(data_path):
    '''
    获取各层文件名
    :param typ:
    :return:
    '''
    layer_data_dir = data_path + "/layer_data"
    bottom = os.listdir(layer_data_dir + "/bottom")
    bottom = [b for b in bottom if b.endswith(".jpg")]

    middle = os.listdir(layer_data_dir + "/middle")
    middle = [b for b in middle if b.endswith(".jpg")]

    top = os.listdir(layer_data_dir + "/top")
    top = [b for b in top if b.endswith(".jpg")]

    others_path = layer_data_dir + "/others"
    if os.path.exists(others_path):
        others = os.listdir(others_path)
        others = [b for b in others if b.endswith(".jpg")]
    else:
        others = []
    return bottom, middle, top, others


def convert_voc_annotation(data_path, data_type, anno_path, aug_name,use_difficult_bbox=True):
    # classes = ["beefsteak", "cartooncookies", "chickenwings", "chiffoncake", "cookies",
    #            "cranberrycookies", "cupcake", "eggtart", "nofood", "peanuts",
    #            "pizza", "porkchops", "purplesweetpotato", "roastedchicken", "toast",
    #            "potatos", "potatom", "potatol", "sweetpotatos", "sweetpotatom", "sweetpotatol"] # 21分类
    # classes = ["beefsteak", "cartooncookies", "chickenwings", "chiffoncake", "cookies",
    #            "cranberrycookies", "cupcake", "eggtart", "nofood", "peanuts",
    #            "pizza", "porkchops", "purplesweetpotato", "roastedchicken", "toast",
    #            "potatos", "potatom", "potatol", "sweetpotatos", "sweetpotatom", "sweetpotatol", "potatocut",
    #            "sweetpotatocut"]  # 23分类
    # classes = ["beefsteak", "cartooncookies", "chickenwings", "chiffoncake", "cookies",
    #            "cranberrycookies", "cupcake", "eggtart", "nofood", "peanuts",
    #            "pizza", "porkchops", "purplesweetpotato", "roastedchicken", "toast"]  # 15分类
    # classes = ["beefsteak", "cartooncookies", "chickenwings", "chiffoncake6", "chiffoncake8",
    #            "cookies", "cranberrycookies", "cupcake", "eggtart", "eggtartbig",
    #            "nofood", "peanuts", "pizzafour", "pizzaone", "pizzasix",
    #            "pizzatwo", "porkchops", "potatocut", "potatol", "potatom",
    #            "potatos", "sweetpotatocut", "sweetpotatol", "sweetpotatom", "sweetpotatos",
    #            "roastedchicken", "toast"]  # 27分类
    # classes = ["beefsteak", "cartooncookies", "chickenwings", "chiffoncake6", "chiffoncake8",
    #            "cookies", "cranberrycookies", "cupcake", "eggtart", "eggtartbig",
    #            "nofood", "peanuts", "pizzafour", "pizzaone", "pizzasix",
    #            "pizzatwo", "porkchops", "potatocut", "potatol", "potatom",
    #            "potatos", "sweetpotatocut", "sweetpotatol", "sweetpotatom", "sweetpotatos",
    #            "roastedchicken", "toast", "sweetpotato_others", "pizza_others",
    #            "potato_others"]  # 30分类,加入了sweetpotato_others,pizza_others,potato_others
    classes = ["beefsteak", "cartooncookies", "chickenwings", "chiffoncake6", "chiffoncake8",
               "cookies", "cranberrycookies", "cupcake", "eggtart", "eggtartbig",
               "nofood", "peanuts", "pizzafour", "pizzaone", "pizzasix",
               "pizzatwo", "porkchops", "potatocut", "potatol", "potatom",
               "potatos", "sweetpotatocut", "sweetpotatol", "sweetpotatom", "sweetpotatos",
               "roastedchicken", "toast", "sweetpotato_others", "pizza_others",
               "potato_others", "chestnut", "cornone", "corntwo", "drumsticks", "taro",
               "steamedbread", "eggplant", "eggplant_cut_sauce", "bread", "container_nonhigh"
        , "container", "fish", "hotdog", "redshrimp", "shrimp", "strand"]  # 原30分类，加入二期17类
    # classes = ["nofood","chestnut", "cornone", "corntwo", "drumsticks", "taro",
    #            "steamedbread", "eggplant", "eggplant_cut_sauce", "bread", "container_nonhigh",
    #            "container", "roastedchicken", "fish", "hotdog", "redshrimp",
    #            "shrimp", "strand"]  # 仅二期17类
    # classes = ["nofood","chestnut", "cornone", "corntwo", "drumsticks", "taro",
    #            "steamedbread"]  # 仅二期6类
    # img_inds_file = os.path.join(data_path, 'ImageSets', 'Main', data_type + '.txt')

    img_inds_file = data_path + '/ImageSets' + '/Main/' + '{}.txt'.format(data_type)
    with open(img_inds_file, 'r') as f:
        txt = f.readlines()
        image_inds = [line.strip() for line in txt]
    random.shuffle(image_inds)

    bottom, middle, top, others = get_layer(data_path)

    with open(anno_path, 'a') as f:
        for image_ind in tqdm(image_inds):
            st = "/"
            image_path = (data_path, 'JPGImages{}'.format(aug_name), image_ind + '{}.jpg'.format(aug_name))  # 增强图片
            image_path = st.join(image_path)

            annotation = image_path
            if image_ind + ".jpg" in bottom:
                layer_label = 0
            elif image_ind + ".jpg" in middle:
                layer_label = 1
            elif image_ind + ".jpg" in top:
                layer_label = 2
            elif image_ind + ".jpg" in others:
                layer_label = 3
            else:
                print(image_path)
                print("error")
                continue

            annotation += ' ' + str(layer_label)  # annotation中写入烤层的标签

            label_path = (data_path, 'Annotations{}'.format(aug_name), image_ind + '{}.xml'.format(aug_name))  # 增强数据
            label_path = st.join(label_path)
            root = ET.parse(label_path).getroot()
            objects = root.findall('object')
            for obj in objects:
                # difficult = obj.find('difficult').text.strip()
                # if (not use_difficult_bbox) and (int(difficult) == 1):
                #     continue
                bbox = obj.find('bndbox')
                label_name = obj.find('name').text.lower()

                # if "enwings" in label_name:
                #     label_name = "chickenwings"
                class_ind = classes.index(label_name.strip())
                xmin = bbox.find('xmin').text.strip()
                xmax = bbox.find('xmax').text.strip()
                ymin = bbox.find('ymin').text.strip()
                ymax = bbox.find('ymax').text.strip()
                annotation += ' ' + ','.join([xmin, ymin, xmax, ymax, str(class_ind)])
            f.write(annotation + "\n")
    return len(image_inds)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path",
                        default="E:/DataSets/KXDataAll")
    parser.add_argument("--train_annotation",
                        default="E:/DataSets/KXDataAll/dataset_for_model/train_all0513_resize_l.txt")
    parser.add_argument("--test_annotation",
                        default="E:/DataSets/KXDataAll/dataset_for_model/test_all0513_resize_l.txt")
    # parser.add_argument("--val_annotation",
    #                     default="E:/kx_detection/multi_detection/data/dataset/202003/val0318.txt")
    parser.add_argument("--aug_name",
                        default="_resize_l")
    flags = parser.parse_args()
    #
    if os.path.exists(flags.train_annotation): os.remove(flags.train_annotation)
    if os.path.exists(flags.test_annotation): os.remove(flags.test_annotation)
    # if os.path.exists(flags.val_annotation): os.remove(flags.val_annotation)
    # # #
    num1 = convert_voc_annotation(flags.data_path, 'train',
                                  flags.train_annotation,flags.aug_name, False)
    num2 = convert_voc_annotation(flags.data_path, 'test',
                                  flags.test_annotation,flags.aug_name, False)
