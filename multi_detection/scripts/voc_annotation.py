# -*- coding: utf-8 -*-
import os
import argparse
import xml.etree.ElementTree as ET
import random
from tqdm import tqdm
from multi_detection.core.config import cfg


def get_layer(data_path, typ):
    '''
    获取各层文件名
    :param typ:
    :return:
    '''
    layer_data_dir = data_path
    # bottom = os.listdir(layer_data_dir + "/layer_data/bottom")
    bottom = os.listdir(layer_data_dir + "/layer_data/{}/bottom".format(typ))
    bottom = [b for b in bottom if b.endswith(".jpg")]

    middle = os.listdir(layer_data_dir + "/layer_data/{}/middle".format(typ))
    # middle = os.listdir(layer_data_dir + "/layer_data/middle")
    middle = [b for b in middle if b.endswith(".jpg")]

    top = os.listdir(layer_data_dir + "/layer_data/{}/top".format(typ))
    # top = os.listdir(layer_data_dir + "/layer_data/top")
    top = [b for b in top if b.endswith(".jpg")]

    others_path = layer_data_dir + "/layer_data/{}/others".format(typ)
    # others_path = layer_data_dir + "/layer_data/others"
    if os.path.exists(others_path):
        others = os.listdir(others_path)
        others = [b for b in others if b.endswith(".jpg")]
    else:
        others = []
    return bottom, middle, top, others


def convert_voc_annotation(data_path, data_type, anno_path, use_difficult_bbox=True):
    def read_class_names(class_file_name):
        '''loads class name from a file'''
        names = []
        with open(class_file_name, 'r') as data:
            for name in data:
                names.append(name.strip('\n'))
        return names

    classes = read_class_names(cfg.YOLO.CLASSES)
    print(classes)

    img_inds_file = data_path + '/ImageSets' + '/Main/' + '{}.txt'.format(data_type)
    with open(img_inds_file, 'r') as f:
        txt = f.readlines()
        image_inds = [line.strip() for line in txt]
    random.shuffle(image_inds)

    bottom, middle, top, others = get_layer(data_path, data_type)
    print(len(bottom))
    print(len(middle))
    print(len(top))
    print(len(others))

    with open(anno_path, 'a') as f:
        for image_ind in tqdm(image_inds):
            st = "/"
            image_path = (data_path, 'JPGImages', image_ind + '.jpg')  # 原图片
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

            label_path = (data_path, 'Annotations', image_ind + '.xml')  # 原数据
            label_path = st.join(label_path)
            if os.path.exists(label_path):
                root = ET.parse(label_path).getroot()
                objects = root.findall('object')
                for obj in objects:
                    # difficult = obj.find('difficult').text.strip()
                    # if (not use_difficult_bbox) and (int(difficult) == 1):
                    #     continue
                    bbox = obj.find('bndbox')
                    label_name = obj.find('name').text.lower()

                    if "pizza_others" in label_name:
                        break
                    if "sweetpotato_others" in label_name:
                        break
                    if "potato_others" in label_name:
                        break
                    if "eggplant_cut" in label_name and "eggplant_cut_sauce" not in label_name:
                        break
                    if "pizzasix" in label_name:
                        label_name = "pizzacut"
                    if "duck" in label_name:
                        label_name = "roastedchicken"
                    if "stand" in label_name:
                        label_name = "strand"
                    if "pizzafour" in label_name:
                        label_name = "pizzacut"
                    if "no container_nonhigh" in label_name:
                        label_name = "container_nonhigh"
                    if "no food" in label_name:
                        label_name = "nofood"
                    if "no bread" in label_name:
                        break
                    class_ind = classes.index(label_name.strip())
                    xmin = bbox.find('xmin').text.strip()
                    xmax = bbox.find('xmax').text.strip()
                    ymin = bbox.find('ymin').text.strip()
                    ymax = bbox.find('ymax').text.strip()
                    annotation += ' ' + ','.join([xmin, ymin, xmax, ymax, str(class_ind)])
                f.write(annotation + "\n")

    return len(image_inds)


def check_txt(txt_path):
    '''
    检查是否含有无标签框数据，若有，去除
    :return:
    '''
    train_txt_file = open(txt_path, "r")
    train_txt_files = train_txt_file.readlines()
    train_all_list = []
    for txt_file_one in train_txt_files:
        if len(txt_file_one.split(" ")) > 2:
            train_all_list.append(txt_file_one)
        else:
            print(txt_file_one)
    file = open(txt_path, "w")
    for i in train_all_list:
        file.write(i)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path",
                        default="F:/serve_data/for_model/202101_03")
    parser.add_argument("--train_annotation",
                        default="F:/serve_data/for_model/202101_03/train42.txt")
    parser.add_argument("--test_annotation",
                        default="F:/serve_data/for_model/202101_03/test42.txt")
    parser.add_argument("--val_annotation",
                        default="F:/serve_data/for_model/202101_03/val42.txt")
    flags = parser.parse_args()
    #
    if os.path.exists(flags.train_annotation): os.remove(flags.train_annotation)
    if os.path.exists(flags.test_annotation): os.remove(flags.test_annotation)
    if os.path.exists(flags.val_annotation): os.remove(flags.val_annotation)
    # # #
    num1 = convert_voc_annotation(flags.data_path, 'train',
                                  flags.train_annotation, False)
    num2 = convert_voc_annotation(flags.data_path, 'test',
                                  flags.test_annotation, False)
    num3 = convert_voc_annotation(flags.data_path, 'val',
                                  flags.val_annotation, False)
    # print(
    #     '=> The number of image for train is: %d\tThe number of image for test is:%d\tThe number of image for val is:%d' % (
    #         num1, num2, num3))
    #
    check_txt(flags.test_annotation)
    check_txt(flags.train_annotation)
    check_txt(flags.val_annotation)
