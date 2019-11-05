# -*- coding: utf-8 -*-
import os
import argparse
import xml.etree.ElementTree as ET
import random


def get_layer(typ):
    '''
    获取各层文件名
    :param typ:
    :return:
    '''
    bottom = os.listdir("E:/layer_data/XandOld_layer1105_{}/bottom".format(typ))
    bottom = [b for b in bottom if b.endswith(".jpg")]

    middle = os.listdir("E:/layer_data/XandOld_layer1105_{}/middle".format(typ))
    middle = [b for b in middle if b.endswith(".jpg")]

    top = os.listdir("E:/layer_data/XandOld_layer1105_{}/top".format(typ))
    top = [b for b in top if b.endswith(".jpg")]

    others = os.listdir("E:/layer_data/XandOld_layer1105_{}/others".format(typ))
    others = [b for b in others if b.endswith(".jpg")]
    return bottom, middle, top, others


def convert_voc_annotation(data_path, data_type, anno_path, use_difficult_bbox=True):
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
    classes = ["beefsteak", "cartooncookies", "chickenwings", "chiffoncake6", "chiffoncake8",
               "cookies", "cranberrycookies", "cupcake", "eggtart", "eggtartbig",
               "nofood", "peanuts", "pizzafour", "pizzaone", "pizzasix",
               "pizzatwo", "porkchops", "potatocut", "potatol", "potatom",
               "potatos", "sweetpotatocut", "sweetpotatol", "sweetpotatom", "sweetpotatos",
               "roastedchicken", "toast"]  # 27分类
    # img_inds_file = os.path.join(data_path, 'ImageSets', 'Main', data_type + '.txt')

    img_inds_file = data_path + '/ImageSets' + '/Main/' + '{}.txt'.format(data_type)
    with open(img_inds_file, 'r') as f:
        txt = f.readlines()
        image_inds = [line.strip() for line in txt]
    random.shuffle(image_inds)

    bottom, middle, top, others = get_layer(data_type)

    with open(anno_path, 'a') as f:
        for image_ind in image_inds:
            # image_path = os.path.join(data_path, 'JPEGImages', image_ind + '.jpg')
            # print("image_path::::::::", image_path)
            st = "/"
            image_path = (data_path, 'JPGImages', image_ind + '.jpg')  # 原图片
            # image_path = (data_path, 'JPGImages', image_ind + '_pad.jpg')   #padding图片
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

            annotation += ' ' + str(layer_label)  # annotation中写入烤层的标签

            label_path = (data_path, 'Annotations', image_ind + '.xml')  # 原数据
            # label_path = (data_path, 'Annotations', image_ind + '_pad.xml')  #padding数据
            label_path = st.join(label_path)
            # label_path = os.path.join(data_path, 'Annotations', image_ind + '.xml')
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
    parser.add_argument("--data_path", default="E:/DataSets/KX_FOODSets_model_data/XandOldData1105")
    parser.add_argument("--train_annotation",
                        default="E:/kx_detection/multi_detection/data/dataset/foodSets1105_XandOld_train27.txt")
    parser.add_argument("--test_annotation",
                        default="E:/kx_detection/multi_detection/data/dataset/foodSets1105_XandOld_test27.txt")
    parser.add_argument("--val_annotation",
                        default="E:/kx_detection/multi_detection/data/dataset/foodSets1105_XandOld_val27.txt")
    flags = parser.parse_args()

    if os.path.exists(flags.train_annotation): os.remove(flags.train_annotation)
    if os.path.exists(flags.test_annotation): os.remove(flags.test_annotation)
    if os.path.exists(flags.val_annotation): os.remove(flags.val_annotation)
    # #
    num1 = convert_voc_annotation(flags.data_path, 'train',
                                  flags.train_annotation, False)
    num2 = convert_voc_annotation(flags.data_path, 'test',
                                  flags.test_annotation, False)
    num3 = convert_voc_annotation(flags.data_path, 'val',
                                  flags.val_annotation, False)
    print(
        '=> The number of image for train is: %d\tThe number of image for test is:%d\tThe number of image for val is:%d' % (
            num1, num2, num3))
