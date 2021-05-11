# -*- coding: utf-8 -*-
# @Time    : 2020/7/28
# @Author  : sunyihuan
# @File    : generate_xml_form_ckpt_predict.py
'''
ckpt文件预测结果
识别错误、置信度低，则输出xml文件，同时拷贝原图数据
'''

import cv2
import numpy as np
import tensorflow as tf
import multi_detection.core.utils as utils
from multi_detection.food_correct_utils import correct_bboxes, get_potatoml
from xml.dom.minidom import *
import os
from tqdm import tqdm

CLASSES_name = "E:/kx_detection/multi_detection/data/classes/food40.names"


def read_class_names(class_file_name):
    '''loads class name from a file'''
    names = {}
    with open(class_file_name, 'r') as data:
        for ID, name in enumerate(data):
            names[ID] = name.strip('\n')
    return names


def generate_xml(img_size, bboxes, save_dir, xml_name, target):
    '''
    生成xml文件
    :param img_size:
    :param bboxes:
    :param save_dir:
    :param xml_name:
    :param target:
    :return:
    '''
    (img_width, img_height, img_channel) = img_size
    # 创建一个文档对象
    doc = Document()

    # 创建一个根节点
    root = doc.createElement('annotation')

    # 根节点加入到tree
    doc.appendChild(root)

    # 创建二级节点
    fodler = doc.createElement('fodler')
    fodler.appendChild(doc.createTextNode('1'))  # 添加文本节点

    filename = doc.createElement('filename')
    filename.appendChild(doc.createTextNode('xxxx.jpg'))  # 添加文本节点

    path = doc.createElement('path')
    path.appendChild(doc.createTextNode('./xxxx.jpg'))  # 添加文本节点

    source = doc.createElement('source')
    name = doc.createElement('database')
    name.appendChild(doc.createTextNode('Unknown'))  # 添加文本节点
    source.appendChild(name)  # 添加文本节点

    size = doc.createElement('size')
    width = doc.createElement('width')
    width.appendChild(doc.createTextNode(str(img_width)))  # 添加图片width
    height = doc.createElement('height')
    height.appendChild(doc.createTextNode(str(img_height)))  # 添加图片height
    channel = doc.createElement('depth')
    channel.appendChild(doc.createTextNode(str(img_channel)))  # 添加图片channel
    size.appendChild(height)
    size.appendChild(width)
    size.appendChild(channel)

    segmented = doc.createElement('segmented')
    segmented.appendChild(doc.createTextNode('0'))
    root.appendChild(fodler)  # fodler加入到根节点
    root.appendChild(filename)  # filename加入到根节点
    root.appendChild(path)  # path加入到根节点
    root.appendChild(source)  # source加入到根节点
    root.appendChild(size)  # source加入到根节点
    root.appendChild(segmented)  # segmented加入到根节点

    for i in range(len(bboxes)):
        object = doc.createElement('object')
        name = doc.createElement('name')
        name.appendChild(doc.createTextNode(str(target)))
        object.appendChild(name)
        pose = doc.createElement('pose')
        pose.appendChild(doc.createTextNode("Unspecified"))
        object.appendChild(pose)
        truncated = doc.createElement('truncated')
        truncated.appendChild(doc.createTextNode("0"))
        object.appendChild(truncated)
        difficult = doc.createElement('difficult')
        difficult.appendChild(doc.createTextNode("0"))
        object.appendChild(difficult)
        bndbox = doc.createElement('bndbox')
        xmin = doc.createElement("xmin")
        xmin.appendChild(doc.createTextNode(str(int(bboxes[i][0]))))
        bndbox.appendChild(xmin)
        ymin = doc.createElement("ymin")
        ymin.appendChild(doc.createTextNode(str(int(bboxes[i][1]))))
        bndbox.appendChild(ymin)
        xmax = doc.createElement("xmax")
        xmax.appendChild(doc.createTextNode(str(int(bboxes[i][2]))))
        bndbox.appendChild(xmax)
        ymax = doc.createElement("ymax")
        ymax.appendChild(doc.createTextNode(str(int(bboxes[i][3]))))
        bndbox.appendChild(ymax)
        # difficult.appendChild(doc.createTextNode("0"))
        object.appendChild(bndbox)

        root.appendChild(object)  # object加入到根节点

    # 存成xml文件
    print('{}/{}.xml'.format(save_dir, xml_name))
    fp = open('{}/{}.xml'.format(save_dir, xml_name), 'w', encoding='utf-8')
    doc.writexml(fp, indent='', addindent='\t', newl='\n', encoding='utf-8')
    fp.close()


def rename_for_jpgandxml(filedir, c, typ):
    if typ == "jpg":
        for xxxx in os.listdir(filedir):
            if xxxx.endswith("jpg"):
                file_name = xxxx.split("_")[0] + "_{}".format(c) + ".jpg"
                os.rename(filedir + "/" + xxxx, filedir + "/" + file_name)
    else:
        for xxxx in os.listdir(filedir):
            if xxxx.endswith("xml"):
                file_name = xxxx.split("_")[0] + "_{}".format(c) + ".xml"
                os.rename(filedir + "/" + xxxx, filedir + "/" + file_name)


class YoloPredict(object):
    '''
    预测结果
    '''

    def __init__(self):
        self.input_size = 416  # 输入图片尺寸（默认正方形）
        self.num_classes = 40  # 种类数
        self.score_cls_threshold = 0.001
        self.score_threshold = 0.45
        self.iou_threshold = 0.5
        self.top_n = 5
        self.weight_file = "E:/模型交付版本/multi_yolov3_predict-20210129/checkpoint/yolov3_train_loss=5.9575.ckpt-151"  # ckpt文件地址
        # self.weight_file = "./checkpoint/yolov3_train_loss=4.7681.ckpt-80"
        self.write_image = True  # 是否画图
        self.show_label = True  # 是否显示标签

        graph = tf.Graph()
        with graph.as_default():
            self.saver = tf.train.import_meta_graph("{}.meta".format(self.weight_file))
            self.sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True))
            self.saver.restore(self.sess, self.weight_file)

            self.input = graph.get_tensor_by_name("define_input/input_data:0")
            self.trainable = graph.get_tensor_by_name("define_input/training:0")

            self.pred_sbbox = graph.get_tensor_by_name("define_loss/pred_sbbox/concat_2:0")
            self.pred_mbbox = graph.get_tensor_by_name("define_loss/pred_mbbox/concat_2:0")
            self.pred_lbbox = graph.get_tensor_by_name("define_loss/pred_lbbox/concat_2:0")

            self.layer_num = graph.get_tensor_by_name("define_loss/layer_classes:0")

    def get_top_cls(self, pred_bbox, org_h, org_w, top_n):
        '''
        获取top_n，类别和得分
        :param pred_bbox:所有框
        :param org_h:高
        :param org_w:宽
        :param top_n:top数
        :return:按置信度前top_n个，输出类别、置信度，
        例如
        [(18, 0.9916), (19, 0.0105), (15, 0.0038), (1, 0.0018), (5, 0.0016), (13, 0.0011)]
        '''
        bboxes = utils.postprocess_boxes(pred_bbox, (org_h, org_w), self.input_size, self.score_cls_threshold)
        classes_in_img = list(set(bboxes[:, 5]))
        best_bboxes = {}
        for cls in classes_in_img:
            cls_mask = (bboxes[:, 5] == cls)
            cls_bboxes = bboxes[cls_mask]
            best_score = 0
            for i in range(len(cls_bboxes)):
                if cls_bboxes[i][-2] > best_score:
                    best_score = cls_bboxes[i][-2]
            if int(cls) not in best_bboxes.keys():
                best_bboxes[int(cls)] = round(best_score, 4)
        best_bboxes = sorted(best_bboxes.items(), key=lambda best_bboxes: best_bboxes[1], reverse=True)
        return best_bboxes[:top_n]

    def predict(self, image):
        org_image = np.copy(image)
        org_h, org_w, org_c = org_image.shape

        image_data = utils.image_preporcess(image, [self.input_size, self.input_size])
        image_data = image_data[np.newaxis, ...]

        pred_sbbox, pred_mbbox, pred_lbbox, layer_n = self.sess.run(
            [self.pred_sbbox, self.pred_mbbox, self.pred_lbbox, self.layer_num],
            feed_dict={
                self.input: image_data,
                self.trainable: False
            }
        )

        pred_bbox = np.concatenate([np.reshape(pred_sbbox, (-1, 5 + self.num_classes)),
                                    np.reshape(pred_mbbox, (-1, 5 + self.num_classes)),
                                    np.reshape(pred_lbbox, (-1, 5 + self.num_classes))], axis=0)
        best_bboxes = self.get_top_cls(pred_bbox, org_h, org_w, self.top_n)  # 获取top_n类别和置信度

        bboxes = utils.postprocess_boxes(pred_bbox, (org_h, org_w), self.input_size, self.score_threshold)
        bboxes = utils.nms(bboxes, self.iou_threshold)

        return org_h, org_w, org_c, bboxes, layer_n, best_bboxes

    def result(self, image_path):
        image = cv2.imread(image_path)  # 图片读取
        org_h, org_w, org_c, bboxes_pr, layer_n, best_bboxes = self.predict(image)  # 预测结果
        bboxes_pr, layer_n, best_bboxes = correct_bboxes(bboxes_pr, layer_n, best_bboxes)
        print(bboxes_pr)
        print(layer_n)
        return org_h, org_w, org_c, bboxes_pr, layer_n


if __name__ == '__main__':
    import time
    import shutil

    classes = read_class_names(CLASSES_name)
    classes[40] = "ptatom"
    classes[41] = "sweetpotatom"
    classes[101] = "chiffon_size4"
    start_time = time.time()
    Y = YoloPredict()
    dir_root = "F:serve_data/OVEN/202104/covert_jpg"
    img_root = dir_root + "/yuantu"
    # cls_list = ["potatos", "chestnut", "chickenwings", "porkchops"]
    cls_list = os.listdir(img_root)
    xml_root = dir_root + "/Annotations"
    for c in cls_list:
        if c != " " and c != "others":
            img_dir = img_root + "/" + c
            xml_dir = xml_root + "/" + c
            img_anno_dir = dir_root + "/JPGImages" + "/" + c
            if not os.path.exists(img_anno_dir):
                os.mkdir(img_anno_dir)
            if not os.path.exists(xml_dir):
                os.mkdir(xml_dir)
            for img in tqdm(os.listdir(img_dir)):
                if img.endswith(".jpg"):
                    img_path = img_dir + "/" + img

                    org_h, org_w, org_c, bboxes_pr, layer_n = Y.result(img_path)
                    if len(bboxes_pr) > 0:  # 有标签框
                        cls_name = classes[int(bboxes_pr[0][-1])]  # 预测类别
                        score = bboxes_pr[0][-2]  # 置信度
                        if score >= 0.8:
                            if cls_name != c:  # 分值高于0.8，结果不一致的，输出xml
                                print("0.8 write")
                                shutil.copy(img_path, img_anno_dir + "/" + img)
                                generate_xml((org_w, org_h, org_c), bboxes_pr, xml_dir, img.split(".jpg")[0], c)
                        else:  # 分值低于0.8直接输出xml
                            print("write")
                            shutil.copy(img_path, img_anno_dir + "/" + img)
                            generate_xml((org_w, org_h, org_c), bboxes_pr, xml_dir, img.split(".jpg")[0], c)

                    else:
                        print("no result")
                        shutil.copy(img_path, img_anno_dir + "/" + img)
            # 统一修改命名  jpg前和xml前加入类别名称
            rename_for_jpgandxml(img_anno_dir, c, "jpg")
            rename_for_jpgandxml(xml_dir, c, "xml")
