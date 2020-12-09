# -*- coding: utf-8 -*-
# @Time    : 2020/7/28
# @Author  : sunyihuan
# @File    : generate_xml_form_ckpt_predict.py
'''
ckpt文件预测结果直接输出xml文件
'''

import cv2
import numpy as np
import tensorflow as tf
import multi_detection.core.utils as utils
from multi_detection.food_correct_utils import correct_bboxes, get_potatoml
from xml.dom.minidom import *
import os
from tqdm import tqdm


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
        self.weight_file = "E:/ckpt_dirs/Food_detection/multi_food5/20201123/yolov3_train_loss=6.5091.ckpt-128"  # ckpt文件地址
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
        # if self.write_image:
        #     image = utils.draw_bbox(image, bboxes_pr, show_label=self.show_label)
        #     drawed_img_save_to_path = str(image_path).split("/")[-1]
        #     cv2.imwrite(drawed_img_save_to_path, image)


if __name__ == '__main__':
    import time

    start_time = time.time()
    Y = YoloPredict()
    img_root = "F:/serve_data/202012030843/JPGImages"
    # cls_list = ["potatos", "chestnut", "chickenwings", "porkchops"]
    cls_list = os.listdir(img_root)
    xml_root = "F:/serve_data/202012030843/Annotations"
    for c in cls_list:
        if c !="chips":
            img_dir = img_root + "/" + c
            xml_dir = xml_root + "/" + c
            if not os.path.exists(xml_dir):
                os.mkdir(xml_dir)
            for img in tqdm(os.listdir(img_dir)):
                if img.endswith(".jpg"):
                    img_path = img_dir + "/" + img

                    org_h, org_w, org_c, bboxes_pr, layer_n = Y.result(img_path)
                    generate_xml((org_w, org_h, org_c), bboxes_pr, xml_dir, img.split(".jpg")[0], c)
