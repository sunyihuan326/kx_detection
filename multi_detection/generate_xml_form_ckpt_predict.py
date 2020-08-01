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
        self.input_size = 320  # 输入图片尺寸（默认正方形）
        self.num_classes = 22  # 种类数
        self.score_threshold = 0.45
        self.iou_threshold = 0.5
        self.weight_file = "E:/ckpt_dirs/Food_detection/multi_food3/20200717/yolov3_train_loss=4.9898.ckpt-197"  # ckpt文件地址
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

        bboxes = utils.postprocess_boxes(pred_bbox, (org_h, org_w), self.input_size, self.score_threshold)
        bboxes = utils.nms(bboxes, self.iou_threshold)

        return org_h, org_w, org_c, bboxes, layer_n

    def result(self, image_path):
        image = cv2.imread(image_path)  # 图片读取
        org_h, org_w, org_c, bboxes_pr, layer_n = self.predict(image)  # 预测结果
        bboxes_pr, layer_n = correct_bboxes(bboxes_pr, layer_n)
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
    img_root = "E:/WLS_originalData/3660bucai(annotation)/JPGImages/X1"
    # cls_list = ["potatos", "chestnut", "chickenwings", "porkchops"]
    cls_list = ["chiffoncake8"]
    xml_root = "E:/WLS_originalData/3660bucai(annotation)/Annotations"
    for c in cls_list:
        img_dir = img_root + "/" + c
        xml_dir = xml_root + "/" + c
        if not os.path.exists(xml_dir):
            print("1")
            os.mkdir(xml_dir)
        for img in tqdm(os.listdir(img_dir)):
            if img.endswith(".jpg"):
                img_path = img_dir + "/" + img
                org_h, org_w, org_c, bboxes_pr, layer_n = Y.result(img_path)
                generate_xml((org_w, org_h, org_c), bboxes_pr, xml_dir, img.split(".jpg")[0], c)
