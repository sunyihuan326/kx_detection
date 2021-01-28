# -*- encoding: utf-8 -*-

"""
预测一张图片结果
@File    : ckpt_predict.py
@Time    : 2019/8/19 15:45
@Author  : sunyihuan
"""

import cv2
import numpy as np
import tensorflow as tf
import multi_detection.core.utils as utils
from multi_detection.food_correct_utils import correct_bboxes, get_potatoml
from multi_detection.core.utils import postprocess_boxes_conf as postprocess_box


class YoloPredict(object):
    '''
    预测结果
    '''

    def __init__(self):
        self.input_size = 416  # 输入图片尺寸（默认正方形）
        self.num_classes = 40  # 种类数
        self.score_cls_threshold = 0.001
        self.score_threshold = 0.4
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
        bboxes = postprocess_box(pred_bbox, (org_h, org_w), self.input_size, self.score_cls_threshold)
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
        org_h, org_w, _ = org_image.shape

        image_data = utils.image_preporcess(image, [self.input_size, self.input_size])
        image_data = image_data[np.newaxis, ...]

        pred_sbbox, pred_mbbox, pred_lbbox, layer_ = self.sess.run(
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

        bboxes = postprocess_box(pred_bbox, (org_h, org_w), self.input_size, self.score_threshold)
        bboxes = utils.nms(bboxes, self.iou_threshold)
        layer_n = layer_[0]  # 烤层结果

        return bboxes, layer_n, best_bboxes

    def result(self, image_path, save_dir=""):
        image = cv2.imread(image_path)  # 图片读取
        bboxes_pr, layer_n, best_bboxes = self.predict(image)
        print("top_n类被及置信度：", best_bboxes)
        print("食材结果：", bboxes_pr)
        print("烤层结果：", layer_n)

        bboxes_pr, layer_n, best_bboxes = correct_bboxes(bboxes_pr, layer_n, best_bboxes)  # 矫正输出结果
        bboxes_pr, layer_n = get_potatoml(bboxes_pr, layer_n)  # 根据输出结果对中大红薯，中大土豆做输出

        print("top_n类被及置信度：", best_bboxes)
        print(bboxes_pr)
        print(layer_n)
        if self.write_image:
            image = utils.draw_bbox(image, bboxes_pr, show_label=self.show_label)
            drawed_img_save_to_path = save_dir + "/" + str(image_path).split("/")[-1]
            cv2.imwrite(drawed_img_save_to_path, image)


if __name__ == '__main__':
    import os
    # img_path = "C:/Users/sunyihuan/Desktop/t/16_bottom_test2020_zhengchang_canju_nofood.jpg"  # 图片地址
    # save_dir = "C:/Users/sunyihuan/Desktop/t_detection"
    # if not os.path.exists(save_dir): os.mkdir(save_dir)
    Y = YoloPredict()
    # Y.result(img_path,save_dir)


    img_dir = "C:/Users/sunyihuan/Desktop/te_0108"
    save_dir = "C:/Users/sunyihuan/Desktop/te_0108_detection"
    if not os.path.exists(save_dir): os.mkdir(save_dir)
    for img in os.listdir(img_dir):
        if img.endswith(".jpg"):
            img_path = img_dir + "/" + img
            Y.result(img_path, save_dir)
