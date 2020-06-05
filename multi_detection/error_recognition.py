# -*- coding: utf-8 -*-
# @Time    : 2020/5/19
# @Author  : sunyihuan
# @File    : error_recognition.py

'''
测试误识别率
'''
import cv2
import numpy as np
import tensorflow as tf
import multi_detection.core.utils as utils
import os
import time
from multi_detection.food_correct_utils import correct_bboxes
from tqdm import tqdm

# gpu限制
gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.2)
config = tf.ConfigProto(gpu_options=gpu_options)


class YoloPredict(object):
    '''
    预测结果
    '''

    def __init__(self):
        self.input_size = 416  # 输入图片尺寸（默认正方形）
        self.num_classes = 30  # 种类数
        self.score_threshold = 0.45
        self.iou_threshold = 0.5
        self.pb_file = "E:/multi_yolov3_predict-20191220/checkpoint/yolov3_1220.pb"  # ckpt文件地址
        self.write_image = True  # 是否画图
        self.show_label = True  # 是否显示标签

        graph = tf.Graph()
        with graph.as_default():
            output_graph_def = tf.GraphDef()
            with open(self.pb_file, "rb") as f:
                output_graph_def.ParseFromString(f.read())
                _ = tf.import_graph_def(output_graph_def, name="")

            self.sess = tf.Session(config=config)

            # 输入
            self.input = self.sess.graph.get_tensor_by_name("define_input/input_data:0")
            self.trainable = self.sess.graph.get_tensor_by_name("define_input/training:0")

            # 输出
            self.pred_sbbox = self.sess.graph.get_tensor_by_name("define_loss/pred_sbbox/concat_2:0")
            self.pred_mbbox = self.sess.graph.get_tensor_by_name("define_loss/pred_mbbox/concat_2:0")
            self.pred_lbbox = self.sess.graph.get_tensor_by_name("define_loss/pred_lbbox/concat_2:0")

            # 烤层
            self.layer_num = graph.get_tensor_by_name("define_loss/layer_classes:0")

    def predict(self, image):
        org_image = np.copy(image)
        org_h, org_w, _ = org_image.shape

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

        return bboxes, layer_n

    def result(self, image_path, save_dir="./img_detection"):
        image = cv2.imread(image_path)  # 图片读取
        # image = utils.white_balance(image)  # 图片白平衡处理
        bboxes_pr, layer_n = self.predict(image)  # 预测结果
        return bboxes_pr, layer_n


if __name__ == '__main__':
    start_time = time.time()
    img_dir = "C:/Users/sunyihuan/Desktop/JPGImages"  # 图片文件地址
    Y = YoloPredict()
    end_time0 = time.time()
    print("model loading time:", end_time0 - start_time)
    error_c = 0
    save_dir="C:/Users/sunyihuan/Desktop/JPGImages_det"
    for img in tqdm(os.listdir(img_dir)):
        # print(img)
        img_path = img_dir + "/" + img
        end_time1 = time.time()
        bboxes_pr, layer_n = Y.result(img_path)
        end_time2 = time.time()
        # print("predict time:", end_time2 - end_time1)
        if len(bboxes_pr) != 0:
            print(img)
            bboxes_pr, layer_n = correct_bboxes(bboxes_pr, layer_n)
            if len(bboxes_pr) != 0:
                error_c += 1
                image = cv2.imread(img_path)
                image = utils.draw_bbox(image, bboxes_pr)
                drawed_img_save_to_path = str(img_path).split("/")[-1]
                drawed_img_save_to_path = save_dir + "/" + drawed_img_save_to_path.split(".jpg")[0] + "_" + str(
                    layer_n[0]) + ".jpg"
                # print(drawed_img_save_to_path)
                cv2.imwrite(drawed_img_save_to_path, image)
    print(error_c)