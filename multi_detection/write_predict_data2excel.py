# -*- coding: utf-8 -*-
# @Time    : 2020/6/5
# @Author  : sunyihuan
# @File    : write_predict_data2excel.py
'''
读取文件夹下所有图片的预测数据
长、宽、类别等
'''

import cv2
import numpy as np
import tensorflow as tf
import multi_detection.core.utils as utils
import os
import time
from multi_detection.food_correct_utils import correct_bboxes
import xlwt

# gpu限制
gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.5)
config = tf.ConfigProto(gpu_options=gpu_options)


class YoloPredict(object):
    '''
    预测结果
    '''

    def __init__(self):
        self.input_size = 416  # 输入图片尺寸（默认正方形）
        self.num_classes = 40  # 种类数
        self.score_threshold = 0.45
        self.iou_threshold = 0.5
        self.weight_file = "E:/ckpt_dirs/Food_detection/multi_food5/20200914/yolov3_train_loss=6.9178.ckpt-95"  # ckpt文件地址
        # self.weight_file = "./checkpoint/yolov3_train_loss=4.7681.ckpt-80"
        self.write_image = True  # 是否画图
        self.show_label = True  # 是否显示标签

        graph = tf.Graph()
        with graph.as_default():
            # 模型加载
            self.saver = tf.train.import_meta_graph("{}.meta".format(self.weight_file))
            self.sess = tf.Session(config=config)
            self.saver.restore(self.sess, self.weight_file)

            # 输入
            self.input = graph.get_tensor_by_name("define_input/input_data:0")
            self.trainable = graph.get_tensor_by_name("define_input/training:0")

            # 输出检测结果
            self.pred_sbbox = graph.get_tensor_by_name("define_loss/pred_sbbox/concat_2:0")
            self.pred_mbbox = graph.get_tensor_by_name("define_loss/pred_mbbox/concat_2:0")
            self.pred_lbbox = graph.get_tensor_by_name("define_loss/pred_lbbox/concat_2:0")

            # 输出烤层结果
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

    def result(self, image_path, save_dir="C:/Users/sunyihuan/Desktop/X5_test/X6_0610/chiffoncake6_detect"):
        image = cv2.imread(image_path)  # 图片读取
        # image = utils.white_balance(image)  # 图片白平衡处理
        bboxes_pr, layer_n = self.predict(image)  # 预测结果
        if not os.path.exists(save_dir): os.mkdir(save_dir)
        if self.write_image:
            image = utils.draw_bbox(image, bboxes_pr, show_label=self.show_label)
            drawed_img_save_to_path = str(image_path).split("/")[-1]
            drawed_img_save_to_path = save_dir + "/" + drawed_img_save_to_path.split(".jpg")[0] + "_" + str(
                layer_n[0]) + ".jpg"
            # print(drawed_img_save_to_path)
            cv2.imwrite(drawed_img_save_to_path, image)

        return bboxes_pr, layer_n


if __name__ == '__main__':
    start_time = time.time()
    img_dir = "C:/Users/sunyihuan/Desktop/JPGImages_chiffon4"  # 图片文件地址
    Y = YoloPredict()
    end_time0 = time.time()
    print("model loading time:", end_time0 - start_time)
    classes = ["beefsteak", "cartooncookies", "chickenwings", "chiffoncake6", "cookies",
               "cranberrycookies", "cupcake", "eggtart", "nofood", "peanuts",
               "pizzacut", "pizzaone", "pizzatwo", "porkchops", "potatocut",
               "potatol", "potatos", "sweetpotatocut", "sweetpotatol", "sweetpotatos",
               "roastedchicken", "toast", "potatom", "sweetpotatom"]
    wk = xlwt.Workbook("chiffon_predict")
    sheet = wk.add_sheet("result")
    sheet.write(0, 0, "jpg_name")
    sheet.write(0, 1, "xmin")
    sheet.write(0, 2, "ymin")
    sheet.write(0, 3, "xmax")
    sheet.write(0, 4, "ymax")
    sheet.write(0, 5, "class")
    sheet.write(0, 6, "layer")
    for i, img in enumerate(os.listdir(img_dir)):
        if img.endswith("jpg"):
            img_path = img_dir + "/" + img
            end_time1 = time.time()
            bboxes_p, layer_ = Y.result(img_path)
            bboxes_pr, layer_n = correct_bboxes(bboxes_p, layer_)  # 矫正输出结果
            if len(bboxes_pr) == 0:
                sheet.write(i + 1, 0, img)
            else:
                pre = bboxes_pr[0][-1]
                sheet.write(i + 1, 0, img)
                sheet.write(i + 1, 1, int(bboxes_pr[0][0]))
                sheet.write(i + 1, 2, int(bboxes_pr[0][1]))
                sheet.write(i + 1, 3, int(bboxes_pr[0][2]))
                sheet.write(i + 1, 4, int(bboxes_pr[0][3]))
                sheet.write(i + 1, 5, pre)
                sheet.write(i + 1, 6, int(layer_n[0]))
    wk.save("C:/Users/sunyihuan/Desktop/JPGImages_chiffon4/chiffon_predict.xls")
