# -*- coding: utf-8 -*-
# @Time    : 2020/3/25
# @Author  : sunyihuan
# @File    : print_img_predict_score.py

'''
输出图片预测后的得分，每张图片1个得分
规则为：无任何结果、结果错误score得分为0，其他按正确框最高分

'''

import numpy as np
import tensorflow as tf
import multi_detection.core.utils as utils
from multi_detection.food_correct_utils import correct_bboxes
import time
import xlwt
import cv2
from tqdm import tqdm

import os

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.9  # 占用GPU的显存

def he_foods(pre):
    '''
    针对合并的类别判断输出是否在合并类别内
    :param pre:
    :return:
    '''
    if pre in [8, 9] and classes_id[classes[i]] in [8, 9]:  # 合并蛋挞
        rigth_label = True
    elif pre in [12, 14] and classes_id[classes[i]] in [12, 14]:  # 合并四分之一披萨、六分之一披萨
        rigth_label = True
    elif pre in [18, 19] and classes_id[classes[i]] in [18, 19]:  # 合并中土豆、大土豆
        rigth_label = True
    elif pre in [22, 23] and classes_id[classes[i]] in [22, 23]:  # 合并中红薯、大红薯
        rigth_label = True
    else:
        rigth_label = False
    return rigth_label

class YoloTest(object):
    def __init__(self):
        self.input_size = 416  # 输入图片尺寸（默认正方形）
        self.num_classes = 30  # 种类数
        self.score_threshold = 0.1
        self.iou_threshold = 0.5
        self.weight_file = "E:/ckpt_dirs/Food_detection/local/20191216/yolov3_train_loss=4.7698.ckpt-80"  # ckpt文件地址
        # self.weight_file = "./checkpoint/yolov3_train_loss=6.2933.ckpt-36"
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
        '''
        预测结果
        :param image: 图片数据，shape为[800,600,3]
        :return:
            bboxes：食材检测预测框结果，格式为：[x_min, y_min, x_max, y_max, probability, cls_id],
            layer_n[0]：烤层检测结果，0：最下层、1：中间层、2：最上层、3：其他
        '''
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

        return bboxes, layer_n[0]


if __name__ == '__main__':
    start_time = time.time()
    Y = YoloTest()  # 加载模型
    end0_time = time.time()
    print("model loading time:", end0_time - start_time)

    classes = ["Beefsteak", "CartoonCookies", "Cookies", "CupCake", "Pizzafour",
               "Pizzatwo", "Pizzaone", "Pizzasix", "ChickenWings", "ChiffonCake6",
               "ChiffonCake8", "CranberryCookies", "eggtarts", "eggtartl", "nofood",
               "Peanuts", "PorkChops", "PotatoCut", "Potatol", "Potatom",
               "Potatos", "RoastedChicken", "SweetPotatoCut", "SweetPotatol", "SweetPotatom",
               "SweetPotatoS", "Toast"]
    # classes = ["CartoonCookies"]
    #
    classes_id = {"CartoonCookies": 1, "Cookies": 5, "CupCake": 7, "Beefsteak": 0, "ChickenWings": 2,
                  "ChiffonCake6": 3, "ChiffonCake8": 4, "CranberryCookies": 6, "eggtarts": 8, "eggtartl": 9,
                  "nofood": 10, "Peanuts": 11, "PorkChops": 16, "PotatoCut": 17, "Potatol": 18,
                  "Potatom": 19, "Potatos": 20, "SweetPotatoCut": 21, "SweetPotatol": 22, "SweetPotatom": 23,
                  "Pizzafour": 12, "Pizzaone": 13, "Pizzasix": 14, "RoastedChicken": 25,
                  "Pizzatwo": 15, "SweetPotatoS": 24, "Toast": 26, "sweetpotato_others": 27, "pizza_others": 28,
                  "potato_others": 29 }
    new_classes = {v: k for k, v in classes_id.items()}

    jpgs_count_all = 0
    layer_jpgs_acc = 0
    food_jpgs_acc = 0

    jpg_root_dir = "E:/check_2_phase/JPGImages"
    workbook = xlwt.Workbook(encoding='utf-8')
    sheet1 = workbook.add_sheet("test score")
    for i in tqdm(range(len(classes))):
        c = classes[i].lower()
        sheet1.write(0, i, "{}_score".format(c))
        jpg_i = 0
        for l_num, l in enumerate(["bottom", "middle", "top", "others"]):
            for jpg_num, jpg in enumerate(os.listdir(jpg_root_dir + "/" + c + "/" + l)):
                if jpg.endswith(".jpg"):
                    jpg_i += 1
                    image_path = jpg_root_dir + "/" + c + "/" + l + "/" + jpg  # 图片地址
                    image = cv2.imread(image_path)  # 图片读取
                    bboxes_pr, layer_n = Y.predict(image)  # 预测每一张结果并保存
                    # bboxes_pr, layer_n = correct_bboxes(bboxes_pr, layer_n)  # 矫正输出结果
                    if len(bboxes_pr) == 0:  # 无任何输出结果，score为0
                        score = 0
                    else:
                        scores_b = [0]
                        for b in bboxes_pr:
                            if b[-1] == classes_id[classes[i]]:  # 若有正确结果，添加score得分
                                scores_b.append(b[-2])
                        score = max(scores_b)
                    sheet1.write(jpg_i + 1, i, score)
    workbook.save(jpg_root_dir+"/all_he_score.xls")