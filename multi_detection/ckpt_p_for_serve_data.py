# -*- coding: utf-8 -*-
# @Time    : 2021/3/25
# @Author  : sunyihuan
# @File    : ckpt_p_for_serve_data.py
'''
将服务端数据，按单类输出预测结果
并分为：高置信度、低置信度，无结果

'''

import cv2
import numpy as np
import tensorflow as tf
import multi_detection.core.utils as utils
from multi_detection.food_correct_utils import correct_bboxes, get_potatoml
import os
from tqdm import tqdm

id_2_name = {0: "牛排", 1: "卡通饼干", 2: "鸡翅", 3: "戚风蛋糕", 4: "戚风蛋糕", 5: "曲奇饼干"
    , 6: "蔓越莓饼干", 7: "纸杯蛋糕", 8: "蛋挞", 9: "空", 10: "花生米"
    , 11: "披萨", 12: "披萨", 13: "披萨", 14: "排骨", 15: "土豆切"
    , 16: "大土豆", 17: "小土豆", 18: "红薯切", 19: "大红薯", 20: "小红薯"
    , 21: "烤鸡", 22: "吐司", 23: "板栗", 24: "玉米", 25: "玉米"
    , 26: "鸡腿", 27: "芋头", 28: "小馒头", 29: "整个茄子", 30: "切开茄子"
    , 31: "吐司面包", 32: "餐具", 33: "餐具", 34: "鱼", 35: "热狗"
    , 36: "虾", 37: "虾", 38: "烤肉串", 39: "锡纸", 101: "戚风蛋糕"
    , 40: "大土豆", 41: "大红薯"}

classes_id39 = {"cartooncookies": 1, "cookies": 5, "cupcake": 7, "beefsteak": 0, "chickenwings": 2,
                "chiffoncake6": 3, "chiffoncake8": 4, "cranberrycookies": 6, "eggtart": 8,
                "nofood": 9, "peanuts": 10, "porkchops": 14, "potatocut": 15, "potatol": 16,
                "potatos": 17, "sweetpotatocut": 18, "sweetpotatol": 19,
                "pizzacut": 11, "pizzaone": 12, "roastedchicken": 21,
                "pizzatwo": 13, "sweetpotatos": 20, "toast": 22, "chestnut": 23, "cornone": 24, "corntwo": 25,
                "drumsticks": 26,
                "taro": 27, "steamedbread": 28, "eggplant": 29, "eggplant_cut_sauce": 30, "bread": 31,
                "container_nonhigh": 32,
                "container": 33, "duck": 21, "fish": 34, "hotdog": 35, "redshrimp": 36,
                "shrimp": 37, "strand": 38, "xizhi": 39, "chiffon_4": 101, "potatom": 40, "sweetpotatom": 41}


def he_foods(pre):
    '''
    针对合并的类别判断输出是否在合并类别内
    :param pre:
    :return:
    '''
    if pre in [3, 4, 101] and classes_id39[c] in [3, 4, 101]:  # 合并戚风
        rigth_label = True
    # if pre in [3, 4, 6] and classes_id39[c] in [3, 4, 6]:  # 合并虾
    #     rigth_label = True
    # elif pre in [10 + 1, 11 + 1, 12 + 1] and classes_id39[c] in [10 + 1, 11 + 1, 12 + 1]:  # 合并披萨
    #     rigth_label = True
    # elif pre in [14 + 1, 15 + 1, 16 + 1] and classes_id39[c] in [14 + 1, 15 + 1, 16 + 1]:  # 合并土豆、土豆
    #     rigth_label = True
    # elif pre in [17 + 1, 18 + 1, 19 + 1] and classes_id39[c] in [17 + 1, 18 + 1, 19 + 1]:  # 合并红薯
    #     rigth_label = True
    # elif pre in [1, 6] and classes_id39[c] in [1, 6]:  # 合并卡通饼干、蔓越莓饼干
    #     rigth_label = True
    # elif pre in [24, 25] and classes_id39[c] in [25, 24]:  # 合并玉米
    #     rigth_label = True
    elif pre in [32, 33] and classes_id39[c] in [32, 33]:  # 合并器皿
        rigth_label = True
    elif pre in [36, 37] and classes_id39[c] in [36, 37]:  # 合并虾
        rigth_label = True
    else:
        rigth_label = False
    # rigth_label = False
    return rigth_label


class YoloPredic(object):
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
        self.pb_file = "E:/模型交付版本/multi_yolov3_predict-20210129/checkpoint/yolov3_0129.pb"  # pb文件地址

        graph = tf.Graph()
        with graph.as_default():
            output_graph_def = tf.GraphDef()
            with open(self.pb_file, "rb") as f:
                output_graph_def.ParseFromString(f.read())
                _ = tf.import_graph_def(output_graph_def, name="")

            self.sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True))

            self.input = self.sess.graph.get_tensor_by_name("define_input/input_data:0")
            self.trainable = self.sess.graph.get_tensor_by_name("define_input/training:0")

            self.pred_sbbox = self.sess.graph.get_tensor_by_name("define_loss/pred_sbbox/concat_2:0")
            self.pred_mbbox = self.sess.graph.get_tensor_by_name("define_loss/pred_mbbox/concat_2:0")
            self.pred_lbbox = self.sess.graph.get_tensor_by_name("define_loss/pred_lbbox/concat_2:0")

            self.layer_num = self.sess.graph.get_tensor_by_name("define_loss/layer_classes:0")

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
        bboxes = utils.postprocess_boxes(pred_bbox, (org_h, org_w), self.input_size, self.score_threshold)
        bboxes = utils.nms(bboxes, self.iou_threshold)
        layer_n = layer_[0]  # 烤层结果

        return bboxes, layer_n, best_bboxes


if __name__ == '__main__':
    img_root = "F:/serve_data/JPGImages"  # 图片地址
    save_root = "F:/serve_data/JPGImages_detetction"
    if not os.path.exists(save_root): os.mkdir(save_root)
    import time

    start_time = time.time()
    Y = YoloPredic()
    end_time0 = time.time()
    new_classes = {v: k for k, v in classes_id39.items()}
    print("加载时间：", end_time0 - start_time)
    cls_list = os.listdir(img_root)
    nore = 0
    error_nu = 0
    rig_0 = 0
    for c in tqdm(cls_list):
        img_dir = img_root + "/" + c
        save_dir = save_root + "/" + c
        if not os.path.exists(save_dir): os.mkdir(save_dir)
        for img in tqdm(os.listdir(img_dir)):
            img_path = img_dir + "/" + img
            image = cv2.imread(img_path)  # 图片读取
            bboxes, layer_n, best_bboxes = Y.predict(image)
            bboxes, layer_n, best_bboxes = correct_bboxes(bboxes, layer_n, best_bboxes)  # 矫正输出结果
            bboxes, layer_n = get_potatoml(bboxes, layer_n)  # 根据输出结果对中大红薯，中大土豆做输出
            he = he_foods(bboxes)
            # print(img, bboxes)

            image_detect = utils.draw_bbox(image, bboxes, show_label=True)
            if len(bboxes) > 0:
                cls = int(bboxes[0][-1])
                score = round(bboxes[0][-2], 2)
                cls_name = new_classes[int(cls)]

                print(cls, classes_id39[c], he)

                save_dir_cls = save_dir + "/{}".format(cls_name)
                if not os.path.exists(save_dir_cls): os.mkdir(save_dir_cls)

                if score >= 0.8:
                    if cls != classes_id39[c]:
                        if he:
                            rig_0 += 1
                        else:
                            error_nu += 1
                    else:
                        rig_0 += 1
                else:
                    nore += 1
            else:
                nore += 1
    print(error_nu, rig_0, nore)
