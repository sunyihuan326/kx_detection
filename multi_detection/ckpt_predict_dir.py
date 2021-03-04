# -*- encoding: utf-8 -*-

"""
预测一个文件夹图片结果
@File    : ckpt_predict.py
@Time    : 2019/12/16 15:45
@Author  : sunyihuan
"""

import cv2
import numpy as np
import tensorflow as tf
import multi_detection.core.utils as utils
import os
import time
from multi_detection.food_correct_utils import correct_bboxes, get_potatoml
import shutil
from tqdm import tqdm

# gpu限制
config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.4


class YoloPredict(object):
    '''
    预测结果
    '''

    def __init__(self):
        self.input_size = 416  # 输入图片尺寸（默认正方形）
        self.num_classes = 40  # 种类数
        self.score_cls_threshold = 0.001
        self.score_threshold = 0.8
        self.iou_threshold = 0.5
        self.top_n = 5
        self.weight_file = "E:/ckpt_dirs/Food_detection/multi_food5/20210224/yolov3_train_loss=5.9418.ckpt-148" # ckpt文件地址
        # self.weight_file = "./checkpoint/yolov3_train_loss=4.7681.ckpt-80"
        self.write_image = False  # 是否画图
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
        best_bboxes = self.get_top_cls(pred_bbox, org_h, org_w, self.top_n)  # 获取top_n类别和置信度
        bboxes = utils.postprocess_boxes(pred_bbox, (org_h, org_w), self.input_size, self.score_threshold)
        bboxes = utils.nms(bboxes, self.iou_threshold)

        return bboxes, layer_n[0], best_bboxes

    def result(self, image_path, save_dir):
        '''
        得出预测结果并保存
        :param image_path: 图片地址
        :param save_dir: 预测结果原图标注框，保存地址
        :return:
        '''
        image = cv2.imread(image_path)  # 图片读取
        # image = utils.white_balance(image)  # 图片白平衡处理
        bboxes_pr, layer_n, best_bboxes = self.predict(image)  # 预测结果


        if self.write_image:
            image = utils.draw_bbox(image, bboxes_pr, show_label=self.show_label)
            drawed_img_save_to_path = str(image_path).split("/")[-1]
            drawed_img_save_to_path = str(drawed_img_save_to_path).split(".")[0] + "_" + str(
                layer_n) + ".jpg"  # 图片保存地址，烤层结果在命名中
            # cv2.imshow('Detection result', image)
            cv2.imwrite(save_dir + "/" + drawed_img_save_to_path, image)  # 保存图片
        return bboxes_pr, layer_n, best_bboxes


if __name__ == '__main__':
    start_time = time.time()

    img_root = "F:/serve_data/OVEN/nofood148"  # 图片文件地址

    layer_data_root = "F:/serve_data/OVEN/nofood148_layer_data"
    if not os.path.exists(layer_data_root): os.mkdir(layer_data_root)
    save_root = "F:/serve_data/OVEN/nofood148_detection"
    if not os.path.exists(save_root): os.mkdir(save_root)
    Y = YoloPredict()
    end_time0 = time.time()
    print("model loading time:", end_time0 - start_time)
    # cls = ["beefsteak", "cartooncookies", "chickenwings", "chiffoncake6", "chiffoncake8",
    #        "cookies", "cranberrycookies", "cupcake", "eggtart", "peanuts",
    #        "pizzacut", "pizzaone", "pizzatwo", "porkchops", "potatocut", "potatol",
    #        "potatos", "sweetpotatocut", "sweetpotatol", "sweetpotatos",
    #        "roastedchicken", "toast", "chestnut", "cornone", "corntwo", "drumsticks", "taro",
    #        "steamedbread", "eggplant", "eggplant_cut_sauce", "bread", "container_nonhigh",
    #        "container", "fish", "hotdog", "redshrimp",
    #        "shrimp", "strand"]
    # cls = ["cornone", "eggplant", "fish", "nofood", "potatol", "roastedchicken", "shrimp", "toast"]
    # cls = ["container",  "fish", "nofood", "roastedchicken", "shrimp", "toast"]
    # cls = os.listdir(img_root)

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

    cls=[""]
    new_classes = {v: k for k, v in classes_id39.items()}
    layer_id = {0: "bottom", 1: "middle", 2: "top", 3: "others"}
    for c in cls:
        img_dir = img_root + "/" + c
        save_dir = save_root + "/" + c
        layer_data_dir=layer_data_root+"/"+c
        if not os.path.exists(layer_data_dir): os.mkdir(layer_data_dir)
        for img in tqdm(os.listdir(img_dir)):
            if img.endswith("jpg"):
                img_path = img_dir + "/" + img
                end_time1 = time.time()
                bboxes_p, layer_, best_bboxes = Y.result(img_path, save_dir)
                bboxes_pr, layer_n, best_bboxes = correct_bboxes(bboxes_p, layer_, best_bboxes)  # 矫正输出结果
                bboxes_pr, layer_n = get_potatoml(bboxes_pr, layer_n)  # 根据输出结果对中大红薯，中大土豆做输出

                print(bboxes_pr)
                print(layer_n)
                # 烤层分到对应文件夹
                if not os.path.exists(layer_data_dir + "/" + layer_id[layer_n]): os.mkdir(
                    layer_data_dir + "/" + layer_id[layer_n])
                shutil.copy(img_path, layer_data_dir + "/" + layer_id[layer_n] + "/"+img)

                # 食材分到对应文件夹
                if len(bboxes_pr) == 0:
                    if not os.path.exists(img_dir + "/noresult"): os.mkdir(img_dir + "/noresult")
                    shutil.move(img_path, img_dir + "/noresult" + "/" + img)
                else:
                    pre = int(bboxes_pr[0][-1])
                    if not os.path.exists(img_dir + "/" + new_classes[pre]): os.mkdir(img_dir + "/" + new_classes[pre])
                    shutil.move(img_path, img_dir + "/" + new_classes[pre] + "/" + img)

    end_time1 = time.time()
    print("all data time:", end_time1 - end_time0)
