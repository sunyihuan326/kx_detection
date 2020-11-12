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
    if pre in [3, 4, 42] and classes_id[c] in [3, 4, 42]:  # 合并戚风
        rigth_label = True
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
    elif pre in [32, 33] and classes_id[c] in [32, 33]:  # 合并器皿
        rigth_label = True
    elif pre in [36, 37] and classes_id[c] in [36, 37]:  # 合并虾
        rigth_label = True
    else:
        rigth_label = False
    # rigth_label = False
    return rigth_label


class YoloTest(object):
    def __init__(self):
        self.input_size = 416  # 输入图片尺寸（默认正方形）
        self.num_classes = 40  # 种类数
        self.score_cls_threshold = 0.001
        self.score_threshold = 0.4
        self.iou_threshold = 0.5
        self.top_n = 5
        self.weight_file = "E:/ckpt_dirs/Food_detection/multi_food5/20201111/yolov3_train_loss=6.4953.ckpt-112"  # ckpt文件地址
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

    classes = ["beefsteak", "cartooncookies", "chickenwings", "chiffoncake6", "chiffoncake8",
               "cookies", "cranberrycookies", "cupcake", "eggtart", "nofood",
               "peanuts", "pizzacut", "pizzaone", "pizzatwo", "porkchops",
               "potatocut", "potatol", "potatos", "roastedchicken", "sweetpotatocut",
               "sweetpotatol", "sweetpotatos", "toast", "chestnut", "cornone",
               "corntwo", "drumsticks", "taro", "steamedbread", "eggplant",
               "eggplant_cut_sauce", "bread", "container", "duck", "fish",
               "hotdog", "shrimp", "strand"]
    # classes=["strand"]
    #
    classes_id = {"cartooncookies": 1, "cookies": 5, "cupcake": 7, "beefsteak": 0, "chickenwings": 2,
                  "chiffoncake6": 3, "chiffoncake8": 4, "cranberrycookies": 6, "eggtart": 8,
                  "nofood": 9, "peanuts": 10, "porkchops": 14, "potatocut": 15, "potatol": 16,
                  "potatom": 16, "potatos": 17, "sweetpotatocut": 18, "sweetpotatol": 19,
                  "pizzacut": 11, "pizzaone": 12, "roastedchicken": 21,
                  "pizzatwo": 13, "sweetpotatos": 20, "toast": 22, "chestnut": 23, "cornone": 24, "corntwo": 25,
                  "drumsticks": 26,
                  "taro": 27, "steamedbread": 28, "eggplant": 29, "eggplant_cut_sauce": 30, "bread": 31,
                  "container_nonhigh": 32, "container": 33, "duck": 21, "fish": 34, "hotdog": 35, "redshrimp": 36,
                  "shrimp": 37, "strand": 38, "xizhi": 39, "small_fish": 40, "chiffon4": 42}
    layer_id = {"bottom": 0, "middle": 1, "top": 2, "others": 3}
    new_classes = {v: k for k, v in classes_id.items()}

    jpgs_count_all = 0
    layer_jpgs_acc = 0
    food_jpgs_acc = 0

    jpg_root_dir = "F:/test_from_yejing_202010/TXKX_all_20201019_rename_all"
    workbook = xlwt.Workbook(encoding='utf-8')
    sheet1 = workbook.add_sheet("test score")
    sheet1.write(0, 0, "jpg_name")
    sheet1.write(0, 1, "food_true_cls")
    sheet1.write(0, 2, "layer_true_cls")
    sheet1.write(0, 3, "food_pre_cls")
    sheet1.write(0, 4, "layer_pre_cls")
    sheet1.write(0, 5, "score")
    jpg_i = 0
    for i in tqdm(range(len(classes))):
        c = classes[i].lower()
        # sheet1.write(0, i, "{}_score".format(c))
        # if c not in ["nofood", "potatom", "sweetpotatom"]:
        #     for jpg_num, jpg in enumerate(os.listdir(jpg_root_dir + "/" + c)):
        #         if jpg.endswith(".jpg"):
        #             jpg_i += 1
        #             image_path = jpg_root_dir + "/" + c + "/" + jpg  # 图片地址
        #             image = cv2.imread(image_path)  # 图片读取
        #             bboxes_pr, layer_n = Y.predict(image)  # 预测每一张结果并保存
        #             bboxes_pr, layer_n = correct_bboxes(bboxes_pr, layer_n)  # 矫正输出结果
        #             if len(bboxes_pr) == 0:  # 无任何输出结果，score为0
        #                 score = 0
        #             else:
        #                 scores_b = [0]
        #                 for b in bboxes_pr:
        #                     if b[-1] == classes_id[classes[i]]:  # 若有正确结果，添加score得分
        #                         scores_b.append(b[-2])
        #                 score = max(scores_b)
        #             sheet1.write(jpg_i + 1, i, score)
        for l_num, l in enumerate(["bottom", "middle", "top", "others"]):  # 有分层数据
            for jpg_num, jpg in enumerate(os.listdir(jpg_root_dir + "/" + c + "/" + l)):
                if jpg.endswith(".jpg"):
                    jpg_i += 1
                    image_path = jpg_root_dir + "/" + c + "/" + l + "/" + jpg  # 图片地址
                    image = cv2.imread(image_path)  # 图片读取
                    bboxes_pr, layer_n = Y.predict(image)  # 预测每一张结果并保存
                    bboxes_pr, layer_n = correct_bboxes(bboxes_pr, layer_n)  # 矫正输出结果
                    if len(bboxes_pr) == 0:  # 无任何输出结果，score为0
                        score = 0
                        pre = ""
                    else:
                        scores_b = [0]
                        for b in bboxes_pr:
                            if b[-1] == classes_id[classes[i]]:  # 若有正确结果，添加score得分
                                scores_b.append(b[-2])
                            else:
                                right_label = he_foods(b[-1])
                                scores_b.append(b[-2])
                        score = max(scores_b)
                        pre = bboxes_pr[0][-1]
                    sheet1.write(jpg_i + 1, 0, jpg)
                    sheet1.write(jpg_i + 1, 1, classes_id[c])
                    sheet1.write(jpg_i + 1, 2, l_num)
                    sheet1.write(jpg_i + 1, 3, pre)
                    sheet1.write(jpg_i + 1, 4, str(layer_n))
                    sheet1.write(jpg_i + 1, 5, score)
    workbook.save(jpg_root_dir + "/all_he_score0914.xls")
