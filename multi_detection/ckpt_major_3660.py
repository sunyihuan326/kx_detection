# -*- coding: utf-8 -*-
# @Time    : 2020/7/24
# @Author  : sunyihuan
# @File    : ckpt_major_3660.py
'''
数据未分层
数据格式为：
   dataroot
       xxxx
       xxxx
       xxxx

输出：
    食材准确率、topn准确率、大类准确率等
'''
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
gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.2)
config = tf.ConfigProto(gpu_options=gpu_options)


def he_foods(pre):
    '''
    针对合并的类别判断输出是否在合并类别内
    :param pre:
    :return:
    '''
    # if pre in [3, 4, 6] and classes_id39[c] in [3, 4, 6]:  # 合并戚风，纸杯蛋糕
    #     rigth_label = True
    # elif pre in [10 + 1, 11 + 1, 12 + 1] and classes_id39[c] in [10 + 1, 11 + 1, 12 + 1]:  # 合并披萨
    #     rigth_label = True
    # elif pre in [14 + 1, 15 + 1, 16 + 1] and classes_id39[c] in [14 + 1, 15 + 1, 16 + 1]:  # 合并土豆、土豆
    #     rigth_label = True
    # elif pre in [17 + 1, 18 + 1, 19 + 1] and classes_id39[c] in [17 + 1, 18 + 1, 19 + 1]:  # 合并红薯
    #     rigth_label = True
    # elif pre in [1, 4 + 1, 5 + 1] and classes_id39[c] in [1, 4, 5]:  # 合并饼干
    #     rigth_label = True
    # elif pre in [24, 25] and classes_id39[c] in [25, 24]:  # 合并玉米
    #     rigth_label = True
    # elif pre in [32, 33] and classes_id39[c] in [32, 33]:  # 合并器皿
    #     rigth_label = True
    # elif pre in [36, 37] and classes_id39[c] in [36, 37]:  # 合并虾
    #     rigth_label = True
    # else:
    #     rigth_label = False
    rigth_label = False
    return rigth_label


class YoloPredict(object):
    def __init__(self):
        self.input_size = 416  # 输入图片尺寸（默认正方形）
        self.num_classes = 40  # 种类数
        self.top_n = 10
        self.score_cls_threshold = 0.0000001
        self.score_threshold = 0.45
        self.iou_threshold = 0.5
        self.weight_file = "E:/ckpt_dirs/Food_detection/multi_food5/20200914/yolov3_train_loss=6.9178.ckpt-95"   # ckpt文件地址
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

        return org_h, org_w, pred_bbox, bboxes, layer_n

    def result(self, image_path, save_dir):
        '''
        得出预测结果并保存
        :param image_path: 图片地址
        :param save_dir: 预测结果原图标注框，保存地址
        :return:
        '''
        image = cv2.imread(image_path)  # 图片读取
        # image = utils.white_balance(image)  # 图片白平衡处理
        org_h, org_w, pred_bbox, bboxes, layer_n = self.predict(image)  # 预测结果
        # print(bboxes_pr)
        # print(layer_n)

        if self.write_image:
            image = utils.draw_bbox(image, bboxes, show_label=self.show_label)
            drawed_img_save_to_path = str(image_path).split("/")[-1]
            drawed_img_save_to_path = str(drawed_img_save_to_path).split(".")[0] + "_" + str(
                layer_n) + ".jpg"  # 图片保存地址，烤层结果在命名中
            # cv2.imshow('Detection result', image)
            cv2.imwrite(save_dir + "/" + drawed_img_save_to_path, image)  # 保存图片
        return org_h, org_w, pred_bbox, bboxes, layer_n


if __name__ == '__main__':
    start_time = time.time()
    img_root = "E:/WLS_originalData/all_test_data/all_original_data"  # 图片文件地址
    save_root = "E:/WLS_originalData/all_test_data/all_original_data_0914_detection5"
    img_error_root = "E:/WLS_originalData/all_test_data/all_original_data_0914_error5"
    img_error_detection_root = "E:/WLS_originalData/all_test_data/all_original_data_0914_error_detect5"
    img_noresult_root = "E:/WLS_originalData/all_test_data/all_original_data_0914_noresult5"
    if not os.path.exists(save_root): os.mkdir(save_root)
    if not os.path.exists(img_error_root): os.mkdir(img_error_root)
    if not os.path.exists(img_noresult_root): os.mkdir(img_noresult_root)
    if not os.path.exists(img_error_detection_root): os.mkdir(img_error_detection_root)
    Y = YoloPredict()
    end_time0 = time.time()
    print("model loading time:", end_time0 - start_time)
    clses = ["beefsteak", "cartooncookies", "chickenwings", "chiffoncake6", "chiffoncake8",
             "cookies", "cranberrycookies", "cupcake", "eggtart", "nofood", "peanuts",
             "pizzacut", "pizzaone", "pizzatwo", "porkchops", "potatocut",
             "potatol", "potatos", "sweetpotatocut", "sweetpotatol", "sweetpotatos",
             "roastedchicken", "toast", "chestnut", "cornone", "corntwo",
             "drumsticks", "taro", "steamedbread", "eggplant", "eggplant_cut_sauce",
             "bread", "container_nonhigh", "container", "fish", "hotdog",
             "redshrimp", "shrimp", "strand", "duck"]

    classes_id39 = {"cartooncookies": 1, "cookies": 5, "cupcake": 7, "beefsteak": 0, "chickenwings": 2,
                    "chiffoncake6": 3, "chiffoncake8": 4, "cranberrycookies": 6, "eggtart": 8,
                    "nofood": 9, "peanuts": 10, "porkchops": 14, "potatocut": 15, "potatol": 16,
                    "potatos": 17, "sweetpotatocut": 18, "sweetpotatol": 19, "pizzacut": 11, "pizzaone": 12,
                    "roastedchicken": 21,
                    "pizzatwo": 13, "sweetpotatos": 20, "toast": 22, "chestnut": 23, "cornone": 24, "corntwo": 25,
                    "drumsticks": 26,
                    "taro": 27, "steamedbread": 28, "eggplant": 29, "eggplant_cut_sauce": 30, "bread": 31,
                    "container_nonhigh": 32,
                    "container": 33, "duck": 21, "fish": 34, "hotdog": 35, "redshrimp": 36,
                    "shrimp": 37, "strand": 38,"xizhi":39}

    all_jpg = 0
    acc_jpg = 0
    noresults = 0
    food_top3_acc_nums = 0  # top3正确数
    food_top5_acc_nums = 0  # top5正确数
    food_top8_acc_nums = 0  # top8正确数
    food_top10_acc_nums = 0  # top10正确数
    food_acc_major = 0  # 大类正确数
    for c in clses:
        if c not in ["nofood", "sweetpotatom", "potatom"]:
            img_dir = img_root + "/" + c
            save_dir = save_root + "/" + c
            img_error_dir = img_error_root + "/" + c
            img_noresult_dir = img_noresult_root + "/" + c
            img_error_detect_dir = img_error_detection_root + "/" + c
            if not os.path.exists(save_dir): os.mkdir(save_dir)
            if not os.path.exists(img_error_dir): os.mkdir(img_error_dir)
            if not os.path.exists(img_noresult_dir): os.mkdir(img_noresult_dir)
            if not os.path.exists(img_error_detect_dir): os.mkdir(img_error_detect_dir)
            for img in tqdm(os.listdir(img_dir)):
                if img.endswith("jpg"):
                    all_jpg += 1
                    img_path = img_dir + "/" + img

                    end_time1 = time.time()
                    try:
                        org_h, org_w, pred_bbox, bboxes, layer_n = Y.result(img_path, save_dir)
                        best_bboxes_3 = Y.get_top_cls(pred_bbox, org_h, org_w, 3)  # 获取top_n类别和置信度
                        best_bboxes_5 = Y.get_top_cls(pred_bbox, org_h, org_w, 5)  # 获取top_n类别和置信度
                        best_bboxes_8 = Y.get_top_cls(pred_bbox, org_h, org_w, 9)  # 获取top_n类别和置信度
                        best_bboxes_10 = Y.get_top_cls(pred_bbox, org_h, org_w, 10)  # 获取top_n类别和置信度

                        if len(bboxes) == 0:
                            noresults += 1
                            shutil.copy(img_path, img_noresult_dir + "/" + img)
                        else:
                            bboxes_pr, layer_n = correct_bboxes(bboxes, layer_n)  # 矫正输出结果
                            # bboxes_pr, layer_n = get_potatoml(bboxes_pr, layer_n)  # 根据输出结果对中大红薯，中大土豆做输出
                            if len(bboxes_pr) == 0:
                                noresults += 1
                                # if not os.path.exists(img_dir + "/noresult"): os.mkdir(img_dir + "/noresult")
                                # shutil.move(img_path, img_dir + "/noresult" + "/" + img)
                            else:
                                pre = int(bboxes_pr[0][-1])
                                # print(pre)
                                # print(clses[pre])
                                # print(classes_id22[clses[pre]])

                                if classes_id39[c] in dict(best_bboxes_3).keys():
                                    # food_topn_acc_b+=1
                                    food_top3_acc_nums += 1
                                if classes_id39[c] in dict(best_bboxes_5).keys():
                                    food_top5_acc_nums += 1
                                if classes_id39[c] in dict(best_bboxes_8).keys():
                                    food_top8_acc_nums += 1
                                if classes_id39[c] in dict(best_bboxes_10).keys():
                                    food_top10_acc_nums += 1
                                else:
                                    print(classes_id39[c], dict(best_bboxes_3).keys())

                                if pre == int(classes_id39[c]):
                                    acc_jpg += 1
                                    food_acc_major += 1
                                else:
                                    shutil.copy(img_path, img_error_dir + "/" + img)
                                    right_label = he_foods(pre)
                                    if right_label:  # 合并后结果正确
                                        food_acc_major += 1
                                    else:
                                        # 图片保存地址，烤层结果在命名中
                                        drawed_img_save_to_path = str(img).split(".")[0] + "_" + str(
                                            layer_n) + ".jpg"
                                        shutil.copy(save_dir + "/" + drawed_img_save_to_path,
                                                    img_error_detect_dir + "/" + drawed_img_save_to_path)

                                    # if not os.path.exists(img_dir + "/" + str(clses[pre])): os.mkdir(
                                    #     img_dir + "/" + str(clses[pre]))
                                    # shutil.move(img_path, img_dir + "/" + str(clses[pre]) + "/" + img)
                    except:
                        print(img_path)
    print("正确数：", acc_jpg)
    print("无任何结果数：", noresults)
    print("top3正确数结果数：", food_top3_acc_nums)
    print("top5正确数结果数：", food_top5_acc_nums)
    print("top8正确数结果数：", food_top8_acc_nums)
    print("top10正确数结果数：", food_top10_acc_nums)
    print("大类正确结果数：", food_acc_major)
    print("总数：", all_jpg)
    print("正确率：：：", acc_jpg / all_jpg)
    print("top3正确率：：：：", food_top3_acc_nums / all_jpg)
    print("top5正确率：：：：", food_top5_acc_nums / all_jpg)
    print("top8正确率：：：：", food_top8_acc_nums / all_jpg)
    print("top10正确率：：：：", food_top10_acc_nums / all_jpg)
    print("大类正确率：：：：", food_acc_major / all_jpg)
    print("无任何结果占比：：：：", noresults / all_jpg)
