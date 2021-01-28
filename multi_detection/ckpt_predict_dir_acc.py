# -*- encoding: utf-8 -*-

"""
预测一个根文件下所有文件夹结果
并打印准确率

@File    : ckpt_predict.py
@Time    : 2020/7/13 11:28
@Author  : sunyihuan

修改于：2021/1/25 11:28
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
gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.2)
config = tf.ConfigProto(gpu_options=gpu_options)


class YoloPredict(object):
    '''
    预测结果
    '''

    def __init__(self):
        self.input_size = 416  # 输入图片尺寸（默认正方形）
        self.num_classes = 40  # 种类数
        self.score_cls_threshold = 0.001
        self.score_threshold = 0.3
        self.iou_threshold = 0.5
        self.top_n = 5
        self.weight_file = "E:/ckpt_dirs/Food_detection/multi_food5/20201123/yolov3_train_loss=6.5091.ckpt-128"  # ckpt文件地址
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
        bboxes = utils.postprocess_boxes_conf(pred_bbox, (org_h, org_w), self.input_size, self.score_cls_threshold)
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

    def result(self, image_path):
        '''
        得出预测结果并保存
        :param image_path: 图片地址
        :param save_dir: 预测结果原图标注框，保存地址
        :return:
        '''
        image = cv2.imread(image_path)  # 图片读取
        # image = utils.white_balance(image)  # 图片白平衡处理
        bboxes_pr, layer_n, best_bboxes = self.predict(image)  # 预测结果
        # print(bboxes_pr)
        # print(layer_n)
        return bboxes_pr, layer_n, best_bboxes


if __name__ == '__main__':
    start_time = time.time()
    img_root = "F:/serve_data/OVEN/for_test"  # 图片文件地址
    save_root = "F:/serve_data/OVEN/for_test_detection"
    model_tag = "multi5_1123"
    save_dir = save_root + "/" + model_tag
    if not os.path.exists(save_dir): os.mkdir(save_dir)
    Y = YoloPredict()
    end_time0 = time.time()
    print("model loading time:", end_time0 - start_time)

    classes_id40 = {"cartooncookies": 1, "cookies": 5, "cupcake": 7, "beefsteak": 0, "chickenwings": 2,
                    "chiffoncake6": 3, "chiffoncake8": 4, "cranberrycookies": 6, "eggtart": 8,
                    "nofood": 9, "peanuts": 10, "porkchops": 14, "potatocut": 15, "potatol": 16,
                    "potatom": 16, "potatos": 17, "sweetpotatocut": 18, "sweetpotatol": 19,
                    "pizzacut": 11, "pizzaone": 12, "roastedchicken": 21,
                    "pizzatwo": 13, "sweetpotatos": 20, "toast": 22, "chestnut": 23, "cornone": 24, "corntwo": 25,
                    "drumsticks": 26,
                    "taro": 27, "steamedbread": 28, "eggplant": 29, "eggplant_cut_sauce": 30, "bread": 31,
                    "container_nonhigh": 32, "container": 33, "duck": 21, "fish": 34, "hotdog": 35, "redshrimp": 36,
                    "shrimp": 37, "strand": 38, "xizhi": 39, "ptatom": 40, "sweetpotatom": 41, "chiffon_size4": 101}

    all_jpg = 0
    acc_80_jpg = 0
    acc_60_jpg = 0
    acc_30_jpg = 0
    acc_00_jpg = 0
    error_jpg = 0
    clses = os.listdir(img_root)
    for c in tqdm(clses):
        if c not in ["hu", "others"]:
            img_dir = img_root + "/" + c
            for img in os.listdir(img_dir):
                if img.endswith("jpg"):
                    all_jpg += 1
                    img_path = img_dir + "/" + img
                    end_time1 = time.time()
                    bboxes_p, layer_, best_bboxes = Y.result(img_path)
                    bboxes_pr, layer_n, best_bboxes = correct_bboxes(bboxes_p, layer_, best_bboxes)  # 矫正输出结果
                    bboxes_pr, layer_n = get_potatoml(bboxes_pr, layer_n)  # 根据输出结果对中大红薯，中大土豆做输出

                    # 图片保存
                    image = cv2.imread(img_path)
                    if len(bboxes_pr) == 0:
                        acc_00_jpg += 1
                        if not os.path.exists(save_dir + "/noresult"): os.mkdir(save_dir + "/noresult")
                        shutil.copy(img_path, save_dir + "/noresult" + "/" + img)
                    else:
                        image = utils.draw_bbox(image, bboxes_pr, show_label=True)  # 画图
                        drawed_img_save_to_path = str(img_path).split("/")[-1]  # 保存文件命名

                        pre = int(bboxes_pr[0][-1])
                        score = bboxes_pr[0][-2]
                        if pre != int(classes_id40[c]):  # 结果错误，保存带框图片
                            error_jpg += 1
                            if not os.path.exists(save_dir + "/error"): os.mkdir(save_dir + "/error")
                            cv2.imwrite(save_dir + "/error/" + drawed_img_save_to_path, image)  # 保存带框图
                        else:
                            if score >= 0.6 and score < 0.8:  # 低分结果、保存
                                acc_60_jpg += 1
                                if not os.path.exists(save_dir + "/low"): os.mkdir(save_dir + "/low")
                                cv2.imwrite(save_dir + "/low/" + drawed_img_save_to_path, image)  # 保存带框图
                            elif score < 0.6:  # 超低分保存画框图
                                acc_30_jpg += 1
                                if not os.path.exists(save_dir + "/lower"): os.mkdir(save_dir + "/lower")
                                cv2.imwrite(save_dir + "/lower/" + drawed_img_save_to_path, image)  # 保存带框图
                            else:
                                acc_80_jpg += 1
                                print("right:", img_path)
    print("所以图片：", all_jpg)
    print("分值大于0.8且正确：", acc_80_jpg)
    print("分值大于0.6小于0.8且正确：", acc_60_jpg)
    print("分值大于0.3小于0.6且正确：", acc_30_jpg)
    print("无任何结果：", acc_00_jpg)
    print("识别错误：", error_jpg)
    print("大于0.8准确率：", round(acc_80_jpg / all_jpg, 2))
    print("大于0.6准确率：", round(acc_60_jpg / all_jpg, 2))
    print("大于0.3准确率：", round(acc_30_jpg / all_jpg, 2))
