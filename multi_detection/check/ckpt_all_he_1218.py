# -*- encoding: utf-8 -*-

"""
@File    : ckpt_all_he_1218.py
@Time    : 2019/12/19 18:16
@Author  : sunyihuan
"""

'''
ckpt文件预测某一文件夹下各类所有图片烤层结果、食材结果
并输出各准确率至excel表格中

此结果合并大蛋挞、小蛋挞；四分之一披萨、六分之一披萨
采用JPGImages_he下数据

说明：correct_bboxes使用李志鹏12月17日修订版
'''

import cv2
import numpy as np
import tensorflow as tf
import multi_detection.core.utils as utils
import os
import shutil
from tqdm import tqdm
import xlwt
from sklearn.metrics import confusion_matrix


def correct_bboxes(bboxes_pr, layer_n):
    '''
    bboxes_pr结果矫正
    :param bboxes_pr: 模型预测结果，格式为[x_min, y_min, x_max, y_max, probability, cls_id]
    :param layer_n:
    :return:
    '''
    num_label = len(bboxes_pr)
    # 未检测食材
    if num_label == 0:
        return bboxes_pr, layer_n

    # 检测到一个食材
    elif num_label == 1:
        if bboxes_pr[0][4] < 0.45:
            if bboxes_pr[0][5] == 10:  # 低分nofood
                bboxes_pr[0][4] = 0.75
            elif bboxes_pr[0][5] == 11:  # 低分花生米
                bboxes_pr[0][4] = 0.85
            elif bboxes_pr[0][5] == 25:  # 低分整鸡
                bboxes_pr[0][4] = 0.75
            else:
                del bboxes_pr[0]

        # else:
        #    if bboxes_pr[0][4] < 0.9 and bboxes_pr[0][4] >= 0.6:
        #        bboxes_pr[0][4] = 0.9

        return bboxes_pr, layer_n

    # 检测到多个食材
    else:
        new_bboxes_pr = []
        for i in range(len(bboxes_pr)):
            if bboxes_pr[i][4] >= 0.45:
                new_bboxes_pr.append(bboxes_pr[i])

        new_num_label = len(new_bboxes_pr)
        if new_num_label == 0:
            return new_bboxes_pr, layer_n
        same_label = True
        for i in range(new_num_label):
            if i == (new_num_label - 1):
                break
            if new_bboxes_pr[i][5] == new_bboxes_pr[i + 1][5]:
                continue
            else:
                same_label = False

        sumProb = 0.
        # 多个食材，同一标签
        if same_label:
            new_bboxes_pr[0][4] = 0.98
            return new_bboxes_pr, layer_n
        # 多个食材，非同一标签
        else:
            problist = list(map(lambda x: x[4], new_bboxes_pr))
            labellist = list(map(lambda x: x[5], new_bboxes_pr))

            labeldict = {}
            for key in labellist:
                labeldict[key] = labeldict.get(key, 0) + 1
                # 按同种食材label数量降序排列
            s_labeldict = sorted(labeldict.items(), key=lambda x: x[1], reverse=True)

            n_name = len(s_labeldict)
            name1 = s_labeldict[0][0]
            num_name1 = s_labeldict[0][1]
            name2 = s_labeldict[1][0]
            num_name2 = s_labeldict[1][1]

            # 优先处理食材特例
            if n_name == 2:
                # 如果鸡翅中检测到了排骨，默认单一食材为鸡翅
                if (name1 == 2 and name2 == 16) or (name1 == 16 and name2 == 2):
                    for i in range(new_num_label):
                        new_bboxes_pr[i][5] = 2
                    return new_bboxes_pr, layer_n
                # 如果对切土豆中检测到了大土豆，默认单一食材为对切土豆
                if (name1 == 17 and name2 == 18) or (name1 == 18 and name2 == 17):
                    for i in range(new_num_label):
                        new_bboxes_pr[i][5] = 17
                    return new_bboxes_pr, layer_n
                # 如果对切红薯中检测到了大红薯，默认单一食材为对切红薯
                if (name1 == 21 and name2 == 22) or (name1 == 22 and name2 == 21):
                    for i in range(new_num_label):
                        new_bboxes_pr[i][5] = 21
                    return new_bboxes_pr, layer_n
                # 如果对切红薯中检测到了中红薯，默认单一食材为对切红薯
                if (name1 == 21 and name2 == 23) or (name1 == 23 and name2 == 21):
                    for i in range(new_num_label):
                        new_bboxes_pr[i][5] = 21
                    return new_bboxes_pr, layer_n

            # 数量最多label对应的食材占比0.7以上
            if num_name1 / new_num_label > 0.7:
                name1_bboxes_pr = []
                for i in range(new_num_label):
                    if name1 == new_bboxes_pr[i][5]:
                        name1_bboxes_pr.append(new_bboxes_pr[i])

                name1_bboxes_pr[0][4] = 0.95
                return name1_bboxes_pr, layer_n

            # 按各个label的probability降序排序
            else:
                new_bboxes_pr = sorted(new_bboxes_pr, key=lambda x: x[4], reverse=True)
                for i in range(len(new_bboxes_pr)):
                    new_bboxes_pr[i][4] = new_bboxes_pr[i][4] * 0.9
                return new_bboxes_pr, layer_n


class YoloTest(object):
    def __init__(self):
        self.input_size = 416  # 输入图片尺寸（默认正方形）
        self.num_classes = 30  # 种类数
        self.score_threshold = 0.1
        self.iou_threshold = 0.5
        self.weight_file = "E:/ckpt_dirs/Food_detection/local/20191216/yolov3_train_loss=4.7698.ckpt-80"  # ckpt文件地址
        self.write_image = True  # 是否画图
        self.show_label = True  # 是否显示标签

        graph = tf.Graph()
        with graph.as_default():
            # 模型加载
            self.saver = tf.train.import_meta_graph("{}.meta".format(self.weight_file))
            self.sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True))
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

    def result(self, image_path, save_dir):
        '''
        得出预测结果并保存
        :param image_path: 图片地址
        :param save_dir: 预测结果原图标注框，保存地址
        :return:
        '''
        image = cv2.imread(image_path)  # 图片读取
        image = utils.white_balance(image)  # 图片白平衡处理
        bboxes_pr, layer_n = self.predict(image)  # 预测结果
        # print(bboxes_pr)
        # print(layer_n)

        if self.write_image:
            image = utils.draw_bbox(image, bboxes_pr, show_label=self.show_label)
            drawed_img_save_to_path = str(image_path).split("/")[-1]
            drawed_img_save_to_path = str(drawed_img_save_to_path).split(".")[0] + "_" + str(
                layer_n) + ".jpg"  # 图片保存地址，烤层结果在命名中
            # cv2.imshow('Detection result', image)
            cv2.imwrite(save_dir + "/" + drawed_img_save_to_path, image)  # 保存图片
        return bboxes_pr, layer_n


if __name__ == '__main__':
    img_dir = "E:/test_from_ye/JPGImages_he"  # 文件夹地址
    save_dir = "E:/test_from_ye/detection_he_local_1216"  # 图片保存地址
    if not os.path.exists(save_dir): os.mkdir(save_dir)

    layer_error_dir = "E:/test_from_ye/layer_error_he_local_1216"  # 预测结果错误保存地址
    if not os.path.exists(layer_error_dir): os.mkdir(layer_error_dir)

    fooderror_dir = "E:/test_from_ye/food_error_he_local_1216"  # 食材预测结果错误保存地址
    if not os.path.exists(fooderror_dir): os.mkdir(fooderror_dir)

    no_result_dir = "E:/test_from_ye/no_result_he_local_1216"  # 无任何输出结果保存地址
    if not os.path.exists(no_result_dir): os.mkdir(no_result_dir)

    Y = YoloTest()  # 加载模型

    classes = ["Beefsteak", "CartoonCookies", "Cookies", "CupCake", "Pizzabits",
               "Pizzatwo", "Pizzaone", "ChickenWings", "ChiffonCake6",
               "ChiffonCake8", "CranberryCookies", "eggtart", "nofood",
               "Peanuts", "PorkChops", "PotatoCut", "Potatol", "Potatom",
               "Potatos", "RoastedChicken", "SweetPotatoCut", "SweetPotatol", "SweetPotatom",
               "SweetPotatoS", "Toast"]

    # ab_classes = ["Pizzafour", "Pizzatwo", "Pizzaone", "Pizzasix",
    #               "PotatoCut", "Potatol", "Potatom",
    #               "RoastedChicken",
    #               "SweetPotatoCut", "SweetPotatol", "SweetPotatom", "SweetPotatoS",
    #               "Toast"]

    classes_id = {"CartoonCookies": 1, "Cookies": 5, "CupCake": 7, "Beefsteak": 0, "ChickenWings": 2,
                  "ChiffonCake6": 3, "ChiffonCake8": 4, "CranberryCookies": 6, "eggtart": [8, 9],
                  "nofood": 10, "Peanuts": 11, "PorkChops": 16, "PotatoCut": 17, "Potatol": 18,
                  "Potatom": 19, "Potatos": 20, "SweetPotatoCut": 21, "SweetPotatol": 22, "SweetPotatom": 23,
                  "Pizzabits": [12, 14], "Pizzaone": 13, "RoastedChicken": 25,
                  "Pizzatwo": 15, "SweetPotatoS": 24, "Toast": 26}
    jpgs_count_all = 0
    layer_jpgs_acc = 0
    food_jpgs_acc = 0

    workbook = xlwt.Workbook(encoding='utf-8')
    sheet1 = workbook.add_sheet("multi_food")
    sheet1.write(0, 0, "classes")
    sheet1.write(1, 0, "classes")

    sheet1.write(1, 1, "bottom_layer_acc")
    sheet1.write(1, 2, "bottom_food_acc")
    sheet1.write(1, 3, "middle_layer_acc")
    sheet1.write(1, 4, "middle_food_acc")
    sheet1.write(1, 5, "top_layer_acc")
    sheet1.write(1, 6, "top_food_acc")
    sheet1.write(1, 7, "others_layer_acc")
    sheet1.write(1, 8, "others_food_acc")
    sheet1.write(1, 9, "jpgs_all")
    sheet1.write(1, 10, "layer_acc")
    sheet1.write(1, 11, "food_acc")
    sheet1.write(1, 12, "no_result_nums")

    layer_img_true = []
    layer_img_pre = []

    food_img_true = []
    food_img_pre = []
    for i in range(len(classes)):
        c = classes[i].lower()

        error_noresults = 0  # 食材无任何输出结果统计
        food_acc = 0  # 食材正确数量统计
        layer_acc = 0  # 烤层准确数统计
        all_jpgs = 0  # 图片总数统计

        # 各层统计
        layer_acc_b = 0  # 最下层--烤层
        layer_acc_m = 0  # 中层--烤层
        layer_acc_t = 0  # 最上层--烤层
        layer_acc_o = 0  # 其他--烤层

        food_acc_b = 0  # 最下层--食材
        food_acc_m = 0  # 中层--食材
        food_acc_t = 0  # 最上层--食材
        food_acc_o = 0  # 其他--食材

        img_dirs = img_dir + "/" + c
        layer_error_c_dirs = layer_error_dir + "/" + c
        if os.path.exists(layer_error_c_dirs): shutil.rmtree(layer_error_c_dirs)
        os.mkdir(layer_error_c_dirs)

        fooderror_dirs = fooderror_dir + "/" + c
        if os.path.exists(fooderror_dirs): shutil.rmtree(fooderror_dirs)
        os.mkdir(fooderror_dirs)

        noresult_dir = no_result_dir + "/" + c
        if os.path.exists(noresult_dir): shutil.rmtree(noresult_dir)
        os.mkdir(noresult_dir)

        save_c_dir = save_dir + "/" + c
        if os.path.exists(save_c_dir): shutil.rmtree(save_c_dir)
        os.mkdir(save_c_dir)

        for file in tqdm(os.listdir(img_dirs + "/bottom")):  # 底层
            if file.endswith("jpg"):
                all_jpgs += 1  # 统计总jpg图片数量
                image_path = img_dirs + "/bottom" + "/" + file
                bboxes_pr, layer_n = Y.result(image_path, save_c_dir)  # 预测每一张结果并保存

                layer_img_true.append(0)  # 烤层真实结果
                layer_img_pre.append(layer_n)  # 烤层预测结果
                if layer_n != 0:
                    shutil.copy(image_path,
                                layer_error_c_dirs + "/" + file.split(".jpg")[0] + "_" + str(layer_n) + ".jpg")
                else:
                    layer_acc_b += 1  # 最下层烤层正确+1
                    layer_acc += 1  # 烤层正确+1

                bboxes_pr, layer_n = correct_bboxes(bboxes_pr, layer_n)  # 矫正输出结果
                if len(bboxes_pr) == 0:  # 无任何结果返回，输出并统计+1
                    error_noresults += 1
                    shutil.copy(image_path, noresult_dir + "/" + file)
                else:
                    # bboxes_pr, layer_n = correct_bboxes(bboxes_pr, layer_n)  # 矫正输出结果
                    pre = bboxes_pr[0][-1]
                    food_img_pre.append(pre)
                    food_img_true.append(classes_id[classes[i]])

                    if pre in [8, 9]:
                        if classes_id[classes[i]] == [8, 9]:
                            food_acc_b += 1
                            food_acc += 1
                        else:
                            drawed_img_save_to_path = str(image_path).split("/")[-1]
                            drawed_img_save_to_path = str(drawed_img_save_to_path).split(".")[0] + "_" + str(
                                layer_n) + ".jpg"  # 图片保存地址，烤层结果在命名中
                            shutil.copy(save_c_dir + "/" + drawed_img_save_to_path,
                                        fooderror_dirs + "/" + file.split(".jpg")[0] + "_" + str(layer_n) + ".jpg")
                    elif pre in [12, 14]:
                        if classes_id[classes[i]] == [12, 14]:
                            food_acc_b += 1
                            food_acc += 1
                        else:
                            drawed_img_save_to_path = str(image_path).split("/")[-1]
                            drawed_img_save_to_path = str(drawed_img_save_to_path).split(".")[0] + "_" + str(
                                layer_n) + ".jpg"  # 图片保存地址，烤层结果在命名中
                            shutil.copy(save_c_dir + "/" + drawed_img_save_to_path,
                                        fooderror_dirs + "/" + file.split(".jpg")[0] + "_" + str(layer_n) + ".jpg")
                    else:
                        if pre == classes_id[classes[i]]:  # 若结果正确，食材正确数+1
                            food_acc_m += 1
                            food_acc += 1
                        else:
                            drawed_img_save_to_path = str(image_path).split("/")[-1]
                            drawed_img_save_to_path = str(drawed_img_save_to_path).split(".")[0] + "_" + str(
                                layer_n) + ".jpg"  # 图片保存地址，烤层结果在命名中
                            shutil.copy(save_c_dir + "/" + drawed_img_save_to_path,
                                        fooderror_dirs + "/" + file.split(".jpg")[0] + "_" + str(layer_n) + ".jpg")

        if len(os.listdir(img_dirs + "/bottom")) == 0:  # 判断是否有值
            layer_bottom_acc = 0
            food_bottom_acc = 0
        else:
            layer_bottom_acc = round(layer_acc_b / len(os.listdir(img_dirs + "/bottom")), 2)
            food_bottom_acc = round(food_acc_b / len(os.listdir(img_dirs + "/bottom")), 2)
        sheet1.write(i + 2, 1, layer_bottom_acc)  # 下层烤层准确率写入
        sheet1.write(i + 2, 2, food_bottom_acc)  # 下层食材准确率写入

        for file in tqdm(os.listdir(img_dirs + "/middle")):  # 中层
            if file.endswith("jpg"):
                all_jpgs += 1  # 统计总jpg图片数量
                image_path = img_dirs + "/middle" + "/" + file
                bboxes_pr, layer_n = Y.result(image_path, save_c_dir)  # 预测每一张结果并保存

                layer_img_true.append(1)  # 烤层真实结果
                layer_img_pre.append(layer_n)  # 烤层预测结果
                if layer_n != 1:  # 判断烤层是否为1
                    shutil.copy(image_path,
                                layer_error_c_dirs + "/" + file.split(".jpg")[0] + "_" + str(layer_n) + ".jpg")
                else:
                    layer_acc_m += 1  # 中层烤层正确+1
                    layer_acc += 1  # 烤层正确+1

                bboxes_pr, layer_n = correct_bboxes(bboxes_pr, layer_n)  # 矫正输出结果
                if len(bboxes_pr) == 0:  # 无任何结果返回，输出并统计+1
                    error_noresults += 1
                    shutil.copy(image_path, noresult_dir + "/" + file)
                else:
                    # bboxes_pr, layer_n = correct_bboxes(bboxes_pr, layer_n)  # 矫正输出结果
                    pre = bboxes_pr[0][-1]
                    food_img_pre.append(pre)
                    food_img_true.append(classes_id[classes[i]])

                    if pre == classes_id[classes[i]]:  # 若结果正确，食材正确数+1
                        food_acc_m += 1
                        food_acc += 1
                    else:
                        if pre in [8, 9] and classes_id[classes[i]] in [8, 9]:
                            food_acc_m += 1
                            food_acc += 1
                        if pre in [12, 14] and classes_id[classes[i]] in [12, 14]:
                            food_acc_m += 1
                            food_acc += 1
                        else:
                            drawed_img_save_to_path = str(image_path).split("/")[-1]
                            drawed_img_save_to_path = str(drawed_img_save_to_path).split(".")[0] + "_" + str(
                                layer_n) + ".jpg"  # 图片保存地址，烤层结果在命名中
                            shutil.copy(save_c_dir + "/" + drawed_img_save_to_path,
                                        fooderror_dirs + "/" + file.split(".jpg")[0] + "_" + str(layer_n) + ".jpg")

        if len(os.listdir(img_dirs + "/middle")) == 0:  # 判断是否有值
            layer_middle_acc = 0
            food_middle_acc = 0
        else:
            layer_middle_acc = round(layer_acc_m / len(os.listdir(img_dirs + "/middle")), 2)
            food_middle_acc = round(food_acc_m / len(os.listdir(img_dirs + "/middle")), 2)
        sheet1.write(i + 2, 3, layer_middle_acc)  # 中层烤层准确率写入
        sheet1.write(i + 2, 4, food_middle_acc)  # 中层食材准确率写入

        for file in tqdm(os.listdir(img_dirs + "/top")):
            if file.endswith("jpg"):
                all_jpgs += 1  # 统计总jpg图片数量
                image_path = img_dirs + "/top" + "/" + file
                bboxes_pr, layer_n = Y.result(image_path, save_c_dir)  # 预测每一张结果并保存

                layer_img_true.append(2)  # 烤层真实结果
                layer_img_pre.append(layer_n)  # 烤层预测结果
                if layer_n != 2:  # 判断烤层是否为2
                    shutil.copy(image_path,
                                layer_error_c_dirs + "/" + file.split(".jpg")[0] + "_" + str(layer_n) + ".jpg")
                else:
                    layer_acc_t += 1  # 上层烤层正确+1
                    layer_acc += 1  # 烤层正确+1

                bboxes_pr, layer_n = correct_bboxes(bboxes_pr, layer_n)  # 矫正输出结果
                if len(bboxes_pr) == 0:  # 无任何结果返回，输出并统计+1
                    error_noresults += 1
                    shutil.copy(image_path, noresult_dir + "/" + file)
                else:
                    # bboxes_pr, layer_n = correct_bboxes(bboxes_pr, layer_n)  # 矫正输出结果
                    pre = bboxes_pr[0][-1]
                    food_img_pre.append(pre)
                    food_img_true.append(classes_id[classes[i]])

                    if pre == classes_id[classes[i]]:  # 若结果正确，食材正确数+1
                        food_acc_t += 1
                        food_acc += 1
                    else:
                        if pre in [8, 9] and classes_id[classes[i]] in [8, 9]:
                            food_acc_t += 1
                            food_acc += 1
                        if pre in [12, 14] and classes_id[classes[i]] in [12, 14]:
                            food_acc_t += 1
                            food_acc += 1
                        else:
                            drawed_img_save_to_path = str(image_path).split("/")[-1]
                            drawed_img_save_to_path = str(drawed_img_save_to_path).split(".")[0] + "_" + str(
                                layer_n) + ".jpg"  # 图片保存地址，烤层结果在命名中
                            shutil.copy(save_c_dir + "/" + drawed_img_save_to_path,
                                        fooderror_dirs + "/" + file.split(".jpg")[0] + "_" + str(layer_n) + ".jpg")

        if len(os.listdir(img_dirs + "/top")) == 0:  # 判断是否有值
            layer_top_acc = 0
            food_top_acc = 0
        else:
            layer_top_acc = round(layer_acc_t / len(os.listdir(img_dirs + "/top")), 2)
            food_top_acc = round(food_acc_t / len(os.listdir(img_dirs + "/top")), 2)
        sheet1.write(i + 2, 5, layer_top_acc)  # 上层烤层准确率写入
        sheet1.write(i + 2, 6, food_top_acc)  # 上层食材准确率写入

        for file in tqdm(os.listdir(img_dirs + "/others")):
            if file.endswith("jpg"):
                all_jpgs += 1  # 统计总jpg图片数量
                image_path = img_dirs + "/others" + "/" + file
                bboxes_pr, layer_n = Y.result(image_path, save_c_dir)  # 预测每一张结果并保存

                layer_img_true.append(3)  # 烤层真实结果
                layer_img_pre.append(layer_n)  # 烤层预测结果
                if layer_n != 3:  # 判断烤层是否为3
                    shutil.copy(image_path,
                                layer_error_c_dirs + "/" + file.split(".jpg")[0] + "_" + str(layer_n) + ".jpg")
                else:
                    layer_acc_o += 1  # 上层烤层正确+1
                    layer_acc += 1  # 烤层正确+1

                bboxes_pr, layer_n = correct_bboxes(bboxes_pr, layer_n)  # 矫正输出结果
                if len(bboxes_pr) == 0:  # 无任何结果返回，输出并统计+1
                    error_noresults += 1
                    shutil.copy(image_path, noresult_dir + "/" + file)
                else:
                    # bboxes_pr, layer_n = correct_bboxes(bboxes_pr, layer_n)  # 矫正输出结果
                    pre = bboxes_pr[0][-1]
                    food_img_pre.append(pre)
                    food_img_true.append(classes_id[classes[i]])

                    if pre == classes_id[classes[i]]:  # 若结果正确，食材正确数+1
                        food_acc_o += 1
                        food_acc += 1
                    else:
                        if pre in [8, 9] and classes_id[classes[i]] in [8, 9]:
                            food_acc_o += 1
                            food_acc += 1
                        if pre in [12, 14] and classes_id[classes[i]] in [12, 14]:
                            food_acc_o += 1
                            food_acc += 1
                        else:
                            drawed_img_save_to_path = str(image_path).split("/")[-1]
                            drawed_img_save_to_path = str(drawed_img_save_to_path).split(".")[0] + "_" + str(
                                layer_n) + ".jpg"  # 图片保存地址，烤层结果在命名中
                            shutil.copy(save_c_dir + "/" + drawed_img_save_to_path,
                                        fooderror_dirs + "/" + file.split(".jpg")[0] + "_" + str(layer_n) + ".jpg")

        if len(os.listdir(img_dirs + "/others")) == 0:  # 判断是否有值
            layer_others_acc = 0
            food_others_acc = 0
        else:
            layer_others_acc = round(layer_acc_o / len(os.listdir(img_dirs + "/others")), 2)
            food_others_acc = round(food_acc_o / len(os.listdir(img_dirs + "/others")), 2)
        sheet1.write(i + 2, 7, layer_others_acc)  # 烤层-其他，准确率写入
        sheet1.write(i + 2, 8, food_others_acc)  # 烤层-其他食材准确率写入

        sheet1.write(i + 2, 0, c)

        sheet1.write(i + 2, 9, all_jpgs)
        sheet1.write(i + 2, 10, round((layer_acc / all_jpgs) * 100, 2))
        sheet1.write(i + 2, 11, round((food_acc / all_jpgs) * 100, 2))

        sheet1.write(i + 2, 12, error_noresults)

        print("food name:", c)
        print("layer accuracy:", round((layer_acc / all_jpgs) * 100, 2))  # 输出烤层正确数
        jpgs_count_all += all_jpgs
        layer_jpgs_acc += layer_acc
        food_jpgs_acc += food_acc
    print("all layer accuracy:", round((layer_jpgs_acc / jpgs_count_all) * 100, 2))  # 输出烤层正确数
    print("all food accuracy:", round((food_jpgs_acc / jpgs_count_all) * 100, 2))  # 输出食材正确数

    layer_conf = confusion_matrix(y_pred=layer_img_pre, y_true=layer_img_true)
    food_conf = confusion_matrix(y_pred=food_img_pre, y_true=food_img_true)

    print(layer_conf)
    print(sum(sum(layer_conf)))
    print(food_conf)
    print(sum(sum(food_conf)))

    sheet1.write(35, 1, jpgs_count_all)
    sheet1.write(35, 2, layer_jpgs_acc)
    sheet1.write(35, 3, food_jpgs_acc)
    sheet1.write(35, 4, round((layer_jpgs_acc / jpgs_count_all) * 100, 2))
    sheet1.write(35, 5, round((food_jpgs_acc / jpgs_count_all) * 100, 2))

    workbook.save("E:/test_from_ye/all_he_he_local_1216.xls")
