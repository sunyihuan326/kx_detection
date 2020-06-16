# -*- encoding: utf-8 -*-

"""
@File    : ckpt_all_he_1218.py
@Time    : 2019/12/18 14:39
@Author  : sunyihuan
"""

'''
ckpt文件预测某一文件夹下各类所有图片烤层结果、食材结果
并输出各准确率至excel表格中

此结果合并大蛋挞、小蛋挞；四分之一披萨、六分之一披萨


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
import time
from sklearn.metrics import confusion_matrix
from multi_detection.food_correct_utils import correct_bboxes

# gpu限制
gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.8)
config = tf.ConfigProto(gpu_options=gpu_options)


def he_foods(pre):
    '''
    针对合并的类别判断输出是否在合并类别内
    :param pre:
    :return:
    '''
    # if pre in [8, 9] and classes_id[classes[i]] in [8, 9]:  # 合并蛋挞
    #     rigth_label = True
    # elif pre in [12, 14] and classes_id[classes[i]] in [12, 14]:  # 合并四分之一披萨、六分之一披萨
    #     rigth_label = True
    # elif pre in [18, 19] and classes_id[classes[i]] in [18, 19]:  # 合并中土豆、大土豆
    #     rigth_label = True
    # elif pre in [22, 23] and classes_id[classes[i]] in [22, 23]:  # 合并中红薯、大红薯
    #     rigth_label = True
    # else:
    #     rigth_label = False
    rigth_label = False
    return rigth_label


class YoloTest(object):
    def __init__(self):
        self.input_size = 416  # 输入图片尺寸（默认正方形）
        self.num_classes = 23  # 种类数
        self.score_threshold = 0.3
        self.iou_threshold = 0.5
        self.weight_file = "E:/ckpt_dirs/Food_detection/multi_food/20200609/yolov3_train_loss=4.9670.ckpt-154"  # ckpt文件地址
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

            # 食材结果
            self.food_class = graph.get_tensor_by_name("define_loss/food_classes:0")

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

        pred_sbbox, pred_mbbox, pred_lbbox, layer_n, food_class = self.sess.run(
            [self.pred_sbbox, self.pred_mbbox, self.pred_lbbox, self.layer_num, self.food_class],
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

        return bboxes, layer_n[0],food_class[0]

    def result(self, image_path, save_dir):
        '''
        得出预测结果并保存
        :param image_path: 图片地址
        :param save_dir: 预测结果原图标注框，保存地址
        :return:
        '''
        image = cv2.imread(image_path)  # 图片读取
        # image = utils.white_balance(image)  # 图片白平衡处理
        bboxes_pr, layer_n,food_class = self.predict(image)  # 预测结果
        # print(bboxes_pr)
        # print(layer_n)

        if self.write_image:
            image = utils.draw_bbox(image, bboxes_pr, show_label=self.show_label)
            drawed_img_save_to_path = str(image_path).split("/")[-1]
            drawed_img_save_to_path = str(drawed_img_save_to_path).split(".")[0] + "_" + str(
                layer_n) + ".jpg"  # 图片保存地址，烤层结果在命名中
            # cv2.imshow('Detection result', image)
            cv2.imwrite(save_dir + "/" + drawed_img_save_to_path, image)  # 保存图片
        return bboxes_pr, layer_n,food_class


if __name__ == '__main__':

    classes_label22 = ["beefsteak", "cartooncookies", "chickenwings", "chiffoncake6", "chiffoncake8",
                       "cookies", "cranberrycookies", "cupcake", "eggtartl", "eggtarts",
                       "nofood", "peanuts", "pizzafour", "pizzaone", "pizzasix",
                       "pizzatwo", "porkchops", "potatocut", "potatol", "potatom",
                       "potatos", "sweetpotatocut", "sweetpotatol", "sweetpotatom", "sweetpotatos",
                       "roastedchicken", "toast", ]

    classes_label46 = ["beefsteak", "cartooncookies", "chickenwings", "chiffoncake6", "chiffoncake8",
                       "cookies", "cranberrycookies", "cupcake", "eggtartl", "eggtarts",
                       "nofood", "peanuts", "pizzafour", "pizzaone", "pizzasix",
                       "pizzatwo", "porkchops", "potatocut", "potatol", "potatom",
                       "potatos", "sweetpotatocut", "sweetpotatol", "sweetpotatom", "sweetpotatos",
                       "roastedchicken", "toast",
                       "chestnut", "cornone", "corntwo", "drumsticks", "taro",
                       "steamedbread", "eggplant", "eggplant_cut_sauce", "bread", "container_nonhigh",
                       "container", "duck", "fish", "hotdog", "redshrimp",
                       "shrimp", "strand"]
    classes_label18 = ["chestnut", "cornone", "corntwo", "drumsticks", "taro",
                       "steamedbread", "eggplant", "eggplant_cut_sauce", "bread",
                       "container_nonhigh", "container", "duck", "fish", "hotdog",
                       "redshrimp", "shrimp", "strand"]

    # classes = ["RoastedChicken"]

    # ab_classes = ["Pizzafour", "Pizzatwo", "Pizzaone", "Pizzasix",
    #               "PotatoCut", "Potatol", "Potatom",
    #               "RoastedChicken",
    #               "SweetPotatoCut", "SweetPotatol", "SweetPotatom", "SweetPotatoS",
    #               "Toast"]
    # classes_id = {"CartoonCookies": 1, "Cookies": 5, "CupCake": 7, "Beefsteak": 0, "ChickenWings": 2,
    #               "ChiffonCake6": 3, "ChiffonCake8": 4, "CranberryCookies": 6, "eggtarts": 8, "eggtartl": 9,
    #               "nofood": 10, "Peanuts": 11, "PorkChops": 16, "PotatoCut": 17, "Potatol": 18,
    #               "Potatom": 19, "Potatos": 20, "SweetPotatoCut": 21, "SweetPotatol": 22, "SweetPotatom": 23,
    #               "Pizzafour": 12, "Pizzaone": 13, "Pizzasix": 14, "RoastedChicken": 25,
    #               "Pizzatwo": 15, "SweetPotatoS": 24, "Toast": 26, "sweetpotato_others": 27, "pizza_others": 28,
    #               "potato_others": 29, "chestnut": 30, "cornone": 31, "corntwo": 32, "drumsticks": 33,
    #               "taro": 34, "steamedbread": 35,}
    classes_id18 = {"chestnut": 1, "cornone": 2, "corntwo": 3, "drumsticks": 4, "taro": 5,
                    "nofood": 0, "steamedbread": 6, "eggplant": 7, "eggplant_cut_sauce": 8, "bread": 9,
                    "container_nonhigh": 10, "container": 11, "duck": 12, "fish": 13, "hotdog": 14,
                    "redshrimp": 15, "shrimp": 16, "strand": 17}
    classes_id46 = {"cartooncookies": 1, "cookies": 5, "cupcake": 7, "beefsteak": 0, "chickenwings": 2,
                    "chiffoncake6": 3, "chiffoncake8": 4, "cranberrycookies": 6, "eggtarts": 8, "eggtartl": 9,
                    "nofood": 10, "peanuts": 11, "porkchops": 16, "potatocut": 17, "potatol": 18,
                    "potatom": 19, "potatos": 20, "sweetpotatocut": 21, "sweetpotatol": 22, "sweetpotatom": 23,
                    "pizzafour": 12, "pizzaone": 13, "pizzasix": 14, "roastedchicken": 25,
                    "pizzatwo": 15, "sweetpotatos": 24, "toast": 26, "sweetpotato_others": 27, "pizza_others": 28,
                    "potato_others": 29, "chestnut": 30, "cornone": 31, "corntwo": 32, "drumsticks": 33,
                    "taro": 34, "steamedbread": 35, "eggplant": 36, "eggplant_cut_sauce": 37, "bread": 38,
                    "container_nonhigh": 39,
                    "container": 40, "duck": 25, "fish": 41, "hotdog": 42, "redshrimp": 43,
                    "shrimp": 44, "strand": 45}
    classes_id22 = {"cartooncookies": 1, "cookies": 4, "cupcake": 6, "beefsteak": 0, "chickenwings": 2,
                    "chiffoncake6": 3, "chiffoncake8": 3, "cranberrycookies": 5, "eggtarts": 7, "eggtartl": 7,
                    "nofood": 8, "peanuts": 9, "porkchops": 13, "potatocut": 14, "potatol": 15,
                    "potatom": 15, "potatos": 16, "sweetpotatocut": 17, "sweetpotatol": 18, "sweetpotatom": 18,
                    "pizzafour": 10, "pizzaone": 11, "pizzasix": 10, "roastedchicken": 20,
                    "pizzatwo": 12, "sweetpotatos": 19, "toast": 21}
    classes_id23 = {"cartooncookies": 1, "cookies": 5, "cupcake": 7, "beefsteak": 0, "chickenwings": 2,
                    "chiffoncake6": 3, "chiffoncake8": 4, "cranberrycookies": 6, "eggtarts": 8, "eggtartl": 8,
                    "nofood": 9, "peanuts": 10, "porkchops": 14, "potatocut": 15, "potatol": 16,
                    "potatom": 16, "potatos": 17, "sweetpotatocut": 18, "sweetpotatol": 19, "sweetpotatom": 19,
                    "pizzafour": 11, "pizzaone": 12, "pizzasix": 11, "roastedchicken": 21,
                    "pizzatwo": 13, "sweetpotatos": 20, "toast": 22}
    # 需要修改
    classes_id = classes_id23  #######
    classes = classes_label22  #######
    mode = "multi_0609"  #######
    tag = ""
    img_dir = "E:/check_2_phase/JPGImages"  # 文件夹地址
    save_dir = "E:/check_2_phase/detection_{0}{1}".format(mode, tag)  # 图片保存地址
    if not os.path.exists(save_dir): os.mkdir(save_dir)

    layer_error_dir = "E:/check_2_phase/layer_error_{0}{1}".format(mode, tag)  # 预测结果错误保存地址
    if not os.path.exists(layer_error_dir): os.mkdir(layer_error_dir)

    fooderror_dir = "E:/check_2_phase/food_error_{0}{1}".format(mode, tag)  # 食材预测结果错误保存地址
    if not os.path.exists(fooderror_dir): os.mkdir(fooderror_dir)

    no_result_dir = "E:/check_2_phase/no_result_{0}{1}".format(mode, tag)  # 无任何输出结果保存地址
    if not os.path.exists(no_result_dir): os.mkdir(no_result_dir)

    start_time = time.time()
    Y = YoloTest()  # 加载模型
    end0_time = time.time()
    print("model loading time:", end0_time - start_time)
    new_classes = {v: k for k, v in classes_id.items()}

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
    sheet1.write(1, 10, "layer_right_nums")
    sheet1.write(1, 11, "food_right_nums")
    sheet1.write(1, 12, "layer_acc")
    sheet1.write(1, 13, "food_acc")
    sheet1.write(1, 14, "no_result_nums")
    sheet1.write(1, 15, "layer_and_food_right")

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

        c_layer_right_list = []
        c_food_right_list = []
        # 底层结果查看
        for file in tqdm(os.listdir(img_dirs + "/bottom")):  # 底层
            if file.endswith("jpg"):
                all_jpgs += 1  # 统计总jpg图片数量
                image_path = img_dirs + "/bottom" + "/" + file
                bboxes_pr, layer_n ,food_class= Y.result(image_path, save_c_dir)  # 预测每一张结果并保存

                layer_img_true.append(0)  # 烤层真实结果
                layer_img_pre.append(layer_n)  # 烤层预测结果
                if layer_n != 0:
                    shutil.copy(image_path,
                                layer_error_c_dirs + "/" + file.split(".jpg")[0] + "_" + str(layer_n) + ".jpg")
                else:
                    layer_acc_b += 1  # 最下层烤层正确+1
                    layer_acc += 1  # 烤层正确+1
                    c_layer_right_list.append(str(c) + "/" + file)  # 正确将名字写入c_layer_right_list中

                bboxes_pr, layer_n = correct_bboxes(bboxes_pr, layer_n)  # 矫正输出结果
                if len(bboxes_pr) == 0:  # 无任何结果返回，输出并统计+1
                    error_noresults += 1
                    shutil.copy(image_path, noresult_dir + "/" + file)
                else:
                    # pre = int(bboxes_pr[0][-1])
                    pre=int(food_class)
                    food_img_pre.append(pre)
                    food_img_true.append(classes_id[classes[i]])

                    if pre == classes_id[classes[i]]:  # 若结果正确，食材正确数+1
                        food_acc_b += 1
                        food_acc += 1
                        c_food_right_list.append(str(c) + "/" + file)  # 食材正确将名字写入c_food_right_list中
                    else:
                        right_label = he_foods(pre)
                        if right_label:  # 合并后结果正确
                            food_acc_b += 1
                            food_acc += 1
                            c_food_right_list.append(str(c) + "/" + file)  # 食材正确将名字写入c_food_right_list中
                        else:
                            drawed_img_save_to_path = str(image_path).split("/")[-1]
                            drawed_img_save_to_path = str(drawed_img_save_to_path).split(".")[0] + "_" + str(
                                layer_n) + ".jpg"  # 图片保存地址，烤层结果在命名中
                            shutil.copy(save_c_dir + "/" + drawed_img_save_to_path,
                                        fooderror_dirs + "/" + file.split(".jpg")[0] + "_" + str(layer_n) + "_" + str(
                                            pre) + ".jpg")
                            shutil.copy(image_path, fooderror_dirs + "/" + file.split(".jpg")[0] + "_{}.jpg".format(
                                new_classes[pre]))

        if len(os.listdir(img_dirs + "/bottom")) == 0:  # 判断是否有值
            layer_bottom_acc = 0
            food_bottom_acc = 0
        else:
            layer_bottom_acc = round(layer_acc_b / len(os.listdir(img_dirs + "/bottom")), 2)
            food_bottom_acc = round(food_acc_b / len(os.listdir(img_dirs + "/bottom")), 2)
        sheet1.write(i + 2, 1, layer_bottom_acc)  # 下层烤层准确率写入
        sheet1.write(i + 2, 2, food_bottom_acc)  # 下层食材准确率写入
        # 中层结果查看
        for file in tqdm(os.listdir(img_dirs + "/middle")):  # 中层
            if file.endswith("jpg"):
                all_jpgs += 1  # 统计总jpg图片数量
                image_path = img_dirs + "/middle" + "/" + file
                bboxes_pr, layer_n,food_class = Y.result(image_path, save_c_dir)  # 预测每一张结果并保存

                layer_img_true.append(1)  # 烤层真实结果
                layer_img_pre.append(layer_n)  # 烤层预测结果
                if layer_n != 1:  # 判断烤层是否为1
                    shutil.copy(image_path,
                                layer_error_c_dirs + "/" + file.split(".jpg")[0] + "_" + str(layer_n) + ".jpg")
                else:
                    layer_acc_m += 1  # 中层烤层正确+1
                    layer_acc += 1  # 烤层正确+1
                    c_layer_right_list.append(str(c) + "/" + file)  # 正确将名字写入c_layer_right_list中

                bboxes_pr, layer_n= correct_bboxes(bboxes_pr, layer_n)  # 矫正输出结果
                if len(bboxes_pr) == 0:  # 无任何结果返回，输出并统计+1
                    error_noresults += 1
                    shutil.copy(image_path, noresult_dir + "/" + file)
                else:
                    # pre = int(bboxes_pr[0][-1])
                    pre = int(food_class)
                    food_img_pre.append(pre)
                    food_img_true.append(classes_id[classes[i]])

                    if pre == classes_id[classes[i]]:  # 若结果正确，食材正确数+1
                        food_acc_m += 1
                        food_acc += 1
                        c_food_right_list.append(str(c) + "/" + file)  # 食材正确将名字写入c_food_right_list中
                    else:
                        right_label_m = he_foods(pre)
                        if right_label_m:  # 合并后结果正确
                            food_acc_m += 1
                            food_acc += 1
                            c_food_right_list.append(str(c) + "/" + file)  # 食材正确将名字写入c_food_right_list中
                        else:
                            drawed_img_save_to_path = str(image_path).split("/")[-1]
                            drawed_img_save_to_path = str(drawed_img_save_to_path).split(".")[0] + "_" + str(
                                layer_n) + ".jpg"  # 图片保存地址，烤层结果在命名中
                            shutil.copy(save_c_dir + "/" + drawed_img_save_to_path,
                                        fooderror_dirs + "/" + file.split(".jpg")[0] + "_" + str(layer_n) + "_" + str(
                                            pre) + ".jpg")
                            shutil.copy(image_path, fooderror_dirs + "/" + file.split(".jpg")[0] + "_{}.jpg".format(
                                new_classes[pre]))

        if len(os.listdir(img_dirs + "/middle")) == 0:  # 判断是否有值
            layer_middle_acc = 0
            food_middle_acc = 0
        else:
            layer_middle_acc = round(layer_acc_m / len(os.listdir(img_dirs + "/middle")), 2)
            food_middle_acc = round(food_acc_m / len(os.listdir(img_dirs + "/middle")), 2)
        sheet1.write(i + 2, 3, layer_middle_acc)  # 中层烤层准确率写入
        sheet1.write(i + 2, 4, food_middle_acc)  # 中层食材准确率写入
        # 上层结果查看
        for file in tqdm(os.listdir(img_dirs + "/top")):  # 上层
            if file.endswith("jpg"):
                all_jpgs += 1  # 统计总jpg图片数量
                image_path = img_dirs + "/top" + "/" + file
                bboxes_pr, layer_n,food_class = Y.result(image_path, save_c_dir)  # 预测每一张结果并保存

                layer_img_true.append(2)  # 烤层真实结果
                layer_img_pre.append(layer_n)  # 烤层预测结果
                if layer_n != 2:  # 判断烤层是否为2
                    shutil.copy(image_path,
                                layer_error_c_dirs + "/" + file.split(".jpg")[0] + "_" + str(layer_n) + ".jpg")
                else:
                    layer_acc_t += 1  # 上层烤层正确+1
                    layer_acc += 1  # 烤层正确+1
                    c_layer_right_list.append(str(c) + "/" + file)  # 正确将名字写入c_layer_right_list中

                bboxes_pr, layer_n = correct_bboxes(bboxes_pr, layer_n)  # 矫正输出结果
                if len(bboxes_pr) == 0:  # 无任何结果返回，输出并统计+1
                    error_noresults += 1
                    shutil.copy(image_path, noresult_dir + "/" + file)
                else:
                    bboxes_pr, layer_n = correct_bboxes(bboxes_pr, layer_n)  # 矫正输出结果
                    if len(bboxes_pr) == 0:
                        error_noresults += 1
                        shutil.copy(image_path, noresult_dir + "/" + file)
                    else:
                        # pre = int(bboxes_pr[0][-1])
                        pre = int(food_class)
                        food_img_pre.append(pre)
                        food_img_true.append(classes_id[classes[i]])

                        if pre == classes_id[classes[i]]:  # 若结果正确，食材正确数+1
                            food_acc_t += 1
                            food_acc += 1
                            c_food_right_list.append(str(c) + "/" + file)  # 食材正确将名字写入c_food_right_list中
                        else:
                            right_label_t = he_foods(pre)
                            if right_label_t:  # 合并后结果正确
                                food_acc_t += 1
                                food_acc += 1
                                c_food_right_list.append(str(c) + "/" + file)  # 食材正确将名字写入c_food_right_list中
                            else:
                                drawed_img_save_to_path = str(image_path).split("/")[-1]
                                drawed_img_save_to_path = str(drawed_img_save_to_path).split(".")[0] + "_" + str(
                                    layer_n) + ".jpg"  # 图片保存地址，烤层结果在命名中
                                shutil.copy(save_c_dir + "/" + drawed_img_save_to_path,
                                            fooderror_dirs + "/" + file.split(".jpg")[0] + "_" + str(
                                                layer_n) + "_" + str(
                                                pre) + ".jpg")
                                shutil.copy(image_path, fooderror_dirs + "/" + file.split(".jpg")[0] + "_{}.jpg".format(
                                    new_classes[pre]))

        if len(os.listdir(img_dirs + "/top")) == 0:  # 判断是否有值
            layer_top_acc = 0
            food_top_acc = 0
        else:
            layer_top_acc = round(layer_acc_t / len(os.listdir(img_dirs + "/top")), 2)
            food_top_acc = round(food_acc_t / len(os.listdir(img_dirs + "/top")), 2)
        sheet1.write(i + 2, 5, layer_top_acc)  # 上层烤层准确率写入
        sheet1.write(i + 2, 6, food_top_acc)  # 上层食材准确率写入
        # 其他层结果查看
        for file in tqdm(os.listdir(img_dirs + "/others")):  # 其他层
            if file.endswith("jpg"):
                all_jpgs += 1  # 统计总jpg图片数量
                image_path = img_dirs + "/others" + "/" + file
                bboxes_pr, layer_n ,food_class= Y.result(image_path, save_c_dir)  # 预测每一张结果并保存

                layer_img_true.append(3)  # 烤层真实结果
                layer_img_pre.append(layer_n)  # 烤层预测结果
                if layer_n != 3:  # 判断烤层是否为3
                    shutil.copy(image_path,
                                layer_error_c_dirs + "/" + file.split(".jpg")[0] + "_" + str(layer_n) + ".jpg")
                else:
                    layer_acc_o += 1  # 上层烤层正确+1
                    layer_acc += 1  # 烤层正确+1
                    c_layer_right_list.append(str(c) + "/" + file)  # 正确将名字写入c_layer_right_list中

                bboxes_pr, layer_n = correct_bboxes(bboxes_pr, layer_n)  # 矫正输出结果
                if len(bboxes_pr) == 0:  # 无任何结果返回，输出并统计+1
                    error_noresults += 1
                    shutil.copy(image_path, noresult_dir + "/" + file)
                else:
                    # pre = int(bboxes_pr[0][-1])
                    pre = int(food_class)
                    food_img_pre.append(pre)
                    food_img_true.append(classes_id[classes[i]])

                    if pre == classes_id[classes[i]]:  # 若结果正确，食材正确数+1
                        food_acc_o += 1
                        food_acc += 1
                        c_food_right_list.append(str(c) + "/" + file)  # 食材正确将名字写入c_food_right_list中
                    else:
                        right_label_o = he_foods(pre)
                        if right_label_o:  # 合并后结果正确
                            food_acc_o += 1
                            food_acc += 1
                            c_food_right_list.append(str(c) + "/" + file)  # 食材正确将名字写入c_food_right_list中
                        else:
                            drawed_img_save_to_path = str(image_path).split("/")[-1]
                            drawed_img_save_to_path = str(drawed_img_save_to_path).split(".")[0] + "_" + str(
                                layer_n) + ".jpg"  # 图片保存地址，烤层结果在命名中
                            shutil.copy(save_c_dir + "/" + drawed_img_save_to_path,
                                        fooderror_dirs + "/" + file.split(".jpg")[0] + "_" + str(layer_n) + "_" + str(
                                            pre) + ".jpg")
                            shutil.copy(image_path, fooderror_dirs + "/" + file.split(".jpg")[0] + ".jpg".format(
                                new_classes[pre]))

        if len(os.listdir(img_dirs + "/others")) == 0:  # 判断是否有值
            layer_others_acc = 0
            food_others_acc = 0
        else:
            layer_others_acc = round(layer_acc_o / len(os.listdir(img_dirs + "/others")), 2)
            food_others_acc = round(food_acc_o / len(os.listdir(img_dirs + "/others")), 2)
        sheet1.write(i + 2, 7, layer_others_acc)  # 烤层-其他，准确率写入
        sheet1.write(i + 2, 8, food_others_acc)  # 烤层-其他食材准确率写入

        sheet1.write(i + 2, 0, c)

        sheet1.write(i + 2, 9, all_jpgs)  # 写入正确总数
        sheet1.write(i + 2, 10, layer_acc)  # 写入烤层正确数
        sheet1.write(i + 2, 11, food_acc)  # 写入食材正确数

        sheet1.write(i + 2, 12, round((layer_acc / all_jpgs) * 100, 2))
        sheet1.write(i + 2, 13, round((food_acc / all_jpgs) * 100, 2))
        sheet1.write(i + 2, 14, error_noresults)

        # 烤层和烤盘均正确数量
        layer_and_food_right = set(c_food_right_list) & set(c_layer_right_list)
        sheet1.write(i + 2, 15, len(list(layer_and_food_right)))

        print("food name:", c)
        print("layer accuracy:", round((layer_acc / all_jpgs) * 100, 2))  # 输出烤层正确数
        jpgs_count_all += all_jpgs
        layer_jpgs_acc += layer_acc
        food_jpgs_acc += food_acc
    print("all layer accuracy:", round((layer_jpgs_acc / jpgs_count_all) * 100, 2))  # 输出烤层正确数
    print("all food accuracy:", round((food_jpgs_acc / jpgs_count_all) * 100, 2))  # 输出食材正确数

    layer_conf = confusion_matrix(y_pred=layer_img_pre, y_true=layer_img_true)
    food_conf = confusion_matrix(y_pred=food_img_pre, y_true=food_img_true,
                                 labels=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21,
                                         22, 23])

    sheet2 = workbook.add_sheet("food_confusion_matrix")
    for i in range(23):
        sheet2.write(i + 1, 0, classes[i])
        sheet2.write(0, i + 1, classes[i])
    for i in range(food_conf.shape[0]):
        for j in range(food_conf.shape[1]):
            sheet2.write(i + 1, j + 1, str(food_conf[i, j]))

    print(layer_conf)
    print(sum(sum(layer_conf)))
    print(food_conf)
    print(sum(sum(food_conf)))

    sheet1.write(45, 1, jpgs_count_all)
    sheet1.write(45, 2, layer_jpgs_acc)
    sheet1.write(45, 3, food_jpgs_acc)
    sheet1.write(45, 4, round((layer_jpgs_acc / jpgs_count_all) * 100, 2))
    sheet1.write(55, 5, round((food_jpgs_acc / jpgs_count_all) * 100, 2))

    workbook.save("E:/check_2_phase/all_he_{0}{1}.xls".format(mode, tag))

    end_time = time.time()
    print("all jpgs time:", end_time - end0_time)
