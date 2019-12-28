# -*- encoding: utf-8 -*-

"""
@File    : ckpt_he_for_orignal.py
@Time    : 2019/12/16 16:16
@Author  : sunyihuan
"""

'''
for normal
仅输出各类烤层结果、食材结果

ckpt文件预测某一文件夹下所有图片结果
并输出食材类别准确率结果
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
        self.score_threshold = 0.45
        self.iou_threshold = 0.5
        self.weight_file = "E:/ckpt_dirs/Food_detection/local/20191216/yolov3_train_loss=4.7698.ckpt-80"   # ckpt文件地址
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
    img_dir = "E:/test_from_ye/JPGImages_abnormal"  # 文件夹地址
    save_dir = "E:/test_from_ye/detection_local_abnormal1216"  # 预测结果标出保存地址
    if not os.path.exists(save_dir): os.mkdir(save_dir)
    Y = YoloTest()  # 加载模型

    food_error_dir = "E:/test_from_ye/fooderror_local_abnormal1216"  # 预测结果错误保存地址
    if not os.path.exists(food_error_dir): os.mkdir(food_error_dir)

    noresult_dir = "E:/test_from_ye/noresult_local_abnormal1216"
    if not os.path.exists(noresult_dir): os.mkdir(noresult_dir)

    classes = ["Beefsteak", "CartoonCookies", "Cookies", "CupCake", "Pizzafour",
               "Pizzatwo", "Pizzaone", "Pizzasix", "ChickenWings", "ChiffonCake6",
               "ChiffonCake8", "CranberryCookies", "eggtarts", "eggtartl", "nofood",
               "Peanuts", "PorkChops", "PotatoCut", "Potatol", "Potatom",
               "Potatos", "RoastedChicken", "SweetPotatoCut", "SweetPotatol", "SweetPotatom",
               "SweetPotatoS", "Toast"]
    ab_classes = ["Pizzafour", "Pizzatwo", "Pizzaone", "Pizzasix",
                  "PotatoCut", "Potatol", "Potatom",
                  "RoastedChicken",
                  "SweetPotatoCut", "SweetPotatol", "SweetPotatom", "SweetPotatoS",
                  "Toast"]
    # classes = ["potatol", "potatom", "sweetpotatom", "sweetpotatol"]
    # classes = ["potatol", "potatom", "sweetpotatom"]
    # classes = ["nofood"]
    # classes_id = {"roast_white": 25}

    classes_id = {"CartoonCookies": 1, "Cookies": 5, "CupCake": 7, "Beefsteak": 0, "ChickenWings": 2,
                  "ChiffonCake6": 3, "ChiffonCake8": 4, "CranberryCookies": 6, "eggtarts": 8, "eggtartl": 9,
                  "nofood": 10, "Peanuts": 11, "PorkChops": 16, "PotatoCut": 17, "Potatol": 18,
                  "Potatom": 19, "Potatos": 20, "SweetPotatoCut": 21, "SweetPotatol": 22, "SweetPotatom": 23,
                  "Pizzafour": 12, "Pizzaone": 13, "Pizzasix": 14, "RoastedChicken": 25,
                  "Pizzatwo": 15, "SweetPotatoS": 24, "Toast": 26}
    jpgs_count_all = 0
    jpgs_acc = 0
    all_noresults = 0

    workbook = xlwt.Workbook(encoding='utf-8')
    sheet1 = workbook.add_sheet("all_he_orignal")
    sheet1.write(0, 0, "classes")
    sheet1.write(0, 1, "food_right_nums")
    sheet1.write(0, 2, "jpgs_all")
    sheet1.write(0, 3, "food_acc")
    sheet1.write(0, 4, "noresult")

    img_true = []
    img_pre = []
    for i in range(len(ab_classes)):
        c = ab_classes[i]
        error_noresults = 0  # 无任何结果统计
        food_acc = 0  # 食材准确数统计
        all_jpgs = 0  # 图片总数统计
        img_dirs = img_dir + "/" + c
        save_dirs = save_dir + "/" + c
        if os.path.exists(save_dirs): shutil.rmtree(save_dirs)
        os.mkdir(save_dirs)

        fooderror_dirs = food_error_dir + "/" + c
        if os.path.exists(fooderror_dirs): shutil.rmtree(fooderror_dirs)
        os.mkdir(fooderror_dirs)

        noresult_c_dirs = noresult_dir + "/" + c
        if os.path.exists(noresult_c_dirs): shutil.rmtree(noresult_c_dirs)
        os.mkdir(noresult_c_dirs)

        for file in tqdm(os.listdir(img_dirs)):
            if file.endswith("jpg"):
                all_jpgs += 1  # 统计总jpg图片数量
                image_path = img_dirs + "/" + file
                bboxes_pr, layer_n = Y.result(image_path, save_dirs)  # 预测每一张结果并保存
                # try:
                #     bboxes_pr, layer_n = Y.result(image_path, save_dirs)  # 预测每一张结果并保存
                # except:
                #     pass
                bboxes_pr, layer_n = correct_bboxes(bboxes_pr, layer_n)  # 矫正输出结果

                drawed_img_save_to_path = str(image_path).split("/")[-1]
                drawed_img_save_to_path = str(drawed_img_save_to_path).split(".")[0] + "_" + str(
                    layer_n) + ".jpg"  # 图片保存地址，烤层结果在命名中

                if len(bboxes_pr) == 0:  # 无任何结果返回，输出并统计+1
                    error_noresults += 1
                    shutil.copy(save_dirs + "/" + drawed_img_save_to_path, noresult_c_dirs + "/" + file)
                else:
                    # bboxes_pr, layer_n = correct_bboxes(bboxes_pr, layer_n)  # 矫正输出结果
                    pre = bboxes_pr[0][-1]
                    img_pre.append(pre)
                    img_true.append(classes_id[c])

                    if pre == classes_id[c]:  # 若结果正确，食材正确数+1
                        food_acc += 1
                    else:
                        if pre in [8, 9] and classes_id[classes[i]] in [8, 9]:
                            food_acc += 1
                        if pre in [12, 14] and classes_id[classes[i]] in [12, 14]:
                            food_acc += 1
                        else:

                            shutil.copy(save_dirs + "/" + drawed_img_save_to_path,
                                        fooderror_dirs + "/" + file.split(".jpg")[0] + "_" + str(layer_n) + ".jpg")
                    # else:
                    #     print(pre)
        sheet1.write(i + 1, 0, c)

        sheet1.write(i + 1, 1, food_acc)
        sheet1.write(i + 1, 2, all_jpgs)
        sheet1.write(i + 1, 3, round((food_acc / all_jpgs) * 100, 2))
        sheet1.write(i + 1, 4, error_noresults)

        print("food name:", c)
        print("food accuracy:", round((food_acc / all_jpgs) * 100, 2))  # 输出食材正确数
        print("no result:", error_noresults)  # 输出无任何结果总数
        jpgs_count_all += all_jpgs
        jpgs_acc += food_acc
        all_noresults += error_noresults
    print("all food accuracy:", round((jpgs_acc / jpgs_count_all) * 100, 2))  # 输出食材正确数
    print("all no result:", all_noresults)  # 输出无任何结果总数

    conf = confusion_matrix(y_pred=img_pre, y_true=img_true)

    print(conf)
    print(sum(sum(conf)))
    sheet1.write(35, 1, jpgs_acc)
    sheet1.write(35, 2, jpgs_count_all)
    sheet1.write(35, 3, round((jpgs_acc / jpgs_count_all) * 100, 2))
    sheet1.write(35, 4, all_noresults)

    workbook.save("E:/test_from_ye/multi_he_local_abnormal1216.xls")
