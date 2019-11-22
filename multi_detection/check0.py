# -*- encoding: utf-8 -*-

"""
@File    : check0.py
@Time    : 2019/11/11 10:21
@Author  : sunyihuan
"""

import cv2
import shutil
from tqdm import tqdm
import numpy as np
import tensorflow as tf
import multi_detection.core.utils as utils
from multi_detection.core.config import cfg
from sklearn.metrics import confusion_matrix

import os

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.07  # 占用GPU的显存


class YoloTest(object):
    def __init__(self):
        self.input_size = cfg.TEST.INPUT_SIZE
        self.anchor_per_scale = cfg.YOLO.ANCHOR_PER_SCALE
        self.classes = utils.read_class_names(cfg.YOLO.CLASSES)
        self.num_classes = len(self.classes)
        self.anchors = np.array(utils.get_anchors(cfg.YOLO.ANCHORS))
        self.score_threshold = cfg.TEST.SCORE_THRESHOLD
        self.iou_threshold = cfg.TEST.IOU_THRESHOLD
        self.moving_ave_decay = cfg.YOLO.MOVING_AVE_DECAY
        self.annotation_path = cfg.TEST.ANNOT_PATH
        self.weight_file = "E:/ckpt_dirs/Food_detection/multi_food/20191028/yolov3_train_loss=4.9485.ckpt-165"
        self.write_image = cfg.TEST.WRITE_IMAGE
        self.write_image_path = cfg.TEST.WRITE_IMAGE_PATH
        self.show_label = cfg.TEST.SHOW_LABEL

        self.count_nums = 0

        graph = tf.Graph()
        with graph.as_default():
            self.saver = tf.train.import_meta_graph("{}.meta".format(self.weight_file))
            self.sess = tf.Session(config=config)
            self.saver.restore(self.sess, self.weight_file)

            self.input = graph.get_tensor_by_name("define_input/input_data:0")
            self.trainable = graph.get_tensor_by_name("define_input/training:0")

            self.pred_sbbox = graph.get_tensor_by_name("define_loss/pred_sbbox/concat_2:0")
            self.pred_mbbox = graph.get_tensor_by_name("define_loss/pred_mbbox/concat_2:0")
            self.pred_lbbox = graph.get_tensor_by_name("define_loss/pred_lbbox/concat_2:0")

            self.layer_num = graph.get_tensor_by_name("define_loss/layer_classes:0")

    def predict(self, image):
        '''

        :param image: 图片数据
        :return:   std_result：食材结果[name, flag, num_label]
                  , layer_n:烤层结果，0：下层、1：中层、2：上层、3：其他
        '''
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

        bboxes = utils.postprocess_boxes(pred_bbox, (org_h, org_w), self.input_size, 0.45)
        bboxes = utils.nms(bboxes, self.iou_threshold)
        bboxes_pr = bboxes  # 检测框结果
        layer_n = layer_[0]  # 烤层结果
        return bboxes_pr, layer_n

    def evaluate(self):
        error_layer_dir = "./mAP/error_layer"  # 烤层错误图片地址
        food_name_dir = "./mAP/food_name_error"  # 食材名称错误图片地址
        noresult_dir = "./mAP/noresult"  # 食材名称错误图片地址
        result_1dir = "./mAP/result1"  # 食材判断仅为1个框

        if os.path.exists(self.write_image_path): shutil.rmtree(self.write_image_path)
        if os.path.exists(error_layer_dir): shutil.rmtree(error_layer_dir)
        if os.path.exists(food_name_dir): shutil.rmtree(food_name_dir)
        if os.path.exists(noresult_dir): shutil.rmtree(noresult_dir)
        if os.path.exists(result_1dir): shutil.rmtree(result_1dir)
        os.mkdir(self.write_image_path)
        os.mkdir(error_layer_dir)
        os.mkdir(food_name_dir)
        os.mkdir(noresult_dir)
        os.mkdir(result_1dir)

        layer_pre = []
        layer_true = []

        food_name_pre = []
        food_name_true = []

        error_noresults = 0  # 输出无结果的nums
        error_c = 0

        all_data = {}

        with open(self.annotation_path, 'r') as annotation_file:
            for line in tqdm(annotation_file):
                annotation = line.strip().split()
                image_path = annotation[0]

                img_layer_true = annotation[1]  # 获取标准layer
                layer_true.append(int(img_layer_true))  # 写入到layer_true

                image_name = image_path.split('/')[-1]
                image = cv2.imread(image_path)

                print('=> ground truth of %s:' % image_name)

                bboxes_pr, layer = self.predict(image)

                layer_pre.append(layer)  # 预测layer写入到layer_pre

                if self.write_image:  # 预测结果画图
                    image = utils.draw_bbox(image, bboxes_pr, show_label=self.show_label)
                    # drawed_img_save_to_path = os.path.join(self.write_image_path, image_name)
                    drawed_img_save_to_path = self.write_image_path + "/" + image_name

                    cv2.imwrite(drawed_img_save_to_path, image)

                if int(layer) != int(img_layer_true):  # 将layer错误的图片拷贝至error_layer
                    print("layer nums error:::::   ")
                    print(layer, img_layer_true)
                    shutil.copy(image_path,
                                error_layer_dir + "/" + image_name.split(".")[0] + "_" + str(layer) + ".jpg")

                if len(bboxes_pr) == 0:  # 无任何结果输出
                    print("no result:::")
                    error_noresults += 1
                    shutil.copy(self.write_image_path + "/" + image_name,
                                noresult_dir + "/" + image_name.split(".")[0] + ".jpg")


                else:
                    if len(bboxes_pr) == 1:
                        if bboxes_pr[0][4] < 0.9 and bboxes_pr[0][4] >= 0.45:
                            bboxes_pr[0][4] = 0.9
                        img_food_name_true = int(str(annotation[2]).split(",")[-1])  # 获取标准food_name
                        food_name_true.append(int(img_food_name_true))  # 写入到food_name_true

                        img_food_name_pre = int(bboxes_pr[0][-1])  # 食材类别结果
                        food_name_pre.append(img_food_name_pre)  # 预测结果写入到food_name_pre

                        if img_food_name_true != img_food_name_pre:  # 结果错误，图片拷贝至food_name_dir
                            error_c += 1  # 错误统计
                            shutil.copy(image_path,
                                        food_name_dir + "/" + image_name.split(".")[0] + "_" + str(
                                            img_food_name_pre) + ".jpg")

                    else:  # 预测cls不唯一
                        same_label = True
                        for i in range(len(bboxes_pr)):
                            if i == (len(bboxes_pr) - 1):
                                break
                            if bboxes_pr[i][5] == bboxes_pr[i + 1][5]:
                                continue
                            else:
                                same_label = False

                        sumProb = 0.
                        # 多个食材，同一标签
                        if same_label:
                            # for i in range(num_label):
                            #    sumProb += bboxes_pr[i][4]
                            # avrProb = sumProb/num_label
                            # bboxes_pr[0][4] = avrProb
                            bboxes_pr[0][4] = 0.98
                        # 多个食材，非同一标签
                        else:
                            problist = list(map(lambda x: x[4], bboxes_pr))
                            labellist = list(map(lambda x: x[5], bboxes_pr))

                            labeldict = {}
                            for key in labellist:
                                labeldict[key] = labeldict.get(key, 0) + 1
                                # 按同种食材label数量降序排列
                            s_labeldict = sorted(labeldict.items(), key=lambda x: x[1], reverse=True)

                            n_name = len(s_labeldict)
                            name1 = s_labeldict[0][0]
                            num_name1 = s_labeldict[0][1]

                            # 数量最多label对应的食材占比0.7以上
                            if num_name1 / len(bboxes_pr) > 0.7:
                                num_label0 = []
                                for i in range(len(bboxes_pr)):
                                    if name1 == bboxes_pr[i][5]:
                                        num_label0.append(bboxes_pr[i])
                                num_label0[0][4] = 0.95

                            # 按各个label的probability降序排序
                            else:
                                # 计数
                                self.count_nums += 1
                                bboxes_pr = sorted(bboxes_pr, key=lambda x: x[4], reverse=True)
                                for i in range(len(bboxes_pr)):
                                    bboxes_pr[i][4] = bboxes_pr[i][4] * 0.9
                                shutil.copy(self.write_image_path + "/" + image_name,
                                            result_1dir + "/" + image_name)

        # print(all_data)
        # for c in all_data.keys():
        #     print(c, ":  ", all_data[c][0], round(all_data[c][1] / all_data[c][0], 2))

        print("无任何结果数量：", error_noresults)
        print("错误数量：", error_c)

        layer_matrix = confusion_matrix(layer_true, layer_pre, labels=[0, 1, 2, 3])
        print("烤层检测混淆矩阵：")
        print(layer_matrix)

        food_name_matrix = confusion_matrix(food_name_true, food_name_pre)
        print("食材检测混淆矩阵：")
        print(food_name_matrix)


if __name__ == '__main__':
    import time

    s = time.time()
    Y = YoloTest()
    s_l_time = time.time()
    print("model load time:", s_l_time - s)
    Y.evaluate()
    e = time.time()
    print("predict all time::::", e - s_l_time)

    print(Y.count_nums)
