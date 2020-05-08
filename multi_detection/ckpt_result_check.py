# -*- encoding: utf-8 -*-

"""
test集、val集直接查看矫正后的结果

@File    : ckpt_result_check.py
@Time    : 2019/11/6 10:52
@Author  : sunyihuan
"""

import cv2
import os
import shutil
from tqdm import tqdm
import numpy as np
import tensorflow as tf
import multi_detection.core.utils as utils
from multi_detection.core.config import cfg
from sklearn.metrics import confusion_matrix


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
        self.weight_file = "E:/ckpt_dirs/Food_detection/multi_food/20191108/yolov3_train_loss=5.2653.ckpt-93"
        self.write_image = cfg.TEST.WRITE_IMAGE
        self.write_image_path = cfg.TEST.WRITE_IMAGE_PATH
        self.show_label = cfg.TEST.SHOW_LABEL

        graph = tf.Graph()
        with graph.as_default():
            self.saver = tf.train.import_meta_graph("{}.meta".format(self.weight_file))
            self.sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True))
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

        num_label = len(bboxes_pr)
        # 未检测食材
        if num_label == 0:
            return bboxes_pr, layer_n

        # 检测到一个食材
        elif num_label == 1:
            return bboxes_pr, layer_n

        # 检测到多个食材
        else:
            same_label = True
            for i in range(num_label):
                if i == (num_label - 1):
                    break
                if bboxes_pr[i][5] == bboxes_pr[i + 1][5]:
                    continue
                else:
                    same_label = False
            # 多个食材，同一标签
            if same_label:
                sumProb = 0.
                for i in range(num_label):
                    sumProb += bboxes_pr[i][4]
                avrProb = sumProb / num_label
                bboxes_pr[0][4] = avrProb
                return bboxes_pr, layer_n
            # 多个食材，非同一标签
            else:
                sumProb = 0.
                problist = list(map(lambda x: x[4], bboxes_pr))
                labellist = list(map(lambda x: x[5], bboxes_pr))

                labeldict = {}
                for key in labellist:
                    labeldict[key] = labeldict.get(key, 0) + 1
                    # 按同种食材label数量降序排列
                s_labeldict = sorted(labeldict.items(), key=lambda x: x[1], reverse=True)

                # n_name = len(s_labeldict)
                name1 = s_labeldict[0][0]
                num_name1 = s_labeldict[0][1]

                # 数量最多label对应的食材占比0.7以上
                if num_name1 / num_label > 0.7:
                    num_label0 = []
                    for i in range(num_label):
                        if name1 == bboxes_pr[i][5]:
                            num_label0.append(bboxes_pr[i])
                    for i in range(len(num_label0)):
                        sumProb += num_label0[i][4]
                    avrProb = sumProb / num_label
                    num_label0[0][4] = avrProb
                    return num_label0, layer_n

                # 按各个label的probability降序排序
                else:
                    bboxes_pr = sorted(bboxes_pr, key=lambda x: x[4], reverse=True)
                    return bboxes_pr, layer_n

    def evaluate(self):
        # predicted_dir_path = './mAP/predicted'
        # ground_truth_dir_path = './mAP/ground-truth'
        error_layer_dir = "./mAP/error_layer"  # 烤层错误图片地址
        food_name_dir = "./mAP/food_name_error"  # 食材名称错误图片地址
        noresult_dir = "./mAP/noresult"  # 食材名称错误图片地址
        # flag3_error = "./mAP/flag3"  # flag=3文件拷贝地址
        # if os.path.exists(predicted_dir_path): shutil.rmtree(predicted_dir_path)
        # if os.path.exists(ground_truth_dir_path): shutil.rmtree(ground_truth_dir_path)
        if os.path.exists(self.write_image_path): shutil.rmtree(self.write_image_path)
        if os.path.exists(error_layer_dir): shutil.rmtree(error_layer_dir)
        if os.path.exists(food_name_dir): shutil.rmtree(food_name_dir)
        if os.path.exists(noresult_dir): shutil.rmtree(noresult_dir)
        # if os.path.exists(flag3_error): shutil.rmtree(flag3_error)
        # os.mkdir(predicted_dir_path)
        # os.mkdir(ground_truth_dir_path)
        os.mkdir(self.write_image_path)
        os.mkdir(error_layer_dir)
        os.mkdir(food_name_dir)
        os.mkdir(noresult_dir)
        # os.mkdir(flag3_error)

        layer_pre = []
        layer_true = []

        food_name_pre = []
        food_name_true = []

        error_noresults = 0  # 输出无结果的nums
        error_c = 0

        with open(self.annotation_path, 'r') as annotation_file:
            for line in tqdm(annotation_file):
                annotation = line.strip().split()
                image_path = annotation[0]

                img_layer_true = annotation[1]  # 获取标准layer
                layer_true.append(int(img_layer_true))  # 写入到layer_true

                image_name = image_path.split('/')[-1]
                image = cv2.imread(image_path)
                # image = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)   # RGB空间转为HSV空间
                # bbox_data_gt = np.array([list(map(int, box.split(','))) for box in annotation[2:]])
                #
                # if len(bbox_data_gt) == 0:
                #     bboxes_gt = []
                #     classes_gt = []
                # else:
                #     bboxes_gt, classes_gt = bbox_data_gt[:, :4], bbox_data_gt[:, 4]
                # ground_truth_path = os.path.join(ground_truth_dir_path, image_name.split(".")[0] + '.txt')
                #
                print('=> ground truth of %s:' % image_name)
                # num_bbox_gt = len(bboxes_gt)
                # with open(ground_truth_path, 'w') as f:
                #     for i in range(num_bbox_gt):
                #         class_name = self.classes[classes_gt[i]]
                #         xmin, ymin, xmax, ymax = list(map(str, bboxes_gt[i]))
                #         bbox_mess = ' '.join([class_name, xmin, ymin, xmax, ymax]) + '\n'
                #         f.write(bbox_mess)
                #         print('\t' + str(bbox_mess).strip())
                # print('=> predict result of %s:' % image_name)
                bboxes_pr, layer = self.predict(image)

                layer_pre.append(layer)  # 预测layer写入到layer_pre

                if self.write_image:
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
                    shutil.copy(image_path,
                                noresult_dir + "/" + image_name.split(".")[0] + ".jpg")

                else:
                    img_food_name_true = int(str(annotation[2]).split(",")[-1])  # 获取标准food_name
                    food_name_true.append(int(img_food_name_true))  # 写入到food_name_true

                    img_food_name_pre = int(bboxes_pr[0][-1])  # 食材类别结果
                    food_name_pre.append(img_food_name_pre)  # 预测结果写入到food_name_pre

                    if img_food_name_true != img_food_name_pre:  # 结果错误，图片拷贝至food_name_dir
                        error_c += 1  # 错误统计
                        shutil.copy(image_path,
                                    food_name_dir + "/" + image_name.split(".")[0] + "_" + str(
                                        img_food_name_pre) + ".jpg")
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
