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
gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.2)
config = tf.ConfigProto(gpu_options=gpu_options)


class YoloPredict(object):
    '''
    预测结果
    '''

    def __init__(self):
        self.input_size = 320  # 输入图片尺寸（默认正方形）
        self.num_classes = 22  # 种类数
        self.score_threshold = 0.45
        self.iou_threshold = 0.5
        self.weight_file = "E:/ckpt_dirs/Food_detection/multi_food3/20200604_22class/yolov3_train_loss=4.9799.ckpt-158"  # ckpt文件地址
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

        return bboxes, layer_n

    def result(self, image_path, save_dir="F:/20200720_data_test_detect"):
        image = cv2.imread(image_path)  # 图片读取
        # image = utils.white_balance(image)  # 图片白平衡处理
        bboxes_pr, layer_n = self.predict(image)  # 预测结果
        if not os.path.exists(save_dir): os.mkdir(save_dir)
        if self.write_image:
            image = utils.draw_bbox(image, bboxes_pr, show_label=self.show_label)
            drawed_img_save_to_path = str(image_path).split("/")[-1]
            drawed_img_save_to_path = save_dir + "/" + drawed_img_save_to_path.split(".jpg")[0] + "_" + str(
                layer_n[0]) + ".jpg"
            # print(drawed_img_save_to_path)
            cv2.imwrite(drawed_img_save_to_path, image)

        return bboxes_pr, layer_n


if __name__ == '__main__':
    start_time = time.time()
    img_root = "F:/20200720_data_test"  # 图片文件地址
    Y = YoloPredict()
    end_time0 = time.time()
    print("model loading time:", end_time0 - start_time)

    for c in ["qifeng", "zhibei", "eggtart", "peanuts", "toast",]:
        img_dir = img_root + "/" + c
        classes = ["beefsteak", "cartooncookies", "chickenwings", "chiffoncake6", "cookies",
                   "cranberrycookies", "cupcake", "eggtart", "nofood", "peanuts",
                   "pizzacut", "pizzaone", "pizzatwo", "porkchops", "potatocut",
                   "potatol", "potatos", "sweetpotatocut", "sweetpotatol", "sweetpotatos",
                   "roastedchicken", "toast", "potatom", "sweetpotatom", "chiffoncake8"]
        for img in tqdm(os.listdir(img_dir)):
            if img.endswith("jpg"):
                img_path = img_dir + "/" + img
                end_time1 = time.time()
                bboxes_p, layer_ = Y.result(img_path)
                bboxes_pr, layer_n = correct_bboxes(bboxes_p, layer_)  # 矫正输出结果
                bboxes_pr, layer_n = get_potatoml(bboxes_pr, layer_n)  # 根据输出结果对中大红薯，中大土豆做输出
                print(bboxes_pr)
                if len(bboxes_pr) == 0:
                    if not os.path.exists(img_dir + "/noresult"): os.mkdir(img_dir + "/noresult")
                    shutil.move(img_path, img_dir + "/noresult" + "/" + img)
                else:
                    pre = int(bboxes_pr[0][-1])
                    if not os.path.exists(img_dir + "/" + classes[pre]): os.mkdir(img_dir + "/" + classes[pre])
                    shutil.move(img_path, img_dir + "/" + classes[pre] + "/" + img)
                # if pre==3:
                #     if not os.path.exists(img_dir + "/" +"chiffoncake6"): os.mkdir(img_dir + "/" + "chiffoncake6")
                #     shutil.move(img_path, img_dir + "/" + "chiffoncake6"+ "/" + img)
                # else:
                #     if not os.path.exists(img_dir + "/" +"chiffoncake8"): os.mkdir(img_dir + "/" + "chiffoncake8")
                #     shutil.move(img_path, img_dir + "/" + "chiffoncake8"+ "/" + img)
        # try:
        #     img_path = img_dir + "/" + img
        #     end_time1 = time.time()
        #     bboxes_p, layer_ = Y.result(img_path)
        #     bboxes_pr, layer_n = correct_bboxes(bboxes_p, layer_)  # 矫正输出结果
        #     print(bboxes_pr)
        #     if len(bboxes_pr)==0:
        #         if not os.path.exists(img_dir+"/noresult"):os.mkdir(img_dir+"/noresult")
        #         shutil.copy(img_path,img_dir+"/noresult"+"/"+img)
        #     else:
        #         pre = bboxes_pr[0][-1]
        #         if not os.path.exists(img_dir+"/"+classes[pre]):os.mkdir(img_dir+"/"+classes[pre])
        #         shutil.copy(img_path, img_dir +"/"+classes[pre]+"/"+img)
        # except:
        #     print(img)
