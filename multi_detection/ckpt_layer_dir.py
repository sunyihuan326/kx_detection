# -*- coding: utf-8 -*-
# @Time    : 2020/11/19
# @Author  : sunyihuan
# @File    : ckpt_layer_dir.py
'''
将文件夹下按烤层识别结果，分为对应的文件夹

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
        self.weight_file = "E:/ckpt_dirs/Food_detection/multi_food5/20210129/yolov3_train_loss=5.9575.ckpt-151"  # ckpt文件地址
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

            # 输出烤层结果
            self.layer_num = graph.get_tensor_by_name("define_loss/layer_classes:0")

    def predict(self, image):
        image = cv2.imread(image)  # 图片读取
        org_image = np.copy(image)
        org_h, org_w, _ = org_image.shape

        image_data = utils.image_preporcess(image, [self.input_size, self.input_size])
        image_data = image_data[np.newaxis, ...]

        layer_ = self.sess.run(
            [self.layer_num],
            feed_dict={
                self.input: image_data,
                self.trainable: False
            }
        )

        layer_n = layer_  # 烤层结果

        return layer_n


if __name__ == '__main__':
    start_time = time.time()

    img_root = "F:/serve_data/202101-03formodel/exrtact_file/layer_data"  # 图片文件地址
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
    cls = os.listdir(img_root)
    layer_name = {"0": "bottom", "1": "middle", "2": "top", "3": "others"}
    for c in cls:
        img_dir = img_root + "/" + c
        for img in tqdm(os.listdir(img_dir)):
            if img.endswith("jpg"):
                img_path = img_dir + "/" + img

                end_time1 = time.time()
                layer_ = Y.predict(img_path)
                layer_r = str(layer_[0][0])
                # print(layer_name[layer_r])
                if not os.path.exists(img_dir + "/" + layer_name[layer_r]):os.mkdir(img_dir + "/" + layer_name[layer_r])

                shutil.move(img_path, img_dir + "/" + layer_name[layer_r] + "/" + img)
