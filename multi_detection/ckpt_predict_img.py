# -*- encoding: utf-8 -*-

"""
查看分类前的特征图

@File    : ckpt_predict.py
@Time    : 2019/8/19 15:45
@Author  : sunyihuan
"""

import cv2
import numpy as np
import tensorflow as tf
import multi_detection.core.utils as utils
import matplotlib.pyplot as plt

class YoloPredict(object):
    '''
    预测结果
    '''

    def __init__(self):
        self.input_size = 416  # 输入图片尺寸（默认正方形）
        self.num_classes = 46  # 种类数
        self.score_threshold = 0.1
        self.iou_threshold = 0.5
        self.weight_file = "E:/multi_yolov3_predict-20191220/checkpoint/yolov3_1220.ckpt"  # ckpt文件地址
        self.write_image = True  # 是否画图
        self.show_label = True  # 是否显示标签

        graph = tf.Graph()
        with graph.as_default():
            self.saver = tf.train.import_meta_graph("{}.meta".format(self.weight_file))
            self.sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True))
            self.saver.restore(self.sess, self.weight_file)

            self.input = graph.get_tensor_by_name("define_input/input_data:0")
            self.trainable = graph.get_tensor_by_name("define_input/training:0")

            self.pred_sbbox = graph.get_tensor_by_name("define_loss/conv_sobj_branch/Relu:0")
            self.pred_mbbox = graph.get_tensor_by_name("define_loss/conv_mobj_branch/Relu:0")
            self.pred_lbbox = graph.get_tensor_by_name("define_loss/conv_lobj_branch/Relu:0")

            self.layer_num = graph.get_tensor_by_name("define_loss/layer_classes:0")

    def predict(self, image_path):
        image = cv2.imread(image_path)  # 图片读取
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
        print(pred_sbbox.shape)
        print(pred_sbbox[0,:,:,1])
        plt.figure(1)
        ax1 = plt.subplot(2, 2, 1)
        plt.sca(ax1)
        plt.imshow(np.array(pred_sbbox[0,:,:,-3]),cmap=plt.cm.gray)   #画图！！！！！！！！！！！！！！
        ax2 = plt.subplot(2, 2, 2)
        plt.sca(ax2)
        plt.imshow(np.array(pred_sbbox[0, :, :,-2]), cmap=plt.cm.gray)
        # plt.imshow(np.array(pred_sbbox[0, :, :, 6]), cmap=plt.cm.gray)
        # plt.show()
        ax3 = plt.subplot(2,2, 3)
        plt.sca(ax3)
        plt.imshow(np.array(pred_sbbox[0, :, :, -1]), cmap=plt.cm.gray)
        plt.show()

        # pred_bbox = np.concatenate([np.reshape(pred_sbbox, (-1, 5 + self.num_classes)),
        #                             np.reshape(pred_mbbox, (-1, 5 + self.num_classes)),
        #                             np.reshape(pred_lbbox, (-1, 5 + self.num_classes))], axis=0)
        #
        # bboxes = utils.postprocess_boxes(pred_bbox, (org_h, org_w), self.input_size, self.score_threshold)
        # bboxes = utils.nms(bboxes, self.iou_threshold)


if __name__ == '__main__':
    import time

    start_time = time.time()
    img_path = "C:/Users/sunyihuan/Desktop/0518/11/20200518_064441746.jpg" # 图片地址
    Y = YoloPredict()
    end_time0 = time.time()

    print("model loading time:", end_time0 - start_time)
    Y.predict(img_path)
    end_time1 = time.time()
    print("predict time:", end_time1 - end_time0)
