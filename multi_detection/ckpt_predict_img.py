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
import matplotlib
import os


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
        self.write_image = True  # 是否画图
        self.show_label = True  # 是否显示标签

        graph = tf.Graph()
        with graph.as_default():
            self.saver = tf.train.import_meta_graph("{}.meta".format(self.weight_file))
            self.sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True))
            self.saver.restore(self.sess, self.weight_file)

            self.input = graph.get_tensor_by_name("define_input/input_data:0")
            self.trainable = graph.get_tensor_by_name("define_input/training:0")

            # self.pred_sbbox = graph.get_tensor_by_name("define_loss/conv_sobj_branch/Relu:0")
            # self.pred_mbbox = graph.get_tensor_by_name("define_loss/conv_mobj_branch/Relu:0")
            # self.pred_lbbox = graph.get_tensor_by_name("define_loss/conv_lobj_branch/Relu:0")

            self.pred_sbbox = graph.get_tensor_by_name("define_loss/pred_sbbox/concat_2:0")
            self.pred_mbbox = graph.get_tensor_by_name("define_loss/pred_mbbox/concat_2:0")
            self.pred_lbbox = graph.get_tensor_by_name("define_loss/pred_lbbox/concat_2:0")

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

        print("pred_bbox.shape::::::", pred_sbbox.shape)
        # pred_sbbox=np.sum(np.array(pred_sbbox),axis=-1)
        # print(pred_sbbox[0, :, :])
        # matplotlib.image.imsave("E:/kx_detection/multi_detection/pre_feature/sbbox/one_all.jpg",
        #                                                     np.array(pred_sbbox[0, :, :]))
        # print(pred_sbbox.shape)
        # print(pred_sbbox.shape[-1])
        # for z in range(pred_sbbox.shape[-1]):
        #     print(z)
        #     plt.imshow(np.array(pred_sbbox[0, :, :, z - 1]))
        #     matplotlib.image.imsave("E:/kx_detection/multi_detection/pre_feature/sbbox/{}.jpg".format(z),
        #                             np.array(pred_sbbox[0, :, :, z - 1]))
        #
        # for z in range(pred_mbbox.shape[-1]):
        #     print(z)
        #     plt.imshow(np.array(pred_mbbox[0, :, :, z - 1]))
        #     matplotlib.image.imsave("E:/kx_detection/multi_detection/pre_feature/mbbox/{}.jpg".format(z),
        #                             np.array(pred_mbbox[0, :, :, z - 1]))
        # for z in range(pred_lbbox.shape[-1]):
        #     print(z)
        #     plt.imshow(np.array(pred_lbbox[0, :, :, z - 1]))
        #     matplotlib.image.imsave("E:/kx_detection/multi_detection/pre_feature/lbbox/{}.jpg".format(z),
        #                             np.array(pred_lbbox[0, :, :, z - 1]))

        # pred_bbox = np.concatenate([np.reshape(pred_sbbox, (-1, 5 + self.num_classes)),
        #                             np.reshape(pred_mbbox, (-1, 5 + self.num_classes)),
        #                             np.reshape(pred_lbbox, (-1, 5 + self.num_classes))], axis=0)
        #
        # bboxes = utils.postprocess_boxes(pred_bbox, (org_h, org_w), self.input_size, self.score_threshold)
        # bboxes = utils.nms(bboxes, self.iou_threshold)

    def show_in_one(self, images, save_jpg_name, show_size=(640, 640), blank_size=1, window_name="merge"):
        small_h, small_w = images[0].shape[:2]
        column = int(show_size[1] / (small_w + blank_size))
        row = int(show_size[0] / (small_h + blank_size))
        shape = [show_size[0], show_size[1]]
        for i in range(2, len(images[0].shape)):
            shape.append(images[0].shape[i])

        merge_img = np.zeros(tuple(shape), images[0].dtype)

        max_count = len(images)
        count = 0
        for i in range(row):
            if count >= max_count:
                break
            for j in range(column):
                if count < max_count:
                    im = images[count]
                    t_h_start = i * (small_h + blank_size)
                    t_w_start = j * (small_w + blank_size)
                    t_h_end = t_h_start + im.shape[0]
                    t_w_end = t_w_start + im.shape[1]
                    merge_img[t_h_start:t_h_end, t_w_start:t_w_end] = im
                    count = count + 1
                else:
                    break
        if count < max_count:
            print("ingnore count %s" % (max_count - count))
        cv2.namedWindow(window_name)
        cv2.imshow(window_name, merge_img)
        cv2.imwrite("E:/kx_detection/multi_detection/pre_feature/{}.jpg".format(save_jpg_name), merge_img)


if __name__ == '__main__':
    import time

    start_time = time.time()
    img_path = "C:/Users/sunyihuan/Desktop/test_all/20200623143818.jpg"  # 图片地址
    Y = YoloPredict()
    end_time0 = time.time()

    print("model loading time:", end_time0 - start_time)
    Y.predict(img_path)
    end_time1 = time.time()
    print("predict time:", end_time1 - end_time0)
    path="E:/kx_detection/multi_detection/pre_feature"

    # show_sizes={"sbbox":(656,656),"mbbox":(462,462),"lbbox":(352,352)}
    # for p in ["sbbox","mbbox","lbbox"]:
    #     debug_images = []
    #     for infile in os.listdir(path+"/"+p):
    #         infile = path + "/"+p+"/" + infile
    #         ext = os.path.splitext(infile)[1][1:]  # get the filename extenstion
    #         if ext == "png" or ext == "jpg" or ext == "bmp" or ext == "tiff" or ext == "pbm":
    #             print(infile)
    #             img = cv2.imread(infile)
    #             if img is None:
    #                 continue
    #             else:
    #                 debug_images.append(img)
    #     print(debug_images)
        # Y.show_in_one(debug_images,p,show_sizes[p])
