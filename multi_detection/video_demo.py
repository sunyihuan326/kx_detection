#! /usr/bin/env python
# coding=utf-8
'''
视频预测demo

'''
import cv2
import time
import numpy as np
import multi_detection.core.utils as utils
import tensorflow as tf
from PIL import Image

return_elements = ["define_input/input_data:0", "define_input/training:0",
                   "define_loss/pred_sbbox/concat_2:0", "define_loss/pred_mbbox/concat_2:0",
                   "define_loss/pred_lbbox/concat_2:0"]
pb_file = "E:/ckpt_dirs/Food_detection/multi_food2/20200507/yolo_model.pb"
video_path = "E:/kx_detection/multi_detection/docs/images/eggtart.mp4"
# video_path      = 0
num_classes = 46
input_size = 416
graph = tf.Graph()
return_tensors = utils.read_pb_return_tensors(graph, pb_file, return_elements)

with tf.Session(graph=graph) as sess:
    vid = cv2.VideoCapture(video_path)
    while True:
        return_value, frame = vid.read()
        if return_value:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image = Image.fromarray(frame)
        else:
            raise ValueError("No image!")
        frame_size = frame.shape[:2]
        image_data = utils.image_preporcess(np.copy(frame), [input_size, input_size])
        image_data = image_data[np.newaxis, ...]
        prev_time = time.time()

        pred_sbbox, pred_mbbox, pred_lbbox = sess.run(
            [return_tensors[2], return_tensors[3], return_tensors[4]],
            feed_dict={return_tensors[0]: image_data,
                       return_tensors[1]: False})

        pred_bbox = np.concatenate([np.reshape(pred_sbbox, (-1, 5 + num_classes)),
                                    np.reshape(pred_mbbox, (-1, 5 + num_classes)),
                                    np.reshape(pred_lbbox, (-1, 5 + num_classes))], axis=0)

        bboxes = utils.postprocess_boxes(pred_bbox, frame_size, input_size, 0.3)
        bboxes = utils.nms(bboxes, 0.45, method='nms')
        image_ = utils.draw_bbox(frame, bboxes)

        curr_time = time.time()
        exec_time = curr_time - prev_time
        result = np.asarray(image_)
        info = "time: %.2f ms" % (1000 * exec_time)
        cv2.namedWindow("result", cv2.WINDOW_AUTOSIZE)
        result_ = cv2.cvtColor(image_, cv2.COLOR_RGB2BGR)
        cv2.imshow("result", result_)

        key = cv2.waitKey(1) & 0xFF
        if key == ord(" "):  # 空格键暂停
            cv2.waitKey(0)
        if key == ord("q"):
            break
