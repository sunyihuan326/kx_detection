#! /usr/bin/env python
# coding=utf-8

import cv2
import numpy as np
import multi_detection.core.utils as utils
import tensorflow as tf
from PIL import Image

return_elements = ["input/input_data:0", "pred_sbbox/concat_2:0", "pred_mbbox/concat_2:0", "pred_lbbox/concat_2:0"]
pb_file = "./yolov3_coco.pb"
image_path ="E:/DataSets/KX_FOODSets_model_data/23classes_0808_test/JPGImages/8_Toast.jpg"
num_classes = 23
input_size = 416
graph = tf.Graph()

original_image = cv2.imread(image_path)
original_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)
original_image_size = original_image.shape[:2]
image_data = utils.image_preporcess(np.copy(original_image), [input_size, input_size])
image_data = image_data[np.newaxis, ...]

return_tensors = utils.read_pb_return_tensors(graph, pb_file, return_elements)
print(return_tensors)

with tf.Session(graph=graph) as sess:
    pred_sbbox, pred_mbbox, pred_lbbox = sess.run(
        ['import/pred_sbbox/concat_2:0', 'import/pred_mbbox/concat_2:0', 'import/pred_lbbox/concat_2:0'],
        feed_dict={"import/input/input_data:0": image_data})

pred_bbox = np.concatenate([np.reshape(pred_sbbox, (-1, 5 + num_classes)),
                            np.reshape(pred_mbbox, (-1, 5 + num_classes)),
                            np.reshape(pred_lbbox, (-1, 5 + num_classes))], axis=0)
bboxes = utils.postprocess_boxes(pred_bbox, original_image_size, input_size, 0.45)

bboxes = utils.nms(bboxes, 0.5, method='nms')
image = utils.draw_bbox(original_image, bboxes)
image = Image.fromarray(image)
image.save(image_path.split("/")[-1])
# image.show()
