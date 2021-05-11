# -*- coding: utf-8 -*-
# @Time    : 2021/4/16
# @Author  : sunyihuan
# @File    : tfrecod_data_generate.py
import tensorflow as tf
import os
import numpy as np
from PIL import Image
import sys


def get_files(txt_data):
    image_list = []
    layer_list = []
    bboxes_list = []
    txt_list = open(txt_data, "r").readlines()
    for t in txt_list:
        name = t.strip().split(" ")[0]
        image_list.append(name)
        layer_list.append(int(t.strip().split(" ")[1]))
        bbx = [b for b in t.strip().split(" ")[2:]]
        print(bbx)
        bboxes_list.append(bbx)
    return image_list, layer_list, bboxes_list


def int64_feature(values):
    if not isinstance(values, (tuple, list)):
        values = [values]
    return tf.train.Feature(int64_list=tf.train.Int64List(value=values))


def bytes_feature(values):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[values]))


def float_feature(value):
    return tf.train.Feature(float_list=tf.train.FloatList(value=value))


def image_to_tfexample(image_data, layer, bboxes):
    return tf.train.Example(features=tf.train.Features(feature={
        'image': bytes_feature(image_data),
        'layer': int64_feature(layer),
        "bboxes": float_feature(bboxes)
    }))


def _convert_dataset(image_list, layer_list, bboxes_list, tfrecord_dir):
    """ Convert data to TFRecord format. """
    with tf.Graph().as_default():
        with tf.Session() as sess:
            if not os.path.exists(tfrecord_dir):
                os.makedirs(tfrecord_dir)
            output_filename = os.path.join(tfrecord_dir, "train.tfrecord")
            tfrecord_writer = tf.python_io.TFRecordWriter(output_filename)
            length = len(image_list)
            for i in range(length):  # 图像数据
                image_data = Image.open(image_list[i], 'r')
                image_data = image_data.tobytes()
                label = layer_list[i]
                bboxes = bboxes_list[i]
                example = image_to_tfexample(image_data, label, bboxes)
                tfrecord_writer.write(example.SerializeToString())
                sys.stdout.write('\r>> Converting image %d/%d' % (i + 1, length))
                sys.stdout.flush()

            sys.stdout.write('\n')
            sys.stdout.flush()


if __name__ == "__main__":
    txt_data = "E:/DataSets/model_data/test42.txt"
    image_list, layer_list, bboxes_list = get_files(txt_data)
    _convert_dataset(image_list, layer_list, bboxes_list, "E:/DataSets/model_data")
