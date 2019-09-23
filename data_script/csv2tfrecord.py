# coding:utf-8 
'''
将xml文件读取标注数据保存tfrecord文件
需csv文件
created on 2019/7/17

@author:sunyihuan
'''
from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

import os
import io
import tensorflow as tf
import pandas as pd

from PIL import Image
import dataset_util
from collections import namedtuple, OrderedDict


def class_text_to_int(row_label):
    '''
    classes标签转换为对应数字，int
    :param row_label: 标签name
    :return: 对应标签
    '''
    CLASSES = ["BeefSteak", "CartoonCookies", "ChickenWings", "ChiffonCake", "Cookies",
               "CranberryCookies", "CupCake", "EggTart", "nofood", "Peanuts",
               "Pizza", "PorkChops", "PurpleSweetPotato", "RoastedChicken", "Toast"]
    row_label = row_label.lower()
    if row_label in CLASSES:
        return CLASSES.index(row_label)+1
    else:
        return None


def split(df, group):
    data = namedtuple('data_script', ['filename', 'object'])
    gb = df.groupby(group)
    return [data(filename, gb.get_group(x)) for filename, x in zip(gb.groups.keys(), gb.groups)]


def create_tf_example(group, path):
    with tf.io.gfile.GFile(os.path.join(path, '{}'.format(group.filename)), 'rb') as fid:
        encoded_jpg = fid.read()
    encoded_jpg_io = io.BytesIO(encoded_jpg)
    image = Image.open(encoded_jpg_io)
    width, height = image.size

    filename = group.filename.encode('utf8')
    image_format = b'jpg'
    xmins = []
    xmaxs = []
    ymins = []
    ymaxs = []
    classes_text = []
    classes = []

    for index, row in group.object.iterrows():
        xmins.append(row['xmin'] / width)
        xmaxs.append(row['xmax'] / width)
        ymins.append(row['ymin'] / height)
        ymaxs.append(row['ymax'] / height)
        classes_text.append(row['class'].encode('utf8'))
        classes.append(class_text_to_int(row['class']))

    tf_example = tf.train.Example(features=tf.train.Features(feature={
        'image/height': dataset_util.int64_feature(height),
        'image/width': dataset_util.int64_feature(width),
        'image/filename': dataset_util.bytes_feature(filename),
        'image/source_id': dataset_util.bytes_feature(filename),
        'image/encoded': dataset_util.bytes_feature(encoded_jpg),
        'image/format': dataset_util.bytes_feature(image_format),
        'image/object/bbox/xmin': dataset_util.float_list_feature(xmins),
        'image/object/bbox/xmax': dataset_util.float_list_feature(xmaxs),
        'image/object/bbox/ymin': dataset_util.float_list_feature(ymins),
        'image/object/bbox/ymax': dataset_util.float_list_feature(ymaxs),
        'image/object/class/text': dataset_util.bytes_list_feature(classes_text),
        'image/object/class/label': dataset_util.int64_list_feature(classes),
    }))
    return tf_example


def csv2tfrecord(csv_data_dir, tf_dir, img_dir):
    '''
    将csv文件保存为tfrecord格式
    :param csv_data_dir: csv文件地址
    :param tf_dir: tfrecord文件地址
    :param img_dir: jpg图片地址
    :return:
    '''
    writer = tf.io.TFRecordWriter(tf_dir)
    examples = pd.read_csv(csv_data_dir)
    grouped = split(examples, 'filename')
    for group in grouped:
        tf_example = create_tf_example(group, img_dir)
        writer.write(tf_example.SerializeToString())

    writer.close()
    print('Successfully created the TFRecords: {}'
          .format(os.path.join(tf_dir)))


if __name__ == '__main__':
    csv_data_dir = '/Users/sunyihuan/Desktop/WLS/KX38I95FOODSETS/train_2classes/train_labels.csv'
    tf_dir = "/Users/sunyihuan/Desktop/WLS/KX38I95FOODSETS/train_2classes/train"
    img_dir = "/Users/sunyihuan/Desktop/WLS/KX38I95FOODSETS/train_2classes/JPGImages"
    csv2tfrecord(csv_data_dir, tf_dir, img_dir)
