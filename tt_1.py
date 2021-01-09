# coding:utf-8 
'''
created on 2019-11-16

@author:sunyihuan
'''
#
# print("Hello world!")
# for i in range(100):
#     print(pow(i, 2))

# import numpy as np
# import cv2
# import threading
# def counter(n):
#     cnt = 0
#     for i in range(n):
#         for j in range(i):
#             cnt += j
#     print(cnt)
#
# from io import StringIO
# f = StringIO()
# f.write('hello')
# f.write(' ')
# print(f.getvalue())
from PIL import Image
import cv2

image_path = "C:/Users/sunyihuan/Desktop/5.jpg"
# image = Image.open(image_path)
image = cv2.imread(image_path)

print(image)

import tensorflow as tf


def l1_loss(y_predict, y_true):
    return tf.reduce_mean(tf.abs(y_predict - y_true))


def l2_loss(y_predict, y_true):
    return tf.reduce_mean(tf.square(y_predict - y_true))


def l2_loss0(y_predict, y_true):
    return tf.nn.l2_loss(y_predict - y_true)


def smooth_l1_loss(y_predict, y_true):
    diff = tf.abs(y_predict - y_true)
    less_than_one = tf.cast(tf.less(diff, 1.0), tf.float32)
    smooth_l1_loss = (less_than_one * 0.5 * diff ** 2) + (1.0 - less_than_one) * (diff - 0.5)
    return tf.reduce_mean(smooth_l1_loss)


def huber_loss(y_predict, y_true, delta=1.0):
    diff = tf.abs(y_predict - y_true)
    less_than_one = tf.cast(tf.less(diff, delta), tf.float32)
    huber_loss = (less_than_one * 0.5 * diff ** 2) + (1.0 - less_than_one) * (delta * diff - 0.5 * delta ** 2)
    return tf.reduce_mean(huber_loss)
