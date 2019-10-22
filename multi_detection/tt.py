# -*- encoding: utf-8 -*-

"""
@File    : tt.py
@Time    : 2019/9/3 13:07
@Author  : sunyihuan
"""

# import tensorflow as tf
#
# # a = [3.568412, -5.623, 10.555, 0.333]
# # s = tf.nn.softmax(a)
# #
# # with tf.Session() as sess:
# #     print(sess.run(s))
#
# import time
# import numpy as np
#
# from tensorflow.contrib import slim
# from tensorflow.examples.tutorials.mnist import input_data
#
# mnist = input_data.read_data_sets("mnist/", one_hot=True)
#
#
# def get_available_gpus():
#     """
#     code from http://stackoverflow.com/questions/38559755/how-to-get-current-available-gpus-in-tensorflow
#     """
#     from tensorflow.python.client import device_lib as _device_lib
#     local_device_protos = _device_lib.list_local_devices()
#     return [x.name for x in local_device_protos if x.device_type == 'GPU']
#
#
# num_gpus = len(get_available_gpus())
# print("Available GPU Number :" + str(num_gpus))
#
# num_steps = 1000
# learning_rate = 0.001
# batch_size = 1000
# display_step = 10
#
# num_input = 784
# num_classes = 10
#
#
# def conv_net_with_layers(x, is_training, dropout=0.75):
#     with tf.variable_scope("ConvNet", reuse=tf.AUTO_REUSE):
#         x = tf.reshape(x, [-1, 28, 28, 1])
#         x = tf.layers.conv2d(x, 12, 5, activation=tf.nn.relu)
#         x = tf.layers.max_pooling2d(x, 2, 2)
#         x = tf.layers.conv2d(x, 24, 3, activation=tf.nn.relu)
#         x = tf.layers.max_pooling2d(x, 2, 2)
#         x = tf.layers.flatten(x)
#         x = tf.layers.dense(x, 100)
#         x = tf.layers.dropout(x, rate=dropout, training=is_training)
#         out = tf.layers.dense(x, 10)
#         out = tf.nn.softmax(out) if not is_training else out
#     return out
#
#
# def conv_net(x, is_training):
#     # "updates_collections": None is very import ,without will only get 0.10
#     batch_norm_params = {"is_training": is_training, "decay": 0.9, "updates_collections": None}
#     # ,'variables_collections': [ tf.GraphKeys.TRAINABLE_VARIABLES ]
#     with slim.arg_scope([slim.conv2d, slim.fully_connected],
#                         activation_fn=tf.nn.relu,
#                         weights_initializer=tf.truncated_normal_initializer(mean=0.0, stddev=0.01),
#                         weights_regularizer=slim.l2_regularizer(0.0005),
#                         normalizer_fn=slim.batch_norm, normalizer_params=batch_norm_params):
#         with tf.variable_scope("ConvNet", reuse=tf.AUTO_REUSE):
#             x = tf.reshape(x, [-1, 28, 28, 1])
#             net = slim.conv2d(x, 6, [5, 5], scope="conv_1")
#             net = slim.max_pool2d(net, [2, 2], scope="pool_1")
#             net = slim.conv2d(net, 12, [5, 5], scope="conv_2")
#             net = slim.max_pool2d(net, [2, 2], scope="pool_2")
#             net = slim.flatten(net, scope="flatten")
#             net = slim.fully_connected(net, 100, scope="fc")
#             net = slim.dropout(net, is_training=is_training)
#             net = slim.fully_connected(net, num_classes, scope="prob", activation_fn=None, normalizer_fn=None)
#             return net
#
#
# def average_gradients(tower_grads):
#     average_grads = []
#     for grad_and_vars in zip(*tower_grads):
#         grads = []
#         for g, _ in grad_and_vars:
#             expend_g = tf.expand_dims(g, 0)
#             grads.append(expend_g)
#         grad = tf.concat(grads, 0)
#         grad = tf.reduce_mean(grad, 0)
#         v = grad_and_vars[0][1]
#         grad_and_var = (grad, v)
#         average_grads.append(grad_and_var)
#     return average_grads
#
#
# PS_OPS = ['Variable', 'VariableV2', 'AutoReloadVariable']
#
#
# def assign_to_device(device, ps_device='/cpu:0'):
#     def _assign(op):
#         node_def = op if isinstance(op, tf.NodeDef) else op.node_def
#         if node_def.op in PS_OPS:
#             return "/" + ps_device
#         else:
#             return device
#
#     return _assign
#
#
# def train():
#     with tf.device("/cpu:0"):
#         global_step = tf.train.get_or_create_global_step()
#         tower_grads = []
#         X = tf.placeholder(tf.float32, [None, num_input])
#         Y = tf.placeholder(tf.float32, [None, num_classes])
#         opt = tf.train.AdamOptimizer(learning_rate)
#         with tf.variable_scope(tf.get_variable_scope()):
#             for i in range(num_gpus):
#                 with tf.device(assign_to_device('/gpu:{}'.format(i), ps_device='/cpu:0')):
#                     _x = X[i * batch_size:(i + 1) * batch_size]
#                     _y = Y[i * batch_size:(i + 1) * batch_size]
#                     logits = conv_net(_x, True)
#                     tf.get_variable_scope().reuse_variables()
#                     loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=_y, logits=logits))
#                     grads = opt.compute_gradients(loss)
#                     tower_grads.append(grads)
#                     if i == 0:
#                         logits_test = conv_net(_x, False)
#                         correct_prediction = tf.equal(tf.argmax(logits_test, 1), tf.argmax(_y, 1))
#                         accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
#         grads = average_gradients(tower_grads)
#         train_op = opt.apply_gradients(grads)
#         with tf.Session() as sess:
#             sess.run(tf.global_variables_initializer())
#             for step in range(1, num_steps + 1):
#                 batch_x, batch_y = mnist.train.next_batch(batch_size * num_gpus)
#                 ts = time.time()
#                 sess.run(train_op, feed_dict={X: batch_x, Y: batch_y})
#                 te = time.time() - ts
#                 if step % 10 == 0 or step == 1:
#                     loss_value, acc = sess.run([loss, accuracy], feed_dict={X: batch_x, Y: batch_y})
#                     print("Step:" + str(step) + ":" + str(loss_value) + " " + str(acc) + ", %i Examples/sec" % int(
#                         len(batch_x) / te))
#             print("Done")
#             print("Testing Accuracy:",
#                   np.mean([sess.run(accuracy, feed_dict={X: mnist.test.images[i:i + batch_size],
#                                                          Y: mnist.test.labels[i:i + batch_size]}) for i in
#                            range(0, len(mnist.test.images), batch_size)]))
#
#
# def train_single():
#     X = tf.placeholder(tf.float32, [None, num_input])
#     Y = tf.placeholder(tf.float32, [None, num_classes])
#     logits = conv_net(X, True)
#     loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=Y, logits=logits))
#     opt = tf.train.AdamOptimizer(learning_rate)
#     train_op = opt.minimize(loss)
#     logits_test = conv_net(X, False)
#     correct_prediction = tf.equal(tf.argmax(logits_test, 1), tf.argmax(Y, 1))
#     accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
#     with tf.Session() as sess:
#         sess.run(tf.global_variables_initializer())
#         for step in range(1, num_steps + 1):
#             batch_x, batch_y = mnist.train.next_batch(batch_size)
#             sess.run(train_op, feed_dict={X: batch_x, Y: batch_y})
#             if step % display_step == 0 or step == 1:
#                 loss_value, acc = sess.run([loss, accuracy], feed_dict={X: batch_x, Y: batch_y})
#                 print("Step:" + str(step) + ":" + str(loss_value) + " " + str(acc))
#         print("Done")
#         print("Testing Accuracy:", np.mean([sess.run(accuracy, feed_dict={X: mnist.test.images[i:i + batch_size],
#                                                                           Y: mnist.test.labels[i:i + batch_size]}) for i
#                                             in
#                                             range(0, len(mnist.test.images), batch_size)]))
#
#
# if __name__ == "__main__":
#     train_single()
#
# import cv2
#
# img_path = "C:/Users/sunyihuan/Desktop/test/3.jpg"
#
# im = cv2.imread(img_path)
# print(im.shape)
# print(im)
# hsv = cv2.cvtColor(im, cv2.COLOR_RGB2HSV)
# print(hsv.shape)
# print(hsv)
# f = open("E:/DataSets/KX_FOODSets_model_data/26classes_0920_padding/ImageSets/Main/val.txt", "r")
#
# all_txt_name = "E:/DataSets/KX_FOODSets_model_data/26classes_0920_padding/ImageSets/Main/val_pad.txt"
# file = open(all_txt_name, "w")
# txt_files = f.readlines()
# for txt_file_one in txt_files:
#     txt_file_one = str(txt_file_one.strip())
#     txt_file_one = txt_file_one + "_pad" + "\n"
#     file.write(txt_file_one)
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
        self.num_classes = 26  # 种类数
        self.score_threshold = 0.45
        self.iou_threshold = 0.5
        self.weight_file = "E:/ckpt_dirs/Food_detection/multi_food/20191015/yolov3_train_loss=4.5853.ckpt-300"  # ckpt文件地址
        self.write_image = True  # 是否画图
        self.show_label = True  # 是否显示标签

        graph = tf.Graph()
        with graph.as_default():
            self.saver = tf.train.import_meta_graph("{}.meta".format(self.weight_file))
            self.sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True))
            self.saver.restore(self.sess, self.weight_file)

            self.input = graph.get_tensor_by_name("define_input/input_data:0")
            self.trainable = graph.get_tensor_by_name("define_input/training:0")

            # self.conv56 = graph.get_tensor_by_name("define_loss/conv_mbbox/Relu:0")

            self.conv56 = graph.get_tensor_by_name("define_loss/conv_mbbox/BiasAdd: 0")

    def predict(self, image):
        org_image = np.copy(image)
        org_h, org_w, _ = org_image.shape

        image_data = utils.image_preporcess(image, [self.input_size, self.input_size])
        image_data = image_data[np.newaxis, ...]

        conv56 = self.sess.run(
            [self.conv56],
            feed_dict={
                self.input: image_data,
                self.trainable: False
            }
        )

        return conv56

    def result(self, image_path):
        image = cv2.imread(image_path)  # 图片读取
        mbbox = self.predict(image)  # 预测结果
        _, __, w, h, c = np.array(mbbox).shape
        mbbox = np.reshape(mbbox, [w, h, c])
        plt.figure(figsize=(6, 6), dpi=80)
        for i in range(20):
            plt.figure(i + 1)
            plt.imshow(mbbox[:, :, i])
        plt.show()


if __name__ == '__main__':
    img_path = "E:/kx_detection/multi_detection/docs/images/344_chickenwings.jpg"  # 图片地址
    Y = YoloPredict()
    Y.result(img_path)
