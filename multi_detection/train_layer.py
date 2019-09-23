# -*- encoding: utf-8 -*-

"""
@File    : train_layer.py
@Time    : 2019/9/2 16:28
@Author  : sunyihuan
"""

import os
import time
import shutil
import numpy as np
import tensorflow as tf
from tensorflow.python.framework import graph_util
import multi_detection.core.utils as utils
from tqdm import tqdm
from multi_detection.core.dataset import Dataset
from multi_detection.core.yolov3 import YOLOV3
from multi_detection.core.config import cfg
import multi_detection.core.common as common


class YoloTrain(object):
    def __init__(self):
        self.anchor_per_scale = cfg.YOLO.ANCHOR_PER_SCALE
        self.classes = utils.read_class_names(cfg.YOLO.CLASSES)
        self.num_classes = len(self.classes)
        self.learn_rate_init = cfg.TRAIN.LEARN_RATE_INIT
        self.learn_rate_end = cfg.TRAIN.LEARN_RATE_END
        self.first_stage_epochs = cfg.TRAIN.FISRT_STAGE_EPOCHS
        self.second_stage_epochs = cfg.TRAIN.SECOND_STAGE_EPOCHS
        self.warmup_periods = cfg.TRAIN.WARMUP_EPOCHS
        self.initial_weight = cfg.TRAIN.INITIAL_WEIGHT
        self.time = time.strftime('%Y-%m-%d-%H-%M-%S', time.localtime(time.time()))
        self.moving_ave_decay = cfg.YOLO.MOVING_AVE_DECAY
        self.train_input_size = cfg.TRAIN.INPUT_SIZE
        self.max_bbox_per_scale = 150
        self.train_logdir = "./data/log/train"
        self.trainset = Dataset('train')
        self.testset = Dataset('test')
        self.steps_per_period = len(self.trainset)
        self.sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True))

        with tf.name_scope('define_input'):
            self.input_data = tf.placeholder(dtype=tf.float32,
                                             shape=(None, self.train_input_size, self.train_input_size, 3),
                                             name='input_data')
            self.layer_label = tf.placeholder(dtype=tf.float32, name='layer_label')
            self.label_sbbox = tf.placeholder(dtype=tf.float32, name='label_sbbox')
            self.label_mbbox = tf.placeholder(dtype=tf.float32, name='label_mbbox')
            self.label_lbbox = tf.placeholder(dtype=tf.float32, name='label_lbbox')
            self.true_sbboxes = tf.placeholder(dtype=tf.float32, name='sbboxes')
            self.true_mbboxes = tf.placeholder(dtype=tf.float32, name='mbboxes')
            self.true_lbboxes = tf.placeholder(dtype=tf.float32, name='lbboxes')
            self.trainable = tf.placeholder(dtype=tf.bool, name='training')

        with tf.name_scope("model"):
            input_data = common.convolutional(self.input_data, filters_shape=(3, 3, 3, 32), trainable=self.trainable,
                                              name='conv0')
            print(input_data)
            input_data = common.convolutional(input_data, filters_shape=(3, 3, 32, 64),
                                              trainable=self.trainable, name='conv1', downsample=True)
            print(input_data)
            # out9 = input_data
            # out = common.convolutional(out9, (1, 1, 64, 128), self.trainable, 'out1')
            # print(out)
            out = tf.layers.flatten(input_data, name="flatten")
            self.out = tf.layers.dense(out, 4, name="dense_out")  # layer 输出

        with tf.name_scope("define_loss"):
            self.predict_op = tf.argmax(input=self.out, axis=1, name='layer_classes')
            print("layer_nums::", self.predict_op)

            layer_label = tf.cast(self.layer_label, tf.int32)
            Y = tf.one_hot(layer_label, depth=4, axis=1, dtype=tf.float32)
            self.layer_loss = tf.losses.softmax_cross_entropy(Y, self.out, scope='LOSS', weights=1.0)

            self.net_var = tf.global_variables()
            print("layer_loss::::")
            print(self.layer_loss)

        with tf.name_scope('learn_rate'):
            self.global_step = tf.Variable(1.0, dtype=tf.float64, trainable=False, name='global_step')
            self.learn_rate = tf.train.exponential_decay(self.learn_rate_init, global_step=self.global_step,
                                                         decay_steps=1000, decay_rate=0.9)
            self.learn_rate = tf.maximum(self.learn_rate, self.learn_rate_end)

        self._optimizer = tf.train.AdamOptimizer(self.learn_rate).minimize(self.layer_loss,global_step=self.global_step)

        with tf.name_scope('loader_and_saver'):
            self.loader = tf.train.Saver(self.net_var)
            self.saver = tf.train.Saver(tf.global_variables(), max_to_keep=10)

        with tf.name_scope('summary'):
            tf.summary.scalar("learn_rate", self.learn_rate)
            tf.summary.scalar("total_loss", self.layer_loss)

            test_logdir = "./data/log/test"
            if os.path.exists(test_logdir): shutil.rmtree(test_logdir)
            os.mkdir(test_logdir)
            if os.path.exists(self.train_logdir): shutil.rmtree(self.train_logdir)
            os.mkdir(self.train_logdir)

            self.write_op = tf.summary.merge_all()
            self.train_summary_writer = tf.summary.FileWriter(self.train_logdir, graph=self.sess.graph)
            self.test_summary_writer = tf.summary.FileWriter(test_logdir, graph=self.sess.graph)

        # 计算参数
        flops = tf.profiler.profile(self.sess.graph, options=tf.profiler.ProfileOptionBuilder.float_operation())
        params = tf.profiler.profile(self.sess.graph,
                                     options=tf.profiler.ProfileOptionBuilder.trainable_variables_parameter())
        print('FLOPs: {};    Trainable params: {}'.format(flops.total_float_ops, params.total_parameters))

    def train(self):
        self.sess.run(tf.global_variables_initializer())
        try:
            print('=> Restoring weights from: %s ... ' % self.initial_weight)
            self.loader.restore(self.sess, self.initial_weight)
        except:
            print('=> %s does not exist !!!' % self.initial_weight)
            print('=> Now it starts to train YOLOV3 from scratch ...')
            self.first_stage_epochs = 0

        for epoch in range(1, 1 + self.first_stage_epochs + self.second_stage_epochs):
            train_op = self._optimizer
            pbar = tqdm(self.trainset)
            train_epoch_loss, test_epoch_loss = [], []

            for train_data in pbar:
                _, summary, train_step_loss, global_step_val, layer_loss_v, layer_o = self.sess.run(
                    [train_op, self.write_op, self.layer_loss, self.global_step,
                     self.layer_loss, self.out], feed_dict={
                        self.input_data: train_data[0],
                        self.layer_label: train_data[1],
                        self.label_sbbox: train_data[2],
                        self.label_mbbox: train_data[3],
                        self.label_lbbox: train_data[4],
                        self.true_sbboxes: train_data[5],
                        self.true_mbboxes: train_data[6],
                        self.true_lbboxes: train_data[7],
                        self.trainable: True,
                    })
                # print("layer_loss:::")
                # print(layer_loss_v)
                print("true:::::")
                print(train_data[1])
                print("predict:::")
                print(layer_o)

                train_epoch_loss.append(train_step_loss)
                self.train_summary_writer.add_summary(summary, global_step_val)
                pbar.set_description("train loss: %.2f" % train_step_loss)

            # 保存模型
            train_epoch_loss = np.mean(train_epoch_loss)
            ckpt_file = "./checkpoint/yolov3_train_loss=%.4f.ckpt" % train_epoch_loss
            self.saver.save(self.sess, ckpt_file, global_step=epoch)
            #
            # output = ["define_loss/pred_sbbox/concat_2", "define_loss/pred_mbbox/concat_2",
            #           "define_loss/pred_lbbox/concat_2", "define_loss/layer_classes"]
            # constant_graph = graph_util.convert_variables_to_constants(self.sess, self.sess.graph_def, output)
            # with tf.gfile.GFile('model/yolo_model.pb', mode='wb') as f:
            #     f.write(constant_graph.SerializeToString())

            # 保存为pb用于tf_serving
            # export_dir = "E:/Joyoung_WLS_github/tf_yolov3/pb"
            # tf.saved_model.simple_save(self.sess, export_dir, inputs={"input": self.input_data},
            #                            outputs={"pred_sbbox": self.pred_sbbox, "pred_mbbox": self.pred_mbbox,
            #                                     "pre_lbbox": self.pred_lbbox})

            par_test = tqdm(self.testset)
            for test_data in par_test:
                test_step_loss, test_summary, test_layer_o = self.sess.run(
                    [self.layer_loss, self.write_op, self.out],
                    feed_dict={
                        self.input_data: test_data[0],
                        self.layer_label: test_data[1],
                        self.label_sbbox: test_data[2],
                        self.label_mbbox: test_data[3],
                        self.label_lbbox: test_data[4],
                        self.true_sbboxes: test_data[5],
                        self.true_mbboxes: test_data[6],
                        self.true_lbboxes: test_data[7],
                        self.trainable: False,
                    })

                test_epoch_loss.append(test_step_loss)
                self.test_summary_writer.add_summary(test_summary)
                par_test.set_description("test loss: %.2f" % test_step_loss)
                print("true:::::")
                print(test_data[1])
                print("test prediction")
                print(test_layer_o)

            train_epoch_loss, test_epoch_loss = np.mean(train_epoch_loss), np.mean(test_epoch_loss)
            ckpt_file = "./checkpoint/yolov3_test_loss=%.4f.ckpt" % test_epoch_loss
            log_time = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time()))
            print("=> Epoch: %2d Time: %s Train loss: %.2f Test loss: %.2f Saving %s ..."
                  % (epoch, log_time, train_epoch_loss, test_epoch_loss, ckpt_file))


if __name__ == '__main__':
    YoloTrain().train()
