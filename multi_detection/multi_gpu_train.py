# -*- encoding: utf-8 -*-

"""
多gpu训练，未完成！！！！！！！
@File    : multi_gpu_train.py
@Time    : 2019/11/7 14:39
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


def average_gradients(tower_grads):
    average_grads = []
    for grad_and_vars in zip(*tower_grads):
        grads = []
        for g, _ in grad_and_vars:
            expend_g = tf.expand_dims(g, 0)
            grads.append(expend_g)
        grad = tf.concat(grads, 0)
        grad = tf.reduce_mean(grad, 0)
        v = grad_and_vars[0][1]
        grad_and_var = (grad, v)
        average_grads.append(grad_and_var)
    return average_grads


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
        self.tower_grads = []

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

        with tf.name_scope('learn_rate'):
            self.global_step = tf.Variable(1.0, dtype=tf.float64, trainable=False, name='global_step')
            self.learn_rate = tf.train.exponential_decay(self.learn_rate_init, global_step=self.global_step,
                                                         decay_steps=1000, decay_rate=0.9)
            self.learn_rate = tf.maximum(self.learn_rate, self.learn_rate_end)
            global_step_update = tf.assign_add(self.global_step, 1.0)

        with tf.variable_scope(tf.get_variable_scope(), reuse=tf.AUTO_REUSE):
            self.opt = tf.train.GradientDescentOptimizer(self.learn_rate)

        with tf.name_scope("define_loss",tf.variable_scope("define_loss", reuse=tf.AUTO_REUSE)):
            for i in range(4):
                with tf.device('/gpu:{}'.format(i + 4)):
                    self.input_data_i = self.input_data[i * cfg.TRAIN.BATCH_SIZE:(i + 1) * cfg.TRAIN.BATCH_SIZE]
                    self.layer_label_i = self.layer_label[i * cfg.TRAIN.BATCH_SIZE:(i + 1) * cfg.TRAIN.BATCH_SIZE]
                    self.label_sbbox_i = self.label_sbbox[i * cfg.TRAIN.BATCH_SIZE:(i + 1) * cfg.TRAIN.BATCH_SIZE]
                    self.label_mbbox_i = self.label_mbbox[i * cfg.TRAIN.BATCH_SIZE:(i + 1) * cfg.TRAIN.BATCH_SIZE]
                    self.label_lbbox_i = self.label_lbbox[i * cfg.TRAIN.BATCH_SIZE:(i + 1) * cfg.TRAIN.BATCH_SIZE]
                    self.true_sbboxes_i = self.true_sbboxes[i * cfg.TRAIN.BATCH_SIZE:(i + 1) * cfg.TRAIN.BATCH_SIZE]
                    self.true_mbboxes_i = self.true_mbboxes[i * cfg.TRAIN.BATCH_SIZE:(i + 1) * cfg.TRAIN.BATCH_SIZE]
                    self.true_lbboxes_i = self.true_lbboxes[i * cfg.TRAIN.BATCH_SIZE:(i + 1) * cfg.TRAIN.BATCH_SIZE]

                    self.model = YOLOV3(self.input_data_i, self.trainable)
                    self.layer_out = self.model.out
                    self.net_var = tf.global_variables()
                    self.giou_loss, self.conf_loss, self.prob_loss, self.giou, self.bbox_loss_scale = self.model.compute_loss(
                        self.label_sbbox_i, self.label_mbbox_i, self.label_lbbox_i,
                        self.true_sbboxes_i, self.true_mbboxes_i, self.true_lbboxes_i)

                    # self.layer_loss = self.model.layer_loss(self.layer_label_i)
                    # self.layer_loss = tf.cond(self.layer_loss > 0.01, lambda: self.layer_loss, lambda: 0.0)
                    self.layer_loss=0
                    self.loss = self.giou_loss + self.conf_loss + 2 * self.prob_loss + 10 * self.layer_loss

                    tf.get_variable_scope().reuse_variables()
                    grads = self.opt.compute_gradients(self.loss)
                    self.tower_grads.append(grads)
        grads = average_gradients(self.tower_grads)
        self.train_op = self.opt.apply_gradients(grads)

        with tf.name_scope('loader_and_saver'):
            self.loader = tf.train.Saver(self.net_var)
            self.saver = tf.train.Saver(tf.global_variables(), max_to_keep=10)

        with tf.name_scope('summary'):
            tf.summary.scalar("learn_rate", self.learn_rate)
            tf.summary.scalar("giou_loss", self.giou_loss)
            tf.summary.scalar("conf_loss", self.conf_loss)
            tf.summary.scalar("prob_loss", self.prob_loss)
            tf.summary.scalar("layer_loss", self.layer_loss)
            tf.summary.scalar("total_loss", self.loss)

            test_logdir = "./data/log/test"
            if os.path.exists(test_logdir): shutil.rmtree(test_logdir)
            os.mkdir(test_logdir)
            if os.path.exists(self.train_logdir): shutil.rmtree(self.train_logdir)
            os.mkdir(self.train_logdir)

            self.write_op = tf.summary.merge_all()
            self.train_summary_writer = tf.summary.FileWriter(self.train_logdir, graph=self.sess.graph)
            self.test_summary_writer = tf.summary.FileWriter(test_logdir, graph=self.sess.graph)

        # 参数量计算
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
            pbar = tqdm(self.trainset)
            train_epoch_loss, test_epoch_loss = [], []

            for train_data in pbar:
                _, summary, train_step_loss, global_step_val, gi, bbo, layer_loss_v, layer_o = self.sess.run(
                    [self.train_op, self.write_op, self.loss, self.global_step, self.giou, self.bbox_loss_scale,
                     self.layer_loss, self.layer_out], feed_dict={
                        # _, summary, train_step_loss, global_step_val, gi, bbo, = self.sess.run(
                        #     [train_op, self.write_op, self.loss, self.global_step, self.giou, self.bbox_loss_scale,
                        #      ], feed_dict={
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
                train_epoch_loss.append(train_step_loss)
                self.train_summary_writer.add_summary(summary, global_step_val)
                pbar.set_description("train loss: %.2f" % train_step_loss)

            # 保存模型
            train_epoch_loss = np.mean(train_epoch_loss)
            ckpt_file = "./checkpoint/yolov3_train_loss=%.4f.ckpt" % train_epoch_loss
            self.saver.save(self.sess, ckpt_file, global_step=epoch)

            output = ["define_loss/pred_sbbox/concat_2", "define_loss/pred_mbbox/concat_2",
                      "define_loss/pred_lbbox/concat_2", "define_loss/layer_classes"]
            constant_graph = graph_util.convert_variables_to_constants(self.sess, self.sess.graph_def, output)
            with tf.gfile.GFile('./model/yolo_model.pb', mode='wb') as f:
                f.write(constant_graph.SerializeToString())

            par_test = tqdm(self.testset)
            for test_data in par_test:
                test_step_loss, test_summary, test_step_giou_loss, test_step_prob_loss, test_step_conf_loss, test_layer_o = self.sess.run(
                    [self.loss, self.write_op, self.giou_loss, self.prob_loss, self.conf_loss, self.layer_out],
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

            train_epoch_loss, test_epoch_loss = np.mean(train_epoch_loss), np.mean(test_epoch_loss)
            ckpt_file = "./checkpoint/yolov3_test_loss=%.4f.ckpt" % test_epoch_loss
            log_time = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time()))
            print("=> Epoch: %2d Time: %s Train loss: %.2f Test loss: %.2f Saving %s ..."
                  % (epoch, log_time, train_epoch_loss, test_epoch_loss, ckpt_file))


if __name__ == '__main__':
    YoloTrain().train()
