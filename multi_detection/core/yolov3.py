#! /usr/bin/env python
# coding=utf-8
# ================================================================


import numpy as np
import tensorflow as tf
import multi_detection.core.utils as utils
import multi_detection.core.common as common
import multi_detection.core.backbone as backbone
from multi_detection.core.config import cfg
import math


class YOLOV3(object):
    """Implement tensoflow yolov3 here"""

    def __init__(self, input_data, trainable):

        self.trainable = trainable
        self.classes = utils.read_class_names(cfg.YOLO.CLASSES)
        self.num_class = len(self.classes)
        self.strides = np.array(cfg.YOLO.STRIDES)
        self.anchors = utils.get_anchors(cfg.YOLO.ANCHORS)
        self.anchor_per_scale = cfg.YOLO.ANCHOR_PER_SCALE
        self.iou_loss_thresh = cfg.YOLO.IOU_LOSS_THRESH
        self.upsample_method = cfg.YOLO.UPSAMPLE_METHOD
        self.layer_nums = cfg.YOLO.LAYER_NUMS

        try:
            self.conv_lbbox, self.conv_mbbox, self.conv_sbbox, self.out = self.__build_nework(input_data)
        except:
            raise NotImplementedError("Can not build up yolov3 network!")
        print(self.out)
        self.predict_op = tf.argmax(input=self.out, axis=1, name='layer_classes')
        print("layer_nums::", self.predict_op)

        with tf.variable_scope('pred_sbbox'):
            self.pred_sbbox = self.decode(self.conv_sbbox, self.anchors[0], self.strides[0])

        with tf.variable_scope('pred_mbbox'):
            self.pred_mbbox = self.decode(self.conv_mbbox, self.anchors[1], self.strides[1])

        with tf.variable_scope('pred_lbbox'):
            self.pred_lbbox = self.decode(self.conv_lbbox, self.anchors[2], self.strides[2])

    def __build_nework(self, input_data):

        route_1, route_2, input_data = backbone.darknet53(input_data, self.trainable)
        out = tf.layers.flatten(input_data)
        # out = tf.layers.dense(out, 400, activation=tf.nn.relu)
        out = tf.layers.dense(out, self.layer_nums)  # layer 输出

        input_data = common.convolutional(input_data, (1, 1, 1024, 512), self.trainable, 'conv52')
        input_data = common.convolutional(input_data, (3, 3, 512, 1024), self.trainable, 'conv53')
        input_data = common.convolutional(input_data, (1, 1, 1024, 512), self.trainable, 'conv54')
        # input_data = common.convolutional(input_data, (3, 3, 512, 1024), self.trainable, 'conv55')
        # input_data = common.convolutional(input_data, (1, 1, 1024, 512), self.trainable, 'conv56')

        conv_lobj_branch = common.convolutional(input_data, (3, 3, 512, 1024), self.trainable, name='conv_lobj_branch')
        conv_lbbox = common.convolutional(conv_lobj_branch, (1, 1, 1024, 3 * (self.num_class + 5)),
                                          trainable=self.trainable, name='conv_lbbox', activate=False, bn=False)

        input_data = common.convolutional(input_data, (1, 1, 512, 256), self.trainable, 'conv57')
        input_data = common.upsample(input_data, name='upsample0', method=self.upsample_method)

        with tf.variable_scope('route_1'):
            input_data = tf.concat([input_data, route_2], axis=-1)

        input_data = common.convolutional(input_data, (1, 1, 768, 256), self.trainable, 'conv58')
        input_data = common.convolutional(input_data, (3, 3, 256, 512), self.trainable, 'conv59')
        input_data = common.convolutional(input_data, (1, 1, 512, 256), self.trainable, 'conv60')
        # input_data = common.convolutional(input_data, (3, 3, 256, 512), self.trainable, 'conv61')
        # input_data = common.convolutional(input_data, (1, 1, 512, 256), self.trainable, 'conv62')

        conv_mobj_branch = common.convolutional(input_data, (3, 3, 256, 512), self.trainable, name='conv_mobj_branch')
        conv_mbbox = common.convolutional(conv_mobj_branch, (1, 1, 512, 3 * (self.num_class + 5)),
                                          trainable=self.trainable, name='conv_mbbox', activate=False, bn=False)

        input_data = common.convolutional(input_data, (1, 1, 256, 128), self.trainable, 'conv63')
        input_data = common.upsample(input_data, name='upsample1', method=self.upsample_method)

        with tf.variable_scope('route_2'):
            input_data = tf.concat([input_data, route_1], axis=-1)

        input_data = common.convolutional(input_data, (1, 1, 384, 128), self.trainable, 'conv64')
        input_data = common.convolutional(input_data, (3, 3, 128, 256), self.trainable, 'conv65')
        input_data = common.convolutional(input_data, (1, 1, 256, 128), self.trainable, 'conv66')
        # input_data = common.convolutional(input_data, (3, 3, 128, 256), self.trainable, 'conv67')
        # input_data = common.convolutional(input_data, (1, 1, 256, 128), self.trainable, 'conv68')

        conv_sobj_branch = common.convolutional(input_data, (3, 3, 128, 256), self.trainable, name='conv_sobj_branch')
        conv_sbbox = common.convolutional(conv_sobj_branch, (1, 1, 256, 3 * (self.num_class + 5)),
                                          trainable=self.trainable, name='conv_sbbox', activate=False, bn=False)
        return conv_lbbox, conv_mbbox, conv_sbbox, out

    def decode(self, conv_output, anchors, stride):
        """
        return tensor of shape [batch_size, output_size, output_size, anchor_per_scale, 5 + num_classes]
               contains (x, y, w, h, score, probability)
        """

        conv_shape = tf.shape(conv_output)
        batch_size = conv_shape[0]
        output_size = conv_shape[1]
        anchor_per_scale = len(anchors)

        conv_output = tf.reshape(conv_output,
                                 (batch_size, output_size, output_size, anchor_per_scale, 5 + self.num_class))

        conv_raw_dxdy = conv_output[:, :, :, :, 0:2]
        conv_raw_dwdh = conv_output[:, :, :, :, 2:4]
        conv_raw_conf = conv_output[:, :, :, :, 4:5]
        conv_raw_prob = conv_output[:, :, :, :, 5:]

        y = tf.tile(tf.range(output_size, dtype=tf.int32)[:, tf.newaxis], [1, output_size])
        x = tf.tile(tf.range(output_size, dtype=tf.int32)[tf.newaxis, :], [output_size, 1])

        xy_grid = tf.concat([x[:, :, tf.newaxis], y[:, :, tf.newaxis]], axis=-1)
        xy_grid = tf.tile(xy_grid[tf.newaxis, :, :, tf.newaxis, :], [batch_size, 1, 1, anchor_per_scale, 1])
        xy_grid = tf.cast(xy_grid, tf.float32)

        pred_xy = (tf.sigmoid(conv_raw_dxdy) + xy_grid) * stride
        pred_wh = (tf.exp(conv_raw_dwdh) * anchors) * stride
        pred_xywh = tf.concat([pred_xy, pred_wh], axis=-1)

        pred_conf = tf.sigmoid(conv_raw_conf)
        pred_prob = tf.sigmoid(conv_raw_prob)

        return tf.concat([pred_xywh, pred_conf, pred_prob], axis=-1)

    def focal(self, target, actual, alpha=1, gamma=2):
        focal_loss = alpha * tf.pow(tf.abs(target - actual), gamma)
        return focal_loss

    def bbox_giou(self, boxes1, boxes2):

        boxes1 = tf.concat([boxes1[..., :2] - boxes1[..., 2:] * 0.5,
                            boxes1[..., :2] + boxes1[..., 2:] * 0.5], axis=-1)
        boxes2 = tf.concat([boxes2[..., :2] - boxes2[..., 2:] * 0.5,
                            boxes2[..., :2] + boxes2[..., 2:] * 0.5], axis=-1)

        boxes1 = tf.concat([tf.minimum(boxes1[..., :2], boxes1[..., 2:]),
                            tf.maximum(boxes1[..., :2], boxes1[..., 2:])], axis=-1)
        boxes2 = tf.concat([tf.minimum(boxes2[..., :2], boxes2[..., 2:]),
                            tf.maximum(boxes2[..., :2], boxes2[..., 2:])], axis=-1)

        boxes1_area = (boxes1[..., 2] - boxes1[..., 0]) * (boxes1[..., 3] - boxes1[..., 1])
        boxes2_area = (boxes2[..., 2] - boxes2[..., 0]) * (boxes2[..., 3] - boxes2[..., 1])

        left_up = tf.maximum(boxes1[..., :2], boxes2[..., :2])
        right_down = tf.minimum(boxes1[..., 2:], boxes2[..., 2:])

        inter_section = tf.maximum(right_down - left_up, 0.0)
        inter_area = inter_section[..., 0] * inter_section[..., 1]
        union_area = tf.maximum(boxes1_area + boxes2_area - inter_area, 1e-5)
        iou = inter_area / union_area

        enclose_left_up = tf.minimum(boxes1[..., :2], boxes2[..., :2])
        enclose_right_down = tf.maximum(boxes1[..., 2:], boxes2[..., 2:])
        enclose = tf.maximum(enclose_right_down - enclose_left_up, 0.0)
        enclose_area = tf.maximum(enclose[..., 0] * enclose[..., 1], 1e-5)
        giou = iou - 1.0 * (enclose_area - union_area) / enclose_area

        return giou

    def bbox_diou(self, boxes1, boxes2):
        '''
        计算两个框的DIoU
        :param boxes1:
        :param boxes2:
        :return:
        '''
        exchange = False
        if boxes1.shape[0] > boxes2.shape[0]:
            boxes1, boxes2 = boxes2, boxes1
            exchange = True
        # #xmin,ymin,xmax,ymax->[:,0],[:,1],[:,2],[:,3]
        w1 = boxes1[..., 2] - boxes1[..., 0]
        h1 = boxes1[..., 3] - boxes1[..., 1]
        w2 = boxes2[..., 2] - boxes2[..., 0]
        h2 = boxes2[..., 3] - boxes2[..., 1]

        area1 = w1 * h1
        area2 = w2 * h2

        center_x1 = (boxes1[..., 2] + boxes1[..., 0]) / 2  # （x1max +x1min）/2
        center_y1 = (boxes1[..., 3] + boxes1[..., 1]) / 2  # (y1max+y1min)/2
        center_x2 = (boxes2[..., 2] + boxes2[..., 0]) / 2
        center_y2 = (boxes2[..., 3] + boxes2[..., 1]) / 2

        inter_max_xy = tf.minimum(boxes1[..., 2:], boxes2[..., 2:])  # min((x1max,y1max ),(x2max,y2max)) ->返回较小一组
        inter_min_xy = tf.maximum(boxes1[..., :2], boxes2[..., :2])  # max((x1min,y1min ),(x2min,y2min))->返回较大的一组
        out_max_xy = tf.maximum(boxes1[..., 2:], boxes2[..., 2:])
        out_min_xy = tf.minimum(boxes1[..., :2], boxes2[..., :2])

        inter = tf.clip_by_value((inter_max_xy - inter_min_xy), clip_value_min=1e-5, clip_value_max=1e5)
        inter_area = inter[..., 0] * inter[..., 1]
        inter_diag = (center_x2 - center_x1) ** 2 + (center_y2 - center_y1) ** 2
        outer = tf.clip_by_value((out_max_xy - out_min_xy), clip_value_min=1e-5, clip_value_max=1e5)
        outer_diag = (outer[..., 0] ** 2) + (outer[..., 1] ** 2)
        union = area1 + area2 - inter_area
        dious = inter_area / tf.maximum(union, 1e-5) - (inter_diag) / tf.maximum(outer_diag, 1e-5)
        dious = tf.clip_by_value(dious, clip_value_min=-1.0, clip_value_max=1.0)
        if exchange:
            dious = dious.T
        return dious

    def bbox_ciou(self, boxes1, boxes2):
        '''
        计算两个框的ciou
        :param boxes1:
        :param boxes2:
        :return:
        '''
        exchange = False
        if boxes1.shape[0] > boxes2.shape[0]:
            boxes1, boxes2 = boxes2, boxes1
            exchange = True

        w1 = boxes1[..., 2] - boxes1[..., 0]
        h1 = boxes1[..., 3] - boxes1[..., 1]
        w2 = boxes2[..., 2] - boxes2[..., 0]
        h2 = boxes2[..., 3] - boxes2[..., 1]

        area1 = w1 * h1
        area2 = w2 * h2

        center_x1 = (boxes1[..., 2] + boxes1[..., 0]) / 2  # （x1max +x1min）/2
        center_y1 = (boxes1[..., 3] + boxes1[..., 1]) / 2  # (y1max+y1min)/2
        center_x2 = (boxes2[..., 2] + boxes2[..., 0]) / 2
        center_y2 = (boxes2[..., 3] + boxes2[..., 1]) / 2

        inter_max_xy = tf.minimum(boxes1[..., 2:], boxes2[..., 2:])  # min((x1max,y1max ),(x2max,y2max)) ->返回较小一组
        inter_min_xy = tf.maximum(boxes1[..., :2], boxes2[..., :2])  # max((x1min,y1min ),(x2min,y2min))->返回较大的一组
        out_max_xy = tf.maximum(boxes1[..., 2:], boxes2[..., 2:])
        out_min_xy = tf.minimum(boxes1[..., :2], boxes2[..., :2])

        inter = tf.clip_by_value((inter_max_xy - inter_min_xy), clip_value_min=1e-5, clip_value_max=10000)
        inter_area = inter[..., 0] * inter[..., 1]
        inter_diag = (center_x2 - center_x1) ** 2 + (center_y2 - center_y1) ** 2
        outer = tf.clip_by_value((out_max_xy - out_min_xy), clip_value_min=1e-5, clip_value_max=10000)
        outer_diag = tf.maximum((outer[..., 0] ** 2) + (outer[..., 1] ** 2), 1e-5)
        union = tf.maximum(area1 + area2 - inter_area, 1e-5)
        u = (inter_diag) / outer_diag
        iou = inter_area / union
        arctan = tf.atan(w2 / tf.maximum(h2, 1e-5)) - tf.atan(w1 / tf.maximum(h1, 1e-5))
        v = (4 / (math.pi ** 2)) * tf.pow(tf.atan(w2 / tf.maximum(h2, 1e-5)) - tf.atan(tf.maximum(h1, 1e-5)), 2)
        alpha = v / tf.maximum((1 - iou + v), 1e-5)
        # w_temp = 2 * w1
        # ar = (8 / (math.pi ** 2)) * arctan * ((w1 - w_temp) * h1)
        cious = iou - u - alpha * v
        cious = tf.clip_by_value(cious, clip_value_min=-1.0, clip_value_max=1.0)
        if exchange:
            cious = cious.T
        return cious

    def bbox_iou(self, boxes1, boxes2):
        '''
        计算两个框的iou
        :param boxes1:
        :param boxes2:
        :return:
        '''

        boxes1_area = boxes1[..., 2] * boxes1[..., 3]
        boxes2_area = boxes2[..., 2] * boxes2[..., 3]

        boxes1 = tf.concat([boxes1[..., :2] - boxes1[..., 2:] * 0.5,
                            boxes1[..., :2] + boxes1[..., 2:] * 0.5], axis=-1)
        boxes2 = tf.concat([boxes2[..., :2] - boxes2[..., 2:] * 0.5,
                            boxes2[..., :2] + boxes2[..., 2:] * 0.5], axis=-1)

        left_up = tf.maximum(boxes1[..., :2], boxes2[..., :2])
        right_down = tf.minimum(boxes1[..., 2:], boxes2[..., 2:])

        inter_section = tf.maximum(right_down - left_up, 0.0)
        inter_area = inter_section[..., 0] * inter_section[..., 1]
        union_area = tf.maximum(boxes1_area + boxes2_area - inter_area, 1e-5)
        iou = 1.0 * inter_area / union_area

        return iou

    def loss_layer(self, conv, pred, label, bboxes, anchors, stride):

        conv_shape = tf.shape(conv)
        batch_size = conv_shape[0]
        output_size = conv_shape[1]
        input_size = stride * output_size
        conv = tf.reshape(conv, (batch_size, output_size, output_size,
                                 self.anchor_per_scale, 5 + self.num_class))
        conv_raw_conf = conv[:, :, :, :, 4:5]
        conv_raw_prob = conv[:, :, :, :, 5:]

        pred_xywh = pred[:, :, :, :, 0:4]
        pred_conf = pred[:, :, :, :, 4:5]

        label_xywh = label[:, :, :, :, 0:4]
        respond_bbox = label[:, :, :, :, 4:5]
        label_prob = label[:, :, :, :, 5:]

        self.giou = tf.expand_dims(self.bbox_giou(pred_xywh, label_xywh), axis=-1)  # giou
        self.diou = tf.expand_dims(self.bbox_diou(pred_xywh, label_xywh), axis=-1)  # diou
        self.ciou = tf.expand_dims(self.bbox_ciou(pred_xywh, label_xywh), axis=-1)  # ciou

        input_size = tf.cast(input_size, tf.float32)

        self.bbox_loss_scale = 2.0 - 1.0 * label_xywh[:, :, :, :, 2:3] * label_xywh[:, :, :, :, 3:4] / (input_size ** 2)

        giou_loss = respond_bbox * self.bbox_loss_scale * (1 - self.giou)  # giou loss
        diou_loss = respond_bbox * self.bbox_loss_scale * (1 - self.diou)  # diou loss
        ciou_loss = respond_bbox * self.bbox_loss_scale * (1 - self.ciou)

        iou = self.bbox_iou(pred_xywh[:, :, :, :, np.newaxis, :], bboxes[:, np.newaxis, np.newaxis, np.newaxis, :, :])
        max_iou = tf.expand_dims(tf.reduce_max(iou, axis=-1), axis=-1)

        respond_bgd = (1.0 - respond_bbox) * tf.cast(max_iou < self.iou_loss_thresh, tf.float32)

        conf_focal = self.focal(respond_bbox, pred_conf)

        conf_loss = conf_focal * (
                respond_bbox * tf.nn.sigmoid_cross_entropy_with_logits(labels=respond_bbox, logits=conv_raw_conf)
                +
                respond_bgd * tf.nn.sigmoid_cross_entropy_with_logits(labels=respond_bbox, logits=conv_raw_conf)
        )

        prob_loss = respond_bbox * tf.nn.sigmoid_cross_entropy_with_logits(labels=label_prob, logits=conv_raw_prob)

        giou_loss = tf.reduce_mean(tf.reduce_sum(giou_loss, axis=[1, 2, 3, 4]))  # giou loss
        diou_loss = tf.reduce_mean(tf.reduce_sum(diou_loss, axis=[1, 2, 3, 4]))  # diou loss
        ciou_loss = tf.reduce_mean(tf.reduce_sum(ciou_loss, axis=[1, 2, 3, 4]))  # ciou loss
        # giou_loss = tf.reduce_mean(tf.reduce_sum(bbox_loss_scale, axis=[1, 2, 3, 4]))
        conf_loss = tf.reduce_mean(tf.reduce_sum(conf_loss, axis=[1, 2, 3, 4]))
        prob_loss = tf.reduce_mean(tf.reduce_sum(prob_loss, axis=[1, 2, 3, 4]))

        return diou_loss, conf_loss, prob_loss, tf.reduce_sum(self.ciou, axis=[1, 2, 3, 4]), tf.reduce_sum(
            self.bbox_loss_scale)

    def compute_loss(self, label_sbbox, label_mbbox, label_lbbox, true_sbbox, true_mbbox, true_lbbox):

        with tf.name_scope('smaller_box_loss'):
            loss_sbbox = self.loss_layer(self.conv_sbbox, self.pred_sbbox, label_sbbox, true_sbbox,
                                         anchors=self.anchors[0], stride=self.strides[0])

        with tf.name_scope('medium_box_loss'):
            loss_mbbox = self.loss_layer(self.conv_mbbox, self.pred_mbbox, label_mbbox, true_mbbox,
                                         anchors=self.anchors[1], stride=self.strides[1])

        with tf.name_scope('bigger_box_loss'):
            loss_lbbox = self.loss_layer(self.conv_lbbox, self.pred_lbbox, label_lbbox, true_lbbox,
                                         anchors=self.anchors[2], stride=self.strides[2])

        with tf.name_scope('giou_loss'):
            giou_loss = loss_sbbox[0] + loss_mbbox[0] + loss_lbbox[0]

        with tf.name_scope('conf_loss'):
            conf_loss = loss_sbbox[1] + loss_mbbox[1] + loss_lbbox[1]

        with tf.name_scope('prob_loss'):
            prob_loss = loss_sbbox[2] + loss_mbbox[2] + loss_lbbox[2]

        with tf.name_scope('giou'):
            giou = loss_sbbox[3]
        with tf.name_scope('bbox_loss_scale'):
            bbox_loss_scale = loss_sbbox[4]

        return giou_loss, conf_loss, prob_loss, giou, bbox_loss_scale

    def layer_loss(self, layer_label):
        layer_label = tf.cast(layer_label, tf.int32)
        # Y = tf.one_hot(layer_label, depth=self.layer_nums, axis=1, dtype=tf.float32)
        cost = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=layer_label, logits=self.out),
                              name='layer_loss')
        # cost=tf.losses.mean_squared_error(labels=layer_label,predictions=self.predict_op)
        return cost
