# coding=utf-8

'''
ckpt文件评估test集

'''
import cv2
import os
import shutil
from tqdm import tqdm
import numpy as np
import tensorflow as tf
import multi_detection.core.utils as utils
from multi_detection.core.config import cfg
from sklearn.metrics import confusion_matrix


class YoloTest(object):
    def __init__(self):
        self.input_size = cfg.TEST.INPUT_SIZE
        self.anchor_per_scale = cfg.YOLO.ANCHOR_PER_SCALE
        self.classes = utils.read_class_names(cfg.YOLO.CLASSES)
        self.num_classes = len(self.classes)
        self.anchors = np.array(utils.get_anchors(cfg.YOLO.ANCHORS))
        self.score_threshold = cfg.TEST.SCORE_THRESHOLD
        self.iou_threshold = cfg.TEST.IOU_THRESHOLD
        self.moving_ave_decay = cfg.YOLO.MOVING_AVE_DECAY
        self.annotation_path = cfg.TEST.ANNOT_PATH
        self.weight_file = "E:/ckpt_dirs/Food_detection/multi_food3/checkpoint/yolov3_train_loss=11.4018.ckpt-98"
        self.write_image = cfg.TEST.WRITE_IMAGE
        self.write_image_path = cfg.TEST.WRITE_IMAGE_PATH
        self.show_label = cfg.TEST.SHOW_LABEL

        graph = tf.Graph()
        with graph.as_default():
            self.saver = tf.train.import_meta_graph("{}.meta".format(self.weight_file))
            self.sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True))
            self.saver.restore(self.sess, self.weight_file)

            self.input = graph.get_tensor_by_name("define_input/input_data:0")
            self.trainable = graph.get_tensor_by_name("define_input/training:0")

            self.pred_sbbox = graph.get_tensor_by_name("define_loss/pred_sbbox/concat_2:0")
            self.pred_mbbox = graph.get_tensor_by_name("define_loss/pred_mbbox/concat_2:0")
            self.pred_lbbox = graph.get_tensor_by_name("define_loss/pred_lbbox/concat_2:0")

            self.layer_num = graph.get_tensor_by_name("define_loss/layer_classes:0")

    def predict(self, image):
        org_image = np.copy(image)
        org_h, org_w, _ = org_image.shape

        image_data = utils.image_preporcess(image, [self.input_size, self.input_size])
        image_data = image_data[np.newaxis, ...]

        pred_sbbox, pred_mbbox, pred_lbbox, layer_ = self.sess.run(
            [self.pred_sbbox, self.pred_mbbox, self.pred_lbbox, self.layer_num],
            feed_dict={
                self.input: image_data,
                self.trainable: False
            }
        )

        pred_bbox = np.concatenate([np.reshape(pred_sbbox, (-1, 5 + self.num_classes)),
                                    np.reshape(pred_mbbox, (-1, 5 + self.num_classes)),
                                    np.reshape(pred_lbbox, (-1, 5 + self.num_classes))], axis=0)

        bboxes = utils.postprocess_boxes(pred_bbox, (org_h, org_w), self.input_size, 0.45)
        bboxes = utils.nms(bboxes, self.iou_threshold)

        return bboxes, layer_

    def evaluate(self):
        predicted_dir_path = './mAP/predicted'
        ground_truth_dir_path = './mAP/ground-truth'
        error_layer_dir = "./mAP/error_layer"
        if os.path.exists(predicted_dir_path): shutil.rmtree(predicted_dir_path)
        if os.path.exists(ground_truth_dir_path): shutil.rmtree(ground_truth_dir_path)
        if os.path.exists(self.write_image_path): shutil.rmtree(self.write_image_path)
        if os.path.exists(error_layer_dir): shutil.rmtree(error_layer_dir)
        os.mkdir(predicted_dir_path)
        os.mkdir(ground_truth_dir_path)
        os.mkdir(self.write_image_path)
        os.mkdir(error_layer_dir)

        layer_pre = []
        layer_true = []

        with open(self.annotation_path, 'r') as annotation_file:
            for line in tqdm(annotation_file):
                annotation = line.strip().split()
                image_path = annotation[0]

                img_layer_true = annotation[1]  # 获取标准layer
                layer_true.append(int(img_layer_true))  # 写入到layer_true

                image_name = image_path.split('/')[-1]
                image = cv2.imread(image_path)
                # image = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)   # RGB空间转为HSV空间
                bbox_data_gt = np.array([list(map(int, box.split(','))) for box in annotation[2:]])

                if len(bbox_data_gt) == 0:
                    bboxes_gt = []
                    classes_gt = []
                else:
                    bboxes_gt, classes_gt = bbox_data_gt[:, :4], bbox_data_gt[:, 4]
                ground_truth_path = os.path.join(ground_truth_dir_path, image_name.split(".")[0] + '.txt')

                print('=> ground truth of %s:' % image_name)
                num_bbox_gt = len(bboxes_gt)
                with open(ground_truth_path, 'w') as f:
                    for i in range(num_bbox_gt):
                        class_name = self.classes[classes_gt[i]]
                        xmin, ymin, xmax, ymax = list(map(str, bboxes_gt[i]))
                        bbox_mess = ' '.join([class_name, xmin, ymin, xmax, ymax]) + '\n'
                        f.write(bbox_mess)
                        print('\t' + str(bbox_mess).strip())
                print('=> predict result of %s:' % image_name)
                predict_result_path = os.path.join(predicted_dir_path, image_name.split(".")[0] + '.txt')
                bboxes_pr, layer_ = self.predict(image)
                layer_pre.append(layer_[0])  # 预测layer写入到layer_pre

                if int(layer_[0]) != int(img_layer_true):  # 将layer错误的图片拷贝至error_layer
                    print("no   ")
                    print(layer_[0], img_layer_true)
                    shutil.copy(image_path,
                                error_layer_dir + "/" + image_name.split(".")[0] + "_" + str(layer_[0]) + ".jpg")

                if self.write_image:
                    image = utils.draw_bbox(image, bboxes_pr, show_label=self.show_label)
                    # drawed_img_save_to_path = os.path.join(self.write_image_path, image_name)
                    drawed_img_save_to_path = self.write_image_path + "/" + image_name

                    cv2.imwrite(drawed_img_save_to_path, image)

                with open(predict_result_path, 'w') as f:
                    for bbox in bboxes_pr:
                        coor = np.array(bbox[:4], dtype=np.int32)
                        score = bbox[4]
                        class_ind = int(bbox[5])
                        class_name = self.classes[class_ind]
                        score = '%.4f' % score
                        xmin, ymin, xmax, ymax = list(map(str, coor))
                        bbox_mess = ' '.join([class_name, score, xmin, ymin, xmax, ymax]) + '\n'
                        f.write(bbox_mess)
                        print('\t' + str(bbox_mess).strip())
        print(layer_true)
        print(layer_pre)
        matrix = confusion_matrix(layer_true, layer_pre, labels=[0, 1, 2, 3])
        print("烤层检测混淆矩阵：")
        print(matrix)


if __name__ == '__main__':
    import time

    s = time.time()
    Y = YoloTest()
    s_l_time = time.time()
    print("model load time:", s_l_time - s)
    Y.evaluate()
    e = time.time()
    print("predict all time::::", e - s_l_time)
