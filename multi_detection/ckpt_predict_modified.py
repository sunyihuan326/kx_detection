# -*- encoding: utf-8 -*-

"""
@File    : ckpt_predict.py
@Time    : 2019/9/18
@Author  : sunyihuan
@Modify  : FreeBird 2019/11/06
"""

import cv2
import numpy as np
import tensorflow as tf
import utils


class YoloPredict(object):
    '''
    预测结果
    '''

    def __init__(self):
        self.input_size = 416  # 输入图片尺寸（默认正方形）
        self.num_classes = utils.class_nums()  # 种类数
        self.score_threshold = 0.45
        self.iou_threshold = 0.5
        self.weight_file = "checkpoint/yolov3_1028.ckpt"  # ckpt文件地址

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

        bboxes = utils.postprocess_boxes(pred_bbox, (org_h, org_w), self.input_size, self.score_threshold)
        bboxes = utils.nms(bboxes, self.iou_threshold)

        return bboxes, layer_[0]

    def result(self, image_path):
        image = cv2.imread(image_path)  # 图片读取
        bboxes_pr, layer_n = self.predict(image)
        # 预测结果,bboxes_pr输出格式为[x_min, y_min, x_max, y_max, probability, cls_id]
                        # cls_id对应标签[0:beefsteak,1:cartooncookies,2:chickenwings,3:chiffoncake6,4:chiffoncake8,
                                       #  5:cookies,6:cranberrycookies,7:cupcake,8:eggtart,9:eggtartbig,10:nofood,
                                       # 11:peanuts,12:pizzafour,13:pizzaone,14:pizzasix,15:pizzatwo,16:porkchops,
                                       # 17:potatocut,18:potatol,19:potatom,20:potatos,
                                       # 21:sweetpotatocut,22:sweetpotatol,23:sweetpotatom,24:sweetpotatos,
                                       # 25:roastedchicken,26:toast]
        # 预测结果,layer_输出结果为0：最下层、1：中间层、2：最上层、3：其他
        
        num_label = len(bboxes_pr)
        # 未检测食材
        if num_label == 0:
            return bboxes_pr, layer_n
        
        # 检测到一个食材    
        elif num_label == 1:
            return bboxes_pr, layer_n
        
        # 检测到多个食材    
        else:
            same_label = True
            for i in range(num_label):
                if i ==(num_label-1):
                    break
                if bboxes_pr[i][5] == bboxes_pr[i+1][5]:
                    continue
                else:
                    same_label = False
            # 多个食材，同一标签        
            if same_label:
                sumProb = 0.
                for i in range(num_label):
                    sumProb += bboxes_pr[i][4]
                avrProb = sumProb/num_label
                bboxes_pr[0][4] = avrProb
                return bboxes_pr, layer_n
            # 多个食材，非同一标签
            else:
                sumProb = 0.
                problist = list(map(lambda x:x[4], bboxes_pr))
                labellist = list(map(lambda x:x[5], bboxes_pr))
                
                labeldict = {}
                for key in labellist:
                    labeldict[key]=labeldict.get(key,0)+1    
                # 按同种食材label数量降序排列
                s_labeldict = sorted(labeldict.items(),key=lambda x:x[1], reverse = True)
                
                #n_name = len(s_labeldict)
                name1 = s_labeldict[0][0]
                num_name1 = s_labeldict[0][1]
                
                # 数量最多label对应的食材占比0.7以上
                if num_name1/num_label>0.7:
                    num_label0 = []
                    for i in range(num_label):
                        if name1 == bboxes_pr[i][5]:
                            num_label0.append(bboxes_pr[i])
                    for i in range(len(num_label0)):
                        sumProb += num_label0[i][4]
                    avrProb = sumProb / num_label
                    num_label0[0][4] = avrProb
                    return num_label0, layer_n
                    
                # 按各个label的probability降序排序
                else:
                    bboxes_pr = sorted(bboxes_pr, key=lambda x: x[4],reverse=True)
                    return bboxes_pr, layer_n
        
        #print("识别个数：", num_label)
        #print("食材结果：", bboxes_pr)
        #print("烤层结果：", layer_n)


if __name__ == '__main__':
    img_path = "data/1_191012size6_Pizzatwo.jpg"  # 预测图片地址
    Y = YoloPredict()
    result, layer = Y.result(img_path)
    print(result)
    print(layer)
