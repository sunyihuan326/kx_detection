# -*- coding: utf-8 -*-
# @Time    : 202003/2/27
# @Author  : sunyihuan
# @File    : food_correct_utils.py

'''
本文件存储预测结果后需要矫正的方法，如：检测框矫正，土豆红薯烤制时间调整等
'''

import math
from multi_detection.core import utils


def get_potatoscale(bboxes_pr, layer_n):
    '''
    最新修改时间：202003/2/27

    获得土豆红薯的大小级别参数
    :param bboxes_pr: 模型预测结果，格式为[x_min, y_min, x_max, y_max, probability, cls_id]
    :param layer_n:
    :return: bboxes_pr: 模型预测结果，格式为[x_min, y_min, x_max, y_max, probability, cls_id, scale_id]
             layer_n:
    '''
    num_label = len(bboxes_pr)

    if layer_n == 1:  # 烤盘中间层
        num_bboxes = 0
        sum_length = 0.
        for i in range(num_label):
            w_box = bboxes_pr[i][2] - bboxes_pr[i][0]
            h_bos = bboxes_pr[i][3] - bboxes_pr[i][1]
            centerx = bboxes_pr[i][0] + w_box // 2
            centery = bboxes_pr[i][1] + h_bos // 2
            # 求box中心点到图像左下角点（0，600）的距离dis
            dis = abs(math.sqrt(centerx ** 2 + (600 - centery) ** 2))
            alpha = dis / 560.
            if alpha < 1.0:
                alpha = 1 - (560 - dis) / 1000
            w = w_box * alpha
            h = h_bos * alpha
            length = round(abs(math.sqrt(w ** 2 + h ** 2)), 2)
            ratio = min(w / h, h / w)
            beta = lambda x: x if x > 0.8 else 0.8
            correct_length = round(length * beta(ratio), 2)

            sum_length += correct_length
            num_bboxes += 1

        ave_length = round(((sum_length / num_bboxes) / 20) * 0.9, 2)
        return ave_length

    elif layer_n == 2:  # 烤盘架最上层
        num_bboxes = 0
        sum_length = 0.
        for i in range(num_label):
            w_box = bboxes_pr[i][2] - bboxes_pr[i][0]
            h_bos = bboxes_pr[i][3] - bboxes_pr[i][1]
            centerx = bboxes_pr[i][0] + w_box // 2
            centery = bboxes_pr[i][1] + h_bos // 2
            # 求box中心点到图像左下角点（0，600）的距离dis
            dis = abs(math.sqrt(centerx ** 2 + (600 - centery) ** 2))
            alpha = dis / 700.
            w = w_box * alpha
            h = h_bos * alpha
            length = round(abs(math.sqrt(w ** 2 + h ** 2)), 2)
            ratio = min(w / h, h / w)
            beta = lambda x: x if x > 0.8 else 0.8
            correct_length = round(length * beta(ratio), 2)

            sum_length += correct_length
            num_bboxes += 1

        ave_length = round(((sum_length / num_bboxes) / 20) * 0.8, 2)
        return ave_length

    else:  # 烤盘最下层、其他层
        num_bboxes = 0
        sum_length = 0.
        for i in range(num_label):
            w_box = bboxes_pr[i][2] - bboxes_pr[i][0]
            h_bos = bboxes_pr[i][3] - bboxes_pr[i][1]
            centerx = bboxes_pr[i][0] + w_box // 2
            centery = bboxes_pr[i][1] + h_bos // 2
            # 求box中心点到图像左下角点（0，600）的距离dis
            dis = abs(math.sqrt(centerx ** 2 + (600 - centery) ** 2))
            # alpha = dis/500.
            alpha = dis / 500.
            if alpha < 1.0:
                alpha = 1 - (500 - dis) / 1000
            w = w_box * alpha
            h = h_bos * alpha
            length = round(abs(math.sqrt(w ** 2 + h ** 2)), 2)
            ratio = min(w / h, h / w)
            beta = lambda x: x if x > 0.8 else 0.8
            correct_length = round(length * beta(ratio), 2)

            sum_length += correct_length
            num_bboxes += 1

        ave_length = round((sum_length / num_bboxes) / 20, 2)
        return ave_length


def get_time(obj_c, obj_length, layer_nums):
    '''
    最新修改时间：202003/2/27

    根据红薯或者土豆长度，得出需要烤制的时间
    :param obj_c: 食材标签
    :param obj_length: 食材长度
    :param layer_nums: 烤层
    :return:work_time:烤制时间
    '''
    if obj_c in [18, 19, 20, 29]:  # 土豆标签
        if layer_nums == 0:  # 最底层
            if obj_length > 15:
                work_time = 105
            elif 8 < obj_length <= 15:
                work_time = math.ceil(7 * obj_length)  # 小数向上取整
            else:
                work_time = 50
        elif layer_nums == 1:  # 中间层
            if obj_length > 15:
                work_time = 95
            elif 8 < obj_length <= 15:
                work_time = math.ceil(6.5 * obj_length)  # 小数向上取整
            else:
                work_time = 50
        elif layer_nums == 2:  # 最上层
            if obj_length > 15:
                work_time = 80
            elif 8 < obj_length <= 15:
                work_time = math.ceil(4 * obj_length + 20)  # 小数向上取整
            else:
                work_time = 50
        else:
            work_time = 105
    elif obj_c in [22, 23, 24, 27]:  # 红薯标签
        if obj_length > 15:  # >15
            work_time = 80
        elif 8 < obj_length <= 15:  # (8,15]
            work_time = math.ceil(5.5 * obj_length)  # 小数向上取整
        else:  # <=8
            work_time = 40
    else:
        work_time = 0
    return work_time


def get_chiffon_size(layer, bboxes):
    '''
    根据烤层和检测框信息，判断戚风尺寸

    :return:
    '''
    xmin = int(bboxes[0][0])
    ymin = int(bboxes[0][1])
    xmax = int(bboxes[0][2])
    ymax = int(bboxes[0][3])
    cls = int(bboxes[0][5])
    x_weight = int(xmax - xmin)
    y_high = int(ymax - ymin)
    size = cls + 3
    if int(layer) == 0:
        if x_weight <= 440:
            size = 4
    elif int(layer) == 1:
        if x_weight <= 485:
            size = 4
    return size


def get_potatoml(bboxes_pr, layer_n):
    '''
    判断大土豆为大土豆或者中土豆
    判断大红薯为大红薯或者中红薯

    :param bboxes_pr:
    :param layer_n:
    :return:
    '''
    if len(bboxes_pr) == 0:  # 无任何结果直接输出
        return bboxes_pr, layer_n
    else:
        cls_list = []  # 标签类别，主要存储土豆红薯标签，用于将大红薯分为中、大，大土豆分为中、大
        for c in bboxes_pr:
            if int(c[-1]) == 16:  # 若输出类别结果为大土豆
                if int(c[-1]) not in cls_list:  # 若标签列表中无大土豆，加入
                    cls_list.append(int(c[-1]))
            elif int(c[-1]) == 17:  # 若输出类别为小土豆
                if 15 in cls_list and int(c[-1]) not in cls_list:  # 若大土豆在标签中，且标签列别中无小土豆，标签加入
                    cls_list.append(int(c[-1]))
            elif int(c[-1]) == 19:  # 若输出类别结果为大红薯
                if int(c[-1]) not in cls_list:  # 若标签列表中无大红薯，标签加入
                    cls_list.append(int(c[-1]))
            elif int(c[-1]) == 20:  # 若输出类别结果为小红薯
                if 18 in cls_list and int(c[-1]) not in cls_list:  # 若大红薯在标签中，且标签列别中无小红薯，标签加入
                    cls_list.append(int(c[-1]))
        if len(cls_list) == 0:  # 若标签类别为空，直接返回结果
            return bboxes_pr, layer_n
        else:  # 若标签结果有值，获取长度、判断大中
            ave_length = get_potatoscale(bboxes_pr, layer_n)
            if int(cls_list[0]) == 16 or cls_list == [17, 16]:  # 标签为大土豆,或者大小土豆
                if int(ave_length) < 14:  # 长度小于14，结果为中土豆，label标签为：40
                    bboxes_pr_new = []
                    for bb in bboxes_pr:
                        bb[-1] = 40
                        bboxes_pr_new.append(bb)
                    return bboxes_pr_new, layer_n
                else:
                    return bboxes_pr, layer_n
            elif int(cls_list[0]) == 19 or cls_list == [20, 19]:  # 标签为大红薯，或者大小红薯
                if int(ave_length) < 18:  # 长度小于18，结果为中红薯，label标签为：41
                    bboxes_pr_new = []
                    for bb in bboxes_pr:
                        bb[-1] = 41
                        bboxes_pr_new.append(bb)
                    return bboxes_pr_new, layer_n
                else:
                    return bboxes_pr, layer_n
            else:
                return bboxes_pr, layer_n

def reduce_score(bboxes_pr, layer_n, best_bboxes, CL_type):
    '''
    降分处理

    by:sunyihuan
     修改时间：2020年11月6日  去掉上层置信度处理
              同步修改best_bboxes中的分值



    若烤层为上层，置信度大于0.9降为0.89

    若检测为戚风，判断大小，若为4寸，置信度打9折

    若不是同一大类，降分处理，置信度打9折

    :param bboxes_pr:
    :param layer_n:
    :return:
    '''
    if int(layer_n) == 2:
        for i in range(len(bboxes_pr)):
            if float(bboxes_pr[i][4]) >= 0.9:
                pass
                # bboxes_pr[i][4] = 0.89
    elif int(bboxes_pr[0][0]) in [3, 4]:
        for i in range(len(bboxes_pr)):
            bboxes_pr[i][4] = bboxes_pr[i][4] * 0.9
        best_bboxes = [[bb[0], bb[1] * 0.9] for bb in best_bboxes]
    elif not CL_type:
        for i in range(len(bboxes_pr)):
            bboxes_pr[i][4] = bboxes_pr[i][4] * 0.9
        best_bboxes = [[bb[0], bb[1] * 0.9] for bb in best_bboxes]
    return bboxes_pr, layer_n, best_bboxes


def correct_bboxes(bboxes_pr, layer_n, best_bboxes):
    '''
    最新修改时间：2020/9/21
    by：sunyihuan
              针对40分类
    修改目的：输出结果为最上层则降低置信度
    处理说明：1、无任何检测框，直接输出
             2、若烤层为最上层，且最高置信度高于0.9，降低为0.89


    最新修改时间：2020/8/26
    by：sunyihuan
              针对39分类
    修改目的：类别和数量控制输出分数，尽量避免出现识别错误直接跳转，
    处理说明：1、无任何检测框，直接输出
             2、仅检测到1个框，对低于0.5分的nofood置信度设置为0.8
             3、检测到多个框，但是同一个标签，针对卡通饼干、曲奇、蔓越莓、排骨数量现置（小于3个）置信度设置为0.75，其余的直接输出结果
             4、检测到多个框，但是标签不一致：（1）先对大小土豆、大小红薯处理，若出现大，则为大
                                            （2）若出现两个种类，排骨、鸡翅，判断为鸡翅
                                            （3）若出现两个种类，排骨、串，判断为串
                                            （4）若出现两个种类，鱼、切开茄子，判断为切开茄子
                                            （5）若出现两个种类，曲奇、鸡翅，判断为曲奇
                                            （6）若标签不在一个大类中，置信度降分处理，若在一个大类，直接输出


    最新修改时间：2020/7/13
    by：sunyihuan
    修改目的：类别和数量控制输出分数，尽量避免出现识别错误直接跳转，
    处理说明：1、无任何检测框，直接输出
             2、仅检测到1个框，对低于0.5分的nofood置信度设置为0.75
             3、检测到多个框，但是同一个标签，针对鸡翅排骨做数量现置（小于3个）置信度设置为0.75，其余的直接输出结果
             4、检测到多个框，但是标签不一致：（1）先对大小土豆、大小红薯处理，若出现大，则为大
                                            （2）若标签不在一个大类中，置信度降分处理，若在一个大类，直接输出

    bboxes_pr结果矫正
    :param bboxes_pr: 模型预测结果，格式为[x_min, y_min, x_max, y_max, probability, cls_id]
    :param layer_n:
    :return:
    '''
    best_bboxes_ = []
    for bb in best_bboxes:
        bb = list(bb)
        best_bboxes_.append(list(bb))
    best_bboxes = best_bboxes_

    num_label = len(bboxes_pr)
    reduce_score_typ = True  # 判断其余标签和第一个标签是否在一个大类中

    # 未检测食材，直接输出
    if num_label == 0:
        if best_bboxes[0][1] < 0.5:  # 若top1置信度低于0.5，直接输出为空
            best_bboxes = []
        return bboxes_pr, layer_n, best_bboxes

    # 检测到一个食材
    elif num_label == 1:  # 小于0.5的nofood输出，其他删除

        if int(bboxes_pr[0][5]) in [3, 4]:  # 判断戚风尺寸，若为4寸，置信度*0.9
            size = get_chiffon_size(layer_n, bboxes_pr)
            if size == 4:
                reduce_score_typ = False
                bboxes_pr[0][5] = 101  # 戚风如果为4寸，则输出类别为101

        bboxes_pr, layer_n, best_bboxes = reduce_score(bboxes_pr, layer_n, best_bboxes, reduce_score_typ)  # 根据烤层对置信度做处理
        if bboxes_pr[0][4] < 0.5:
            if bboxes_pr[0][5] == 9:  # 低分nofood,设置为0.8
                bboxes_pr[0][4] = 0.8
            # else:
            #     del bboxes_pr[0]

        # 置信度低于0.5分值，直接限制
        nnew_bboxes_pr = []
        for i in range(len(bboxes_pr)):
            if bboxes_pr[i][4] > 0.5:
                nnew_bboxes_pr.append(bboxes_pr[i])
        bboxes_pr = nnew_bboxes_pr
        if best_bboxes[0][1] < 0.5:  # 若top1置信度低于0.5，直接输出为空
            best_bboxes = []
        return bboxes_pr, layer_n, best_bboxes

    # 检测到多个食材
    else:
        # 判断标签类别是否为同一个
        new_bboxes_pr = sorted(bboxes_pr, key=lambda x: x[4], reverse=True)  # 对检测框按置信度降序排序

        if int(new_bboxes_pr[0][5]) in [3, 4]:  # 判断戚风尺寸，若为4寸，置信度*0.9
            size = get_chiffon_size(layer_n, bboxes_pr)
            if size == 4:
                reduce_score_typ = False
                for i in range(len(new_bboxes_pr)):
                    new_bboxes_pr[i][5] = 101  # 戚风如果为4寸，则输出类别为101

        new_num_label = len(new_bboxes_pr)
        if new_num_label == 0:
            if best_bboxes[0][1] < 0.5:  # 若top1置信度低于0.5，直接输出为空
                best_bboxes = []
            return new_bboxes_pr, layer_n, best_bboxes

        same_label = True
        for i in range(new_num_label):
            if i == (new_num_label - 1):
                break
            if new_bboxes_pr[i][5] == new_bboxes_pr[i + 1][5]:
                continue
            else:
                same_label = False

        sumProb = 0.
        # 多个食材，同一标签
        if same_label:  # 多个检测框，同一个标签，直接输出结果

            if new_bboxes_pr[0][-1] in [1, 5, 6, 14]:  # 若类别为卡通饼干、曲奇、蔓越莓、排骨，判断数量是否小于3，若小于3降低score分值
                if len(new_bboxes_pr) < 3:
                    reduce_score_typ = False

            new_bboxes_pr, layer_n, best_bboxes = reduce_score(new_bboxes_pr, layer_n, best_bboxes,
                                                           reduce_score_typ)  # 根据烤层及降分标志对置信度做处理

            # 置信度低于0.5分值，直接限制
            nnew_bboxes_pr = []
            for i in range(len(new_bboxes_pr)):
                if new_bboxes_pr[i][4] > 0.5:
                    nnew_bboxes_pr.append(new_bboxes_pr[i])
            if best_bboxes[0][1] < 0.5:  # 若top1置信度低于0.5，直接输出为空
                best_bboxes = []
            return new_bboxes_pr, layer_n, best_bboxes

        # 多个食材，非同一标签
        else:
            labellist = list(map(lambda x: x[5], new_bboxes_pr))

            labeldict = {}
            for key in labellist:
                labeldict[key] = labeldict.get(key, 0) + 1
                # 按同种食材label数量降序排列
            s_labeldict = sorted(labeldict.items(), key=lambda x: x[1], reverse=True)

            name1 = s_labeldict[0][0]
            name2 = s_labeldict[1][0]

            n_nums = len(s_labeldict)

            if n_nums == 2:
                # 如果土豆中检测到了大土豆，默认单一食材为大土豆
                # if (name1 == 17 and name2 == 18) or (name1 == 18 and name2 == 17):
                if (name1 == 16 and name2 == 17) or (name1 == 17 and name2 == 16):
                    for i in range(new_num_label):
                        new_bboxes_pr[i][5] = 16
                    # return new_bboxes_pr, layer_n
                # 如果红薯中检测到了大红薯，默认单一食材为大红薯
                # if (name1 == 21 and name2 == 22) or (name1 == 22 and name2 == 21):
                if (name1 == 19 and name2 == 20) or (name1 == 20 and name2 == 19):
                    for i in range(new_num_label):
                        new_bboxes_pr[i][5] = 19

                # 如果出现串、排骨，判断为串
                if (name1 == 14 and name2 == 38) or (name1 == 38 and name2 == 14):
                    for i in range(new_num_label):
                        new_bboxes_pr[i][5] = 38
                # 如果出现鸡翅、排骨，判断为鸡翅
                if (name1 == 14 and name2 == 2) or (name1 == 2 and name2 == 14):
                    for i in range(new_num_label):
                        new_bboxes_pr[i][5] = 2
                # 如果出现鱼、切开茄子，判断为切开茄子
                if (name1 == 30 and name2 == 34) or (name1 == 34 and name2 == 30):
                    for i in range(new_num_label):
                        new_bboxes_pr[i][5] = 30
                # 如果出现曲奇、鸡翅，判断为曲奇
                if (name1 == 5 and name2 == 2) or (name1 == 2 and name2 == 5):
                    for i in range(new_num_label):
                        new_bboxes_pr[i][5] = 5

            # 12大类，分别为：披萨、土豆、红薯、饼干、肉类、蛋糕、烤鸡、牛排、蛋挞、花生米、吐司、空
            # CL_es = [[10, 11, 12], [14, 15, 16], [17, 18, 19], [1, 4, 5], [2, 13], [3, 6], [20], [0], [7], [9],
            #          [21], [8]]

            # 24大类，分别为：披萨、土豆、红薯、饼干、肉类、蛋糕、烤鸡、牛排、蛋挞、花生米、吐司、空、板栗、玉米、鸡腿、芋头、小馒头、茄子、面包、容器、鱼、热狗、虾、串
            CL_es = [[13, 11, 12], [17, 15, 16], [20, 18, 19], [1, 6, 5], [2, 14], [3, 4, 7], [21], [0], [8], [10],
                     [22], [9], [23], [24, 25], [26], [27], [28], [29, 30], [31], [32, 33], [34], [35], [36, 37], [38]]

            c = int(s_labeldict[0][0])
            cl_es = []
            for i in range(len(CL_es)):
                if c in CL_es[i]:
                    cl_es = CL_es[i]
                    break

            for k in range(len(s_labeldict) - 1):
                if s_labeldict[k + 1][0] not in cl_es:
                    reduce_score_typ = False
                    break
            new_bboxes_pr, layer_n, best_bboxes = reduce_score(new_bboxes_pr, layer_n, best_bboxes,
                                                               reduce_score_typ)  # 根据烤层、不属于同一大类，对置信度做处理
            nnew_bboxes_pr = []
            for i in range(len(new_bboxes_pr)):
                if new_bboxes_pr[i][4] > 0.5:
                    nnew_bboxes_pr.append(new_bboxes_pr[i])
            if best_bboxes[0][1] < 0.5:  # 若top1置信度低于0.5，直接输出为空
                best_bboxes = []
            return nnew_bboxes_pr, layer_n, best_bboxes
