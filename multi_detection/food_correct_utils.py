# -*- coding: utf-8 -*-
# @Time    : 202003/2/27
# @Author  : sunyihuan
# @File    : food_correct_utils.py

'''
本文件存储预测结果后需要矫正的方法，如：检测框矫正，土豆红薯烤制时间调整等
'''

import math


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


def correct_bboxes(bboxes_pr, layer_n):
    '''
    最新修改时间：202003/2/27

    bboxes_pr结果矫正
    :param bboxes_pr: 模型预测结果，格式为[x_min, y_min, x_max, y_max, probability, cls_id]
    :param layer_n:
    :return:
    '''
    num_label = len(bboxes_pr)
    # 未检测食材
    if num_label == 0:
        return bboxes_pr, layer_n

    # 检测到一个食材
    elif num_label == 1:
        if bboxes_pr[0][4] < 0.45:
            if bboxes_pr[0][5] == 10:  # 低分nofood
                bboxes_pr[0][4] = 0.75
            elif bboxes_pr[0][5] == 11:  # 低分花生米
                bboxes_pr[0][4] = 0.85
            elif bboxes_pr[0][5] == 25:  # 低分整鸡
                bboxes_pr[0][4] = 0.75
            else:
                del bboxes_pr[0]

        # else:
        #    if bboxes_pr[0][4] < 0.9 and bboxes_pr[0][4] >= 0.6:
        #        bboxes_pr[0][4] = 0.9

        return bboxes_pr, layer_n

    # 检测到多个食材
    else:
        new_bboxes_pr = []
        for i in range(len(bboxes_pr)):
            if bboxes_pr[i][4] >= 0.45:
                new_bboxes_pr.append(bboxes_pr[i])

        new_num_label = len(new_bboxes_pr)
        if new_num_label == 0:
            return new_bboxes_pr, layer_n
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
        if same_label:
            new_bboxes_pr[0][4] = 0.98
            return new_bboxes_pr, layer_n
        # 多个食材，非同一标签
        else:
            problist = list(map(lambda x: x[4], new_bboxes_pr))
            labellist = list(map(lambda x: x[5], new_bboxes_pr))

            labeldict = {}
            for key in labellist:
                labeldict[key] = labeldict.get(key, 0) + 1
                # 按同种食材label数量降序排列
            s_labeldict = sorted(labeldict.items(), key=lambda x: x[1], reverse=True)

            n_name = len(s_labeldict)
            name1 = s_labeldict[0][0]
            num_name1 = s_labeldict[0][1]
            name2 = s_labeldict[1][0]
            num_name2 = s_labeldict[1][1]

            # 优先处理食材特例
            if n_name == 2:
                # 如果鸡翅中检测到了排骨，默认单一食材为鸡翅
                if (name1 == 2 and name2 == 16) or (name1 == 16 and name2 == 2):
                    for i in range(new_num_label):
                        new_bboxes_pr[i][5] = 2
                    return new_bboxes_pr, layer_n
                # 如果对切土豆中检测到了大土豆，默认单一食材为对切土豆
                if (name1 == 17 and name2 == 18) or (name1 == 18 and name2 == 17):
                    for i in range(new_num_label):
                        new_bboxes_pr[i][5] = 17
                    return new_bboxes_pr, layer_n
                # 如果对切红薯中检测到了大红薯，默认单一食材为对切红薯
                if (name1 == 21 and name2 == 22) or (name1 == 22 and name2 == 21):
                    for i in range(new_num_label):
                        new_bboxes_pr[i][5] = 21
                    return new_bboxes_pr, layer_n
                # 如果对切红薯中检测到了中红薯，默认单一食材为对切红薯
                if (name1 == 21 and name2 == 23) or (name1 == 23 and name2 == 21):
                    for i in range(new_num_label):
                        new_bboxes_pr[i][5] = 21
                    return new_bboxes_pr, layer_n

            # 数量最多label对应的食材占比0.7以上
            if num_name1 / new_num_label > 0.7:
                name1_bboxes_pr = []
                for i in range(new_num_label):
                    if name1 == new_bboxes_pr[i][5]:
                        name1_bboxes_pr.append(new_bboxes_pr[i])

                name1_bboxes_pr[0][4] = 0.95
                return name1_bboxes_pr, layer_n

            # 按各个label的probability降序排序
            else:
                new_bboxes_pr = sorted(new_bboxes_pr, key=lambda x: x[4], reverse=True)
                for i in range(len(new_bboxes_pr)):
                    new_bboxes_pr[i][4] = new_bboxes_pr[i][4] * 0.9
                return new_bboxes_pr, layer_n
