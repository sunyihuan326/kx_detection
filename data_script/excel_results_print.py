# -*- coding: utf-8 -*-
# @Time    : 2020/11/2
# @Author  : sunyihuan
# @File    : excel_results_print.py
'''
根据excel中各图片类别结果，统计各的分段数量
'''
import xlwt
import xlrd


def he_foods(pre, true):
    '''
    针对合并的类别判断输出是否在合并类别内
    :param pre:
    :return:
    '''
    if pre in [3, 4, 42] and true in [3, 4, 42]:  # 合并戚风
        rigth_label = True
    elif pre in [32, 33] and true in [32, 33]:  # 合并器皿
        rigth_label = True
    elif pre in [36, 37] and true in [36, 37]:  # 合并虾
        rigth_label = True
    else:
        rigth_label = False
    # rigth_label = False
    return rigth_label


excel_path = "F:/test_from_yejing_202010/TXKX_all_20201019_rename_all/all_he_score0914.xls"

excel = xlrd.open_workbook(excel_path)
sheet = excel.sheet_by_index(0)
score_90 = 0
score_80 = 0
score_70 = 0
score_60 = 0
score_50 = 0
jpg_all = 0
jpg_90 = 0
jpg_80 = 0
jpg_70 = 0
jpg_60 = 0
jpg_50 = 0

for k in range(sheet.nrows):
    if sheet.cell(k, 0).value.endswith(".jpg"):

        food_true = sheet.cell(k, 1).value
        food_pre = sheet.cell(k, 3).value
        score = float(sheet.cell(k, 5).value)
        # print(food_true, food_pre, score)
        if food_true < 23:
            jpg_all += 1
            if score >= 0.9:
                jpg_90 += 1
                if food_true == food_pre:
                    score_90 += 1
                else:
                    right_label = he_foods(food_pre, food_true)
                    if right_label:
                        score_90 += 1
            elif score < 0.9 and score >= 0.8:
                jpg_80 += 1
                if food_true == food_pre:
                    score_80 += 1
                else:
                    right_label = he_foods(food_pre, food_true)
                    if right_label:
                        score_80 += 1
            elif score < 0.8 and score >= 0.7:
                jpg_70 += 1
                if food_true == food_pre:
                    score_70 += 1
                else:
                    right_label = he_foods(food_pre, food_true)
                    if right_label:
                        score_70 += 1
            elif score < 0.7 and score >= 0.6:
                jpg_60 += 1
                if food_true == food_pre:
                    score_60 += 1
                else:
                    right_label = he_foods(food_pre, food_true)
                    if right_label:
                        score_60 += 1
            elif score < 0.6 and score >= 0.5:
                jpg_50 += 1
                if food_true == food_pre:
                    score_50 += 1
                else:
                    right_label = he_foods(food_pre, food_true)
                    if right_label:
                        score_90 += 1

print(score_50, score_60, score_70, score_80, score_90)
print(jpg_50, jpg_60, jpg_70, jpg_80, jpg_90)

print("图片总数：", jpg_all)
print("置信度[50%，60%)并正确占总比：{}%".format(round(score_50 * 100 / jpg_all, 2)))
print("置信度[60%，70%)并正确占总比：{}%".format(round(score_60 * 100 / jpg_all, 2)))
print("置信度[70%，80%)并正确占总比：{}%".format(round(score_70 * 100 / jpg_all, 2)))
print("置信度[80%，90%)并正确占总比：{}%".format(round(score_80 * 100 / jpg_all, 2)))
print("置信度[90%，)并正确占总比：{}%".format(round(score_90 * 100 / jpg_all, 2)))
