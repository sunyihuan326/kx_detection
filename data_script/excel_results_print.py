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
    elif pre in [11, 12, 13] and true in [11, 12, 13]:  # 合并披萨
        rigth_label = True
    elif pre in [14 + 1, 15 + 1, 16 + 1] and true in [15, 16, 17]:  # 合并土豆
        rigth_label = True
    elif pre in [18, 19, 20] and true in [18, 19, 20]:  # 合并红薯
        rigth_label = True
    else:
        rigth_label = False
    # rigth_label = False
    return rigth_label


class ScoreResults:
    '''
    置信度得分统计
    '''

    def __init__(self, excel_path):
        '''
        excel地址
        :param excel_path:
        '''
        self.excel_path = excel_path
        excel = xlrd.open_workbook(self.excel_path)
        self.sheet = excel.sheet_by_index(0)

    def score_nums(self, nrows_l, nrows_h, low_s, high_s):
        '''
        统计第nrows_l行至nrows_h-1行中得分区间为[low_s，high_s)中的所有正确数量
        :param nrows_l:
        :param nrows_h:
        :param low_s:
        :param high_s:
        :return:
        '''
        jpg_all = 0
        score_nums = 0
        for k in range(nrows_l, nrows_h, 1):
            if self.sheet.cell(k, 0).value.endswith(".jpg"):
                food_true = self.sheet.cell(k, 1).value
                food_pre = self.sheet.cell(k, 3).value
                score = float(self.sheet.cell(k, 5).value)
                jpg_all += 1
                if score >= low_s and score < high_s:
                    score_nums += 1
                    if food_true == food_pre:
                        score_nums += 1
                    else:
                        right_label = he_foods(food_pre, food_true)
                        if right_label:
                            score_nums += 1
        return score_nums

    def classes_nums(self, low_s, high_s):
        '''
        按类别输出分值区间内的数量
        '''

        classes_id39 = {"cartooncookies": 1, "cookies": 5, "cupcake": 7, "beefsteak": 0, "chickenwings": 2,
                        "chiffoncake6": 3, "chiffoncake8": 4, "cranberrycookies": 6, "eggtart": 8,
                        "nofood": 9, "peanuts": 10, "porkchops": 14, "potatocut": 15, "potatol": 16,
                        "potatom": 16, "potatos": 17, "sweetpotatocut": 18, "sweetpotatol": 19,
                        "pizzacut": 11, "pizzaone": 12, "roastedchicken": 21,
                        "pizzatwo": 13, "sweetpotatos": 20, "toast": 22, "chestnut": 23, "cornone": 24, "corntwo": 25,
                        "drumsticks": 26,
                        "taro": 27, "steamedbread": 28, "eggplant": 29, "eggplant_cut_sauce": 30, "bread": 31,
                        "container_nonhigh": 32, "container": 33, "duck": 21, "fish": 34, "hotdog": 35, "redshrimp": 36,
                        "shrimp": 37, "strand": 38, "xizhi": 39, "small_fish": 40, "chiffon4": 42}
        new_classes = {v: k for k, v in classes_id39.items()}

        classes_s = {}
        for k in range(0, self.sheet.nrows, 1):
            if self.sheet.cell(k, 0).value.endswith(".jpg"):
                food_true = self.sheet.cell(k, 1).value
                food_pre = self.sheet.cell(k, 3).value
                score = float(self.sheet.cell(k, 5).value)
                food_true = int(food_true)
                c = new_classes[food_true]
                if c not in classes_s.keys():
                    classes_s[c] = [1, 0]
                else:
                    classes_s[c][0] += 1
                if score >= low_s and score < high_s:
                    if food_true == food_pre:
                        classes_s[c][1] += 1
                    else:
                        right_label = he_foods(food_pre, food_true)
                        if right_label:
                            classes_s[c][1] += 1
        return classes_s


if __name__ == "__main__":
    excel_path = "F:/test_from_yejing_202010/置信度统计/0914模型/all_he_score0914.xls"
    S = ScoreResults(excel_path)
    classes_s = S.classes_nums(0.8, 1.5)

    wk = xlwt.Workbook("结果分析")
    sheet = wk.add_sheet("结果统计")
    sheet.write(0, 0, "类别")
    sheet.write(0, 1, "总数")
    sheet.write(0, 2, "正确数")
    print(len(classes_s))
    for kk, c in enumerate(classes_s.keys()):
        sheet.write(kk + 1, 0, c)
        sheet.write(kk + 1, 1, classes_s[c][0])
        sheet.write(kk + 1, 2, classes_s[c][1])
    wk.save("结果分析80-合并红薯披萨.xls")
