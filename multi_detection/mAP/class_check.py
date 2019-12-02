# -*- encoding: utf-8 -*-

"""
@File    : class_check.py
@Time    : 2019/10/31 13:32
@Author  : sunyihuan
"""
import os
import shutil
from sklearn.metrics import confusion_matrix
import numpy as np

gt_txt_root = "E:/kx_detection/multi_detection/mAP/ground-truth"
pre_txt_root = "E:/kx_detection/multi_detection/mAP/predicted"

# CLASSES = ["beefsteak", "cartooncookies", "chickenwings", "chiffoncake", "cookies",
#            "cranberrycookies", "cupcake", "eggtart", "nofood", "peanuts",
#            "pizza", "porkchops", "purplesweetpotato", "roastedchicken", "toast",
#            "potatos", "potatom", "potatol", "sweetpotatos", "sweetpotatom", "sweetpotatol",
#            "potatocut", "sweetpotatocut", "pizzaone", "pizzatwo", "pizzafour", "pizzasix"]  #26分类
CLASSES = ["beefsteak", "cartooncookies", "chickenwings", "chiffoncake6", "chiffoncake8",
           "cookies", "cranberrycookies", "cupcake", "eggtart", "eggtartbig",
           "nofood", "peanuts", "pizzafour", "pizzaone", "pizzasix",
           "pizzatwo", "porkchops", "potatocut", "potatol", "potatom",
           "potatos", "sweetpotatocut", "sweetpotatol", "sweetpotatom", "sweetpotatos",
           "roastedchicken", "toast"]  # 27分类


def get_accuracy(error_write=True):
    '''
    由txt文件，查看classes准确率
    :param error_write: 是否将错误图片数据写入到error文件中，True/False
    :return: error_,
            error_c,
            acc
    '''

    error_dir = "E:/kx_detection/multi_detection/mAP/correct/"
    if os.path.exists(error_dir): shutil.rmtree(error_dir)
    os.mkdir(error_dir)

    error_dir2 = "E:/kx_detection/multi_detection/mAP/correct1/"
    if os.path.exists(error_dir2): shutil.rmtree(error_dir2)
    os.mkdir(error_dir2)

    error_dir3 = "E:/kx_detection/multi_detection/mAP/correct2/"
    if os.path.exists(error_dir3): shutil.rmtree(error_dir3)
    os.mkdir(error_dir3)

    detection_dir = "E:/kx_detection/multi_detection/data/detection/"

    pre_txt_list = os.listdir(pre_txt_root)

    class_true = []
    class_pre = []
    no_result = {}

    error_c = 0  # 输出标签种类错误的nums
    error_cc = 0  # 披萨、土豆、红薯不需要校验中错误的数量
    error_noresults = 0  # 输出无结果的nums
    need_check_nums = 0  # 需手动check数量
    psp = ["pizzafour", "pizzaone", "pizzasix", "pizzatwo",
           "potatocut", "potatol", "potatom", "potatos",
           "sweetpotatocut", "sweetpotatol", "sweetpotatom", "sweetpotatos"]
    for pre in pre_txt_list:
        if pre.endswith("txt"):
            with open(os.path.join(pre_txt_root, pre), "r") as f:
                with open(os.path.join(gt_txt_root, pre), "r") as fg:  # 读取真实类别
                    for lin in fg.readlines():
                        true_cc = lin.split(" ")[0]  # 真实结果

                all_lines = f.readlines()
                if len(all_lines) > 0:
                    # 预测结果排序
                    pre_c = {}
                    score_list = []
                    cls = []
                    for line in all_lines:
                        c = line.split(" ")[0]
                        cls.append(c)
                        if c not in pre_c.keys():
                            pre_c[c] = 1
                        else:
                            pre_c[c] += 1
                        score_list.append(line.split(" ")[1])
                    pre_cc = sorted(pre_c.items(), key=lambda x: x[1], reverse=True)
                    psp_is = 0  # 默认不在psp中
                    for p_c in cls:
                        if p_c in psp:  # 判断标签中是否有披萨、红薯、土豆
                            psp_is = 1
                            break
                    if psp_is == 1:
                        if len(cls) >= 2:
                            # 预测结果排序
                            if len(pre_cc) == 1:
                                predict_c = pre_cc[0][0]  # 若输出种类为1

                                shutil.copy(detection_dir + pre.split(".")[0] + ".jpg",
                                            error_dir + pre.split(".")[0] + ".jpg")  # 拷贝正确图片到correct

                                shutil.copy(os.path.join(gt_txt_root, pre),
                                            error_dir + pre.split(".")[0] + "_gt.txt")  # 拷贝ground_truth文件
                                shutil.copy(os.path.join(pre_txt_root, pre),
                                            error_dir + pre.split(".")[0] + "_pre.txt")  # 拷贝predicted文件

                                if predict_c != true_cc:
                                    error_cc += 1

                                    # 若排序最高的class为错误类别，写入到error
                                    shutil.copy(detection_dir + pre.split(".")[0] + ".jpg",
                                                error_dir2 + pre.split(".")[0] + ".jpg")  # 拷贝错误图片到error

                                    shutil.copy(os.path.join(gt_txt_root, pre),
                                                error_dir2 + pre.split(".")[0] + "_gt.txt")  # 拷贝ground_truth文件
                                    shutil.copy(os.path.join(pre_txt_root, pre),
                                                error_dir2 + pre.split(".")[0] + "_pre.txt")  # 拷贝predicted文件
                            else:  # 需要手动校验
                                need_check_nums += 1
                                shutil.copy(detection_dir + pre.split(".")[0] + ".jpg",
                                            error_dir3 + pre.split(".")[0] + ".jpg")  # 拷贝错误图片到error3

                                shutil.copy(os.path.join(gt_txt_root, pre),
                                            error_dir3 + pre.split(".")[0] + "_gt.txt")  # 拷贝ground_truth文件
                                shutil.copy(os.path.join(pre_txt_root, pre),
                                            error_dir3 + pre.split(".")[0] + "_pre.txt")  # 拷贝predicted文件
                        else:  # 需要手动校验
                            need_check_nums += 1
                            shutil.copy(detection_dir + pre.split(".")[0] + ".jpg",
                                        error_dir3 + pre.split(".")[0] + ".jpg")  # 拷贝错误图片到error3

                            shutil.copy(os.path.join(gt_txt_root, pre),
                                        error_dir3 + pre.split(".")[0] + "_gt.txt")  # 拷贝ground_truth文件
                            shutil.copy(os.path.join(pre_txt_root, pre),
                                        error_dir3 + pre.split(".")[0] + "_pre.txt")  # 拷贝predicted文件
                    else:
                        if len(pre_cc) == 1:
                            predict_c = pre_cc[0][0]  # 若输出种类为1
                            if predict_c != true_cc:
                                error_c += 1

                                # 若排序最高的class为错误类别，写入到error
                                # shutil.copy(detection_dir + pre.split(".")[0] + ".jpg",
                                #             error_dir + pre.split(".")[0] + ".jpg")  # 拷贝错误图片到error
                                #
                                # shutil.copy(os.path.join(gt_txt_root, pre),
                                #             error_dir + pre.split(".")[0] + "_gt.txt")  # 拷贝ground_truth文件
                                # shutil.copy(os.path.join(pre_txt_root, pre),
                                #             error_dir + pre.split(".")[0] + "_pre.txt")  # 拷贝predicted文件
                        else:
                            if pre_cc[0][1] != pre_cc[1][1]:  # 如果输种类大于1个，最多的只有一类，取数量最多的一个
                                predict_c = pre_cc[0][0]
                                if predict_c != true_cc:
                                    error_c += 1
                                    # 若sort后种类大于1，且数量最多为1，写入到error2
                                    # shutil.copy(detection_dir + pre.split(".")[0] + ".jpg",
                                    #             error_dir2 + pre.split(".")[0] + ".jpg")  # 拷贝错误图片到error2
                                    #
                                    # shutil.copy(os.path.join(gt_txt_root, pre),
                                    #             error_dir2 + pre.split(".")[0] + "_gt.txt")  # 拷贝ground_truth文件
                                    # shutil.copy(os.path.join(pre_txt_root, pre),
                                    #             error_dir2 + pre.split(".")[0] + "_pre.txt")  # 拷贝predicted文件
                            else:  # 若最多种类不唯一，取得分最高的
                                predict_c = all_lines[score_list.index(max(score_list))].split(" ")[0]
                                if predict_c != true_cc:
                                    error_c += 1
                                    # 若sort后种类大于1且score得分最高的class为错误类别，写入到error2
                                    # shutil.copy(detection_dir + pre.split(".")[0] + ".jpg",
                                    #             error_dir3 + pre.split(".")[0] + ".jpg")  # 拷贝错误图片到error3
                                    #
                                    # shutil.copy(os.path.join(gt_txt_root, pre),
                                    #             error_dir3 + pre.split(".")[0] + "_gt.txt")  # 拷贝ground_truth文件
                                    # shutil.copy(os.path.join(pre_txt_root, pre),
                                    #             error_dir3 + pre.split(".")[0] + "_pre.txt")  # 拷贝predicted文件
                else:
                    error_noresults += 1
    return error_c, error_cc, need_check_nums,error_noresults


if __name__ == "__main__":
    error_c, error_cc, need_check_nums,error_noresults = get_accuracy()
    print(error_c)
    print("error_cc：", error_cc)
    print("need_check_nums：", need_check_nums)
    print(error_noresults)
