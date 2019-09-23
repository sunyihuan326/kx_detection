# coding:utf-8
'''
分文件夹
train1、train2、train3、train4、train5、train6、others
其中train1：120   train2：120   train3：120   train4：120  train5：120  train6：120  ，其他的在others中

created on 2019/7/12

@author:sunyihuan
'''

import shutil
import os
import numpy as np

# CLASSES = ["BeefSteak", "CartoonCookies", "ChiffonCake", "Cookies", "CranberryCookies",
# "CupCake", "nofood", "Peanuts", "Pizza", "PorkChops", "RoastedChicken", "Toast"]
CLASSES = ["PorkChops", "Toast"]

DataRoot = "/Users/sunyihuan/Desktop/WLS/KX38I95FOODSETS_0712/JPGImages"


def split_data(train_i_nums):
    '''

    :param train_i_nums: 每个train文件夹中的文件数量，如：120
    :return:
    '''
    assert train_i_nums > 1
    for classes in CLASSES:
        tr1_save_dir = os.path.join(DataRoot, classes, 'train1')
        tr2_save_dir = os.path.join(DataRoot, classes, 'train2')
        tr3_save_dir = os.path.join(DataRoot, classes, 'train3')
        tr4_save_dir = os.path.join(DataRoot, classes, 'train4')
        tr5_save_dir = os.path.join(DataRoot, classes, 'train5')
        tr6_save_dir = os.path.join(DataRoot, classes, 'train6')
        # va_save_dir1 = os.path.join(DataRoot, classes, 'valid1')
        # va_save_dir2 = os.path.join(DataRoot, classes, 'valid2')
        # va_save_dir3 = os.path.join(DataRoot, classes, 'valid3')

        shutil.rmtree(tr1_save_dir, True)
        shutil.rmtree(tr2_save_dir, True)
        shutil.rmtree(tr3_save_dir, True)
        shutil.rmtree(tr4_save_dir, True)
        shutil.rmtree(tr5_save_dir, True)
        shutil.rmtree(tr6_save_dir, True)
        # shutil.rmtree(va_save_dir1, True)
        # shutil.rmtree(va_save_dir2, True)
        # shutil.rmtree(va_save_dir3, True)

        # class_dir = os.path.join(DataRoot, 'fixed-' + classes)
        class_dir = os.path.join(DataRoot, classes)
        file_lists = os.listdir(class_dir)
        os.makedirs(tr1_save_dir, exist_ok=True)
        os.makedirs(tr2_save_dir, exist_ok=True)
        os.makedirs(tr3_save_dir, exist_ok=True)
        os.makedirs(tr4_save_dir, exist_ok=True)
        os.makedirs(tr5_save_dir, exist_ok=True)
        os.makedirs(tr6_save_dir, exist_ok=True)
        # os.makedirs(va_save_dir1, exist_ok=True)
        # os.makedirs(va_save_dir2, exist_ok=True)
        # os.makedirs(va_save_dir3, exist_ok=True)

        num = len(file_lists)
        # valid0_num = int(ratio[0] * num)
        # valid1_num = int(ratio[1] * num)
        # valid2_num = int(ratio[2] * num)
        # valid3_num = int(ratio[3] * num)
        # test_num = int(ratio[4] * num)
        permutation = list(np.random.permutation(num))
        for i, filename in enumerate(file_lists):
            if filename != ".DS_Store":
                file_path = os.path.join(class_dir, filename)
                if os.path.isfile(file_path):
                    if permutation[i] < train_i_nums:
                        save_dir = tr1_save_dir
                    # elif permutation[i] < valid0_num + valid1_num:
                    #         save_dir = va_save_dir1
                    elif permutation[i] < 2 * train_i_nums:
                        save_dir = tr2_save_dir
                    elif permutation[i] < 3 * train_i_nums:
                        save_dir = tr3_save_dir
                    elif permutation[i] < 4 * train_i_nums:
                        save_dir = tr4_save_dir
                    elif permutation[i] < 5 * train_i_nums:
                        save_dir = tr5_save_dir
                    else:
                        save_dir = tr6_save_dir

                    shutil.copy(file_path, save_dir)


if __name__ == "__main__":
    split_data(120)
