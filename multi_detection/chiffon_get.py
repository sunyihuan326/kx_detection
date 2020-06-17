# coding:utf-8 
'''
created on 2020-06-10

@author:sunyihuan
'''

import xlrd


def read_data(excel_filename):
    '''
    读取数据
    :param excel_filename:
    :return:
    '''
    w = xlrd.open_workbook(excel_filename)
    all_data = w.sheet_by_name("result")

    xmin_list = []
    ymin_list = []
    xmax_list = []
    ymax_list = []
    layer_list = []
    for i in range(1000):
        # print(all_data.cell_value(i + 1, 1))
        xmin_list.append(all_data.cell_value(i + 1, 1))
        ymin_list.append(all_data.cell_value(i + 1, 2))
        xmax_list.append(all_data.cell_value(i + 1, 3))
        ymax_list.append(all_data.cell_value(i + 1, 4))
        layer_list.append(all_data.cell_value(i + 1, 6))
    return 1000, layer_list, xmin_list, ymin_list, xmax_list, ymax_list,


def judge_size(layer, xmin, ymin, xmax, ymax):
    '''
    判断尺寸
    :param layer:
    :param xmin:
    :param ymin:
    :param xmax:
    :param ymax:
    :return:
    '''
    x_weigth = int(xmax) - int(xmin)
    y_high = int(ymax) - int(ymin)
    x_middle, y_middle = int(int(xmax) / 2 + int(xmin) / 2), int(int(ymax) / 2 + int(ymin) / 2)
    if x_middle >= 400:
        if layer == 0:
            if x_weigth <= 500:
                size = 6
                # if int(x_weigth * y_high) < 275829:
                #     size = 6
                # else:
                #     size = 8
            else:
                size = 8
        elif layer == 1:
            if x_weigth <= 520:
                size = 6
            else:
                size = 8
        else:
            if x_weigth <= 570:
                size = 6
            else:
                size = 8
    else:
        if layer == 0:
            if x_weigth <= 550:
                size = 6
            else:
                size = 8
        elif layer == 1:
            if x_weigth <= 577:
                size = 6
            else:
                size = 8
        else:
            if x_weigth <= 600:
                size = 6
            else:
                size = 8
    return size


def check_acc(size6_xls_name, size8_xls_name):
    size6_data = read_data(size6_xls_name)
    size8_data = read_data(size8_xls_name)

    size6_count = 0
    size8_count = 0
    size6_all_count = 0
    size8_all_count = 0
    for i in range(size6_data[0] - 1):
        if size6_data[1][i] != "":
            layer = int(size6_data[1][i])
            xmin = int(size6_data[2][i])
            ymin = int(size6_data[3][i])
            xmax = int(size6_data[4][i])
            ymax = int(size6_data[5][i])
            size = judge_size(layer, xmin, ymin, xmax, ymax)

            size6_all_count += 1
            if size == 6:
                size6_count += 1
    print(size6_count, size6_all_count)
    for i in range(size8_data[0] - 1):
        if size8_data[1][i] != "":
            layer = int(size8_data[1][i])
            xmin = int(size8_data[2][i])
            ymin = int(size8_data[3][i])
            xmax = int(size8_data[4][i])
            ymax = int(size8_data[5][i])
            size = judge_size(layer, xmin, ymin, xmax, ymax)

            size8_all_count += 1
            if size == 8:
                size8_count += 1
    print(size8_count, size8_all_count)
    return round(size6_count / size6_all_count, 4), round(size8_count / size8_all_count, 4)


if __name__ == "__main__":
    size6_xls_name = "/Users/sunyihuan/Desktop/all_chiffon_data/traindata/chiffon6_train_predict.xls"
    size8_xls_name = "/Users/sunyihuan/Desktop/all_chiffon_data/traindata/chiffon8_train_predict.xls"
    print(check_acc(size6_xls_name, size8_xls_name))
