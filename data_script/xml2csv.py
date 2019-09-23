# coding:utf-8 
'''
created on 2019/7/17

@author:sunyihuan

将xml写入到csv文件中

'''

from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

import glob
import pandas as pd
import xml.etree.ElementTree as ET


def xml_to_csv(data_dir, csv_save_path, classes_name="ChickenWings"):
    '''
    将xml文件中的数据写入到csv表格中
    :param data_dir: xml文件夹地址
    :param csv_save_path: csv文件保存的地址
    :param classes_name: 校验标签名称，确认和标签类别一致
    :return:
    '''
    xml_list = []
    for xml_file in glob.glob(data_dir + '/*.xml'):
        tree = ET.parse(xml_file)
        root = tree.getroot()
        for member in root.findall('object'):
            # member[0].text = classes_name
            value = (root.find('filename').text,
                     int(root.find('size')[0].text),
                     int(root.find('size')[1].text),
                     member[0].text,
                     int(member[4][0].text),
                     int(member[4][1].text),
                     int(member[4][2].text),
                     int(member[4][3].text)
                     )
            xml_list.append(value)
    column_name = ['filename', 'width', 'height', 'class', 'xmin', 'ymin', 'xmax', 'ymax']
    xml_df = pd.DataFrame(xml_list, columns=column_name)
    xml_df.to_csv('{}_labels.csv'.format(csv_save_path), index=None)


if __name__ == '__main__':
    data_dir = '/Users/sunyihuan/Desktop/WLS/KX38I95FOODSETS/train_2classes/Annotations'
    csv_save_path = "/Users/sunyihuan/Desktop/WLS/KX38I95FOODSETS/train_2classes/train"
    xml_to_csv(data_dir, csv_save_path)
