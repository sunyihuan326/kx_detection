# -*- coding: utf-8 -*-
# @Time    : 2020/7/28
# @Author  : sunyihuan
# @File    : xml_file_generate.py
'''
根据标签框结果，生成xml文件
'''
from xml.dom.minidom import *
import os
from PIL import Image


def generate_xml(img_size, bboxes, save_dir, xml_name):
    (img_width, img_height, img_channel) = img_size
    # 创建一个文档对象
    doc = Document()

    # 创建一个根节点
    root = doc.createElement('annotation')

    # 根节点加入到tree
    doc.appendChild(root)

    # 创建二级节点
    fodler = doc.createElement('fodler')
    fodler.appendChild(doc.createTextNode('1'))  # 添加文本节点

    filename = doc.createElement('filename')
    filename.appendChild(doc.createTextNode('xxxx.jpg'))  # 添加文本节点

    path = doc.createElement('path')
    path.appendChild(doc.createTextNode('./xxxx.jpg'))  # 添加文本节点

    source = doc.createElement('source')
    name = doc.createElement('database')
    name.appendChild(doc.createTextNode('Unknown'))  # 添加文本节点
    source.appendChild(name)  # 添加文本节点

    size = doc.createElement('size')
    width = doc.createElement('width')
    width.appendChild(doc.createTextNode(str(img_width)))  # 添加图片width
    height = doc.createElement('height')
    height.appendChild(doc.createTextNode(str(img_height)))  # 添加图片height
    channel = doc.createElement('depth')
    channel.appendChild(doc.createTextNode(str(img_channel)))  # 添加图片channel
    size.appendChild(height)
    size.appendChild(width)
    size.appendChild(channel)

    segmented = doc.createElement('segmented')
    segmented.appendChild(doc.createTextNode('0'))
    root.appendChild(fodler)  # fodler加入到根节点
    root.appendChild(filename)  # filename加入到根节点
    root.appendChild(path)  # path加入到根节点
    root.appendChild(source)  # source加入到根节点
    root.appendChild(size)  # source加入到根节点
    root.appendChild(segmented)  # segmented加入到根节点

    for i in range(len(bboxes)):
        object = doc.createElement('object')
        name = doc.createElement('name')
        name.appendChild(doc.createTextNode("nofood"))
        object.appendChild(name)
        pose = doc.createElement('pose')
        pose.appendChild(doc.createTextNode("Unspecified"))
        object.appendChild(pose)
        truncated = doc.createElement('truncated')
        truncated.appendChild(doc.createTextNode("0"))
        object.appendChild(truncated)
        difficult = doc.createElement('difficult')
        difficult.appendChild(doc.createTextNode("0"))
        object.appendChild(difficult)
        bndbox = doc.createElement('bndbox')
        xmin = doc.createElement("xmin")
        xmin.appendChild(doc.createTextNode(str(bboxes[i][0])))
        bndbox.appendChild(xmin)
        ymin = doc.createElement("ymin")
        ymin.appendChild(doc.createTextNode(str(bboxes[i][1])))
        bndbox.appendChild(ymin)
        xmax = doc.createElement("xmax")
        xmax.appendChild(doc.createTextNode(str(bboxes[i][2])))
        bndbox.appendChild(xmax)
        ymax = doc.createElement("ymax")
        ymax.appendChild(doc.createTextNode(str(bboxes[i][3])))
        bndbox.appendChild(ymax)
        # difficult.appendChild(doc.createTextNode("0"))
        object.appendChild(bndbox)

        root.appendChild(object)  # object加入到根节点

    # 存成xml文件
    fp = open('{}/{}.xml'.format(save_dir, xml_name), 'w', encoding='utf-8')
    doc.writexml(fp, indent='', addindent='\t', newl='\n', encoding='utf-8')
    fp.close()


if __name__ == "__main__":
    img_root = "E:/DataSets/X_3660_data/bu/20201020/JPGImages"
    xml_root = "E:/DataSets/X_3660_data/bu/20201020/Annotations"
    for img in os.listdir(img_root):
        if img.endswith("jpg") or img.endswith("png"):
            img_name = img_root + "/" + img
            xml_name = img.split(".")[0]

            image = Image.open(img_name)
            org_h, org_w = image.size
            bboxes = [[1, 1, int(org_h) - 1, int(org_w) - 1, 1, 10]]
            generate_xml((org_h, org_w, 3), bboxes, xml_root, xml_name)
