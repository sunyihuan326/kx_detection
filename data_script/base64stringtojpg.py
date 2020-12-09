# -*- coding: utf-8 -*-
# 20201209
import os
import cv2
import base64
import datetime
import numpy as np


def base64stringtojpg(filedir, dstfiledir):
    for f in os.listdir(filedir):
        if f[-4:] == ".jpg":
            data_file_path = os.path.join(filedir, f)
            with open(data_file_path, 'r', encoding='utf-8') as file:
                content = file.read()
                img_b = base64.b64decode(content)
                image = cv2.imdecode(np.frombuffer(img_b, np.uint8), cv2.COLOR_RGB2BGR)
                strfile = os.path.join(dstfiledir, f)
                cv2.imwrite(strfile, image)


# 源/目标
base64stringtojpg("E:/lhf/OSS2JPG/src/", "E:/lhf/OSS2JPG/img/")
