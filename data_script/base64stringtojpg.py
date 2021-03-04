# -*- coding: utf-8 -*-
# 20201209
import os
import cv2
import base64
import datetime
import numpy as np
from tqdm import tqdm


def base64stringtojpg(filedir, dstfiledir):
    for f in tqdm(os.listdir(filedir)):
        if f[-4:] == ".jpg":
            data_file_path = os.path.join(filedir, f)
            with open(data_file_path, 'r', encoding='utf-8') as file:
                content = file.read()
                img_b = base64.b64decode(content)
                image = cv2.imdecode(np.frombuffer(img_b, np.uint8), cv2.COLOR_RGB2BGR)
                strfile = os.path.join(dstfiledir, f)
                cv2.imwrite(strfile, image)


# 源/目标
jpg_dir = "F:/serve_data/OVEN/202101/20210130"
save_dir = "F:/serve_data/OVEN/202101/20210130/convet_jpg"
if not os.path.exists(save_dir): os.mkdir(save_dir)
base64stringtojpg(jpg_dir, save_dir)
# for kk in os.listdir("F:/serve_data/OVEN"):
#     jpg_dir = "F:/serve_data/OVEN/202101/20210130/{}".format(kk)
#     save_dir = "F:/serve_data/OVEN/{}/conver_jpg".format(kk)
#     if not os.path.exists(save_dir): os.mkdir(save_dir)
#     base64stringtojpg(jpg_dir, save_dir)
