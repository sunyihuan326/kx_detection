# -*- coding:utf-8 -*-

import os


class ImageRename():
    def __init__(self, target):
        self.path = 'C:/Users/sunyihuan/Desktop/X-补采/X-3/Cookies/qiaokeli'
        self.target = target

    def rename(self):
        filelist = os.listdir(self.path)
        total_num = len(filelist)

        for i, item in enumerate(filelist):
            if item.endswith('.jpg'):
                src = os.path.join(os.path.abspath(self.path), item)
                filename = str(i + 1)
                # filename = str(item).split(".")[0]
                # print(filename)
                # dst = os.path.join(os.path.abspath(self.path),
                #                    filename + '.jpg')
                # os.rename(src, dst)
                #
                if self.target not in filename:
                    dst = os.path.join(os.path.abspath(self.path),
                                       filename + "_191030" + "X3_qkl_" + self.target + '.jpg')
                    os.rename(src, dst)
                    print('converting %s to %s ...' % (src, dst))


if __name__ == '__main__':
    target = "Cookies"
    newname = ImageRename(target)
    newname.rename()
