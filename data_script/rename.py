# -*- coding:utf-8 -*-

import os


class ImageRename():
    def __init__(self, target):
        self.path = 'E:/WLS_originalData/未标注原图/SweetPotatoS'
        self.target = target

    def rename(self):
        filelist = os.listdir(self.path)
        total_num = len(filelist)

        i = 0
        for item in filelist:
            if item.endswith('.jpg'):
                src = os.path.join(os.path.abspath(self.path), item)
                filename = str(item).split(".")[0]
                print(filename)
                if self.target not in filename:
                    dst = os.path.join(os.path.abspath(self.path),
                                       filename + "_" + self.target + '.jpg')
                    os.rename(src, dst)
                    print('converting %s to %s ...' % (src, dst))
                    i = i + 1
                    print('total %d to rename & converted %d jpgs' % (total_num, i))


if __name__ == '__main__':
    target = "xiao"
    newname = ImageRename(target)
    newname.rename()
