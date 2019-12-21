# -*- coding:utf-8 -*-

import os


class ImageRename():
    def __init__(self, target):
        self.path = "E:/test_from_ye/JPGImages_he/pizzasix/bottom"
        self.target = target

    def rename(self):
        filelist = os.listdir(self.path)
        total_num = len(filelist)

        for i, item in enumerate(filelist):
            if item.endswith('.jpg'):
                src = os.path.join(os.path.abspath(self.path), item)
                # filename = str(i + 1)
                filename = str(item).split(".")[0]
                print(filename)
                dst = os.path.join(os.path.abspath(self.path),
                                   filename + '_s.jpg')
                os.rename(src, dst)
                # #
                # if self.target not in filename:
                #     dst = os.path.join(os.path.abspath(self.path),
                #                        filename + "_191217" + "_X5_" + self.target + '.jpg')
                #     os.rename(src, dst)
                #     print('converting %s to %s ...' % (src, dst))
#

if __name__ == '__main__':
    target = "sweetpotato"
    newname = ImageRename(target)
    newname.rename()
