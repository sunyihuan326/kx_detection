# -*- coding:utf-8 -*-

import os


class ImageRename():
    def __init__(self, target):
        self.path = "C:/Users/sunyihuan/Desktop/test_results_jpg/supply/烤披萨/pizzatwo/中"
        self.target = target

    def rename(self):
        filelist = os.listdir(self.path)
        total_num = len(filelist)

        for i, item in enumerate(filelist):
            if item.endswith('.jpg'):
                src = os.path.join(os.path.abspath(self.path), item)
                filename = str(i + 41)
                # filename = str(item).split(".")[0]
                # print(filename)
                dst = os.path.join(os.path.abspath(self.path),
                                   filename + '.jpg')
                os.rename(src, dst)
                # #
                # if self.target not in filename:
                #     dst = os.path.join(os.path.abspath(self.path),
                #                        filename + "_191118" + "_X1_" + self.target + '.jpg')
                #     os.rename(src, dst)
                #     print('converting %s to %s ...' % (src, dst))


if __name__ == '__main__':
    target = "sweetpotatol"
    newname = ImageRename(target)
    newname.rename()
