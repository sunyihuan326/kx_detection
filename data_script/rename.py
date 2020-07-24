# -*- coding:utf-8 -*-
'''
图片重新命名

'''
import os


class ImageRename():
    def __init__(self, root_path, target):
        '''

        :param root_path: 图片根地址
        :param target: 图片命名中类别标签名
        '''
        self.root_path = root_path
        self.target = target

    def rename0(self, say):
        '''
        将文件夹中排序命名
        :return:
        '''
        for i, item in enumerate(os.listdir(self.root_path)):
            if item.endswith('.jpg'):
                src = os.path.join(os.path.abspath(self.root_path), item)
                filename = str(i + 1)
                # filename = str(item).split(".")[0]
                # print(filename)
                # dst = os.path.join(os.path.abspath(self.path),
                #                    filename + '_s.jpg')
                # os.rename(src, dst)
                # #
                try:
                    # 修改命名，规则为：i_日期_烤箱/其他说明_类别名.jpg
                    dst = os.path.join(os.path.abspath(self.root_path),
                                       filename + "_200608" + "_{}".format(say) + self.target + '.jpg')
                    os.rename(src, dst)
                    print('converting %s to %s ...' % (src, dst))
                except:
                    pass

    def change_jpg_name(self):
        for i, item in enumerate(os.listdir(self.root_path)):
            if item.endswith('.jpg'):
                src = os.path.join(os.path.abspath(self.root_path), item)
                try:
                    # 修改命名，规则为：i_日期_炸锅型号/其他说明_类别名.jpg
                    # tj:托架、xz:锡纸、gyz:硅油纸、jk:净空、cp:瓷盘
                    # qh:浅红、zh:正红、ch:橙红、sh:深红
                    if "cornone" in src:
                        dst = src.replace("cornone", "corntwo")
                        os.rename(src, dst)
                        print('converting %s to %s ...' % (src, dst))
                    else:
                        pass

                except:
                    pass

    def rename(self):
        '''
        将文件中kaojia、kaojia(bujiaxizhi)、kaopan、kaopan(bujiaxizhi)等文件夹下命名

        :return:
        '''
        for k in ["kaopan","xizhi"]:
            path_dir = self.root_path + "/" + k
            for b in ["bottom", "middle", "top"]:
                path_name = path_dir + "/" + b
                filelist = os.listdir(path_name)
                total_num = len(filelist)

                for i, item in enumerate(filelist):
                    if item.endswith('.jpg'):
                        src = os.path.join(os.path.abspath(path_name), item)
                        filename = str(i + 1)
                        try:
                            dst = os.path.join(os.path.abspath(path_name),
                                               filename + "_200723" + "_X1_" + "{}".format(k) + "_{}_".format(
                                                   b) + self.target + '.jpg')
                            os.rename(src, dst)
                            print('converting %s to %s ...' % (src, dst))
                        except:
                            pass
                        # if self.target not in filename:


if __name__ == '__main__':
    path = "/Volumes/SYH/Joyoung/3660摄像头补图202007/X1/xiaotudou"
    target = "potatos"

    newname = ImageRename(path, target)
    newname.rename()
    # newname.rename0("X5_top_")
