# -*- coding:utf-8 -*-
'''
图片重新命名

'''
import os
import shutil


class ImageRename():
    def __init__(self, root_path, target, save_dir):
        '''

        :param root_path: 图片根地址
        :param target: 图片命名中类别标签名
        '''
        self.root_path = root_path
        self.target = target
        self.save_dir = save_dir

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
                dst = os.path.join(os.path.abspath(self.root_path),
                                   filename + "_200805" + "_{}".format(say) + self.target + '.jpg')
                os.rename(src, dst)
                print('converting %s to %s ...' % (src, dst))
                save_name = os.path.join(os.path.abspath(self.save_dir),
                                         filename + "_200805" + "_{}".format(say) + self.target + '.jpg')
                shutil.copy(dst, save_name)
                # try:
                #     # 修改命名，规则为：i_日期_烤箱/其他说明_类别名.jpg
                #     dst = os.path.join(os.path.abspath(self.root_path),
                #                        filename + "_200803" + "_{}".format(say) + self.target + '.jpg')
                #     os.rename(src, dst)
                #     print('converting %s to %s ...' % (src, dst))
                #     save_name = os.path.join(os.path.abspath(self.save_dir),
                #                              filename + "_200803" + "_{}".format(say) + self.target + '.jpg')
                #     shutil.copy(dst, save_name)
                # except:
                #     pass

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
        for k in ["kaojia", "xizhi"]:
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
                                               filename + "_200723" + "_X5_bai_" + "{}".format(k) + "_{}_".format(
                                                   b) + self.target + '.jpg')
                            os.rename(src, dst)
                            print('converting %s to %s ...' % (src, dst))
                        except:
                            pass
                        # if self.target not in filename:


if __name__ == '__main__':
    path = "/Volumes/SYH/Joyoung/炸锅项目/炸锅采图202007/ZG1/jichijian"
    target = "chickenwing_top"
    save_dir = "/Volumes/SYH/Joyoung/炸锅项目/炸锅采图202007/ZG1/jichijian"
    for c in [""]:
        dir_p0 = path + "/" + c
        for k in [""]:
            dir_p = dir_p0 + "/" + k
            for ty in ["tj","xz"]:
                dir_p_ = dir_p + "/" + ty
                newname = ImageRename(dir_p_, target, save_dir)
                newname.rename0("ZG1_{}_{}_{}_".format(c,k, ty))
    # newname.rename0("X5_top_")
