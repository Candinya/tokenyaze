import os
import re

from config import *


def gen_file(root_path, target):
    # 获取该目录下所有的文件名称和目录名称
    dir_or_files = os.listdir(root_path)
    for dir_file in dir_or_files:
        # 获取目录或者文件的路径
        dir_file_path = os.path.join(root_path, dir_file)
        # 判断该路径为文件还是路径
        if os.path.isdir(dir_file_path):
            # 递归获取所有文件和目录的路径
            gen_file(dir_file_path, target)
        else:
            # 是文件，那么读取内容并追加
            text = open(dir_file_path, 'r', encoding='utf-8')
            for eachline in text:
                tag = re.compile(r'/[a-zA-Z0-9]+', re.U)
                eachline = re.sub(tag, '', eachline)
                branckets = re.compile(r'[()\[\]]+', re.U)
                eachline = re.sub(branckets, '', eachline)
                if len(re.sub(r'[\s\n]', '', eachline)) != 0:
                    # 空行就不写出了
                    target.writelines(eachline)
            text.close()
            # print('load finish for ' + dir_file_path)


if __name__ == "__main__":
    # 根目录路径
    root_path = r"data"
    # 合并文件
    target = open(filename, 'w', encoding='utf-8')
    gen_file(root_path, target)
    target.close()
    print('merge complete.')
