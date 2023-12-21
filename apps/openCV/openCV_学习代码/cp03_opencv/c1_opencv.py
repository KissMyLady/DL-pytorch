# coding:utf-8
# Author:mylady
# Datetime:2023/3/27 18:03
import cv2 # pip install opencv-contrib-python
import numpy as np


def test_1():
    rose = cv2.imread('./cat.png')  # 图片，图片是什么？？？数组

    print(rose.shape) # (576, 886, 3) 数组形状，610像素
    print(type(rose)) # numpy数组 <class 'numpy.ndarray'>
    print(rose) # 三个[ 三维数组（彩色图片：高度、宽度、像素红绿蓝）
    # 第一维表示高度 ::-1表示翻转 颠倒，上下颠倒

    # 最后一维，表示颜色 蓝0绿1红2  ::-1 红2绿1蓝0
    cv2.imshow('rose', rose[:, :, [1, 0, 2]])  # 弹出窗口
    cv2.waitKey()  # 等待键盘输入，任意输入，出发这个代码，窗口消失
    cv2.destroyAllWindows()  # 销毁内存
    pass


def main():
    test_1()
    pass


if __name__ == '__main__':
    main()
