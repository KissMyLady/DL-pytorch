# coding:utf-8
# Author:mylady
# Datetime:2023/3/27 15:37
import cv2
import numpy as np


def test_1():
    img = cv2.imread('cat.png')

    # shape属性中包括了三个信息
    # 高度，长度 和 通道数
    print('高度，长度, 通道数', img.shape)

    # 图像占用多大空间
    # 高度 * 长度 * 通道数
    print('图像占用多大空间: ', img.size)

    # 图像中每个元素的位深
    print('图像中每个元素的位深: ', img.dtype)
    pass


def main():
    test_1()
    pass


if __name__ == '__main__':
    main()
