# coding:utf-8
# Author:mylady
# Datetime:2023/3/27 15:35
import cv2
import numpy as np


def test_1():
    img = cv2.imread('./cat.png')

    # 浅拷贝
    img2 = img.view()

    # 深拷贝
    img3 = img.copy()

    # 原图修改
    img[10:100, 10:100] = [0, 0, 255]

    cv2.imshow('img', img)
    cv2.imshow('img2', img2)
    cv2.imshow('img3', img3)

    cv2.waitKey(0)
    cv2.destroyAllWindows()
    pass


def main():
    test_1()
    pass


if __name__ == '__main__':
    main()
