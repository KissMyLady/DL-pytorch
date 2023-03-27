# coding:utf-8
# Author:mylady
# Datetime:2023/3/27 15:40
import cv2
import numpy as np


def test_1():
    img = np.zeros((480, 640, 3), np.uint8)

    b, g, r = cv2.split(img)  # 分割图像的通道
    print(r)

    # 通过组合不同通道, 得到不同的合并颜色
    # b[10:100, 10:100] = 255
    g[10:100, 10:100] = 255
    r[10:100, 10:100] = 255

    img2 = cv2.merge((b, g, r))

    cv2.imshow('img', img)
    cv2.imshow('b', b)
    cv2.imshow('g', g)
    cv2.imshow('r', r)
    cv2.imshow('img2', img2)

    cv2.waitKey(0)
    cv2.destroyAllWindows()
    pass


def main():
    test_1()
    pass


if __name__ == '__main__':
    main()
