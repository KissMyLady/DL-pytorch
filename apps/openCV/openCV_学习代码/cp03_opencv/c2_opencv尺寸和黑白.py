# coding:utf-8
# Author:mylady
# Datetime:2023/3/27 18:06
import numpy as np
import cv2


def test_1():
    rose = cv2.imread('./cat.png')

    rose2 = cv2.resize(rose, (105, 165))  # 图片尺寸的改变

    gray = cv2.cvtColor(rose, code=cv2.COLOR_BGR2GRAY)  # 黑白图片，灰度化处理

    hsv = cv2.cvtColor(rose, code=cv2.COLOR_BGR2HSV)  # brg -> hsv

    cv2.imshow('name-rose', gray)
    cv2.waitKey(0)  # 0 无限等待；1000毫秒 = 1秒之后，自动消失
    cv2.destroyAllWindows()
    pass


def main():
    test_1()
    pass


if __name__ == '__main__':
    main()
