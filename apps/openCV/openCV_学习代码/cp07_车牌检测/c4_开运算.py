# coding:utf-8
# Author:mylady
# Datetime:2023/3/28 20:50
import cv2
import numpy as np


def test_1():
    # 开运算：先进性腐蚀然后进行膨胀
    img = cv2.imread('./data/img2.jpg', flags=cv2.IMREAD_GRAYSCALE)
    result = cv2.morphologyEx(img,
                              op=cv2.MORPH_OPEN,
                              kernel=np.ones(shape=[10, 10], dtype=np.uint8),
                              iterations=1)
    cv2.imshow('raw', img)
    cv2.imshow('morpholoyEx', result)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    pass


def main():
    test_1()
    pass


if __name__ == '__main__':
    main()
