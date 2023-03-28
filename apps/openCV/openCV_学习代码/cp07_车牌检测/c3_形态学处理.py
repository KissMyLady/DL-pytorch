# coding:utf-8
# Author:mylady
# Datetime:2023/3/28 20:49
import cv2
import numpy as np


def test_1():
    img = cv2.imread('./data/img.jpg', flags=cv2.IMREAD_GRAYSCALE)
    img = (img - 255) * 255
    kernel = np.ones(shape=[3, 3], dtype=np.uint8)

    # 腐蚀，由多变少，越是边界上，越容易被腐蚀，去除噪声，图像变小，变细
    img2 = cv2.erode(img, kernel=kernel, iterations=1)

    # 膨胀，图像变粗，变大
    img3 = cv2.dilate(img2, kernel, iterations=1)

    cv2.imshow('raw', img)  # 原图
    cv2.imshow('erode', img2)  # 腐蚀
    cv2.imshow('dilate', img3)  # 膨胀，还原（噪声去掉还原）
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    pass


def main():
    test_1()
    pass


if __name__ == '__main__':
    main()
