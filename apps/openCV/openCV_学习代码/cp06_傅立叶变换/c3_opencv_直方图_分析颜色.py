# coding:utf-8
# Author:mylady
# Datetime:2023/3/28 20:23
import numpy as np
import cv2  # opencv没有提供案例图片
import matplotlib.pyplot as plt
from skimage import data  # pip install scikit-image


def main():
    moon = data.moon()

    plt.hist(moon.ravel(), bins=256)
    plt.show()
    cv2.imshow('moon', moon)

    # 直方图均衡化，可以将图片的明暗对比增强
    moon2 = cv2.equalizeHist(moon)  # 直方图均衡化！平均一下！
    plt.hist(moon2.reshape(-1), bins=256)
    plt.show()

    hist = cv2.calcHist([moon], [0], None, [256], [0, 256])
    # plt.bar(x = np.arange(0,256),height=hist.reshape(-1))
    # plt.show()

    cv2.imshow('moon2', moon2)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    pass


if __name__ == '__main__':
    main()
