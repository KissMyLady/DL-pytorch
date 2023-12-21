# coding:utf-8
# Author:mylady
# Datetime:2023/3/28 20:34
import numpy as np
import cv2
import matplotlib.pyplot as plt


def test_1():
    moon = cv2.imread('./data/moon.png')
    f = np.fft.fft2(moon)  # 大部分都是0，快速傅里叶变换，时域---->频域
    fshift = np.fft.fftshift(f)  # 低通滤波，将低频移动到中心

    # 60*60范围定义为低频波
    row, col = moon.shape[0] // 2, moon.shape[1] // 2
    fshift[row - 60:row + 60, col - 60:col + 60] = 0

    f = np.fft.ifftshift(fshift)

    moon2 = np.fft.ifft2(f)  # 频域 ----->时域
    moon2 = np.abs(moon2)  # 去除虚数，保留实部

    plt.subplot(121)
    plt.imshow(moon, cmap='gray')
    plt.subplot(122)
    plt.imshow(moon2, cmap='gray')

    plt.show()
    pass


def main():
    test_1()
    pass


if __name__ == '__main__':
    main()
