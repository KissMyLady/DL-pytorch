# coding:utf-8
# Author:mylady
# Datetime:2023/3/28 20:33
import cv2
import numpy as np
from matplotlib import pyplot as plt


def test_1():
    img = cv2.imread('./data/moon.png', 0)
    f = np.fft.fft2(img)  # 时域到频域
    # Shift the zero-frequency component to the center of the spectrum
    # 移动0频率的数据到中心
    fshift = np.fft.fftshift(f)
    # print(fshift)
    # # 这里构建振幅图的公式没学过
    magnitude_spectrum1 = 20 * np.log(np.abs(f))  # 对数运算，大幅缩小数据
    magnitude_spectrum2 = 20 * np.log(np.abs(fshift))  # 对数运算，大幅缩小数据

    # 振幅谱图
    # 131表示：1行3列第1个
    plt.figure(figsize=(12, 4))
    plt.subplot(131), plt.imshow(img, cmap='gray')
    plt.title('Input Image')
    plt.subplot(132), plt.imshow(magnitude_spectrum1, cmap='gray')
    plt.title('Magnitude Spectrum')
    plt.subplot(133), plt.imshow(magnitude_spectrum2, cmap='gray')
    plt.title('Magnitude Spectrum shift')
    plt.show()
    pass


def main():
    test_1()
    pass


if __name__ == '__main__':
    main()
