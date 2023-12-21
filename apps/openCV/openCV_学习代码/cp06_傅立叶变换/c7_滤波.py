# coding:utf-8
# Author:mylady
# Datetime:2023/3/28 20:35
import cv2
import numpy as np
import matplotlib.pyplot as plt


def test_1():
    # 加载图片，处理
    moon = cv2.imread('./data/moon.png', flags=cv2.IMREAD_GRAYSCALE)
    moon2 = moon / 255  # float32数据

    # 傅里叶变换，移动，将低频移动到中心
    dft = cv2.dft(moon2, flags=cv2.DFT_COMPLEX_OUTPUT)  # 经过傅里叶变换的频率
    dft_shift = np.fft.fftshift(dft)  # 移动zero-frequency component 低频波 中心
    print(dft_shift.shape)
    # 低通滤波：过滤高频波
    # h,w = moon.shape
    # h2,w2 = h//2,w//2 #中心点
    # mask = np.zeros((h,w,2),dtype=np.uint8)
    # mask[h2-15:h2+15,w2-15:w2+15] = 1
    # dft_shift*=mask # 中心区域15*15不变，其他（高频）变成了0，高频过滤，噪声

    # 高通滤波：过滤低频波(细节)，保留高频波（'噪声'，突兀，变化明显的地方）
    h, w = moon.shape
    h2, w2 = h // 2, w // 2
    dft_shift[h2 - 15:h2 + 15, w2 - 15:w2 + 15] = 0

    # 翻转
    ifft_shift = np.fft.ifftshift(dft_shift)
    result = cv2.idft(ifft_shift)

    # 显示图片
    plt.figure(figsize=(12, 9))
    plt.subplot(121)
    plt.imshow(moon, cmap='gray')
    plt.subplot(122)
    plt.imshow(result[:, :, 0], cmap='gray')  # 实数
    plt.show()

    pass


def main():
    test_1()
    pass


if __name__ == '__main__':
    main()
