# coding:utf-8
# Author:mylady
# Datetime:2023/3/28 20:15
import numpy as np
import cv2
import matplotlib.pyplot as plt


def test_1():
    img = cv2.imread('./data/moon.png', flags=cv2.IMREAD_GRAYSCALE)  # 灰度化图片
    dft = cv2.dft(img / 255, flags=cv2.DFT_COMPLEX_OUTPUT)
    dft_shift = np.fft.fftshift(dft)

    magnitude_spectrum = 20 * np.log(cv2.magnitude(dft_shift[:, :, 0], dft_shift[:, :, 1]))

    plt.subplot(121), plt.imshow(img, cmap='gray')
    plt.title('Input Image')
    plt.subplot(122), plt.imshow(magnitude_spectrum, cmap='gray')
    plt.title('Magnitude Spectrum'), plt.xticks([]), plt.yticks([])
    plt.show()
    pass


def test_2():
    img = cv2.imread('./data/lena.jpeg', flags=cv2.IMREAD_GRAYSCALE)
    img2 = img / 255
    # gray = cv2.cvtColor(img, code=cv2.COLOR_BGR2GRAY)

    # 傅里叶变换，时域----->频域
    dft = cv2.dft(img2, flags=cv2.DFT_COMPLEX_OUTPUT)  # 输出既有实数，又有虚数
    dft_shift = np.fft.fftshift(dft)  # 低频 0_frequency 移动到中心

    # 低通滤波，低频滤波，保留低频中心，去除高频，外边
    h, w = img.shape
    x, y = w // 2, h // 2  # 中心点
    mask = np.zeros(shape=(h, w, 2), dtype=np.uint8)
    mask[y - 20:y + 20, x - 20:x + 20] = 1  # 中心位置是1，边界是0
    dft_shift *= mask  # 中心不变，边界*0变成了，高频，过滤掉
    dft_ishift = np.fft.ifftshift(dft_shift)
    result = cv2.idft(dft_ishift)

    # 高通滤波，高频滤波，保留高频，低频过滤掉，设置成0
    # h,w = img.shape
    # x,y = w//2,h//2 # 图片的中心点坐标
    # dft_shift[y-5:y+5,x-5:x+5] = 0
    # dft_ishift = np.fft.ifftshift(dft_shift) # 0_frequency 复原
    # result = cv2.idft(dft_ishift)

    res = result[:, :, 0]
    # (数组 - min)/(max - min) ----> 0 ~ 1数据，表示颜色
    res = (res - res.min()) / (res.max() - res.min())
    print(res.min(), res.max())
    cv2.imshow('HPF', res)
    cv2.imshow('lena', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    # plt.figure(figsize=(18,9))
    # plt.subplot(121)
    # plt.imshow(img,cmap = 'gray')
    # plt.subplot(122)
    # plt.imshow(result[:,:,0],cmap = 'gray')
    # plt.show()
    pass


def main():
    test_1()
    pass


if __name__ == '__main__':
    main()
