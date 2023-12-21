# coding:utf-8
# Author:mylady
# Datetime:2023/3/28 20:45
import cv2
import numpy as np


# Licence Plate recognize 车牌识别
def lpr(filename):
    # 1、图片加载和预处理
    img = cv2.imread(filename)
    gray_img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)  # 灰度化处理
    GaussianBlur_img = cv2.GaussianBlur(gray_img, (3, 3), sigmaX=0)  # 高斯平滑，高斯模糊
    canny = cv2.Canny(GaussianBlur_img, 150, 255)  # 轮廓检测
    ret, binary_img = cv2.threshold(canny, 127, 255, cv2.THRESH_OTSU)  # 二值化操作

    # 2、形态学运算
    kernel = np.ones((5, 5), np.uint8)
    # 先闭运算将车牌数字部分连接，再开运算较小的部分去掉
    close_img = cv2.morphologyEx(binary_img, cv2.MORPH_CLOSE, kernel)
    open_img = cv2.morphologyEx(close_img, cv2.MORPH_OPEN, kernel)
    # 由于部分图像得到的轮廓边缘不整齐，因此再进行一次膨胀操作
    dilation_img = cv2.dilate(open_img, np.ones(shape=[5, 5], dtype=np.uint8), iterations=3)

    # 3、获取轮廓
    contours, hierarchy = cv2.findContours(dilation_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    rectangles = []  # 获取轮廓的矩形
    for c in contours:
        x = []
        y = []
        for point in c:
            y.append(point[0][0])
            x.append(point[0][1])
        r = [min(y), min(x), max(y), max(x)]
        rectangles.append(r)

    # 4、根据HSV颜色空间查找汽车上车牌位置
    dist_r = []
    max_mean = 0

    for r in rectangles:
        block = img[r[1]:r[3], r[0]:r[2]]  # block块
        hsv = cv2.cvtColor(block, cv2.COLOR_BGR2HSV)
        low = np.array([100, 43, 46])
        up = np.array([124, 255, 255])
        result = cv2.inRange(hsv, low, up)
        # 用计算均值的方式找蓝色最多的区块
        mean = np.mean(result)
        if mean > max_mean:
            max_mean = mean
            dist_r = r
        pass

    # 画出识别结果，由于之前多做了一次膨胀操作，导致矩形框稍大了一些，因此这里对于框架+3-3可以使框架更贴合车牌
    cv2.rectangle(img, (dist_r[0] + 3, dist_r[1]), (dist_r[2] - 3, dist_r[3]), (0, 255, 0), 2)

    # img 截取, 读取牌号

    cv2.imshow("lpr", img)
    cv2.waitKey(0)


def main():
    # 主程序
    lpr('./data/car1.jpeg')
    pass


if __name__ == '__main__':
    main()
