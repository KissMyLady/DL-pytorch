# coding:utf-8
# Author:mylady
# Datetime:2023/3/28 13:35
import cv2
import numpy as np


def main():
    img = cv2.imread('./data/flower.jpg')  # 蓝绿红，适合显示图片
    hsv = cv2.cvtColor(img, code=cv2.COLOR_BGR2HSV)  # 颜色空间，适合计算

    # 轮廓查找，使用的是，颜色值，进行的
    lower_red = (156, 50, 50)  # 浅红色
    upper_red = (180, 255, 255)  # 深红色
    mask = cv2.inRange(hsv, lower_red, upper_red)

    # # 手工绘制轮廓
    # h,w,c = img.shape
    # mask = np.zeros((h, w), dtype=np.uint8)
    # x_data = np.array([124, 169, 208, 285, 307, 260, 175])+110 # 横坐标
    # y_data = np.array([205, 124, 135, 173, 216, 311, 309]) +110# 纵坐标
    # pts = np.c_[x_data,y_data] # 横纵坐标合并，点（x,y）
    # print(pts)
    # cv2.fillPoly(mask, [pts], (255)) # 绘制多边形！

    res = cv2.bitwise_and(img, img, mask=mask)
    cv2.imshow('flower', res)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    pass


if __name__ == '__main__':
    main()
