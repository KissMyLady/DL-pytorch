# coding:utf-8
# Author:mylady
# Datetime:2023/3/28 13:40
import cv2
import numpy as np


def main():
    # findContours()
    # img = cv2.imread('./data/head.jpg')
    img = cv2.imread('./data/nba.jpeg')
    gray = cv2.cvtColor(img, code=cv2.COLOR_BGR2GRAY)  # 中间地带

    # 二值化，只有两个值，离散化，黑纯黑，白纯白
    threshold, gray2 = cv2.threshold(gray, thresh=100, maxval=255, type=cv2.THRESH_OTSU)

    # 可以找到多个轮廓，大轮廓，小轮廓！
    contours, hierarchy = cv2.findContours(gray2, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # areas = [] # 最大轮廓，面积最大的区域
    # area = 0
    # for  i in range(len(contours)):
    #     area = cv2.contourArea(contours[i]) # 计算轮廓面积
    #     areas.append(area)
    # areas = np.asarray(areas)
    # print(areas)
    # index = areas.argsort()
    # print(index)
    # print(len(contours))

    # 纯黑的图片
    h, w, c = img.shape
    result = np.zeros(shape=(h, w, c), dtype=np.uint8)
    cv2.drawContours(result, contours, -1, (0, 0, 255), 5, cv2.LINE_8)
    # print(contours[1])
    # cv2.fillPoly(result,pts = contours[1],color = [0,0,255])

    cv2.imshow('gray', gray)
    cv2.imshow('gray2', gray2)
    cv2.imshow('result', result)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    pass


if __name__ == '__main__':
    main()
