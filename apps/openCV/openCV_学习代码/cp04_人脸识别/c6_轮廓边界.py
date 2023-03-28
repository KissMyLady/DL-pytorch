# coding:utf-8
# Author:mylady
# Datetime:2023/3/28 13:37
import cv2
import numpy as np


def main():
    dog = cv2.imread('./data/head.jpg')
    cv2.imshow('dog_origin', dog)

    gray = cv2.cvtColor(dog, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.GaussianBlur(gray, (5, 5), 0)  # 高斯平滑，高斯模糊
    canny = cv2.Canny(gray2, 150, 200)
    cv2.imshow('dog', gray)
    cv2.imshow('dog2', gray2)
    cv2.imshow('canny', canny)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    pass


if __name__ == '__main__':
    main()
