# coding:utf-8
# Author:mylady
# Datetime:2023/3/28 8:42
import cv2
import numpy as np


def test_1():
    img = cv2.imread('./data/bao.jpeg')
    print(img.shape)  # 高度232，宽度350

    # 马赛克方式一
    # img2 = cv2.resize(img, (35, 23))
    # img3 = cv2.resize(img2, (350, 232))
    # cv2.imshow('bao', img3)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    # 马赛克方式二
    # img2 = cv2.resize(img,(35,23))
    # img3 = np.repeat(img2,10,axis=0) # 重复行
    # img4 = np.repeat(img3,10,axis=1) # 重复列
    # cv2.imshow('bao',img4)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    # 马赛克方式3
    img2 = img[::10,::10]  # 每10个中取出一个像素，细节
    cv2.namedWindow('bao',flags=cv2.WINDOW_NORMAL)
    cv2.resizeWindow('bao',350,232)
    cv2.imshow('bao',img2)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    pass


def main():
    test_1()
    pass


if __name__ == '__main__':
    main()
