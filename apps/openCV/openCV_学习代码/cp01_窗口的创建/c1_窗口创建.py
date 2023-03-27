# coding:utf-8
# Author:mylady
# Datetime:2023/3/26 20:55
import cv2
import matplotlib.pyplot as plt
import numpy as np


def test_1():
    # cv2.namedWindow('new', cv2.WINDOW_AUTOSIZE)
    # WINDOW_NORMAL可以让窗口大小变得可以调节
    cv2.namedWindow('new', cv2.WINDOW_NORMAL)

    # 修改窗口大小
    # cv2.resizeWindow('new', 1920, 1080)
    cv2.resizeWindow('new', 500, 456)
    cv2.imshow('new', 0)

    # waitKey方法表示等待按键, 0表示任何按键, 其他整数表示等待按键的时间,单位是毫秒, 超过时间没有发生按键操作窗口会自动关闭.
    # 会返回按键的ascii的值
    key = cv2.waitKey(0)
    if key == ord('q'):
        cv2.destroyAllWindows()
        pass
    pass


def cv_show(name, img):
    cv2.imshow(name, img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


# 读取图片
def test_2():
    img = cv2.imread('./cat.png')
    # cv2.imshow('cat', img)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    cv_show("zhanShan", img)
    pass


# 保存图片
def test_3():
    cv2.namedWindow('img', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('img', 320, 240)
    img = cv2.imread("./cat.png")

    # 利用while循环优化退出逻辑
    while True:
        cv2.imshow('img', img)
        key = cv2.waitKey(0)

        if (key & 0xFF == ord('q')):
            break
        elif (key & 0xFF == ord('s')):
            print("输入了s, 保存文件")
            cv2.imwrite("./123.png", img)
        else:
            print('其他键, 不做处理: ',key)
        pass
    cv2.destroyAllWindows()
    pass


def main():
    # test_1()  # 窗口展示
    # test_2() # 读取图片
    test_3()  # 写入

    pass


if __name__ == '__main__':
    main()
