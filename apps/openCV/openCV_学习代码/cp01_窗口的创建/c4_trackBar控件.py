# coding:utf-8
# Author:mylady
# Datetime:2023/3/27 14:28
import cv2
import numpy as np


# 定义回调函数
def callback(value):
    print(value)


def test_1():
    # 创建窗口
    cv2.namedWindow('trackbar', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('trackbar', 640, 480)

    # 创建trackbar
    cv2.createTrackbar('R', 'trackbar', 0, 255, callback)
    cv2.createTrackbar('G', 'trackbar', 0, 255, callback)
    cv2.createTrackbar('B', 'trackbar', 0, 255, callback)

    # 创建一个背景图片
    img = np.zeros((480, 640, 3), np.uint8)
    while True:
        # 获取当前trackbar的值
        r = cv2.getTrackbarPos('R', 'trackbar')
        g = cv2.getTrackbarPos('G', 'trackbar')
        b = cv2.getTrackbarPos('B', 'trackbar')

        # 改变背景图颜色
        img[:] = [b, g, r]
        cv2.imshow('trackbar', img)

        key = cv2.waitKey(1)
        if key & 0xFF == ord('q'):
            break
        elif key == 27:
            break

    cv2.destroyAllWindows()
    pass


def main():
    # 根据鼠标动作, 执行更新画布
    test_1()
    pass


if __name__ == '__main__':
    main()
