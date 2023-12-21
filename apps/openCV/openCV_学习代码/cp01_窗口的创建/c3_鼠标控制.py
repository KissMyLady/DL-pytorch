# coding:utf-8
# Author:mylady
# Datetime:2023/3/27 14:17
import cv2
import numpy as np


def mouse_callback(event, x, y, flags, userdata):
    """
    event  事件(鼠标移动, 左键, 右键等),
    x,y    点鼠标的坐标点,
    flags   主要用于组合键,
    userdata 就是上面的 setMouseCallback的 userdata
    """
    coordinate_xy = " x: %s, y: %s " % (x, y)

    print('鼠标事件: ', event, coordinate_xy, 'flags: ', flags, 'userdata: ', userdata)


def test_1():
    cv2.namedWindow('mouse', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('mouse', 640, 360)

    # 设置鼠标回调函数
    cv2.setMouseCallback('mouse', mouse_callback, '123')

    # 显示窗口和背景
    # 生成全黑的图片
    img = np.zeros((360, 640, 3), np.uint8)
    while True:
        cv2.imshow('mouse', img)
        key = cv2.waitKey(1)
        if key & 0xFF == ord('q'):
            break
        elif key == 27:
            break

    cv2.destroyAllWindows()

    pass


def main():
    test_1()
    pass


if __name__ == '__main__':
    main()
