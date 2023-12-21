# coding:utf-8
# Author:mylady
# Datetime:2023/3/27 17:57
# 按下l, 拖动鼠标, 可以绘制直线.
# 按下r, 拖到鼠标, 可以绘制矩形
# 按下c, 拖动鼠标, 可以绘制圆. 拖动的长度可以作为半径.
import cv2
import numpy as np

# 这是一个全局标志, 判断要画什么类型的图.
curshape = 0
startpos = (0, 0)

# 创建背景图
img = np.zeros((480, 640, 3), np.uint8)


# 要监听鼠标的行为, 所以必须通过鼠标回调函数实现.
def mouse_callback(event, x, y, flags, userdata):
    # 引入全局变量
    global curshape, startpos
    # 引入非本层的局部变量用什么关键字nonlocal
    if event == cv2.EVENT_LBUTTONDOWN:
        # 记录起始位置
        startpos = (x, y)

    elif event == 0 and flags == 1:  # 表示按下鼠标左键并移动鼠标

        if curshape == 0:  # 画直线
            cv2.line(img, startpos, (x, y), (0, 0, 255), 1)
        elif curshape == 1:  # 画矩形
            cv2.rectangle(img, startpos, (x, y), (0, 0, 255), 1)
        elif curshape == 2:  # 画圆
            # 注意计算半径
            a = (x - startpos[0])
            b = (y - startpos[1])
            r = int((a ** 2 + b ** 2) ** 0.5)
            # 画圆的时候, 半径必须是整数
            cv2.circle(img, startpos, r, (0, 0, 255), 1)
        else:  # 按其他的按键
            print('暂不支持绘制其他图形')

    elif event == cv2.EVENT_LBUTTONUP:
        # 判断要画什么类型的图.
        if curshape == 0:  # 画直线
            cv2.line(img, startpos, (x, y), (0, 0, 255), 3)
        elif curshape == 1:  # 画矩形
            cv2.rectangle(img, startpos, (x, y), (0, 0, 255), 1)
        elif curshape == 2:  # 画圆
            # 注意计算半径
            a = (x - startpos[0])
            b = (y - startpos[1])
            r = int((a ** 2 + b ** 2) ** 0.5)
            # 画圆的时候, 半径必须是整数
            cv2.circle(img, startpos, r, (0, 0, 255), 1)
        else:  # 按其他的按键
            print('暂不支持绘制其他图形')
    pass


# 创建窗口
cv2.namedWindow('drawshape', cv2.WINDOW_NORMAL)
# 设置鼠标回调函数
cv2.setMouseCallback('drawshape', mouse_callback)

while True:
    cv2.imshow('drawshape', img)
    # 检测按键
    key = cv2.waitKey(1)
    if key == ord('q'):
        break
    elif key == ord('l'):
        curshape = 0
    elif key == ord('r'):
        curshape = 1
    elif key == ord('c'):
        curshape = 2

cv2.destroyAllWindows()


def main():
    pass


if __name__ == '__main__':
    main()
