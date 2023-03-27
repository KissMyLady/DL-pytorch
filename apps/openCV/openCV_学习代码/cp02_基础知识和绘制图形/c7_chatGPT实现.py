# coding:utf-8
# Author:mylady
# Datetime:2023/3/27 16:24
import cv2
import numpy as np

# 设置参数
drawing = False
mode = 0  # 0为画线模式，1为画矩形模式，2为画圆形模式
ix, iy = -1, -1


# 定义回调函数
def draw_shape(event, x, y, flags, param):
    global ix, iy, drawing, mode

    # 鼠标左键按下，开始画图形
    if event == cv2.EVENT_LBUTTONDOWN:
        drawing = False
        ix, iy = x, y

    # 鼠标移动，画图形
    elif event == cv2.EVENT_MOUSEMOVE:
        if drawing is True:
            if mode == 0:
                cv2.line(img, (ix, iy), (x, y), (0, 0, 255), 2)
            elif mode == 1:
                cv2.rectangle(img, (ix, iy), (x, y), (0, 255, 0), 2)
            elif mode == 2:
                radius = int(np.sqrt((x - ix) ** 2 + (y - iy) ** 2))
                cv2.circle(img, (ix, iy), radius, (255, 0, 0), 2)

    # 鼠标左键释放，结束画图形
    elif event == cv2.EVENT_LBUTTONUP:
        drawing = False
        if mode == 0:
            cv2.line(img, (ix, iy), (x, y), (0, 0, 255), 2)
        elif mode == 1:
            cv2.rectangle(img, (ix, iy), (x, y), (0, 255, 0), 2)
        elif mode == 2:
            radius = int(np.sqrt((x - ix) ** 2 + (y - iy) ** 2))
            cv2.circle(img, (ix, iy), radius, (255, 0, 0), 2)


# 创建窗口并显示图片
img = cv2.imread("cat.png")
cv2.namedWindow('image')
cv2.setMouseCallback('image', draw_shape)

while True:
    cv2.imshow('image', img)
    k = cv2.waitKey(1) & 0xFF

    # 按l键切换到画线模式
    if k == ord('l'):
        mode = 0

    # 按r键切换到画矩形模式
    elif k == ord('r'):
        mode = 1

    # 按c键切换到画圆模式
    elif k == ord('c'):
        mode = 2

    # 按esc键退出程序
    elif k == 27:
        break

cv2.destroyAllWindows()


def main():
    """
    实现按l键之后拖动鼠标绘制直线, 按r键之后拖动鼠标绘制矩形, 按r键拖动鼠标绘制圆形
    """
    pass


if __name__ == '__main__':
    main()
