# coding:utf-8
# Author:mylady
# Datetime:2023/3/27 16:24
import cv2

# 设置参数
draw = False
ix, iy = -1, -1
mode = None


# 定义回调函数
def draw_line(event, x, y, flags, param):
    global ix, iy, draw, mode

    # 鼠标左键按下，开始画线或矩形
    if event == cv2.EVENT_LBUTTONDOWN:
        draw = True
        ix, iy = x, y

        if mode == "line":
            cv2.line(img, (ix, iy), (x, y), (0, 0, 255), 2)
        elif mode == "rectangle":
            cv2.rectangle(img, (ix, iy), (x, y), (0, 255, 0), 2)
        elif mode == "cycle":
            cv2.rectangle(img, (ix, iy), (x, y), (0, 255, 0), 2)

    # 鼠标移动，画线或矩形
    elif event == cv2.EVENT_MOUSEMOVE:
        if draw == True:
            if mode == "line":
                cv2.line(img, (ix, iy), (x, y), (255, 0, 0), 2)
            elif mode == "rectangle":
                cv2.rectangle(img, (ix, iy), (x, y), (0, 255, 0), 2)
            elif mode == "cycle":
                cv2.rectangle(img, (ix, iy), (x, y), (0, 255, 0), 2)

    # 鼠标左键释放，结束画线或矩形
    elif event == cv2.EVENT_LBUTTONUP:
        draw = False
        if mode == "line":
            cv2.line(img, (ix, iy), (x, y), (0, 0, 255), 2)
        elif mode == "rectangle":
            cv2.rectangle(img, (ix, iy), (x, y), (0, 255, 0), 2)
        elif mode == "cycle":
            cv2.circle(img, (320, 240), 100, (0, 0, 255))


# 创建窗口并显示图片
img = cv2.imread("./cat.png")
cv2.namedWindow('image')
cv2.setMouseCallback('image', draw_line)


while True:
    cv2.imshow('image', img)
    k = cv2.waitKey(1) & 0xFF

    # 直线
    if k == ord('l'):
        mode = 'line'

    # 矩形
    if k == ord('r'):
        mode = 'rectangle'

    if k == 27:
        break

cv2.destroyAllWindows()


def main():
    """
    实现按l键之后拖动鼠标绘制直线, 按r键之后拖动鼠标绘制矩形, 按r键拖动鼠标绘制圆形
    """
    pass


if __name__ == '__main__':
    main()
