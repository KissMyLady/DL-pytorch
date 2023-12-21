# coding:utf-8
# Author:mylady
# Datetime:2023/3/27 15:23
import cv2


def callback(value):
    pass


# 色彩空间的转换
def test_1():
    cv2.namedWindow('color', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('color', 640, 480)

    img = cv2.imread('./cat.png')

    # 常见的颜色空间转换
    colorspaces = [cv2.COLOR_BGR2RGBA,  # brg -> rgb
                   cv2.COLOR_BGR2BGRA,  # brg -> brg
                   cv2.COLOR_BGR2GRAY,  # brg -> gra
                   cv2.COLOR_BGR2HSV,   # brg -> hsv
                   cv2.COLOR_BGR2YUV,   # brg -> yuv
                   ]

    # 创建点击回调事件
    cv2.createTrackbar('curcolor', 'color', 0, 4, callback)

    while True:
        # 获取事件
        index = cv2.getTrackbarPos('curcolor', 'color')

        # 颜色空间转换API
        cvt_img = cv2.cvtColor(img, colorspaces[index])

        cv2.imshow('color', cvt_img)
        key = cv2.waitKey(10)

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
