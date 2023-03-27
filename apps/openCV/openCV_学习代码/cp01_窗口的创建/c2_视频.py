# coding:utf-8
# Author:mylady
# Datetime:2023/3/27 13:40
import cv2


def test_1():
    # 打开视频文件
    vc = cv2.VideoCapture('./穿越火线女团活动宣传CG完整版.mp4')

    # 打开摄像头
    # vc = cv2.VideoCapture(0)

    # 检查是否正确打开
    if vc.isOpened() is False:
        open = False

        pass
    # 读取视频的一帧.
    open, frame = vc.read()
    while True:
        # 可以读到内容ret返回True
        ret, frame = vc.read()
        # 读到最后frame就是空
        if frame is None:
            break
        if ret == True:
            cv2.imshow('result', frame)
            # 0xFF == 27表示按esc退出键会退出
            if cv2.waitKey(33) & 0xFF == 27:
                break

    vc.release()
    cv2.destroyAllWindows()
    pass


def main():
    test_1()
    pass


if __name__ == '__main__':
    main()
