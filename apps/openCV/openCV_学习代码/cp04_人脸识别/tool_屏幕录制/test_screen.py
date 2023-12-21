# coding:utf-8
# Author:mylady
# Datetime:2023/3/28 15:31
import numpy as np
from PIL import ImageGrab
import cv2


def test_1():
    im = ImageGrab.grab()
    width, high = im.size  # 获取屏幕的宽和高
    fourcc = cv2.VideoWriter_fourcc(*'I420')  # 设置视频编码格式
    fps = 15  # 设置帧率
    video = cv2.VideoWriter('./data/screen_test.avi',
                            fourcc, fps, (width, high)
                            )

    while True:
        im = ImageGrab.grab()
        im_cv = cv2.cvtColor(np.array(im), cv2.COLOR_BGR2RGB)
        # 图像写入
        video.write(im_cv)
        key = cv2.waitKey(10)  # 等待键盘输入的Key 键盘
        if key == ord('q'):
            break
        if key == 27:
            break

    video.release()  # 释放缓存，持久化视频
    pass


def main():
    pass


if __name__ == '__main__':
    main()
