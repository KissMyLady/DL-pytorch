# coding:utf-8
# Author:mylady
# Datetime:2023/3/28 15:16
import cv2
import numpy as np
import datetime


def test_1():
    # 一秒多少帧，24帧
    v = cv2.VideoCapture('./data/2023_03_28_14_42_23_183_广场_僧侣.mp4')
    fourcc = v.get(propId=cv2.CAP_PROP_FOURCC)
    print(fourcc)

    fourcc = cv2.VideoWriter.fourcc('m', 'p', '4', 'v')
    writer = cv2.VideoWriter(filename='./data/save_gray_%s.mp4' % datetime.time(),
                             fourcc=fourcc,  # cv2.VideoWriter.fourcc(*'MP4V'),
                             fps=24,
                             frameSize=(640, 480))

    while True:
        flag, frame = v.read()
        if flag is False:
            break

        gray = cv2.cvtColor(frame, code=cv2.COLOR_BGR2GRAY)
        # gray = np.reshape(gray, [720, 1280, 1])  # 增加一维
        # gray = np.repeat(gray, 3, axis=-1)  # 本来彩色，但是蓝绿红，一样，波动，绚丽色彩
        print('-------gray:', gray.shape)
        print('------frame:', frame.shape)
        cv2.imshow('frame', gray)

        writer.write(gray)  # 存的是黑白
        key = cv2.waitKey(1)  # 等待键盘输入的Key 键盘
        if key == ord('q'):
            break
        if key == 27:
            break
        pass

    cv2.destroyAllWindows()
    v.release()  # 释放，视频流
    writer.release()
    pass


def main():
    test_1()
    pass


if __name__ == '__main__':
    main()
