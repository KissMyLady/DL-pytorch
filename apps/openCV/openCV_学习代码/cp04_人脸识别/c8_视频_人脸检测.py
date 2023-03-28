# coding:utf-8
# Author:mylady
# Datetime:2023/3/28 14:22
import cv2
import numpy as np


def test_1():
    # 一秒多少帧，24帧
    # v = cv2.VideoCapture('./data/ttnk.mp4')
    # v = cv2.VideoCapture('./data/2023_03_28_14_34_25_684.mp4')
    v = cv2.VideoCapture('./data/2023_03_28_14_42_23_183_广场_僧侣.mp4')
    # cv2.VideoWriter()
    fps = v.get(propId=cv2.CAP_PROP_FPS)  # frame(帧) per second
    print(fps)
    w_ = v.get(propId=cv2.CAP_PROP_FRAME_WIDTH)
    h_ = v.get(propId=cv2.CAP_PROP_FRAME_HEIGHT)
    print(int(w_ // 2), int(h_ // 2))
    face_detector = cv2.CascadeClassifier('./data/haarcascade_frontalface_alt.xml')

    while True:
        flag, frame = v.read()
        # 最后一张图片后面，没有图片了，无法读取图片，返回False
        if flag is False:
            break

        # frame = cv2.resize(frame, dsize=(int(w_ // 2), int(h_ // 2)))
        gray = cv2.cvtColor(frame, code=cv2.COLOR_BGR2GRAY)

        # 播放慢了，检测人脸耗时操作，扫描整张图片，图片大，耗时长
        faces = face_detector.detectMultiScale(frame)
        print(faces)
        for x, y, w, h in faces:
            # cv2.rectangle(frame, pt1=(x, y), pt2=(x + w, y + h), color=[0, 0, 255], thickness=2)
            cv2.rectangle(gray,
                          pt1=(x, y),  # 左上角
                          pt2=(x + w, y + h),  # 右下角
                          color=[0, 0, 255],
                          thickness=2)  # 线宽

            # 文本
            cv2.putText(img=gray,
                        text='face',  # 显示文本
                        org=(x, y),  # 位置
                        fontFace=cv2.FONT_HERSHEY_TRIPLEX,  # 字体
                        fontScale=1,  # 字号
                        color=[0, 0, 255],  # 颜色
                        thickness=2)  # 线宽

        cv2.imshow('frame', gray)
        key = cv2.waitKey(10)  # 等待键盘输入的Key 键盘
        if key == ord('q'):
            break
        if key == 27:
            break

    cv2.destroyAllWindows()
    v.release()  # 释放，视频流
    pass


def main():
    test_1()
    pass


if __name__ == '__main__':
    main()
