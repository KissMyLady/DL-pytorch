# coding:utf-8
# Author:mylady
# Datetime:2023/3/28 20:36
import cv2
import numpy as np


def test_1():
    video = cv2.VideoCapture('./data/v.mp4')  # 读取文件，视频，视频流，占内存
    # 视频 由一张图片组成的 顺序进行播放 频率 24帧
    # 肉眼，反应不过来，视频（图片一张显示）
    fps = video.get(propId=cv2.CAP_PROP_FPS)  # 视频帧率 24
    width = video.get(propId=cv2.CAP_PROP_FRAME_WIDTH)
    height = video.get(propId=cv2.CAP_PROP_FRAME_HEIGHT)
    count = video.get(propId=cv2.CAP_PROP_FRAME_COUNT)
    print('----视频帧率：', fps)
    print(width, height, count)
    face_detector = cv2.CascadeClassifier('./data/haarcascade_frontalface_alt.xml')

    while True:
        retval, image = video.read()  # retval boolean表明是否获得了图片，True
        if retval is False:  # 取了最后一张，再读取，没有了
            print('视频读取完成，没有图片！')
            break

        image = cv2.resize(image, (640, 360))
        gray = cv2.cvtColor(image, code=cv2.COLOR_BGR2GRAY)
        faces = face_detector.detectMultiScale(gray,
                                               scaleFactor=1.05,
                                               minNeighbors=3,
                                               minSize=(25, 25),
                                               maxSize=(75, 75),
                                               )  # 耗时操作！扫描整张图片

        for x, y, w, h in faces:
            # cv2.rectangle(image,pt1=(x,y),pt2=(x+w,y+h),color=[0,0,255],thickness=2)
            # face = image[y:y + h, x:x + w]
            # face = face[::10, ::10]
            # face = np.repeat(np.repeat(face, 10, axis=0), 10, axis=1)
            # image[y:y + h, x:x + w] = face[:h, :w]
            pass

            # 画框
            cv2.rectangle(image,
                          pt1=(x, y),
                          pt2=(x + w, y + h),
                          color=[0, 0, 255], thickness=2)
            # 文本
            cv2.putText(img=image,
                        text='face',  # 显示文本
                        org=(x, y),  # 位置
                        fontFace=cv2.FONT_HERSHEY_TRIPLEX,  # 字体
                        fontScale=1,  # 字号
                        color=[0, 0, 255],  # 颜色
                        thickness=2)  # 线宽

        cv2.imshow('ttnk', image)

        key = cv2.waitKey(1)  # 等待键盘ascii 1毫秒
        if key == ord('q'):
            print('用户键盘输入了q，死循环退出，不在显示图片')
            break
        if key == 27:
            break

    # print(image.shape)
    cv2.destroyAllWindows()
    video.release()  # 释放内存
    pass


def main():
    test_1()
    pass


if __name__ == '__main__':
    main()
