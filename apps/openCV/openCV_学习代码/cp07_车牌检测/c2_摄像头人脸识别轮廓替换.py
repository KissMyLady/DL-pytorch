# coding:utf-8
# Author:mylady
# Datetime:2023/3/28 20:47
import numpy as np
import cv2


def test_1():
    # 给视频路径，打开视频
    cap = cv2.VideoCapture(0)  # 打开本机的摄像头
    face_detector = cv2.CascadeClassifier('./data/haarcascade_frontalface_alt.xml')
    head = cv2.imread('./data/head.jpg')

    while True:
        flag, frame = cap.read()  # flag是否读取了图片
        if not flag:
            break

        gray = cv2.cvtColor(frame, code=cv2.COLOR_BGR2GRAY)
        faces = face_detector.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=10)

        for x, y, w, h in faces:
            head2 = cv2.resize(head, dsize=(w, h))
            head2_gray = cv2.cvtColor(head2, code=cv2.COLOR_BGR2GRAY)
            threshold, otsu = cv2.threshold(head2_gray, 100, 255, cv2.THRESH_OTSU)
            contours, hierarchy = cv2.findContours(otsu, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            areas = []

            for c in contours:
                areas.append(cv2.contourArea(c))

            areas = np.asarray(areas)
            index = areas.argsort()
            mask = np.zeros(shape=[h, w], dtype=np.uint8)
            cv2.drawContours(mask, contours, index[-2], (255), thickness=-1)
            for i in range(h):
                for j in range(w):
                    if mask[i, j] == 255:  # !!!
                        frame[i + y, j + x] = head2[i, j]
            pass

        cv2.imshow('face', frame)
        key = cv2.waitKey(1000 // 24)  # 注意是整除//，时间是毫秒，最小1毫秒，必须是整数
        if key == ord('q'):  # 键盘输入q退出
            break
        if key == 27:
            break

    cv2.destroyAllWindows()
    cap.release()
    pass


def main():
    test_1()
    pass


if __name__ == '__main__':
    main()
