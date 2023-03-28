# coding:utf-8
# Author:mylady
# Datetime:2023/3/28 20:30
import numpy as np
import cv2


def test_1():
    han = cv2.imread('./data/han.jpeg')
    head = cv2.imread('./data/head.jpg')

    # 人脸检测
    face_detector = cv2.CascadeClassifier('./data/haarcascade_frontalface_alt.xml')
    han_gray = cv2.cvtColor(han, code=cv2.COLOR_BGR2GRAY)
    faces = face_detector.detectMultiScale(han_gray)
    print("检测到的人脸: ", faces)

    head_gray = cv2.cvtColor(head, cv2.COLOR_BGR2GRAY)

    # 旺旺二进制图片，黑白
    threshold, head_binary = cv2.threshold(head_gray, 50, 255, cv2.THRESH_OTSU)
    contours, hierarchy = cv2.findContours(head_binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    areas = []
    for contour in contours:
        areas.append(cv2.contourArea(contour))

    areas = np.asarray(areas)
    index = areas.argsort()  # 从小到大，倒数第二个，第二大轮廓
    mask = np.zeros_like(head_gray, dtype=np.uint8)  # mask面具
    mask = cv2.drawContours(mask,
                            contours,
                            index[-2],
                            (255, 255, 255),
                            thickness=-1)
    for x, y, w, h in faces:
        mask2 = cv2.resize(mask, (w, h))
        head2 = cv2.resize(head, (w, h))  # 彩色图片
        for i in range(h):
            for j in range(w):
                if (mask2[i, j] == 255).all():
                    han[i + y, j + x] = head2[i, j]
        pass

    cv2.imshow('face', han)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    pass


def main():
    test_1()
    pass


if __name__ == '__main__':
    main()
