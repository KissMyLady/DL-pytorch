# coding:utf-8
# Author:mylady
# Datetime:2023/3/28 20:32
import numpy as np
import cv2


def test_1():
    han = cv2.imread('./data/han.jpeg')
    head = cv2.imread('./data/head.jpg')

    # 人脸检测
    face_detector = cv2.CascadeClassifier('./data/haarcascade_frontalface_alt.xml')
    han_gray = cv2.cvtColor(han, code=cv2.COLOR_BGR2GRAY)
    faces = face_detector.detectMultiScale(han_gray)
    head_gray = cv2.cvtColor(head, cv2.COLOR_BGR2GRAY)

    # 旺旺二进制图片，黑白
    threshold, head_binary = cv2.threshold(head_gray, 50, 255, cv2.THRESH_OTSU)
    contours, hierarchy = cv2.findContours(head_binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    areas = []
    for contour in contours:
        areas.append(cv2.contourArea(contour))
        pass

    areas = np.asarray(areas)
    index = areas.argsort()  # 从小到大，倒数第二个，第二大轮廓
    mask = np.zeros_like(head_gray, dtype=np.uint8)  # mask面具
    mask = cv2.drawContours(mask, contours, index[-2], (255, 255, 255),
                            thickness=-1)

    for x, y, w, h in faces:  # w和h值一样的（正方形），意义不同
        # 位运算比上一个代码中的for循环条件判断高级！！！
        mask2 = cv2.resize(mask, (w, h))
        head2 = cv2.resize(head, (w, h))  # 彩色图片
        mask3 = (mask2 - 255) * 255  # mask3和mask2正好相反
        # mask3 = mask2.copy()
        # for i in range(h):
        #     for j in range(w):
        #         if mask3[i,j] == 255:
        #             mask3[i,j] = 0
        #         else:
        #             mask3[i,j] = 255
        face = han[y:y + h, x:x + w]
        # 黑色地方，没有进行计算，白色的地方进行了与运算
        head3 = cv2.bitwise_and(head2, head2, mask=mask2)
        face = cv2.bitwise_and(face, face, mask=mask3)
        face = cv2.bitwise_or(head3, face)  # 没有给mask 计算所有
        han[y:y + h, x:x + w] = face
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
