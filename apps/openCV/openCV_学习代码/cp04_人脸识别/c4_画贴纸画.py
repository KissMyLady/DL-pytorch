# coding:utf-8
# Author:mylady
# Datetime:2023/3/28 13:34
import cv2
import numpy as np


def test_1():
    img = cv2.imread('./data/han.jpeg')
    gray = cv2.cvtColor(img, code=cv2.COLOR_BGR2GRAY)
    face_detector = cv2.CascadeClassifier('./data/haarcascade_frontalface_alt.xml')
    faces = face_detector.detectMultiScale(gray)
    star = cv2.imread('./data/star.jpg')

    for x, y, w, h in faces:
        cv2.rectangle(img, pt1=(x, y), pt2=(x + w, y + h), color=[0, 0, 255], thickness=2)
        d = (3 * w) // 8
        img[y:y + h // 4, x + d:x + d + w // 4] = cv2.resize(star, (w // 4, h // 4))
        pass

    cv2.imshow('face', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    pass


def test_2():
    img = cv2.imread('./data/bao.jpeg')

    # 检测人脸
    face_detector = cv2.CascadeClassifier('./data/haarcascade_frontalface_alt.xml')
    gray = cv2.cvtColor(img, code=cv2.COLOR_BGR2GRAY)
    faces = face_detector.detectMultiScale(gray)
    print("faces位置: ", faces)
    star = cv2.imread('./data/star.jpg')
    print(star)

    for x, y, w, h in faces:
        # 把五角星，放到额头正中间
        star2 = cv2.resize(star, dsize=(w // 6, h // 6))
        # img[y:y+h//6,x + 5*w//12: 5*w//12+x+w//6] = star2
        for i in range(h // 6):
            for j in range(w // 6):
                if (star2[i, j] > 200).all():  # 大白色
                    pass  # 不赋值
                else:
                    # 不是大白色，赋值
                    img[i + y, j + x + 5 * w // 12] = star2[i, j]
        pass

    cv2.imshow('face', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    pass


def main():
    test_2()
    pass


if __name__ == '__main__':
    main()
