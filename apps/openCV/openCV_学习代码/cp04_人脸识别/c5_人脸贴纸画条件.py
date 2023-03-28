# coding:utf-8
# Author:mylady
# Datetime:2023/3/28 13:35
import cv2


def main():
    img = cv2.imread('./data/han.jpeg')
    gray = cv2.cvtColor(img, code=cv2.COLOR_BGR2GRAY)
    face_detector = cv2.CascadeClassifier('./data/haarcascade_frontalface_alt.xml')
    faces = face_detector.detectMultiScale(gray)
    star = cv2.imread('./data/star2.jpeg')

    for x, y, w, h in faces:
        star_s = cv2.resize(star, (w // 2, h // 2))
        w1 = w // 2
        h1 = h // 2
        for i in range(h1):
            for j in range(w1):  # 遍历 图片数据
                if not (star_s[i, j] > 240).all():  # 红色
                    img[i + y, j + x + w // 4] = star_s[i, j]
        pass

    cv2.imshow('face', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    pass


if __name__ == '__main__':
    main()
