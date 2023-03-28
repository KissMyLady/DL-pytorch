# coding:utf-8
# Author:mylady
# Datetime:2023/3/28 13:20
import numpy as np
import cv2


def test_1():
    img = cv2.imread('./data/bao.jpeg')
    # 1、人为人脸定位
    # 人脸左上角坐标（143，42）；右下角（239，164）（x,y）(宽度、高度)

    # 2、切片获取人脸
    face = img[42:164, 138:244]

    # 3、间隔切片，重复，切片，赋值
    face = face[::7, ::7]  # 每7个中取出一个像素，马赛克
    face = np.repeat(face, 7, axis=0)  # 行方向重复10次
    face = np.repeat(face, 7, axis=1)  # 列方向上重复10次
    img[42:164, 138:244] = face[:122, :106]  # 填充，尺寸一致

    # 4、显示
    cv2.imshow('bao', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    pass


# 实现二
def test_2():
    img = cv2.imread('./data/bao.jpeg')

    # 使用人脸检测, 然后涂抹马赛克
    face_detector = cv2.CascadeClassifier('./data/haarcascade_frontalface_alt.xml')
    gray = cv2.cvtColor(img, code=cv2.COLOR_BGR2GRAY)
    faces = face_detector.detectMultiScale(gray)
    for x, y, w, h in faces:
        # 图片：高度、宽度、像素
        face = img[y:y + h, x:x + w]  # x 横坐标，宽度；y 纵坐标，高度
        face = face[::10, ::10]  # 信息缺失的人脸，模糊人脸
        # # face = cv2.resize(face,dsize = (w,h))
        # face = np.repeat(face,10,axis = 0)
        # face = np.repeat(face, 10, axis=1)
        # img[y:y+h,x:x+w] = face[:h,:w] # 10个中取一个像素，信息变少了
        h2, w2 = face.shape[:2]
        for i in range(h2):
            for j in range(w2):
                # 切片
                # i = 0 y:y+10
                # i = 1 y+10:y+20
                img[i * 10 + y:(i + 1) * 10 + y, j * 10 + x:(j + 1) * 10 + x] = face[i, j]  # 一个像素抵十个像素进行替换

    cv2.imshow('face', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    pass


def main():
    test_2()
    pass


if __name__ == '__main__':
    main()
