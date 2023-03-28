# coding:utf-8
# Author:mylady
# Datetime:2023/3/28 13:28
import numpy as np
import cv2


def main():
    img = cv2.imread('./data/sew2.jpeg')
    gray = cv2.cvtColor(img, code=cv2.COLOR_BGR2GRAY)  # 数据变少
    # 人脸特征详细说明，1万多行，计算机根据这些特征，进行人脸检测
    # 符合其中一部分，算做人脸
    # 级联分类器，检测器，
    face_detector = cv2.CascadeClassifier('./data/haarcascade_frontalface_alt.xml')

    """
    minNeighbors 参数越大，条件越苛刻，参数越小，宽松！
    scaleFactor 参数越大，缩放越大，遗漏人脸，参数越小，细腻，找到人脸。
    """
    # faces = face_detector.detectMultiScale(gray,
    #                                        scaleFactor=1.05,  # 缩放
    #                                        minNeighbors=10,
    #                                        minSize=(60, 60))  # 坐标x,y,w,h
    faces = face_detector.detectMultiScale(gray,
                                           scaleFactor=1.05,
                                           minNeighbors=3,
                                           minSize=(25, 25),
                                           maxSize=(70, 70))
    # faces = face_detector.detectMultiScale(gray, minNeighbors=10,)

    print(faces)
    for x, y, w, h in faces:
        # 矩形
        cv2.rectangle(img,
                      pt1=(x, y),  # 左上角
                      pt2=(x + w, y + h), # 右下角
                      color=[0, 0, 255],
                      thickness=2)  # 线宽

        # 文本
        cv2.putText(img=img,
                    text='face',  # 显示文本
                    org=(x, y),  # 位置
                    fontFace=cv2.FONT_HERSHEY_TRIPLEX,  # 字体
                    fontScale=1,  # 字号
                    color=[0, 0, 255],  # 颜色
                    thickness=2)  # 线宽

        # cv2.circle(img, center=(x + w // 2, y + h // 2),  # 圆心
        #            radius=w // 2,  # 半径
        #            color=[0, 255, 0], thickness=2)

        # face = img[y:y + h, x:x + w]
        # face = face[::10, ::10]
        # face = np.repeat(face, 10, axis=0)
        # face = np.repeat(face, 10, axis=1)
        # img[y:y + h, x:x + w] = face[:h, :w]
        pass

    cv2.imshow('face', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    pass


if __name__ == '__main__':
    main()
