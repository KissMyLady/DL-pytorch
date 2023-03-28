# coding:utf-8
# Author:mylady
# Datetime:2023/3/28 16:07
import cv2
import os
import numpy as np


def load_data():
    listdir = os.listdir('./data/faces_dynamic')

    # 列表生成式
    names = [d for d in listdir if not d.startswith('.')]
    faces = []
    target = [i for i in range(len(names))] * 10

    for dir in names:
        for i in range(1, 11):
            gray = cv2.imread('./data/faces_dynamic/%s/%d.jpg' % (dir, i))  # 三维图片
            gray_ = gray[:, :, 0]  # 二维数组
            gray_ = cv2.equalizeHist(gray_)  # 图片均衡化处理，颜色鲜明
            faces.append(gray_)
        pass

    faces = np.asarray(faces)
    target = np.asarray(target)
    target.sort()  # 排序
    return faces, target, names


def take_photo(path):
    cap = cv2.VideoCapture(0)
    face_detector = cv2.CascadeClassifier('./data/haarcascade_frontalface_alt.xml')
    filename = 1
    flag_write = False
    while True:
        flag, frame = cap.read()
        if not flag:
            break
        gray = cv2.cvtColor(frame, code=cv2.COLOR_BGR2GRAY)
        faces = face_detector.detectMultiScale(gray, minNeighbors=10)
        for x, y, w, h in faces:
            if flag_write:
                face = gray[y:y + h, x:x + w]
                face = cv2.resize(face, dsize=(64, 64))
                cv2.imwrite('./data/faces_dynamic/%s/%d.jpg' % (path, filename), face)
                filename += 1
            cv2.rectangle(frame, pt1=(x, y), pt2=(x + w, y + h), color=[0, 0, 255], thickness=2)
        if filename > 10:
            break
        cv2.imshow('face', frame)
        key = cv2.waitKey(1000 // 24)
        if key == ord('q'):
            break
        if key == ord('w'):
            flag_write = True
    cv2.destroyAllWindows()
    cap.release()


def take_faces():
    while True:
        key = input('请输入文件夹的名字，姓名拼音缩写。如果输入Q，程序退出！')
        if key == 'Q':
            break
        if key == 27:
            break
        # 在faces_dynamic下面创建子文件夹
        os.makedirs('./data/faces_dynamic/%s' % (key), exist_ok=True)
        take_photo(key)


def dynamic_recognizer_face(face_recognizer, names):
    cap = cv2.VideoCapture(0)
    # 人脸检测
    face_detector = cv2.CascadeClassifier('./data/haarcascade_frontalface_alt.xml')
    while True:
        flag, frame = cap.read()
        if not flag:
            break

        gray = cv2.cvtColor(frame, code=cv2.COLOR_BGR2GRAY)
        faces = face_detector.detectMultiScale(gray, minNeighbors=10)

        for x, y, w, h in faces:
            face = gray[y:y + h, x:x + w]
            face = cv2.resize(face, dsize=(64, 64))
            face = cv2.equalizeHist(face)  # 动态采集的数据，进行均衡化处理
            y_, confidence = face_recognizer.predict(face)  # 人脸辨识
            label = names[y_]
            print('这个人是：%s。置信度是：%0.1f' % (label, confidence))
            cv2.rectangle(frame, pt1=(x, y), pt2=(x + w, y + h), color=[0, 0, 255], thickness=2)
            cv2.putText(frame, text=label,
                        org=(x, y - 10),
                        fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                        fontScale=1.5,
                        color=[0, 0, 255],
                        thickness=2)
            pass
        cv2.imshow('face', frame)
        key = cv2.waitKey(1000 // 24)
        if key == ord('q'):
            break
        if key == 27:
            break
        pass

    cv2.destroyAllWindows()
    cap.release()


def main():
    # 1、动态采集人脸

    # take_faces()
    # 2、加载数据，返回目标值
    faces,target,names = load_data()
    print(faces.shape)

    # 3、加载算法
    # face_recognizer = cv2.face.EigenFaceRecognizer_create()
    # face_recognizer = cv2.face.FisherFaceRecognizer_create()
    face_recognizer = cv2.face.LBPHFaceRecognizer_create()

    # 4、算法训练，找到数据和目标值之间的规律
    face_recognizer.train(faces,target)

    # 5、动态加载数据
    dynamic_recognizer_face(face_recognizer,names)
    pass


if __name__ == '__main__':
    main()
