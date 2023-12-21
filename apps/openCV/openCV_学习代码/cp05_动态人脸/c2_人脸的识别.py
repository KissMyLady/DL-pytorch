# coding:utf-8
# Author:mylady
# Datetime:2023/3/28 15:59
import cv2
import os
import numpy as np


def load_data():
    listdir = os.listdir('./data/faces')
    # names = []
    # for d in listdir:
    #     if d.startswith('.'):
    #         pass
    #     else:
    #         names.append(d)
    # print(names)
    # print('-----------','.xxx'.startswith('.'))

    # 列表生成式
    names = [d for d in listdir if not d.startswith('.')]
    faces = []
    target = [i for i in range(len(names))] * 30

    for dir in names:
        for i in range(1, 31):
            gray = cv2.imread('./data/faces/%s/%d.jpg' % (dir, i))  # 三维图片
            gray_ = gray[:, :, 0]  # 二维数组
            gray_ = cv2.resize(gray_, dsize=(64, 64))
            faces.append(gray_)
        pass

    faces = np.asarray(faces)
    target = np.asarray(target)
    target.sort()  # 排序
    return faces, target, names


def split_data(faces, target):
    index = np.arange(870)
    np.random.shuffle(index)  # 洗牌
    faces = faces[index]
    target = target[index]

    # X_train训练，X_test测试
    X_train, X_test = faces[:700], faces[700:]

    # 目标值，y_test是测试数据X_test真实值
    y_train, y_test = target[:700], target[700:]
    return X_train, X_test, y_train, y_test


def test_1():
    # 1、加载数据，返回目标值
    faces, target, names = load_data()
    # 2、数据拆分
    X_train, X_test, y_train, y_test = split_data(faces, target)
    # 3、加载算法
    face_recognizer = cv2.face.EigenFaceRecognizer_create()
    # 4、算法训练，找到数据和目标值之间的规律
    face_recognizer.train(X_train, y_train)
    # 5、使用算法进行预测
    for face in X_test:
        # 返回的预测值是数字
        # 返回两个，第一个是类别，第二个是置信度（距离），越低越好！
        y_, confidence = face_recognizer.predict(face)
        name = names[y_]
        print('y_: %s 这个人是：-------- %s, \t %s' % (y_, name, confidence))
        cv2.imshow('face', face)

        key = cv2.waitKey(0)
        if key == ord('q'):
            break
        if key == 27:
            break
        pass
    cv2.destroyAllWindows()
    pass


def main():
    test_1()
    pass


if __name__ == '__main__':
    main()
