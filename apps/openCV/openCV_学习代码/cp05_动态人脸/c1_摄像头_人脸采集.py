# coding:utf-8
# Author:mylady
# Datetime:2023/3/28 15:55
import cv2


def test_1():
    cap = cv2.VideoCapture(0)
    face_detector = cv2.CascadeClassifier('./data/haarcascade_frontalface_alt.xml')
    filename = 1
    flag_write = False

    while True:
        flag, frame = cap.read()
        if flag is False:
            break

        gray = cv2.cvtColor(frame, code=cv2.COLOR_BGR2GRAY)
        faces = face_detector.detectMultiScale(gray, minNeighbors=10)
        print('----------------', faces)

        for x, y, w, h in faces:
            if flag_write:
                face = gray[y:y + h, x:x + w]  # 获取人脸，保存，30张
                # 调整人脸尺寸，统一尺寸
                face = cv2.resize(face, dsize=(64, 64))
                cv2.imwrite('./lfk/%d.jpg' % (filename), face)
                filename += 1  # 自增
                pass

            cv2.rectangle(frame,
                          pt1=(x, y),
                          pt2=(x + w, y + h),
                          color=[0, 0, 255], thickness=2)
            # 文本
            cv2.putText(img=frame,
                        text='face',  # 显示文本
                        org=(x, y),  # 位置
                        fontFace=cv2.FONT_HERSHEY_TRIPLEX,  # 字体
                        fontScale=1,  # 字号
                        color=[0, 0, 255],  # 颜色
                        thickness=2)  # 线宽
            pass
        if filename > 40:
            break
        cv2.imshow('face', frame)
        key = cv2.waitKey(1000 // 24)
        if key == ord('q'):
            break
        if key == ord('w'):
            flag_write = True  # 表明写入数据


    cv2.destroyAllWindows()
    cap.release()
    pass


def main():
    test_1()
    pass


if __name__ == '__main__':
    main()
