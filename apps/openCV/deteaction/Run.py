# coding:utf-8
# Author:mylady
# Datetime:2023/3/26 13:46
import cv2
from ObjectDetection import ObjectDetection


def run():
    od = ObjectDetection()
    cap = cv2.VideoCapture("../data/test.mp4")

    while True:
        _, frame = cap.read()
        (class_ids, scores, boxes) = od.detect(frame)
        for box in boxes:
            (x, y, w, h) = box
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            pass

        cv2.imshow("car detection", frame)
        key = cv2.waitKey(1)
        if key == 27:
            break

    cap.release()
    cv2.destroyAllWindows()


def main():
    """
    opnCV实现车辆检测
    """
    run()
    pass


if __name__ == '__main__':
    main()
