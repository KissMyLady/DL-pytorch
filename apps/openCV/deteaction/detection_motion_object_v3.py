# coding:utf-8
# Author:mylady
# Datetime:2023/3/26 13:46
import cv2


def test_1():
    """
    使用KNN背景差分器

    """
    # todo 初始化KNN差分器
    bg_subtractor = cv2.createBackgroundSubtractorKNN(detectShadows=True)

    erode_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 5))
    dilate_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (17, 11))

    # cap = cv2.VideoCapture('traffic.flv')
    cap = cv2.VideoCapture("rtsp://admin:YING123ZZ@192.168.1.2:554/h264/ch1/main/av_stream")
    while True:
        success, frame = cap.read()

        fg_mask = bg_subtractor.apply(frame)
        _, thresh = cv2.threshold(fg_mask, 244, 255, cv2.THRESH_BINARY)  # 阈值图像

        # todo 对图片进行腐蚀和扩张
        cv2.erode(thresh, erode_kernel, thresh, iterations=2)
        cv2.dilate(thresh, dilate_kernel, thresh, iterations=2)

        contours, _ = cv2.findContours(thresh,
                                       cv2.RETR_EXTERNAL,
                                       cv2.CHAIN_APPROX_SIMPLE)

        # for c in contours:
        #     if cv2.contourArea(c) > 1000:
        #         x, y, w, h = cv2.boundingRect(c)
        #         cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 255, 0), 2)
        cv2.imshow('knn', fg_mask)
        cv2.imshow('thresh', thresh)
        cv2.imshow('detection', frame)

        # cv2.waitKey(100)
        if cv2.waitKey(1) == 27:
            break

    pass


def main():
    test_1()
    pass


if __name__ == '__main__':
    test_1()
    pass
