# coding:utf-8
# Author:mylady
# Datetime:2023/3/26 13:46
import cv2
from ObjectDetection import ObjectDetection


def run():
    """
    注意调用此方法时, data目录下要有权重模型文件.
    """
    od = ObjectDetection(nmsThreshold=0.4, confThreshold=0.5)
    # 读取本地视频
    # cap = cv2.VideoCapture("../data/2023-02-26-大疆mini2se_庙会.mp4")
    # 读取网络摄像头
    cap = cv2.VideoCapture("rtsp://admin:YING123ZZ@192.168.1.2:554/h264/ch1/main/av_stream")
    # cap = cv2.VideoCapture(0)

    while True:
        _, frame = cap.read()
        (class_ids, scores, boxes) = od.detect(frame)
        for cid, score, box in zip(class_ids, scores, boxes):
            (x, y, w, h) = box
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

            # 文本
            cv2.putText(img=frame,
                        text='%s %.2f' % (od.getObjectName(cid), score),  # 显示文本
                        org=(x, y),  # 位置
                        fontFace=cv2.FONT_HERSHEY_TRIPLEX,  # 字体
                        fontScale=1,  # 字号
                        color=[0, 0, 255],  # 颜色
                        thickness=2)  # 线宽
            pass

        cv2.imshow("object detection", frame)
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
