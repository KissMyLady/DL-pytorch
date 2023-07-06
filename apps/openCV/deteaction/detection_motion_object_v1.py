# coding:utf-8
# Author:mylady
# Datetime:2023/3/26 13:46
import cv2


def test_1():
    """
    参考来源,知乎: https://zhuanlan.zhihu.com/p/512267368
    运动物体检测
    """
    BLUR_RADIUS = 21
    erode_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    dilate_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))

    # 捕捉摄像头的帧
    # cap = cv2.VideoCapture(0)
    cap = cv2.VideoCapture("rtsp://admin:YING123ZZ@192.168.1.2:554/h264/ch1/main/av_stream")
    # 设置视频分辨率
    #cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    #cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    for i in range(10):
        success, frame = cap.read()
        if not success:
            exit(1)
    gray_background = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray_background = cv2.GaussianBlur(gray_background, (BLUR_RADIUS, BLUR_RADIUS), 0)

    while True:
        success, frame = cap.read()
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray_frame = cv2.GaussianBlur(gray_frame, (BLUR_RADIUS, BLUR_RADIUS), 0)

        # 计算当前帧与背景图像之间的差异图像
        diff = cv2.absdiff(gray_background, gray_frame)

        # 使用阈值将差异图像转换为二值图像
        _, thresh = cv2.threshold(diff, 40, 255, cv2.THRESH_BINARY)

        # 对图片进行腐蚀和扩张
        cv2.erode(thresh, erode_kernel, thresh, iterations=2)
        cv2.dilate(thresh, dilate_kernel, thresh, iterations=2)

        contours, _ = cv2.findContours(thresh, # 脱粒
                                       cv2.RETR_EXTERNAL,  # 检索外部
                                       cv2.CHAIN_APPROX_SIMPLE)  # 链_近似_简单

        # 这段代码是在对检测到的轮廓进行处理和绘制矩形框
        #for c in contours:
        #    if cv2.contourArea(c) > 5000:
        #        x, y, w, h = cv2.boundingRect(c)
        #        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 255, 0), 2)

        # 输出检测图像，按Esc键退出
        cv2.imshow('diff', diff)
        cv2.imshow('thresh', thresh)
        cv2.imshow('detection', frame)  # 矩形画框
        if cv2.waitKey(1) == 27:
            break
    pass


def main():
    test_1()
    pass


if __name__ == '__main__':
    test_1()
    pass
