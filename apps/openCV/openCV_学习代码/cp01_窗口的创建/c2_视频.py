# coding:utf-8
# Author:mylady
# Datetime:2023/3/27 13:40
import cv2


# 读取视频
def test_1():
    # 打开视频文件
    vc = cv2.VideoCapture('./穿越火线女团活动宣传CG完整版.mp4')

    # 打开摄像头
    # vc = cv2.VideoCapture(0)

    # 检查是否正确打开
    if vc.isOpened() is False:
        open = False
        pass

    # 读取视频的一帧.
    open, frame = vc.read()
    while True:
        # 可以读到内容ret返回True
        ret, frame = vc.read()
        # 读到最后frame就是空
        if frame is None:
            break
        if ret == True:
            cv2.imshow('result', frame)
            # 0xFF == 27表示按esc退出键会退出
            if cv2.waitKey(33) & 0xFF == 27:
                break
        pass

    vc.release()
    cv2.destroyAllWindows()
    pass


# 读取摄像头
def test_2():
    # 创建窗口
    cv2.namedWindow('video', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('video', 640, 480)

    # 获取视频设备
    cap = cv2.VideoCapture(0)

    while True:
        # 从摄像头读取视频
        ret, frame = cap.read()

        # 将视频帧放在窗口中显示
        cv2.imshow('video', frame)

        # 等待键盘事件, 如果为q,退出
        key = cv2.waitKey(1)
        if key & 0xFF == ord('q'):
            break

    # 释放
    cap.release()
    cv2.destroyAllWindows()
    pass


# 视频录制
def test_3():
    cap = cv2.VideoCapture(0)

    # *mp4v就是解包操作 等同于  'm', 'p', '4', 'v'
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')

    # (640, 480)表示摄像头拍视频, 这个大小搞错了也不行.
    # 主要是这个分辨率.
    vw = cv2.VideoWriter('output.mp4', fourcc, 20, (640, 480))

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            print('can not recive frame, Exiting...')
            break
        vw.write(frame)
        cv2.imshow('frame', frame)

        if cv2.waitKey(1) == ord('q'):
            break

    cap.release()
    # 释放VideoWriter
    vw.release()
    cv2.destroyAllWindows()
    pass


def main():
    # test_1()  # 读取视频
    # test_2()  # 读取摄像头
    test_3()  # 视频录制
    pass


if __name__ == '__main__':
    main()
