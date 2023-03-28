# coding:utf-8
# Author:mylady
# Datetime:2023/3/27 18:10
import cv2
import numpy as np


# 提取图片中的蓝色区域
def test_1():
    img1 = cv2.imread('./cat.png')  # 颜色空间BGR
    img2 = cv2.cvtColor(img1, code=cv2.COLOR_BGR2HSV)  # 颜色空间的转变

    # 定义在HSV颜色空间中蓝色的范围
    lower_blue = np.array([110, 50, 50])  # 浅蓝色
    upper_blue = np.array([130, 255, 255])  # 深蓝色

    # 根据蓝色的范围，标记图片中哪些位置是蓝色
    # inRange 是否在这个范围内 lower_bule ~ upper_blue:蓝色
    # 如果在那么就是255，不然就是0
    mask = cv2.inRange(img2, lower_blue, upper_blue)
    res = cv2.bitwise_and(img1, img1, mask=mask)  # 位运算：与运算！

    # 展示与销毁
    cv2.imshow('name-hsv', res)
    cv2.waitKey(0)
    cv2.destroyAllWindows()  # 键盘输入，窗口，销毁，释放内存
    pass


def test_2():
    # a = np.array([1,2,3,4,5])
    # b = np.array([1,4,5,7,5])
    # c = np.bitwise_and(a,b)
    # 2 ---->0b010
    # 4 ---->0b100
    # 2&4 -->0b000
    # print(bin(2), bin(4), sep = '\n')
    # print(c)

    # a = np.random.randint(0, 10, size=(20, 20, 3))
    # b = np.random.randint(0,10,size = (20,20,3))
    # mask = cv2.inRange(b,np.array([2,5,5]),np.array([4,10,10]))
    # print(mask.dtype,mask.shape)
    # c = cv2.bitwise_and(a,b,mask = mask)
    # print(c)

    a = np.random.randint(0, 10, size=(3, 3, 3))
    b = np.random.randint(0, 10, size=(3, 3, 3))
    print(a, b, sep='\n')
    c = cv2.bitwise_and(a, b, mask=np.array([[1, 0, 0], [0, 0, 0], [0, 0, 0]], dtype=np.uint8))
    # 9 ----> 1001
    # 7 ----> 0111
    #   ----> 0001 ---->数字 1
    pass


def main():
    test_2()
    pass


if __name__ == '__main__':
    main()
