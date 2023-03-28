# coding:utf-8
# Author:mylady
# Datetime:2023/3/28 20:44
import numpy as np
import cv2


def test_1():
    # import pytesseract
    # pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
    # 1、加载图片
    image = cv2.imread('./data/Car Images/9.jpg')
    # 灰度化处理
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # 去除噪声
    gray = cv2.bilateralFilter(gray, 0, 17, 17)
    # 发现边界
    edged = cv2.Canny(gray, 170, 200)
    # 2、找到轮廓，根据轮廓面积，排序，提取面积最大的30个
    cnts, new = cv2.findContours(edged, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    cnts = sorted(cnts, key=cv2.contourArea, reverse=True)[:30]

    # 3、循环找到最有可能是车牌的轮廓
    NumberPlateCnt = None
    for c in cnts:
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.02 * peri, True)
        if len(approx) == 4:  # 选择4个角的轮廓，最有可能是车牌
            NumberPlateCnt = approx  # 这就是车牌索引
            break
        pass

    # 4、绘制轮廓
    print(NumberPlateCnt)
    cv2.drawContours(image, [NumberPlateCnt], -1, (0, 255, 0), 3)
    cv2.imshow("result", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    pass


def main():
    test_1()
    pass


if __name__ == '__main__':
    main()
