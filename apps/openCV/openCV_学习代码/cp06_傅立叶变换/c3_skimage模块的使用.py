from skimage import io  # 导入io模块，以读取目标路径下的图片


"""
skimage即是Scikit-Image,是基于python脚本语言开发的数字图片处理包。 skimage包的全称是scikit-image SciKit (toolkit for SciPy) ，
它对scipy.ndimage进行了扩展，提供了更多的图片处理功能。它是由python语言编写的，由scipy 社区开发和维护。
skimage包由许多的子模块组成，各个子模块提供不同的功能。
"""


def test_1():
    img = io.imread('./data/moon.png')  # 读取dog.jpg文件
    print(type(img))  # 显示类型
    print(img.shape)  # 显示尺寸

    print(img.shape[0])  # 显示高度
    print(img.shape[1])  # 显示宽度
    #print(img.shape[2])  # 显示图片通道数

    print(img.size)  # 显示总像素数
    print(img.max())  # 显示最大像素值
    print(img.min())  # 显示最小像素值
    print(img.mean())  # 像素平均值

    # print(img[0][0])  # 指定像素点的像素值
    # io.imshow(img)  # io模块下显示图像
    io.show()  # 显示图像

    pass


def main():
    test_1()
    pass


if __name__ == '__main__':
    main()
