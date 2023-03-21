# coding:utf-8
# Author:mylady
# Datetime:2023/3/21 9:56
import torchvision
from PIL import Image

import sys

sys.path.append("../..")
# import d2lzh_pytorch as d2l
from apps.chapter import d2lzh_pytorch as d2l


def show_images(imgs, num_rows, num_cols, scale=2):
    figsize = (num_cols * scale, num_rows * scale)
    _, axes = d2l.plt.subplots(num_rows, num_cols, figsize=figsize)
    for i in range(num_rows):
        for j in range(num_cols):
            axes[i][j].imshow(imgs[i * num_cols + j])
            axes[i][j].axes.get_xaxis().set_visible(False)
            axes[i][j].axes.get_yaxis().set_visible(False)
        pass
    return axes


def apply(img, aug, num_rows=2, num_cols=4, scale=1.5):
    Y = [aug(img) for _ in range(num_rows * num_cols)]
    show_images(Y, num_rows, num_cols, scale)


d2l.set_figsize()
img = Image.open('./img/cat1.jpg')

d2l.plt.imshow(img)


def test_1():
    # 实例来实现一半概率的图像水平（左右）翻转
    apply(img, torchvision.transforms.RandomHorizontalFlip())
    pass


def test_2():
    # 上下翻转
    apply(img, torchvision.transforms.RandomVerticalFlip())
    pass


def test_3():
    # 裁剪
    shape_aug = torchvision.transforms.RandomResizedCrop(200,
                                                         scale=(0.1, 1),
                                                         ratio=(0.5, 2)
                                                         )
    apply(img, shape_aug)
    pass


def test_4():
    # 色调变换
    color_aug = torchvision.transforms.ColorJitter(
        brightness=0.5,  # 亮度
        contrast=0.5,  # 对比度
        saturation=0.5,  # 饱和度
        hue=0.5  # 色调
    )

    apply(img, color_aug)
    pass


def test_5():
    # 色调变换
    color_aug = torchvision.transforms.ColorJitter(
        brightness=0.5,  # 亮度
        contrast=0.5,  # 对比度
        saturation=0.5,  # 饱和度
        hue=0.5)  # 色调

    # 裁剪
    shape_aug = torchvision.transforms.RandomResizedCrop(200,
                                                         scale=(0.1, 1),
                                                         ratio=(0.5, 2))
    # 叠加使用
    augs = torchvision.transforms.Compose([
        torchvision.transforms.RandomHorizontalFlip(),
        color_aug,  # 图片颜色变换
        shape_aug  # 图片颜色裁剪
    ])

    apply(img, augs)
    pass


def main():
    test_1()
    pass


if __name__ == '__main__':
    main()
