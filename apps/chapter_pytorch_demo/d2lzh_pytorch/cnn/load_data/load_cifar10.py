# coding:utf-8
# Author:mylady
# Datetime: 2023-07-01 23:46:00
import torch
import torchvision
from torchvision import transforms
from torch.utils.data.dataloader import DataLoader
from d2lzh_pytorch.myUtils import Timer
from d2lzh_pytorch.myUtils import get_dataloader_workers
from matplotlib import pyplot as plt


def show_images(imgs, 
                num_rows, num_cols, 
                titles=None, scale=1.5):
    """Plot a list of images.

    Defined in :numref:`sec_fashion_mnist`"""
    figsize = (num_cols * scale, num_rows * scale)
    _, axes = plt.subplots(num_rows, num_cols, figsize=figsize)
    axes = axes.flatten()
    for i, (ax, img) in enumerate(zip(axes, imgs)):
        if torch.is_tensor(img):
            # Tensor Image
            ax.imshow(img.numpy())
        else:
            # PIL Image
            ax.imshow(img)
        ax.axes.get_xaxis().set_visible(False)
        ax.axes.get_yaxis().set_visible(False)
        if titles:
            ax.set_title(titles[i])
    return axes


def load_cifar10(is_train, augs, batch_size):
    """
    CIFAR-10数据集包含60,000张32x32彩色图像，分为10个类，每类6,000张。有50,000张训练图片和10,000张测试图片。
    """
    dataset = torchvision.datasets.CIFAR10(root="/mnt/aiguo/ai_data/Datasets_on_HHD/CIFAR", 
                                           train=is_train,
                                           transform=augs, 
                                           download=False)

    print("use image augmentation technology, dataset data size is: %s" % len(dataset))

    dataloader = torch.utils.data.DataLoader(dataset, 
                                             batch_size=batch_size,
                                             shuffle=is_train, 
                                             num_workers=get_dataloader_workers())
    return dataloader


def apply(img, aug, num_rows=2, num_cols=4, scale=1.5):
    Y = [aug(img) for _ in range(num_rows * num_cols)]
    show_images(Y, num_rows, num_cols, scale=scale)


# 使用了图像增广的数据集
def get_aug_cifar10(batch_size=256):
    timer = Timer()
    timer.start()

    train_augs = torchvision.transforms.Compose(
        [
            # 随机左右翻转
            torchvision.transforms.RandomHorizontalFlip(),
            # 上下翻转
            torchvision.transforms.RandomVerticalFlip(),
            # 随机裁剪
            # torchvision.transforms.RandomResizedCrop(
            #     (200, 200), scale=(0.1, 1), ratio=(0.5, 2)
            # ),
            # 改变颜色-亮度
            # torchvision.transforms.ColorJitter(brightness=0.5, contrast=0, saturation=0, hue=0),
            # 改变颜色-色调
            torchvision.transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.5),
            # 转换为深度学习框架所要求的格式(批量大小,通道数,高度,宽度)32位浮点数
            # 取值范围0~1
            torchvision.transforms.ToTensor()
        ])

    test_augs = torchvision.transforms.Compose(
        [
            torchvision.transforms.ToTensor()
        ])

    train_iter = load_cifar10(True, train_augs, batch_size)
    test_iter = load_cifar10(False, test_augs, batch_size)

    end_times = timer.stop()
    print("load cifar10 sonsume time: %.2f" % end_times)
    return train_iter, test_iter


def main():
    pass


if __name__ == "__main__":
    main()
    pass