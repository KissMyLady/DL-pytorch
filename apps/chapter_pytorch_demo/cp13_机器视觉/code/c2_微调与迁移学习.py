# coding:utf-8
# Author:mylady
# Datetime:2023/3/21 10:13
import torch
from torch import nn, optim
from torch_package.utils.data import Dataset, DataLoader
import torchvision
from torchvision.datasets import ImageFolder
from torchvision import transforms
from torchvision import models
import os

import sys

sys.path.append("../..")
# import d2lzh_pytorch as d2l
from apps.chapter_pytorch_demo import d2lzh_pytorch as d2l

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 路径, 数据加载
data_dir = '/home/mylady/Datasets'
os.listdir(os.path.join(data_dir, "hotdog"))  # ['train', 'test']

hotdog_train_path = os.path.join(data_dir, 'hotdog/train')
hotdog_test_path = os.path.join(data_dir, 'hotdog/test')

train_imgs = ImageFolder(hotdog_train_path)
test_imgs = ImageFolder(hotdog_test_path)


def load_data(batch_size):
    # 指定RGB三个通道的均值和方差来将图像通道归一化
    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )

    train_augs = transforms.Compose([
        transforms.RandomResizedCrop(size=224),  # 把给定的图片resize
        transforms.RandomHorizontalFlip(),  # 以0.5的概率水平翻转给定的PIL图像
        transforms.ToTensor(),
        normalize
    ])

    test_augs = transforms.Compose([
        transforms.Resize(size=256),  # 把给定的图片resize
        transforms.CenterCrop(size=224),  # 在图片的中间区域进行裁剪
        transforms.ToTensor(),
        normalize  # 用均值和标准差归一化张量图像
    ])

    train_iter = DataLoader(ImageFolder(hotdog_train_path, transform=train_augs),
                            batch_size,
                            shuffle=True)

    test_iter = DataLoader(ImageFolder(hotdog_test_path, transform=test_augs),
                           batch_size)
    return train_iter, test_iter


def train_fine_tuning(net, optimizer, batch_size=128, num_epochs=5):
    train_iter, test_iter = load_data(batch_size)

    # 损失函数
    loss = torch.nn.CrossEntropyLoss()

    # 训练
    d2l.train(train_iter, test_iter,
              net, loss, optimizer,
              device, num_epochs)


def run():
    # 初始化模型
    pretrained_net = models.resnet18(weights=True)  # Net模型继承
    pretrained_net.fc = nn.Linear(512, 2)  # 改为输出为2

    output_params = list(map(id, pretrained_net.fc.parameters()))
    feature_params = filter(lambda p: id(p) not in output_params, pretrained_net.parameters())

    lr = 0.01
    optimizer = optim.SGD([{'params': feature_params},
                           {'params': pretrained_net.fc.parameters(), 'lr': lr * 10}],
                          lr=lr,
                          weight_decay=0.001)
    # 开始训练
    train_fine_tuning(pretrained_net,
                      optimizer
                      )
    pass


if __name__ == '__main__':
    # run()
    pass
