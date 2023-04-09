# coding:utf-8
# Author:mylady
# Datetime:2023/3/19 14:16
import time
import torch
from torch import nn, optim
import torchvision

import sys
sys.path.append("..")
# import d2lzh_pytorch as d2l
from apps.chapter_pytorch_demo import d2lzh_pytorch as d2l


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class AlexNet(nn.Module):

    def __init__(self):
        super(AlexNet, self).__init__()

        self.conv = nn.Sequential(
            # in_channels, out_channels, kernel_size, stride, padding
            nn.Conv2d(1, 96, 11, 4),
            nn.ReLU(),
            nn.MaxPool2d(3, 2),  # kernel_size, stride

            # 减小卷积窗口，使用填充为2来使得输入与输出的高和宽一致，且增大输出通道数
            nn.Conv2d(96, 256, 5, 1, 2),
            nn.ReLU(),
            nn.MaxPool2d(3, 2),

            # 连续3个卷积层，且使用更小的卷积窗口。除了最后的卷积层外，进一步增大了输出通道数。
            # 前两个卷积层后不使用池化层来减小输入的高和宽
            nn.Conv2d(256, 384, 3, 1, 1),
            nn.ReLU(),
            nn.Conv2d(384, 384, 3, 1, 1),
            nn.ReLU(),
            nn.Conv2d(384, 256, 3, 1, 1),
            nn.ReLU(),
            nn.MaxPool2d(3, 2)
        )

        # 这里全连接层的输出个数比LeNet中的大数倍。使用丢弃层来缓解过拟合
        self.fc = nn.Sequential(
            nn.Linear(256 * 5 * 5, 4096),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(4096, 4096),
            nn.ReLU(),
            nn.Dropout(0.5),
            # 输出层。由于这里使用Fashion-MNIST，所以用类别数为10，而非论文中的1000
            nn.Linear(4096, 10),
        )

    def forward(self, img):
        feature = self.conv(img)
        output = self.fc(feature.view(img.shape[0], -1))
        return output


def main():
    batch_size = 128
    train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size, resize=224)

    net = AlexNet()
    lr = 0.001
    num_epochs = 5

    optimizer = torch.optim.Adam(net.parameters(),
                                 lr=lr
                                 )
    # 训练
    d2l.train_ch5(net, train_iter, test_iter,
                  batch_size, optimizer,
                  device,
                  num_epochs
                  )
    pass


if __name__ == '__main__':
    main()
