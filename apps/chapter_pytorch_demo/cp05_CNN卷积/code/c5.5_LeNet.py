# coding:utf-8
# Author:mylady
# Datetime:2023/3/19 13:17
import time
import torch
from torch import nn, optim

import sys

sys.path.append("..")
# import d2lzh_pytorch as d2l
from apps.chapter_pytorch_demo import d2lzh_pytorch as d2l

# 获取设备
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 获取训练数据
batch_size = 256
train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size=batch_size, root="K:\code_big")


class LeNet(nn.Module):

    def __init__(self):
        super(LeNet, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 6, 5),  # in_channels, out_channels, kernel_size
            nn.Sigmoid(),
            nn.MaxPool2d(2, 2),  # kernel_size, stride
            nn.Conv2d(6, 16, 5),
            nn.Sigmoid(),
            nn.MaxPool2d(2, 2)
        )
        self.fc = nn.Sequential(
            nn.Linear(16 * 4 * 4, 120),
            nn.Sigmoid(),
            nn.Linear(120, 84),
            nn.Sigmoid(),
            nn.Linear(84, 10)
        )

    def forward(self, img):
        feature = self.conv(img)
        output = self.fc(feature.view(img.shape[0], -1))
        return output


def evaluate_accuracy(data_iter, net, device=None):
    # 如果没指定device就使用net的device
    if device is None and isinstance(net, torch.nn.Module):
        device = list(net.parameters())[0].device

    acc_sum = 0.0
    n = 0

    with torch.no_grad():
        for X, y in data_iter:
            if isinstance(net, torch.nn.Module):
                # 评估模式, 这会关闭dropout
                net.eval()

                y_hat = net(X.to(device)).argmax(dim=1) == y.to(device)
                acc_sum += (y_hat).float().sum().cpu().item()

                # 改回训练模式
                net.train()
            # 自定义的模型, 3.13节之后不会用到, 不考虑GPU
            else:
                # 如果有is_training这个参数
                if ('is_training' in net.__code__.co_varnames):
                    # 将is_training设置成False
                    res_1 = net(X, is_training=False).argmax(dim=1) == y
                    acc_sum += (res_1).float().sum().item()
                else:
                    res_1 = net(X).argmax(dim=1) == y
                    acc_sum += (res_1).float().sum().item()
                pass
            n += y.shape[0]
            pass
        pass
    return acc_sum / n


def train_ch5(net, train_iter, test_iter, batch_size, optimizer, device, num_epochs):
    net = net.to(device)
    print("training on ", device)

    loss = torch.nn.CrossEntropyLoss()

    for epoch in range(num_epochs):

        train_l_sum = 0.0
        train_acc_sum = 0.0
        n = 0
        batch_count = 0
        start = time.time()

        for X, y in train_iter:
            X = X.to(device)
            y = y.to(device)
            y_hat = net(X)

            l = loss(y_hat, y)
            optimizer.zero_grad()
            l.backward()
            optimizer.step()

            train_l_sum += l.cpu().item()

            train_acc_sum += (y_hat.argmax(dim=1) == y).sum().cpu().item()
            n += y.shape[0]
            batch_count += 1
            pass
        test_acc = evaluate_accuracy(test_iter, net)

        epoch_count = epoch + 1  # 训练批次
        loss_cp = train_l_sum / batch_count  # 损失计算
        train_acc = train_acc_sum / n  # 正确率
        time_consume = time.time() - start  # 耗时

        print('epoch %d, loss %.4f, train acc %.3f, test acc %.3f, time_ %.1f sec'
              % (epoch_count, loss_cp, train_acc, test_acc, time_consume)
              )


def run_1():
    lr = 0.001
    num_epochs = 10

    # 模型构建
    net = LeNet()

    optimizer = torch.optim.Adam(net.parameters(),
                                 lr=lr
                                 )

    # 训练
    train_ch5(net, train_iter, test_iter,
              batch_size, optimizer,
              device,
              num_epochs
              )
    pass


def main():
    run_1()
    pass


if __name__ == '__main__':
    main()
