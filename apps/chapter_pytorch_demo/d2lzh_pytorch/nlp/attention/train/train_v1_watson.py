# coding:utf-8
# Author:mylady
# 2023/7/28 9:42
import sys

sys.path.append("..")
sys.path.append("../..")
sys.path.append("../../..")
sys.path.append("../../../..")

from d2lzh_pytorch import myPolt
from d2lzh_pytorch.torch_package import plot
from matplotlib import pyplot as plt

import torch
from torch import nn

from d2lzh_pytorch.nlp.attention.model.att_net_v1_watson import get_attention


def plot_kernel_reg(y_hat, x_test, y_truth, x_train, y_train):
    plot(X=x_test,
         Y=[y_truth, y_hat],
         xlabel='x',
         ylabel='y',
         legend=['Truth', 'Pred'],
         xlim=[0, 5],
         ylim=[-1, 5])

    plt.plot(x_train,
             y_train,
             'o',
             alpha=0.5)
    pass


def train_net(net, x_train, y_train, keys, values):
    loss = nn.MSELoss(reduction='none')
    trainer = torch.optim.SGD(net.parameters(), lr=0.5)
    animator = myPolt.Animator(xlabel='epoch',
                               ylabel='loss',
                               xlim=[1, 5])
    for epoch in range(5):
        trainer.zero_grad()
        l = loss(net(x_train, keys, values), y_train)
        l.sum().backward()
        trainer.step()
        print(f'epoch {epoch + 1}, loss {float(l.sum()):.6f}')
        animator.add(epoch + 1, float(l.sum()))

    print("train is end << ")
    return net


def predict_attention(net, x_train, y_train, y_truth, n_test, x_test):
    # keys的形状:(n_test，n_train)，每一行包含着相同的训练输入（例如，相同的键）
    keys = x_train.repeat((n_test, 1))
    # value的形状:(n_test，n_train)
    values = y_train.repeat((n_test, 1))

    # 计算
    y_hat = net(x_test, keys, values).unsqueeze(1).detach()

    # 绘制
    plot_kernel_reg(y_hat, x_test, y_truth, x_train, y_train)
    pass


def f(x):
    return 2 * torch.sin(x) + x ** 0.8


def train_c10():
    # 样本数据
    n_train = 50
    x_train, _ = torch.sort(torch.rand(n_train) * 5)
    y_train = f(x_train) + torch.normal(0.0, 0.5, (n_train,))

    # 测试数据
    x_test = torch.arange(0, 5, 0.1)
    y_truth = f(x_test)
    n_test = len(x_test)

    # 在所有样本维度上重复 n 次，以便与 train数据对应。
    X_tile = x_train.repeat((n_train, 1))
    Y_tile = y_train.repeat((n_train, 1))

    keys = X_tile[(1 - torch.eye(n_train)).type(torch.bool)].reshape((n_train, -1))
    values = Y_tile[(1 - torch.eye(n_train)).type(torch.bool)].reshape((n_train, -1))

    # 网络
    net = get_attention()

    print("开始训练 >> ")
    train_net(net, x_train, y_train, keys, values)

    print("开始预测 >> ")
    predict_attention(net, x_train, y_train, y_truth, n_test, x_test)
    pass


def main():
    # 训练 核函数回归 注意力
    train_c10()
    pass


if __name__ == "__main__":
    main()
