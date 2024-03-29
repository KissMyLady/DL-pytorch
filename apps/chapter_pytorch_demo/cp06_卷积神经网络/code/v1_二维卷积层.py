# coding:utf-8
# Author:mylady
# Datetime:2023/3/17 14:17
import torch
from torch import nn

X = torch.tensor([[0, 1, 2],
                  [3, 4, 5],
                  [6, 7, 8]])

K = torch.tensor([[0, 1],
                  [2, 3]])


def corr2d(X, K):
    """
    输入数组X, 卷积核K, 返回输出
    """
    h, w = K.shape
    Y = torch.zeros((X.shape[0] - h + 1, X.shape[1] - w + 1))
    for i in range(Y.shape[0]):
        for j in range(Y.shape[1]):
            Y[i, j] = (X[i: i + h, j: j + w] * K).sum()
    return Y


# 二维卷积层
class Conv2D(nn.Module):

    def __init__(self, kernel_size):
        super(Conv2D, self).__init__()
        self.weight = nn.Parameter(torch.randn(kernel_size))
        self.bias = nn.Parameter(torch.randn(1))

    def forward(self, x):
        return corr2d(x, self.weight) + self.bias


# 二维互相关运算
def test_1():
    y = corr2d(X, K)
    print(y)
    pass


# 边缘检测
def test_2():
    X = torch.ones(6, 8)

    X[:, 2:6] = 0
    print("当前边缘模型是: ")
    print(X)

    # 卷积核
    K = torch.tensor([[1, -1]])

    # 互相关运算
    Y = corr2d(X, K)
    print("\n 边缘检测计算结果: ")
    print(Y)
    pass


# 卷积核的学习
def test_3():
    # 构造一个核数组形状是(1, 2)的二维卷积层
    conv2d = Conv2D(kernel_size=(1, 2))
    X = torch.ones(6, 8)
    X[:, 2:6] = 0

    K = torch.tensor([[1, -1]])
    Y = corr2d(X, K)

    step = 20
    lr = 0.01

    for i in range(step):
        Y_hat = conv2d(X)
        l = ((Y_hat - Y) ** 2).sum()
        l.backward()

        # 梯度下降
        conv2d.weight.data -= lr * conv2d.weight.grad
        conv2d.bias.data -= lr * conv2d.bias.grad

        # 梯度清0
        conv2d.weight.grad.fill_(0)
        conv2d.bias.grad.fill_(0)

        if (i + 1) % 5 == 0:
            print('Step %d, loss %.3f' % (i + 1, l.item()))

    print("weight: ", conv2d.weight.data)
    print("bias: ", conv2d.bias.data)
    pass


def main():
    test_1()  # 二维互相关运算
    test_2()  # 边缘检测
    test_3()  # 卷积核的学习
    pass


if __name__ == '__main__':
    main()
