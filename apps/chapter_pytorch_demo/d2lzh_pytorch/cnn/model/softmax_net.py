import torch
from torch import nn


def Linear_net():
    net = nn.Sequential(
            nn.Flatten(), 
            nn.Linear(784, 10)
          )
    return net


def init_weights(m):
    if type(m) == nn.Linear:
        nn.init.normal_(m.weight, std=0.01)


# 返回一个简单的softmax分类模型
def get_Linear_net():
    net = Linear_net()
    # 初始化参数
    net.apply(init_weights)
    return net


def main():
    pass


if __name__ == "__main__":
    main()
    pass