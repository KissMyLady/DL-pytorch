# coding:utf-8
import torch
from torch import nn
from torch.nn import init
import numpy as np
import sys
from collections import OrderedDict
import time

# import d2lzh_pytorch.torch as d2l
import sys
sys.path.append("../..")
import d2lzh_pytorch.torch as d2l

print(torch.__version__)


class LinearNet(nn.Module):

    def __init__(self, num_inputs, num_outputs):
        super(LinearNet, self).__init__()
        self.linear = nn.Linear(num_inputs, num_outputs)
        
    def forward(self, x): # x shape: (batch, 1, 28, 28)
        y = self.linear(x.view(x.shape[0], -1))
        return y



class FlattenLayer(nn.Module):
    
    def __init__(self):
        super(FlattenLayer, self).__init__()
        
    def forward(self, x): # x shape: (batch, *, *, ...)
        return x.view(x.shape[0], -1)


def main():
        
    # 加载数据
    batch_size = 256

    train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size)

    # 超参设置
    num_inputs = 784
    num_outputs = 10

    # 加载模型
    net = nn.Sequential(
        OrderedDict([
            ('flatten', FlattenLayer()),
            ('linear', nn.Linear(num_inputs, num_outputs))
        ])
    )

    # 模型随机初始化
    init.normal_(net.linear.weight, mean=0, std=0.01)
    init.constant_(net.linear.bias, val=0) 

    # 损失函数
    loss = nn.CrossEntropyLoss()

    # 优化算法
    optimizer = torch.optim.SGD(net.parameters(), lr=0.1)

    # 训练轮数
    num_epochs = 5

    start = time.time()

    # 开始训练
    d2l.train_ch3(net, train_iter, test_iter, loss, 
                  num_epochs, 
                  optimizer)

    print("训练耗时: ", time.time() - start)
    pass


if __name__ == "__main__":
    main()

