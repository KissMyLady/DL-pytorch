# coding:utf-8
# Author:mylady
# Datetime:2023/7/01 17:01
import sys
sys.path.append("..") 

import torch
from torch import nn
import torch.nn.functional as F


class GlobalAvgPool2d(nn.Module):
    # 全局平均池化层可通过将池化窗口形状设置成输入的高和宽实现
    def __init__(self):
        super(GlobalAvgPool2d, self).__init__()

    def forward(self, x):
        return F.avg_pool2d(x, kernel_size=x.size()[2:])


class FlattenLayer(torch.nn.Module):
    def __init__(self):
        super(FlattenLayer, self).__init__()

    def forward(self, x):  # x shape: (batch, *, *, ...)
        return x.view(x.shape[0], -1)


def conv_block(in_channels, out_channels):
    blk = nn.Sequential(nn.BatchNorm2d(in_channels), 
                        nn.ReLU(),
                        nn.Conv2d(in_channels, 
                                  out_channels, 
                                  kernel_size=3, 
                                  padding=1)
                        )
    return blk


# Dense块
class DenseBlock(nn.Module):
    def __init__(self, num_convs, in_channels, out_channels):
        super(DenseBlock, self).__init__()
        net = []
        for i in range(num_convs):
            in_c = in_channels + i * out_channels
            net.append(conv_block(in_c, out_channels))
            pass
        self.net = nn.ModuleList(net)
        # 计算输出通道数
        self.out_channels = in_channels + num_convs * out_channels 

    def forward(self, X):
        for blk in self.net:
            Y = blk(X)
            # 在通道维上将输入和输出连结
            X = torch.cat((X, Y), dim=1)
        return X


# 过渡层
def transition_block(in_channels, out_channels):
    blk = nn.Sequential(nn.BatchNorm2d(in_channels), 
                        nn.ReLU(),
                        nn.Conv2d(in_channels, out_channels, kernel_size=1),
                        nn.AvgPool2d(kernel_size=2, stride=2)
                       )
    
    return blk


def get_DenseNet():
    net = nn.Sequential(nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3),
                        nn.BatchNorm2d(64), 
                        nn.ReLU(),
                        nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
                        )

    num_channels, growth_rate = 64, 32  # num_channels为当前的通道数
    num_convs_in_dense_blocks = [4, 4, 4, 4]

    for i, num_convs in enumerate(num_convs_in_dense_blocks):
        DB = DenseBlock(num_convs, num_channels, growth_rate)
        net.add_module("DenseBlosk_%d" % i, DB)
        
        # 上一个稠密块的输出通道数
        num_channels = DB.out_channels
        
        # 在稠密块之间加入通道数减半的过渡层
        if i != len(num_convs_in_dense_blocks) - 1:
            net.add_module("transition_block_%d" % i, transition_block(num_channels, num_channels // 2))
            num_channels = num_channels // 2
        pass

    net.add_module("BN", nn.BatchNorm2d(num_channels))
    net.add_module("relu", nn.ReLU())
    # GlobalAvgPool2d的输出: (Batch, num_channels, 1, 1)
    net.add_module("global_avg_pool", GlobalAvgPool2d()) 
    net.add_module("fc", nn.Sequential(FlattenLayer(), nn.Linear(num_channels, 10))) 
    return net


def test_1():
    pass


def main():
    pass


if __name__ == "__main__":
    # main()
    pass