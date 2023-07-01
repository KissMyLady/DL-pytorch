import torch
from torch import nn


# LetNet模型封装
def get_LeNet_mode():
    net = nn.Sequential(
        nn.Conv2d(1, 6, kernel_size=5, padding=2), nn.Sigmoid(),
        nn.AvgPool2d(kernel_size=2, stride=2),
        nn.Conv2d(6, 16, kernel_size=5), nn.Sigmoid(),
        nn.AvgPool2d(kernel_size=2, stride=2),
        nn.Flatten(),
        nn.Linear(16 * 5 * 5, 120), nn.Sigmoid(),
        nn.Linear(120, 84), nn.Sigmoid(),
        nn.Linear(84, 10))

    return net

# 加入批量规范化, 用户加速深层网络的收敛速度
def get_LeNet_mode_v2():
    net = nn.Sequential(
        nn.Conv2d(1, 6, kernel_size=5), 
        nn.BatchNorm2d(6), 
        nn.Sigmoid(),
        
        nn.AvgPool2d(kernel_size=2, stride=2),
        nn.Conv2d(6, 16, kernel_size=5), 
        nn.BatchNorm2d(16), 
        nn.Sigmoid(),
        
        nn.AvgPool2d(kernel_size=2, stride=2), nn.Flatten(),
        nn.Linear(256, 120), 
        nn.BatchNorm1d(120), 
        nn.Sigmoid(),
        
        nn.Linear(120, 84), 
        nn.BatchNorm1d(84), 
        nn.Sigmoid(),
        
        nn.Linear(84, 10)
    )
    return net


def main():
    pass
 

if __name__ == "__main__":
    main()
    pass