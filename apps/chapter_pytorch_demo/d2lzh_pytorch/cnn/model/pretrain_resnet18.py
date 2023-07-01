# coding:utf-8
# Author:mylady
# Datetime: 2023-07-02 01:11:00
from torchvision import models
from torch import nn


def get_pretrain_resnet18():
    pretrained_net = models.resnet18(pretrained=True)
    # pretrained_net = models.resnet18(weights=True)
    # pretrained_net = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    pretrained_net.fc = nn.Linear(512, 2)
    return pretrained_net


def main():
    pass


if __name__ == "__main__":
    main()
    pass