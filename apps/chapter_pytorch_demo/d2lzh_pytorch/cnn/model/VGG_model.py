import torch
from torch import nn
import torch.nn.functional as F

import sys
sys.path.append("..")
from d2lzh_pytorch.utils import FlattenLayer


def vgg_block(num_convs, in_channels, out_channels):
    blk = []
    for i in range(num_convs):
        if i == 0:
            blk.append(nn.Conv2d(in_channels,
                                 out_channels,
                                 kernel_size=3,
                                 padding=1))
        else:
            blk.append(nn.Conv2d(out_channels,
                                 out_channels,
                                 kernel_size=3,
                                 padding=1))
        blk.append(nn.ReLU())
        pass

    blk.append(nn.MaxPool2d(kernel_size=2, stride=2))  # 这里会使宽高减半
    return nn.Sequential(*blk)


def vgg(conv_arch, fc_features, fc_hidden_units=4096):
    net = nn.Sequential()

    # 卷积层部分
    for i, (num_convs, in_channels, out_channels) in enumerate(conv_arch):
        # 每经过一个vgg_block都会使宽高减半
        net.add_module("vgg_block_" + str(i + 1),
                       vgg_block(num_convs,
                                 in_channels,
                                 out_channels
                                 ))
        pass

    # 全连接层部分
    net.add_module("fc", nn.Sequential(FlattenLayer(),
                                       nn.Linear(fc_features, fc_hidden_units),
                                       nn.ReLU(),
                                       nn.Dropout(0.5),
                                       nn.Linear(fc_hidden_units,
                                                 fc_hidden_units),
                                       nn.ReLU(),
                                       nn.Dropout(0.5),
                                       nn.Linear(fc_hidden_units, 10)
                                       ))
    return net


def get_VGG_model():
    fc_features = 512 * 7 * 7  # c * w * h
    fc_hidden_units = 4096  # 任意

    ratio = 8
    small_conv_arch = [(1, 1, 64 // ratio),
                       (1, 64 // ratio, 128 // ratio),
                       (2, 128 // ratio, 256 // ratio),
                       (2, 256 // ratio, 512 // ratio),
                       (2, 512 // ratio, 512 // ratio)
                       ]

    net = vgg(small_conv_arch,
              fc_features // ratio,
              fc_hidden_units // ratio
              )
    return net



def main():
    pass
 

if __name__ == "__main__":
    # main()
    pass
