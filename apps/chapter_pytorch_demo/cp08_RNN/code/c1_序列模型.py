# coding:utf-8
# Author:mylady
# Datetime:2023/4/16 20:39
import torch
from torch import nn


import sys
sys.path.append("..")
import d2lzh_pytorch.torch as d2l


def run():
    # 总共产生1000个点
    T = 1000
    time = torch.arange(1, T + 1, dtype=torch.float32)

    x = torch.sin(0.01 * time) + torch.normal(0, 0.2, (T,))

    # 画图
    d2l.plot(time,
             [x],
             'time',
             'x',
             xlim=[1, 1000],
             figsize=(6, 3)
             )

    tau = 4
    features = torch.zeros((T - tau, tau))
    for i in range(tau):
        features[:, i] = x[i: T - tau + i]
        pass

    labels = x[tau:].reshape((-1, 1))

    pass


def main():
    pass


if __name__ == '__main__':
    main()
