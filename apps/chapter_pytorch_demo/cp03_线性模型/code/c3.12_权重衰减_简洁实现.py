# coding:utf-8
# Author:mylady
# Datetime:2023/3/15 9:35
import torch
import numpy as np
import torch.nn as nn

import sys
sys.path.append("../..")
from apps.chapter import d2lzh_pytorch as d2l


n_train = 20
n_test = 100
num_inputs = 200

# 生成数据
true_w = torch.ones(num_inputs, 1) * 0.01
true_b = 0.05

features = torch.randn((120, 200))

labels = torch.matmul(features, true_w) + true_b
labels += torch.tensor(np.random.normal(0, 0.01, size=labels.size()),
                       dtype=torch.float)

train_features = features[:20, :]
test_features  = features[20:, :]

train_labels = labels[:20]
test_labels  = labels[20:]


# 加载数据集
batch_size = 1
num_epochs = 100
lr = 0.003

dataset = torch.utils.data.TensorDataset(train_features, train_labels)
train_iter = torch.utils.data.DataLoader(dataset, batch_size, shuffle=True)

# 定义网络, 损失函数
net = d2l.linreg
loss = d2l.squared_loss


def fit_and_plot_pytorch(wd):
    # 对权重参数衰减。权重名称一般是以weight结尾
    net = nn.Linear(num_inputs, 1)
    nn.init.normal_(net.weight, mean=0, std=1)
    nn.init.normal_(net.bias, mean=0, std=1)

    # 对权重参数衰减
    optimizer_w = torch.optim.SGD(params=[net.weight],
                                  lr=lr,
                                  weight_decay=wd)

    # 不对偏差参数衰减
    optimizer_b = torch.optim.SGD(params=[net.bias],
                                  lr=lr)

    train_ls = []
    test_ls = []
    for _ in range(num_epochs):
        for X, y in train_iter:
            l = loss(net(X), y).mean()
            optimizer_w.zero_grad()
            optimizer_b.zero_grad()

            l.backward()

            # 对两个optimizer实例分别调用step函数，从而分别更新权重和偏差
            optimizer_w.step()
            optimizer_b.step()
            pass

        # 计算 y_hat
        train_y_hat = net(train_features)
        test_y_hat = net(test_features)

        # 计算损失
        res_train = loss(train_y_hat, train_labels).mean().item()
        res_test = loss(test_y_hat, test_labels).mean().item()

        train_ls.append(res_train)
        test_ls.append(res_test)
        pass

    d2l.semilogy(range(1, num_epochs + 1), train_ls, 'epochs', 'loss',
                 range(1, num_epochs + 1), test_ls, ['train', 'test']
                 )
    print('L2 norm of w:', net.weight.data.norm().item())


def main():
    # 传入不同的we, 观察对过拟合的影响
    fit_and_plot_pytorch(0)
    fit_and_plot_pytorch(3)
    pass


if __name__ == '__main__':
    main()
