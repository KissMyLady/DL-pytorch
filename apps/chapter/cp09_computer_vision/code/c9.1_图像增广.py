# coding:utf-8
# Author:mylady
# Datetime:2023/3/21 8:42
import time
import torch
from torch.utils.data import DataLoader
import torchvision

import sys

sys.path.append("../..")
# import d2lzh_pytorch as d2l
from apps.chapter import d2lzh_pytorch as d2l

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def load_cifar10(is_train, augs, batch_size, root="~/Datasets/CIFAR"):
    dataset = torchvision.datasets.CIFAR10(root=root,
                                           train=is_train,
                                           transform=augs,
                                           download=True
                                           )
    num_workers = 0 if sys.platform.startswith('win32') else 4
    return DataLoader(dataset, batch_size=batch_size, shuffle=is_train, num_workers=num_workers)


def train(train_iter, test_iter, net, loss, optimizer, device, num_epochs):
    net = net.to(device)
    print("training on ", device)
    batch_count = 0

    for epoch in range(num_epochs):

        train_l_sum = 0.0
        train_acc_sum = 0.0
        n = 0
        start = time.time()

        for X, y in train_iter:
            X = X.to(device)
            y = y.to(device)
            y_hat = net(X)

            l = loss(y_hat, y)
            optimizer.zero_grad()
            l.backward()
            optimizer.step()

            train_l_sum += l.cpu().item()
            train_acc_sum += (y_hat.argmax(dim=1) == y).sum().cpu().item()

            n += y.shape[0]
            batch_count += 1
            pass

        test_acc = d2l.evaluate_accuracy(test_iter, net)
        print('epoch %d, loss %.4f, train acc %.3f, test acc %.3f, time %.1f sec'
              % (epoch + 1, train_l_sum / batch_count, train_acc_sum / n, test_acc, time.time() - start))
        pass


def train_with_data_aug(lr=0.001):
    batch_size = 256
    net = d2l.resnet18(10)
    num_epochs = 5

    optimizer = torch.optim.Adam(net.parameters(), lr=lr)
    loss = torch.nn.CrossEntropyLoss()

    # 图像增广配置项
    flip_aug = torchvision.transforms.Compose([
        torchvision.transforms.RandomHorizontalFlip(),
        torchvision.transforms.ToTensor()
    ])
    no_aug = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor()
    ])

    # 加载训练数据
    train_iter = load_cifar10(is_train=True, augs=flip_aug, batch_size=batch_size)
    test_iter = load_cifar10(is_train=False, augs=no_aug, batch_size=batch_size)

    # 训练
    train(train_iter, test_iter,
          net,
          loss,
          optimizer,
          device,
          num_epochs)


def run():
    # 训练
    train_with_data_aug()
    pass


if __name__ == '__main__':
    run()
    pass
