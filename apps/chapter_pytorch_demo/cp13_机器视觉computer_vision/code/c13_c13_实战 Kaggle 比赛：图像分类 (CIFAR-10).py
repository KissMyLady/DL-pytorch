# coding:utf-8
# Author:mylady
# Datetime:2023/3/22 20:10
import collections
import math
import os
import shutil
import torch
import torchvision
from torch import nn
from torchvision import models

from d2l import torch as d2l


def loadData():
    """
    数据集从网络下载
    """
    # 加载 kaggle_cifar10_tiny.zip 数据集   32x32像素, 每个图片 3kb大小
    d2l.DATA_HUB['cifar10_tiny'] = (d2l.DATA_URL + 'kaggle_cifar10_tiny.zip',
                                    '2068874e4b9a9f0fb07ebe0ad2b29754449ccacd')
    # 如果使用完整的Kaggle竞赛的数据集，设置demo为False
    demo = True

    if demo:
        data_dir = d2l.download_extract('cifar10_tiny')
    else:
        data_dir = '../data/cifar-10/'
    return data_dir


def read_csv_labels(fname) -> dict:
    """
    读取fname来给标签字典返回一个文件名
    """
    with open(fname, 'r') as f:
        # 跳过文件头行(列名)
        lines = f.readlines()[1:]

    tokens = [l.rstrip().split(',') for l in lines]
    return dict(((name, label) for name, label in tokens))


def copyfile(filename, target_dir):
    """
    filename:   源文件路径
    target_dir: 目标文件或目录路径
    将文件复制到目标目录
    """
    os.makedirs(target_dir, exist_ok=True)
    shutil.copy(filename, target_dir)


def reorg_train_valid(data_dir, labels, valid_ratio):
    """
    将验证集从原始的训练集中拆分出来
    """

    # 训练数据集中样本最少的类别中的样本数 dog类别
    n = collections.Counter(labels.values()).most_common()[-1][1]
    # print("n: ", n)  # 85
    # print("valid_ratio: ", valid_ratio)  # 0.1

    # 验证集中每个类别的样本数
    n_valid_per_label = max(1, math.floor(n * valid_ratio))  # 8

    label_count = dict()

    # 文件列表: ['1.png', .... , '999.png', '1000.png']
    train_file_list = os.listdir(os.path.join(data_dir, 'train'))
    for train_file in train_file_list:

        label = labels[train_file.split('.')[0]]
        fname = os.path.join(data_dir, 'train', train_file)

        target_path_v1 = os.path.join(data_dir, 'train_valid_test', 'train_valid', label)
        copyfile(fname, target_path_v1)

        if (label not in label_count) or (label_count[label] < n_valid_per_label):
            # 保证 train_valid_test/valid/doc , /bird .. 每个文件有8张图片
            target_path_v2 = os.path.join(data_dir, 'train_valid_test', 'valid', label)
            copyfile(fname, target_path_v2)

            label_count[label] = label_count.get(label, 0) + 1
        else:
            # 其余都是训练数据
            target_path_v3 = os.path.join(data_dir, 'train_valid_test', 'train', label)
            copyfile(fname, target_path_v3)
            pass

    return n_valid_per_label


# 重组 测试集
def reorg_test(data_dir):
    """
    在预测期间整理测试集，以方便读取
    """
    for test_file in os.listdir(os.path.join(data_dir, 'test')):
        copyfile(os.path.join(data_dir, 'test', test_file),
                 os.path.join(data_dir, 'train_valid_test', 'test', 'unknown'))
    pass


def reorg_cifar10_data(data_dir, valid_ratio):
    """
    data_dir 路径
    valid_ratio 校验数据比例
    """
    labels = read_csv_labels(os.path.join(data_dir, 'trainLabels.csv'))  # 返回标签字典

    reorg_train_valid(data_dir, labels, valid_ratio)  # 将验证集从原始的训练集中拆分出来
    reorg_test(data_dir)  # 在预测期间整理测试集，以方便读取
    pass


# 图像增广配置
def image_augmentation(data_dir, batch_size=32):
    transform_train = torchvision.transforms.Compose([
        # 在高度和宽度上将图像放大到40像素的正方形
        torchvision.transforms.Resize(40),

        # 随机裁剪出一个高度和宽度均为40像素的正方形图像，
        # 生成一个面积为原始图像面积0.64～1倍的小正方形，
        # 然后将其缩放为高度和宽度均为32像素的正方形
        torchvision.transforms.RandomResizedCrop(32, scale=(0.64, 1.0), ratio=(1.0, 1.0)),
        torchvision.transforms.RandomHorizontalFlip(),
        torchvision.transforms.ToTensor(),

        # 标准化图像的每个通道
        torchvision.transforms.Normalize([0.4914, 0.4822, 0.4465],
                                         [0.2023, 0.1994, 0.2010])
    ])

    transform_test = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize([0.4914, 0.4822, 0.4465],
                                         [0.2023, 0.1994, 0.2010])
    ])

    train_ds, train_valid_ds = [torchvision.datasets.ImageFolder(
        os.path.join(data_dir, 'train_valid_test', folder),
        transform=transform_train) for folder in ['train', 'train_valid']
    ]

    valid_ds, test_ds = [torchvision.datasets.ImageFolder(
        os.path.join(data_dir, 'train_valid_test', folder),
        transform=transform_test) for folder in ['valid', 'test']
    ]

    train_iter, train_valid_iter = [torch.utils.data.DataLoader(dataset,
                                                                batch_size,
                                                                shuffle=True,
                                                                drop_last=True) for dataset in (train_ds, train_valid_ds)
                                    ]

    valid_iter = torch.utils.data.DataLoader(valid_ds,
                                             batch_size,
                                             shuffle=False,
                                             drop_last=True
                                             )

    test_iter = torch.utils.data.DataLoader(test_ds,
                                            batch_size,
                                            shuffle=False,
                                            drop_last=False)
    return train_iter, test_iter, train_valid_iter, valid_iter


# 模型
def get_net():
    num_classes = 10
    # net = d2l.resnet18(num_classes, 3)    # 普通net
    # 微调版net
    # net = models.resnet18(weights=True)
    net = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    net.fc = nn.Linear(512, num_classes)

    return net


# 训练
def train(net, loss,
          train_iter, valid_iter,
          num_epochs, lr, wd, devices, lr_period, lr_decay):
    # 方式一
    # trainer = torch.optim.SGD(net.parameters(),
    #                           lr=lr,
    #                           momentum=0.9,
    #                           weight_decay=wd
    #                           )

    # 方式二, 调用源模型
    output_params = list(map(id, net.fc.parameters()))
    feature_params = filter(lambda p: id(p) not in output_params, net.parameters())
    trainer = torch.optim.SGD([{'params': feature_params},
                               {'params': net.fc.parameters(), 'lr': lr * 10}],
                              lr=lr, weight_decay=0.001)

    scheduler = torch.optim.lr_scheduler.StepLR(trainer, lr_period, lr_decay)

    #
    num_batches = len(train_iter)
    timer = d2l.Timer()

    legend = ['train loss', 'train acc']

    if valid_iter is not None:
        legend.append('valid acc')

    animator = d2l.Animator(xlabel='epoch',
                            xlim=[1, num_epochs],
                            legend=legend
                            )

    net = nn.DataParallel(net, device_ids=devices).to(devices[0])

    for epoch in range(num_epochs):
        net.train()
        metric = d2l.Accumulator(3)

        for i, (features, labels) in enumerate(train_iter):
            # 计时
            timer.start()

            # 训练
            l, acc = d2l.train_batch_ch13(net, features, labels,
                                          loss, trainer, devices)

            # 计数
            metric.add(l, acc, labels.shape[0])

            # 计时停止
            timer.stop()

            # 动画绘制
            if (i + 1) % (num_batches // 5) == 0 or i == num_batches - 1:
                animator.add(epoch + (i + 1) / num_batches,
                             (metric[0] / metric[2], metric[1] / metric[2],
                              None)
                             )
            pass

        if valid_iter is not None:
            valid_acc = d2l.evaluate_accuracy_gpu(net, valid_iter)
            animator.add(epoch + 1, (None, None, valid_acc))
            pass

        # 计时停止
        scheduler.step()

        # epoch for循环结束
        pass

    measures = (f'train loss {metric[0] / metric[2]:.3f}, '
                f'train acc {metric[1] / metric[2]:.3f}')

    if valid_iter is not None:
        measures += f', valid acc {valid_acc:.3f}'

    print(measures + f'\n{metric[2] * num_epochs / timer.sum():.1f}' f' examples/sec on {str(devices)}')


def load_model(PATH, devices, isFullLoad=True):
    """
    返回加载的模型net
    """
    net_models = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    net_models.fc = nn.Linear(512, 10)

    if isFullLoad:
        net_models = torch.load(PATH)  # 读取方式一
    else:
        net_models.load_state_dict(torch.load(PATH))  # 读取方式二

    # 转到GPU上
    net_models = net_models.to(devices[0])
    return net_models


def save_model(net, PATH='kaggle_cifar10_tiny_save.pt', isFullSave=True):
    # 模型保存
    if isFullSave:
        torch.save(net, PATH)  # 全保存
    else:
        torch.save(net.state_dict(), PATH)  # 保存模型参数
    return True


def run():
    data_dir = r"X:\ai_data\kaggle\kaggle_cifar10_tiny"
    labels = read_csv_labels(os.path.join(data_dir, 'trainLabels.csv'))
    print('# 训练样本 :', len(labels))
    print('# 类别 :', len(set(labels.values())))

    demo = True
    batch_size = 32 if demo else 128
    valid_ratio = 0.1  # 比例

    # 文件夹生成
    reorg_cifar10_data(data_dir, valid_ratio)

    # 图像增广
    train_iter, test_iter, train_valid_iter, valid_iter = image_augmentation(data_dir, batch_size)

    # 损失
    loss = nn.CrossEntropyLoss(reduction="none")

    # 训练参数
    devices = d2l.try_all_gpus()  # GPU驱动
    num_epochs = 20
    lr = 2e-4
    wd = 5e-4

    lr_period = 4
    lr_decay = 0.9
    net = get_net()

    # 训练
    train(net, loss,
          train_iter, valid_iter,
          num_epochs,
          lr,
          wd,
          devices,
          lr_period,
          lr_decay
          )
    pass


# 训练数据 文件创建测试
def test_1():
    data_dir = r"X:\ai_data\kaggle\kaggle_cifar10_tiny"
    labels = read_csv_labels(os.path.join(data_dir, 'trainLabels.csv'))
    print('# 训练样本 :', len(labels))
    print('# 类别 :', len(set(labels.values())))

    demo = True
    batch_size = 32 if demo else 128
    valid_ratio = 0.1  # 比例

    # 文件夹生成
    reorg_cifar10_data(data_dir, valid_ratio)
    pass


def main():
    test_1()
    pass


if __name__ == '__main__':
    main()
