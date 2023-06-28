import sys
sys.path.append(".")
sys.path.append("../..")

import torch
from torch import nn
import math
import collections

from d2lzh_pytorch import myUtils
from d2lzh_pytorch import myPolt


argmax = lambda x, *args, **kwargs: x.argmax(*args, **kwargs)
astype = lambda x, *args, **kwargs: x.type(*args, **kwargs)
reduce_sum = lambda x, *args, **kwargs: x.sum(*args, **kwargs)
astype = lambda x, *args, **kwargs: x.type(*args, **kwargs)
size = lambda x, *args, **kwargs: x.numel(*args, **kwargs)


def accuracy(y_hat, y):
    """
        Compute the number of correct predictions.
    """
    if len(y_hat.shape) > 1 and y_hat.shape[1] > 1:
        y_hat = argmax(y_hat, axis=1)
    cmp = astype(y_hat, y.dtype) == y
    return float(reduce_sum(astype(cmp, y.dtype)))


def evaluate_accuracy_gpu(net, data_iter, device=None):
    """
        Compute the accuracy for a model on a dataset using a GPU.
    """
    if isinstance(net, nn.Module):
        net.eval()  # Set the model to evaluation mode
        if not device:
            device = next(iter(net.parameters())).device
    # No. of correct predictions, no. of predictions
    metric = myUtils.Accumulator(2)

    with torch.no_grad():
        for X, y in data_iter:
            if isinstance(X, list):
                # Required for BERT Fine-tuning (to be covered later)
                X = [x.to(device) for x in X]
            else:
                X = X.to(device)
            y = y.to(device)
            metric.add(accuracy(net(X), y), size(y))
    return metric[0] / metric[1]


def train_batch_ch13(net, X, y, loss, trainer, devices):
    """
        Train for a minibatch with mutiple GPUs (defined in Chapter 13).
    """
    if isinstance(X, list):
        # Required for BERT fine-tuning (to be covered later)
        X = [x.to(devices[0]) for x in X]
    else:
        X = X.to(devices[0])
    y = y.to(devices[0])
    net.train()
    trainer.zero_grad()
    pred = net(X)
    l = loss(pred, y)
    l.sum().backward()
    trainer.step()
    train_loss_sum = l.sum()
    train_acc_sum = accuracy(pred, y)
    return train_loss_sum, train_acc_sum


def train_ch13(net, train_iter, test_iter, 
               loss, trainer, 
               num_epochs, devices=myUtils.try_all_gpus()):
    """Train a model with mutiple GPUs (defined in Chapter 13).
    Defined in :numref:`sec_image_augmentation`
    使 用: 在 15 章节
    """
    timer, num_batches = myUtils.Timer(), len(train_iter)
    animator = myPolt.Animator(xlabel='epoch', 
                            xlim=[1, num_epochs], 
                            ylim=[0, 1], 
                            legend=['train loss', 'train acc', 'test acc'])
    net = nn.DataParallel(net, device_ids=devices).to(devices[0])
    for epoch in range(num_epochs):
        # Sum of training loss, sum of training accuracy, no. of examples,
        # no. of predictions
        metric = myUtils.Accumulator(4)
        for i, (features, labels) in enumerate(train_iter):
            timer.start()
            l, acc = train_batch_ch13(net, 
                                      features, 
                                      labels, 
                                      loss, 
                                      trainer, 
                                      devices)
            metric.add(l, acc, labels.shape[0], labels.numel())
            timer.stop()
            if (i + 1) % (num_batches // 5) == 0 or i == num_batches - 1:
                animator.add(epoch + (i + 1) / num_batches,
                             (metric[0] / metric[2], metric[1] / metric[3],
                              None))
        test_acc = evaluate_accuracy_gpu(net, test_iter)
        animator.add(epoch + 1, (None, None, test_acc))

    print('time consuming: %.4f' % timer.sum())
    print(f'loss {metric[0] / metric[2]:.3f}, train acc '
          f'{metric[1] / metric[3]:.3f}, test acc {test_acc:.3f}')
    print(f'{metric[2] * num_epochs / timer.sum():.1f} examples/sec on '
          f'{str(devices)}')


# 使用训练ch13
def test_1():

    # 加载数据
    batch_size = 64
    train_iter, test_iter, vocab = load_data_imdb(batch_size)

    embed_size = 100
    num_hiddens = 100
    num_layers = 2
    devices = myUtils.try_all_gpus()
    
    # 网络构建
    net = BiRNN(len(vocab),
                embed_size,
                num_hiddens,
                num_layers)

    lr = 0.01
    num_epochs = 5
    # 优化器
    trainer = torch.optim.Adam(net.parameters(), lr=lr)
    # 损失函数
    loss = nn.CrossEntropyLoss(reduction="none")

    # 训练
    train_ch13(net, train_iter, test_iter,
            loss,
            trainer,
            num_epochs,
            devices
            )
    pass


def main():
    pass


if __name__ == "__main__":
    # main()
    pass