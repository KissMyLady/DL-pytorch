# coding:utf-8
# Author:mylady
# Datetime:2023/3/8 18:36
import torch
import torchvision
import numpy as np
import sys
import time


def load_data_fashion_mnist(batch_size, resize=None, root='~/Datasets/FashionMNIST'):
    """Download the fashion mnist dataset and then load into memory."""
    trans = []
    if resize:
        trans.append(torchvision.transforms.Resize(size=resize))
        pass

    trans.append(torchvision.transforms.ToTensor())
    transform = torchvision.transforms.Compose(trans)

    mnist_train = torchvision.datasets.FashionMNIST(
        root=root,
        train=True,
        download=True,
        transform=transform
    )
    mnist_test = torchvision.datasets.FashionMNIST(
        root=root,
        train=False,
        download=True,
        transform=transform
    )

    if sys.platform.startswith('win'):
        num_workers = 0  # 0表示不用额外的进程来加速读取数据
    else:
        num_workers = 4

    train_iter = torch.utils.data.DataLoader(
        mnist_train,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers
    )

    test_iter = torch.utils.data.DataLoader(
        mnist_test,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers
    )
    return train_iter, test_iter


batch_size = 256

# 获取和读取数据
train_iter, test_iter = load_data_fashion_mnist(batch_size)

# 初始化模型参数
num_inputs = 784
num_outputs = 10


W = torch.tensor(np.random.normal(0, 0.01, (num_inputs, num_outputs)),
                 dtype=torch.float
                 )

b = torch.zeros(num_outputs,
                dtype=torch.float
                )

W.requires_grad_(requires_grad=True)
b.requires_grad_(requires_grad=True)


#  实现softmax运算
X = torch.tensor([[1, 2, 3],
                  [4, 5, 6]]
                 )
print('实现softmax运算 X.0: ', X.sum(dim=0, keepdim=True))
print('实现softmax运算 X.1: ', X.sum(dim=1, keepdim=True))


def softmax(X):
    X_exp = X.exp()
    partition = X_exp.sum(dim=1, keepdim=True)
    return X_exp / partition


X = torch.rand((2, 5))
X_prob = softmax(X)
print("X_prob: \n", X_prob)
print("X_prob.sum(dim=1): \n", X_prob.sum(dim=1))


# 定义模型
def net(X):
    return softmax(torch.mm(X.view((-1, num_inputs)), W) + b)


# 定义损失函数
y_hat = torch.tensor([[0.1, 0.3, 0.6], [0.3, 0.2, 0.5]])
y = torch.LongTensor([0, 2])
y_hat.gather(1, y.view(-1, 1))


# 定义损失函数
def cross_entropy(y_hat, y):
    return - torch.log(y_hat.gather(1, y.view(-1, 1)))


# 计算分类准确率
def accuracy(y_hat, y):
    return (y_hat.argmax(dim=1) == y).float().mean().item()


print(accuracy(y_hat, y))


# 本函数已保存在d2lzh_pytorch包中方便以后使用。
# 该函数将被逐步改进：它的完整实现将在“图像增广”一节中描述
def evaluate_accuracy(data_iter, net):
    acc_sum, n = 0.0, 0
    for X, y in data_iter:
        acc_sum += (net(X).argmax(dim=1) == y).float().sum().item()
        n += y.shape[0]
    return acc_sum / n


print(evaluate_accuracy(test_iter, net))


# 训练模型
def sgd(params, lr, batch_size):
    # 为了和原书保持一致，这里除以了batch_size，但是应该是不用除的，因为一般用PyTorch计算loss时就默认已经
    # 沿batch维求了平均了。
    for param in params:
        # 注意这里更改param时用的param.data
        param.data -= lr * param.grad / batch_size
        pass


num_epochs, lr = 20, 0.1


# 本函数已保存在d2lzh包中方便以后使用
def train_ch3(net, train_iter, test_iter, loss, num_epochs, batch_size,
              params=None, lr=None, optimizer=None):
    for epoch in range(num_epochs):
        train_l_sum, train_acc_sum, n = 0.0, 0.0, 0
        for X, y in train_iter:
            y_hat = net(X)
            l = loss(y_hat, y).sum()

            # 梯度清零
            if optimizer is not None:
                optimizer.zero_grad()
            elif params is not None and params[0].grad is not None:
                for param in params:
                    param.grad.data.zero_()

            l.backward()
            if optimizer is None:
                sgd(params, lr, batch_size)
            else:
                optimizer.step()  # “softmax回归的简洁实现”一节将用到

            train_l_sum += l.item()
            train_acc_sum += (y_hat.argmax(dim=1) == y).sum().item()

            n += y.shape[0]
            pass

        test_acc = evaluate_accuracy(test_iter, net)
        print('epoch %d, loss %.4f, train acc %.3f, test acc %.3f'
              % (epoch + 1, train_l_sum / n, train_acc_sum / n, test_acc))
        pass
    pass


# 训练


start = time.time()

train_ch3(net,
          train_iter,
          test_iter,
          cross_entropy,
          num_epochs,
          batch_size,
          [W, b],
          lr
          )

print("训练共计耗时: ", time.time() - start)


# 预测
def get_fashion_mnist_labels(labels):
    text_labels = ['t-shirt', 'trouser', 'pullover', 'dress', 'coat',
                   'sandal', 'shirt', 'sneaker', 'bag', 'ankle boot'
                   ]
    return [text_labels[int(i)] for i in labels]


correct_num = 0
error_num = 0

for i, value in enumerate(test_iter):
    X, y = value[0], value[1]

    true_labels = get_fashion_mnist_labels(y.numpy())
    pred_labels = get_fashion_mnist_labels(net(X).argmax(dim=1).numpy())

    # 正确率计算
    for true, pred in zip(true_labels, pred_labels):
        if pred == true:
            correct_num += 1
        else:
            error_num += 1
        pass

print("correct_num: %s" % correct_num)
print("error_num: %s" % error_num)
print("正确率: %s" % (correct_num / (correct_num + error_num)))