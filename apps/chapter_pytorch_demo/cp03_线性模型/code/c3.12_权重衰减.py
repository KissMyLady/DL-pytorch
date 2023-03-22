# coding:utf-8
# Author:mylady
# Datetime:2023/3/15 9:35
import torch
import numpy as np

from matplotlib import pyplot as plt
from IPython import display


# 初始化模型参数
def init_params():
    w = torch.randn((num_inputs, 1), requires_grad=True)
    b = torch.zeros(1, requires_grad=True)
    return [w, b]


def sgd(params, lr, batch_size):
    # 为了和原书保持一致，这里除以了batch_size，但是应该是不用除的，因为一般用PyTorch计算loss时就默认已经
    # 沿batch维求了平均了。
    for param in params:
        # 注意这里更改param时用的param.data
        param.data -= lr * param.grad / batch_size


# 定义L2范数惩罚
def l2_penalty(w):
    return (w ** 2).sum() / 2


def linreg(X, w, b):
    return torch.mm(X, w) + b


def squared_loss(y_hat, y):
    # 注意这里返回的是向量, 另外, pytorch里的MSELoss并没有除以 2
    return ((y_hat - y.view(y_hat.size())) ** 2) / 2


def use_svg_display():
    """Use svg format to display plot in jupyter"""
    display.set_matplotlib_formats('svg')


def set_figsize(figsize=(3.5, 2.5)):
    use_svg_display()
    # 设置图的尺寸
    plt.rcParams['figure.figsize'] = figsize


def semilogy(x_vals, y_vals, x_label, y_label,
             x2_vals=None, y2_vals=None, legend=None,
             figsize=(3.5, 2.5)):

    set_figsize(figsize)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.semilogy(x_vals, y_vals)
    if x2_vals and y2_vals:
        plt.semilogy(x2_vals, y2_vals, linestyle=':')
        plt.legend(legend)
    # plt.show()


n_train = 20
n_test = 100
num_inputs = 200

true_w = torch.ones(num_inputs, 1) * 0.01
true_b = 0.05

features = torch.randn((120, 200))

labels = torch.matmul(features, true_w) + true_b
labels += torch.tensor(np.random.normal(0, 0.01, size=labels.size()),
                       dtype=torch.float
                       )

# 初始化训练数据集
train_features = features[:20, :]
test_features  = features[20:, :]

train_labels = labels[:20]
test_labels  = labels[20:]

# 定义训练和测试
batch_size = 1
num_epochs = 100
lr = 0.003

net = linreg
loss = squared_loss

dataset = torch.utils.data.TensorDataset(train_features, train_labels)
train_iter = torch.utils.data.DataLoader(dataset, batch_size, shuffle=True)


def fit_and_plot(lambd):
    w, b = init_params()
    train_ls = []
    test_ls = []

    for _ in range(num_epochs):
        for X, y in train_iter:
            # 添加了L2范数惩罚项
            l = loss(net(X, w, b), y) + lambd * l2_penalty(w)
            l = l.sum()

            if w.grad is not None:
                w.grad.data.zero_()
                b.grad.data.zero_()

            l.backward()
            sgd([w, b], lr, batch_size)

        # 计算 y_hat
        train_y_hat = net(train_features, w, b)
        test_y_hat = net(test_features, w, b)

        # 计算损失
        res_train = loss(train_y_hat, train_labels).mean().item()
        res_test = loss(test_y_hat, test_labels).mean().item()

        train_ls.append(res_train)
        test_ls.append(res_test)
        pass

    semilogy(range(1, num_epochs + 1), train_ls, 'epochs', 'loss',
            range(1, num_epochs + 1),  test_ls, ['train', 'test'])

    print('L2 norm of w:', w.norm().item())


def main():
    # 传入不同的we, 观察对过拟合的影响
    fit_and_plot(lambd=0)
    fit_and_plot(lambd=3)
    pass


if __name__ == '__main__':
    main()
