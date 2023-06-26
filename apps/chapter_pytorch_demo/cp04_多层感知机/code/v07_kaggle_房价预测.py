# %matplotlib inline
# from d2l import torch as d2l

import torch
from torch import nn
import pandas as pd
import numpy as np
import datetime
import sys

sys.path.append("../..")
import d2lzh_pytorch.torch as d2l

print(torch.__version__)


def get_net(in_features):
    net = nn.Sequential(nn.Linear(in_features, 1))
    return net


def log_rmse(net, loss, features, labels):
    # 为了在取对数时进一步稳定该值，将小于1的值设置为1
    clipped_preds = torch.clamp(net(features), 1, float('inf'))
    rmse = torch.sqrt(loss(torch.log(clipped_preds),
                           torch.log(labels))
                      )
    return rmse.item()


def train(net, loss, train_features, train_labels, test_features, test_labels,
          num_epochs,
          learning_rate,
          weight_decay,
          batch_size):
    train_ls, test_ls = [], []
    train_iter = d2l.load_array((train_features, train_labels), batch_size)

    # 这里使用的是Adam优化算法
    optimizer = torch.optim.Adam(net.parameters(),
                                 lr=learning_rate,
                                 weight_decay=weight_decay)

    for epoch in range(num_epochs):
        for X, y in train_iter:
            optimizer.zero_grad()
            l = loss(net(X), y)
            l.backward()
            optimizer.step()

        train_ls.append(log_rmse(net, loss, train_features, train_labels))
        if test_labels is not None:
            test_ls.append(log_rmse(net, loss, test_features, test_labels))
    return train_ls, test_ls


# @tab all
def get_k_fold_data(k, i, X, y):
    assert k > 1
    fold_size = X.shape[0] // k
    X_train, y_train = None, None

    for j in range(k):
        idx = slice(j * fold_size, (j + 1) * fold_size)
        X_part, y_part = X[idx, :], y[idx]
        if j == i:
            X_valid, y_valid = X_part, y_part
        elif X_train is None:
            X_train, y_train = X_part, y_part
        else:
            X_train = d2l.concat([X_train, X_part], 0)
            y_train = d2l.concat([y_train, y_part], 0)

    return X_train, y_train, X_valid, y_valid


def k_fold(k, X_train, y_train, in_features, loss,
           num_epochs, learning_rate, weight_decay,
           batch_size):
    train_l_sum, valid_l_sum = 0, 0

    for i in range(k):
        data = get_k_fold_data(k, i, X_train, y_train)

        net = get_net(in_features)

        train_ls, valid_ls = train(net, loss, *data,
                                   num_epochs, learning_rate,
                                   weight_decay,
                                   batch_size)

        train_l_sum += train_ls[-1]
        valid_l_sum += valid_ls[-1]

        if i == 0:
            d2l.plot(list(range(1, num_epochs + 1)), [train_ls, valid_ls],
                     xlabel='epoch', ylabel='rmse', xlim=[1, num_epochs],
                     legend=['train', 'valid'], yscale='log')

        print(f'折{i + 1}，训练log rmse{float(train_ls[-1]):f}, '
              f'验证log rmse{float(valid_ls[-1]):f}')
    return train_l_sum / k, valid_l_sum / k


# @tab all
def train_and_pred(loss, train_features, test_features, train_labels, test_data,
                   in_features,
                   num_epochs,
                   lr,
                   weight_decay,
                   batch_size):
    net = get_net(in_features)

    train_ls, _ = train(net, loss, train_features, train_labels, None, None,
                        num_epochs, lr, weight_decay, batch_size
                        )

    d2l.plot(np.arange(1, num_epochs + 1), [train_ls], xlabel='epoch',
             ylabel='log rmse', xlim=[1, num_epochs], yscale='log')

    print(f'训练log rmse：{float(train_ls[-1]):f}')

    # 将网络应用于测试集。
    preds = d2l.numpy(net(test_features))

    # 将其重新格式化以导出到Kaggle
    test_data['SalePrice'] = pd.Series(preds.reshape(1, -1)[0])
    submission = pd.concat([test_data['Id'], test_data['SalePrice']], axis=1)

    str_time = str(datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))
    submission.to_csv('submission_%s.csv' % str_time, index=False)


def load_house_data():
    # 加载模型数据
    train_data = pd.read_csv('/mnt/aiguo/ai_data/Datasets_on_HHD/kaggle_house/train.csv')
    # train_data = pd.read_csv(r'Z:\ai_data\Datasets_on_HHD\d2l_data\kaggle_house\train.csv')

    test_data = pd.read_csv('/mnt/aiguo/ai_data/Datasets_on_HHD/kaggle_house/test.csv')
    # test_data = pd.read_csv(r'Z:\ai_data\Datasets_on_HHD\d2l_data\kaggle_house\test.csv')

    all_features = pd.concat((
        train_data.iloc[:, 1:-1],
        test_data.iloc[:, 1:])
    )

    # @tab all
    # 若无法获得测试数据，则可根据训练数据计算均值和标准差
    numeric_features = all_features.dtypes[all_features.dtypes != 'object'].index
    all_features[numeric_features] = all_features[numeric_features].apply(lambda x: (x - x.mean()) / (x.std()))

    # 在标准化数据之后，所有均值消失，因此我们可以将缺失值设置为0
    all_features[numeric_features] = all_features[numeric_features].fillna(0)

    # “Dummy_na=True”将“na”（缺失值）视为有效的特征值，并为其创建指示符特征
    all_features = pd.get_dummies(all_features, dummy_na=True)
    all_features.shape

    n_train = train_data.shape[0]

    train_features = d2l.tensor(all_features[:n_train].values, dtype=d2l.float32)
    test_features = d2l.tensor(all_features[n_train:].values, dtype=d2l.float32)
    train_labels = d2l.tensor(train_data.SalePrice.values.reshape(-1, 1), dtype=d2l.float32)
    return train_features, train_labels, test_features, test_data


def main():
    # 房价数据加载
    train_features, train_labels, test_features, test_data = load_house_data()

    # 损失函数
    loss = nn.MSELoss()

    # 输入特征
    in_features = train_features.shape[1]

    # 超参数设置
    k = 5
    num_epochs = 100
    lr = 5
    weight_decay = 0
    batch_size = 64

    # 使用k-折叠交叉 训练
    train_l, valid_l = k_fold(k,
                              train_features, train_labels,
                              in_features, loss,
                              num_epochs,
                              lr,
                              weight_decay,
                              batch_size)

    print(f'{k}-折验证: 平均训练log rmse: {float(train_l):f}, '
          f'平均验证log rmse: {float(valid_l):f}')

    # 训练并预测
    train_and_pred(loss, train_features, test_features, train_labels, test_data,
                   in_features,
                   num_epochs,
                   lr,
                   weight_decay,
                   batch_size)
    pass


if __name__ == "__main__":
    main()
    pass
