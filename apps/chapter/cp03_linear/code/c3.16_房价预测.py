# coding:utf-8
# Author:mylady
# Datetime:2023/3/16 11:04

# %matplotlib inline
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import sys

sys.path.append("..")
# import d2lzh_pytorch as d2l
from apps.chapter import d2lzh_pytorch as d2l


print(torch.__version__)

torch.set_default_tensor_type(torch.FloatTensor)

# 加载数据
# train_data = pd.read_csv('~/Datasets/kaggle_house/train.csv')
train_data = pd.read_csv('K:\\code_big\\kaggle_house\\train.csv')
# test_data = pd.read_csv('~/Datasets/kaggle_house/test.csv')
test_data = pd.read_csv('K:\\code_big\\kaggle_house\\test.csv')

# 不要 id, y 标签数据
trainData_not_label = train_data.iloc[:, 1:-1]

# 不要 id 数据. (test没y数据)
testData_not_label = test_data.iloc[:, 1:]

# 得到全部特征
all_features = pd.concat((trainData_not_label,
                          testData_not_label
                         ))

# 这里只提取 数字类型的特征
numeric_features = all_features.dtypes[all_features.dtypes != 'object'].index


# 只有36行的 数字类型特征
features_36 = all_features[numeric_features]


# 特征值标准化
all_features[numeric_features] = features_36.apply(
    lambda x: (x - x.mean()) / (x.std())
)

# 标准化后，每个数值特征的均值变为 0，所以可以直接用 0 来替换缺失值
all_features[numeric_features] = all_features[numeric_features].fillna(0)

# dummy_na=True将缺失值也当作合法的特征值并为其创建指示特征
# 将 object 等描述性特征, 转成 int类型特征
all_features = pd.get_dummies(all_features, dummy_na=True)


# 训练-测试数据
n_train = train_data.shape[0]  # 1460

train_data_1460 = all_features[:n_train].values  # 1460 条
test_data_1459 = all_features[n_train:].values   # 1459 条

train_features = torch.tensor(train_data_1460, dtype=torch.float)
test_features  = torch.tensor(test_data_1459, dtype=torch.float)

train_labels = torch.tensor(train_data.SalePrice.values,
                            dtype=torch.float
                           ).view(-1, 1)


def get_net(feature_num):
    net = nn.Linear(feature_num, 1)
    # 一并初始化模型参数
    for param in net.parameters():
        nn.init.normal_(param, mean=0, std=0.01)
    return net


loss = torch.nn.MSELoss()


# 对数均方根误差
def log_rmse(net, features, labels):
    with torch.no_grad():
        # 将小于1的值设成1，使得取对数时数值更稳定
        clipped_preds = torch.max(net(features), torch.tensor(1.0))
        rmse = torch.sqrt(loss(clipped_preds.log(), labels.log()))
        pass
    return rmse.item()


def train(net, train_features, train_labels,
          test_features, test_labels,
          num_epochs, learning_rate,
          weight_decay,
          batch_size
          ):
    train_ls = []
    test_ls = []

    dataset = torch.utils.data.TensorDataset(train_features, train_labels)
    train_iter = torch.utils.data.DataLoader(dataset, batch_size, shuffle=True)

    # 这里使用了Adam优化算法
    optimizer = torch.optim.Adam(params=net.parameters(),
                                 lr=learning_rate,
                                 weight_decay=weight_decay  # 权重衰减
                                 )
    net = net.float()

    # 训练
    for epoch in range(num_epochs):
        for X, y in train_iter:
            l = loss(net(X.float()), y.float())
            optimizer.zero_grad()
            l.backward()
            optimizer.step()
            pass

        # 损失计算
        loss_t = log_rmse(net, train_features, train_labels)
        train_ls.append(loss_t)

        if test_labels is not None:
            loss_t = log_rmse(net, test_features, test_labels)
            test_ls.append(loss_t)
        pass

    return train_ls, test_ls


# 𝐾 折交叉验证
def get_k_fold_data(k, i, X, y):
    # 返回第i折交叉验证时所需要的训练和验证数据
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
            X_train = torch.cat((X_train, X_part), dim=0)
            y_train = torch.cat((y_train, y_part), dim=0)
        pass

    return X_train, y_train, X_valid, y_valid


# 训练K次, 并返回训练和验证的平均误差
def k_fold(k, X_train, y_train, num_epochs,
           learning_rate, weight_decay, batch_size):
    train_l_sum = 0
    valid_l_sum = 0

    for i in range(k):
        data = get_k_fold_data(k, i, X_train, y_train)
        net = get_net(X_train.shape[1])

        train_ls, valid_ls = train(net,
                                   *data,
                                   num_epochs,
                                   learning_rate,
                                   weight_decay,
                                   batch_size
                                   )

        train_l_sum += train_ls[-1]
        valid_l_sum += valid_ls[-1]
        if i == 0:
            d2l.semilogy(range(1, num_epochs + 1), train_ls, 'epochs', 'rmse',
                         range(1, num_epochs + 1), valid_ls,
                         ['train', 'valid'])
        print('fold %d, train rmse %f, valid rmse %f' % (i, train_ls[-1], valid_ls[-1]))
        pass

    return train_l_sum / k, valid_l_sum / k


def run_v1():
    # 启动
    k = 5
    num_epochs = 100
    lr = 5
    weight_decay = 0
    batch_size = 64

    train_l, valid_l = k_fold(k,
                              train_features,
                              train_labels,
                              num_epochs,
                              lr,
                              weight_decay, batch_size
                              )
    print('%d-fold validation: avg train rmse %f, avg valid rmse %f' % (k, train_l, valid_l))
    pass


def train_and_pred(train_features, test_features, train_labels, test_data,
                   num_epochs, lr, weight_decay, batch_size):
    # 模型选择
    net = get_net(train_features.shape[1])

    # 训练
    train_ls, _ = train(net, train_features, train_labels, None, None,
                        num_epochs, lr,
                        weight_decay,
                        batch_size
                        )

    d2l.semilogy(range(1, num_epochs + 1), train_ls, 'epochs', 'rmse')
    print('train rmse %f' % train_ls[-1])
    preds = net(test_features).detach().numpy()
    test_data['SalePrice'] = pd.Series(preds.reshape(1, -1)[0])
    submission = pd.concat([test_data['Id'], test_data['SalePrice']], axis=1)
    submission.to_csv('./submission.csv', index=False)
    pass


# 执行
def run_v2():
    num_epochs = 100
    lr = 5
    weight_decay = 0
    batch_size = 64

    train_and_pred(train_features, test_features,
                   train_labels, test_data,
                   num_epochs, lr,
                   weight_decay, batch_size
                   )
    pass


def main():
    run_v1()
    run_v2()
    pass


if __name__ == '__main__':
    main()
