# coding:utf-8
# Author:mylady
# Datetime:2023/7/01 16:40
import torch
import time
from d2lzh_pytorch.myUtils import Timer


def evaluate_accuracy(data_iter, net, device=None):
    if device is None and isinstance(net, torch.nn.Module):
        # 如果没指定device就使用net的device
        device = list(net.parameters())[0].device

    acc_sum, n = 0.0, 0
    with torch.no_grad():
        for X, y in data_iter:
            if isinstance(net, torch.nn.Module):
                net.eval()  # 评估模式, 这会关闭dropout
                acc_sum += (net(X.to(device)).argmax(dim=1) == y.to(device)).float().sum().cpu().item()
                net.train()  # 改回训练模式
            else:  
                # 自定义的模型, 3.13节之后不会用到, 不考虑GPU
                if ('is_training' in net.__code__.co_varnames):  # 如果有is_training这个参数
                    # 将is_training设置成False
                    acc_sum += (net(X, is_training=False).argmax(dim=1) == y).float().sum().item()
                else:
                    acc_sum += (net(X).argmax(dim=1) == y).float().sum().item()
            n += y.shape[0]
        pass
    return acc_sum / n


# CNN的训练模块
def train_ch5(net, train_iter, test_iter, 
              batch_size, optimizer, device, num_epochs):
    timer = Timer()
    net = net.to(device)
    print("training on ", device)

    loss = torch.nn.CrossEntropyLoss()
    for epoch in range(num_epochs):
        timer.start()

        train_l_sum = 0.0  # 训练损失
        train_acc_sum = 0.0
        n = 0
        batch_count = 0  # 当前数据集 ,训练了多少轮
        for X, y in train_iter:
            X = X.to(device)
            y = y.to(device)
            # 计算
            y_hat = net(X)
            # 损失
            l = loss(y_hat, y)
            # 梯度下降
            optimizer.zero_grad()
            l.backward()
            optimizer.step()

            train_l_sum += l.cpu().item()
            train_acc_sum += (y_hat.argmax(dim=1) == y).sum().cpu().item()
            n += y.shape[0]
            batch_count += 1
            pass

        loss_ = train_l_sum / batch_count   # 总损失 / 数据训练次数
        train_acc = train_acc_sum / n  # 正确数 / 总数
        test_acc = evaluate_accuracy(test_iter, net)  # 测试
        # time_sec = time.time() - t_start  # 耗时
        time_sec = timer.stop()

        print('epoch %d, loss %.4f, train acc %.3f, test acc %.3f, time %.1f sec'
              % (epoch + 1, loss_, train_acc, test_acc, time_sec))
        pass

    # 总耗时
    sum_time = timer.sum()
    print("the train sum of time is %.2f" % sum_time)
    pass


def test_1():
    pass


def main():
    pass


if __name__ == "__main__":
    main()
    pass