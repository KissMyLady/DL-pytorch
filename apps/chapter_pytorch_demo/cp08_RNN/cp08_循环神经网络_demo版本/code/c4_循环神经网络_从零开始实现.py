# coding:utf-8
# Author:mylady
# Datetime:2023/4/7 21:03
import time
import math
import numpy as np
import torch
from torch import nn, optim
import torch_package.nn.functional as F

# import sys
# sys.path.append("..")
import apps.chapter_pytorch_demo.d2lzh_pytorch as d2l

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# 将词映射成向量
def one_hot(x, n_class, dtype=torch.float32):
    x = x.long()
    res = torch.zeros(x.shape[0],
                      n_class,
                      dtype=dtype,
                      device=x.device)
    res.scatter_(1, x.view(-1, 1), 1)
    return res


def to_onehot(X, n_class):
    # X shape: (batch, seq_len), output: seq_len elements of (batch, n_class)
    return [one_hot(X[:, i], n_class) for i in range(X.shape[1])]


# 加载歌词数据
(corpus_indices,
 char_to_idx,
 idx_to_char,
 vocab_size) = d2l.load_data_jay_lyrics(read_file_path=r"../../data/jaychou_lyrics.txt")

num_inputs = vocab_size
num_hiddens = 256
num_outputs = vocab_size


# 参数初始化
def get_params():
    def _one(shape):
        ts = torch.tensor(np.random.normal(0, 0.01, size=shape), device=device, dtype=torch.float32)
        return torch.nn.Parameter(ts, requires_grad=True)

    # 隐藏层参数
    W_xh = _one((num_inputs, num_hiddens))
    W_hh = _one((num_hiddens, num_hiddens))
    b_h = torch.nn.Parameter(torch.zeros(num_hiddens, device=device, requires_grad=True))

    # 输出层参数
    W_hq = _one((num_hiddens, num_outputs))
    b_q = torch.nn.Parameter(torch.zeros(num_outputs, device=device, requires_grad=True))

    return nn.ParameterList([W_xh, W_hh, b_h, W_hq, b_q])


# 初始化参数
def init_rnn_state(batch_size, num_hiddens, device):
    return (torch.zeros((batch_size, num_hiddens), device=device),)


# 模型
def rnn(inputs, state, params):
    """

    输出: Y, H
    """
    # inputs和outputs皆为num_steps个形状为(batch_size, vocab_size)的矩阵
    W_xh, W_hh, b_h, W_hq, b_q = params
    H, = state
    outputs = []

    for X in inputs:
        H = torch.tanh(torch.matmul(X, W_xh) + torch.matmul(H, W_hh) + b_h)  # 隐藏层计算
        Y = torch.matmul(H, W_hq) + b_q  # 输出层
        outputs.append(Y)
        pass

    return outputs, (H,)


def predict_rnn(prefix, num_chars, rnn, params,
                init_rnn_state, num_hiddens,
                vocab_size, device, idx_to_char, char_to_idx):
    state = init_rnn_state(1, num_hiddens, device)
    output = [char_to_idx[prefix[0]]]

    for t in range(num_chars + len(prefix) - 1):
        # 将上一时间步的输出作为当前时间步的输入
        X = to_onehot(torch.tensor([[output[-1]]], device=device), vocab_size)

        # 计算输出和更新隐藏状态
        (Y, state) = rnn(X, state, params)

        # 下一个时间步的输入是prefix里的字符或者当前的最佳预测字符
        if t < len(prefix) - 1:
            output.append(char_to_idx[prefix[t + 1]])
        else:
            output.append(int(Y[0].argmax(dim=1).item()))

    return ''.join([idx_to_char[i] for i in output])


# 裁剪梯度
def grad_clipping(params, theta, device):
    norm = torch.tensor([0.0], device=device)
    for param in params:
        norm += (param.grad.data ** 2).sum()
    norm = norm.sqrt().item()
    if norm > theta:
        for param in params:
            param.grad.data *= (theta / norm)


# 训练并预测
def train_and_predict_rnn(rnn, get_params,
                          init_rnn_state, num_hiddens,
                          vocab_size, device,
                          corpus_indices, idx_to_char,
                          char_to_idx,
                          is_random_iter,
                          num_epochs, num_steps,
                          lr, clipping_theta,
                          batch_size, pred_period,
                          pred_len, prefixes):
    # 数据加载方式
    if is_random_iter:
        data_iter_fn = d2l.data_iter_random
    else:
        data_iter_fn = d2l.data_iter_consecutive

    params = get_params()
    loss = nn.CrossEntropyLoss()

    for epoch in range(num_epochs):
        if not is_random_iter:
            # 如使用相邻采样，在epoch开始时初始化隐藏状态
            state = init_rnn_state(batch_size, num_hiddens, device)

        l_sum, n, start = 0.0, 0, time.time()
        data_iter = data_iter_fn(corpus_indices, batch_size, num_steps, device)

        for X, Y in data_iter:
            # 如使用随机采样，在每个小批量更新前初始化隐藏状态
            if is_random_iter:
                state = init_rnn_state(batch_size, num_hiddens, device)
            else:
                # 否则需要使用detach函数从计算图分离隐藏状态, 这是为了
                # 使模型参数的梯度计算只依赖一次迭代读取的小批量序列(防止梯度计算开销太大)
                for s in state:
                    s.detach_()
            inputs = to_onehot(X, vocab_size)

            # outputs有num_steps个形状为(batch_size, vocab_size)的矩阵
            (outputs, state) = rnn(inputs, state, params)

            # 拼接之后形状为(num_steps * batch_size, vocab_size)
            outputs = torch.cat(outputs, dim=0)

            # Y的形状是(batch_size, num_steps)，转置后再变成长度为
            # batch * num_steps 的向量，这样跟输出的行一一对应
            y = torch.transpose(Y, 0, 1).contiguous().view(-1)

            # 使用交叉熵损失计算平均分类误差
            l = loss(outputs, y.long())

            # 梯度清0
            if params[0].grad is not None:
                for param in params:
                    param.grad.data.zero_()

            l.backward()
            grad_clipping(params, clipping_theta, device)  # 裁剪梯度
            d2l.sgd(params, lr, 1)  # 因为误差已经取过均值，梯度不用再做平均
            l_sum += l.item() * y.shape[0]
            n += y.shape[0]

        if (epoch + 1) % pred_period == 0:
            print('epoch %d, perplexity %f, time %.2f sec' % (
                epoch + 1, math.exp(l_sum / n),
                time.time() - start)
                  )

            for prefix in prefixes:
                print(' -', predict_rnn(prefix, pred_len, rnn,
                                        params, init_rnn_state,
                                        num_hiddens, vocab_size,
                                        device, idx_to_char, char_to_idx))


def test_1():
    params = get_params()
    predictStr = predict_rnn('分开', 10, rnn, params,
                             init_rnn_state, num_hiddens,
                             vocab_size, device,
                             idx_to_char, char_to_idx
                             )
    print("predictStr: ", predictStr)
    pass


# 下面采用随机采样训练模型并创作歌词
def test_2():
    num_epochs = 250
    num_steps = 35
    batch_size = 32
    lr = 1e2
    clipping_theta = 1e-2
    pred_period = 50
    pred_len = 50
    prefixes = ['分开', '不分开']

    # 训练, 并创作歌词
    train_and_predict_rnn(rnn, get_params,
                          init_rnn_state, num_hiddens,
                          vocab_size, device,
                          corpus_indices, idx_to_char, char_to_idx,
                          True,
                          num_epochs, num_steps, lr,
                          clipping_theta, batch_size,
                          pred_period, pred_len,
                          prefixes)
    pass


# 接下来采用相邻采样训练模型并创作歌词
def test_3():
    num_epochs = 250
    num_steps = 35
    batch_size = 32
    lr = 1e2
    clipping_theta = 1e-2
    pred_period = 50
    pred_len = 50
    prefixes = ['分开', '不分开']

    train_and_predict_rnn(rnn, get_params,
                          init_rnn_state, num_hiddens,
                          vocab_size, device,
                          corpus_indices, idx_to_char,
                          char_to_idx,
                          False,
                          num_epochs, num_steps, lr,
                          clipping_theta, batch_size,
                          pred_period, pred_len,
                          prefixes)
    pass


def main():
    # test_2()  # 随机采样训练, 并创作歌词
    test_3()  # 相邻采样训练, 并创作歌词
    pass


if __name__ == '__main__':
    main()
