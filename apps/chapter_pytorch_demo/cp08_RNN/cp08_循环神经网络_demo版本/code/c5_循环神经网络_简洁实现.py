# coding:utf-8
# Author:mylady
# Datetime:2023/4/7 22:40
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

# 加载歌词数据
(corpus_indices,
 char_to_idx,
 idx_to_char,
 vocab_size) = d2l.load_data_jay_lyrics()

num_hiddens = 256

# RNN 简洁定义
rnn_layer = nn.RNN(input_size=vocab_size, hidden_size=num_hiddens)

num_steps = 35
batch_size = 2
state = None

X = torch.rand(num_steps, batch_size, vocab_size)

# 计算后, 返回输出和隐藏状态
Y, state_new = rnn_layer(X, state)

print(Y.shape, len(state_new), state_new[0].shape)


class RNNModel(nn.Module):

    def __init__(self, rnn_layer, vocab_size):
        super(RNNModel, self).__init__()
        self.rnn = rnn_layer
        self.hidden_size = rnn_layer.hidden_size * (2 if rnn_layer.bidirectional else 1)
        self.vocab_size = vocab_size
        self.dense = nn.Linear(self.hidden_size, vocab_size)
        self.state = None

    def forward(self, inputs, state):  # inputs: (batch, seq_len)

        # 获取 one-hot 向量表示
        X = d2l.to_onehot(inputs, self.vocab_size)  # X是个list
        Y, self.state = self.rnn(torch.stack(X), state)

        # 全连接层会首先将Y的形状变成(num_steps * batch_size, num_hiddens)，它的输出
        # 形状为(num_steps * batch_size, vocab_size)
        output = self.dense(Y.view(-1, Y.shape[-1]))
        return output, self.state


def predict_rnn_pytorch(prefix, num_chars, model, vocab_size, device,
                        idx_to_char, char_to_idx
                        ):
    state = None
    output = [char_to_idx[prefix[0]]]  # output会记录prefix加上输出

    for t in range(num_chars + len(prefix) - 1):
        X = torch.tensor([output[-1]], device=device).view(1, 1)

        if state is not None:
            if isinstance(state, tuple):  # LSTM, state:(h, c)
                state = (state[0].to(device), state[1].to(device))
            else:
                state = state.to(device)
            pass

        (Y, state) = model(X, state)
        if t < len(prefix) - 1:
            output.append(char_to_idx[prefix[t + 1]])
        else:
            output.append(int(Y.argmax(dim=1).item()))
    return ''.join([idx_to_char[i] for i in output])


def train_and_predict_rnn_pytorch(model, num_hiddens, vocab_size, device,
                                  corpus_indices, idx_to_char, char_to_idx,
                                  num_epochs, num_steps, lr, clipping_theta,
                                  batch_size, pred_period, pred_len, prefixes):
    loss = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    model.to(device)
    state = None

    for epoch in range(num_epochs):
        l_sum, n, start = 0.0, 0, time.time()

        # 相邻采样
        data_iter = d2l.data_iter_consecutive(corpus_indices,
                                              batch_size,
                                              num_steps,
                                              device)

        for X, Y in data_iter:
            if state is not None:
                # 使用detach函数从计算图分离隐藏状态, 这是为了
                # 使模型参数的梯度计算只依赖一次迭代读取的小批量序列(防止梯度计算开销太大)
                if isinstance(state, tuple):  # LSTM, state:(h, c)
                    state = (state[0].detach(), state[1].detach())
                else:
                    state = state.detach()

            # output: 形状为(num_steps * batch_size, vocab_size)
            (output, state) = model(X, state)

            # Y的形状是(batch_size, num_steps)，转置后再变成长度为
            # batch * num_steps 的向量，这样跟输出的行一一对应
            y = torch.transpose(Y, 0, 1).contiguous().view(-1)
            l = loss(output, y.long())

            optimizer.zero_grad()
            l.backward()

            # 梯度裁剪
            d2l.grad_clipping(model.parameters(), clipping_theta, device)
            optimizer.step()

            l_sum += l.item() * y.shape[0]
            n += y.shape[0]
            pass

        try:
            perplexity = math.exp(l_sum / n)
        except OverflowError:
            perplexity = float('inf')

        if (epoch + 1) % pred_period == 0:
            print('epoch %d, perplexity %f, time %.2f sec' % (epoch + 1,
                                                              perplexity,
                                                              time.time() - start))
            for prefix in prefixes:
                print(' -', predict_rnn_pytorch(prefix, pred_len,
                                                model, vocab_size,
                                                device, idx_to_char,
                                                char_to_idx))
                pass
            pass
        # 结束epoch循环
        pass
    pass


def test_1():
    model = RNNModel(rnn_layer, vocab_size).to(device)

    # 预测
    predict_rnn_pytorch('分开', 10, model, vocab_size,
                        device,
                        idx_to_char,
                        char_to_idx
                        )
    pass


def test_2():
    model = RNNModel(rnn_layer, vocab_size).to(device)

    num_epochs = 250
    batch_size = 32
    lr = 1e-3
    clipping_theta = 1e-2

    pred_period = 50
    pred_len = 50
    prefixes = ['分开', '不分开']

    # 训练
    train_and_predict_rnn_pytorch(model, num_hiddens, vocab_size, device,
                                  corpus_indices, idx_to_char, char_to_idx,
                                  num_epochs, num_steps, lr, clipping_theta,
                                  batch_size, pred_period, pred_len, prefixes
                                  )
    pass


def main():
    test_1()
    pass


if __name__ == '__main__':
    main()
