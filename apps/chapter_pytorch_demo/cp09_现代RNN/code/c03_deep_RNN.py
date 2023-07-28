# coding:utf-8
# Author:mylady
# Datetime:2023/4/23 5:21
from torch_package import nn
# from d2l import torch as d2l

import sys

sys.path.append("../..")
import d2lzh_pytorch.torch as d2l

import d2lzh_pytorch.nlp.load_chinese_txt_data as load_chinese  # 加载中文训练数据
import d2lzh_pytorch.rnn_train_chinese as train_chinese  # 加载训练模块
from d2lzh_pytorch.rnn_model import RNNModel  # 加载RNN模型

device = d2l.try_gpu()


def run():
    batch_size, num_steps = 32, 35

    # 加载数据
    # train_iter, vocab = d2l.load_data_time_machine(batch_size, num_steps)
    txtPath = "/mnt/g2t/ai_data/txtBook/贾平凹-山本.txt"
    stopwords_file = "/mnt/g2t/ai_data/txtBook/stopwords.txt"
    train_iter, vocab = load_chinese.load_data_time_machine(batch_size,
                                                            num_steps,
                                                            txtPath,
                                                            stopwords_file)

    vocab_size = len(vocab)
    num_hiddens = 256
    num_layers = 2
    num_inputs = vocab_size

    # LSTM 长短期记忆网络
    lstm_layer = nn.LSTM(num_inputs,
                         num_hiddens,
                         num_layers)

    # 模型定义
    model = RNNModel(lstm_layer,
                     len(vocab))

    # 转移到GPU
    model = model.to(device)

    # 训练
    num_epochs = 1000
    lr = 0.2
    train_chinese.train_ch8(model,
                            train_iter,
                            vocab,
                            lr * 1.0,
                            num_epochs,
                            device)
    pass


def main():
    run()
    pass


if __name__ == '__main__':
    main()
