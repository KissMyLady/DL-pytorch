# coding:utf-8
# Author:mylady
# Datetime:2023/4/23 4:02
from torch import nn
# from d2l import torch as d2l

import sys

sys.path.append("../..")
import d2lzh_pytorch.torch as d2l

import d2lzh_pytorch.nlp.load_chinese_txt_data as load_chinese  # 加载中文训练数据
import d2lzh_pytorch.rnn_train_chinese as train_chinese  # 加载训练模块
import d2lzh_pytorch.rnn_model as RNNModel  # 加载RNN模型


# 驱动
device = d2l.try_gpu()


def run():
    batch_size = 32
    num_steps = 35

    # 加载数据
    # train_iter, vocab = d2l.load_data_time_machine(batch_size, num_steps)
    txtPath = "/mnt/g2t/ai_data/txtBook/贾平凹-山本.txt"
    stopwords_file = "/mnt/g2t/ai_data/txtBook/stopwords.txt"
    train_iter, vocab = load_chinese.load_data_time_machine(batch_size,
                                                            num_steps,
                                                            txtPath,
                                                            stopwords_file)
    # 网络层定义
    num_hiddens = 256
    rnn_layer = nn.RNN(len(vocab), num_hiddens)

    # RNN模型
    net = RNNModel(rnn_layer, vocab_size=len(vocab))
    net = net.to(device)

    num_epochs = 500
    lr = 1

    # 开始训练
    train_chinese.train_ch8(net,
                            train_iter,
                            vocab,
                            lr,
                            num_epochs,
                            device
                            )

    # 预测
    train_chinese.predict_ch8('陆菊人', 20,
                              net, vocab, device
                              )

    pass


def main():
    run()
    pass


if __name__ == '__main__':
    main()
