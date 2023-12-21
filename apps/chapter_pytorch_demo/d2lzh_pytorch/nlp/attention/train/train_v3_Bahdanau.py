# coding:utf-8
# Author:mylady
# 2023/7/28 11:20
import sys

sys.path.append("..")
sys.path.append("../..")
sys.path.append("../../..")
sys.path.append("../../../..")

from d2lzh_pytorch import myPolt
from d2lzh_pytorch.torch_package import plot
from matplotlib import pyplot as plt

import torch
from torch import nn

from d2lzh_pytorch.nlp.attention.model.att_net_v3_Bahdanau import get_net
from d2lzh_pytorch.myUtils import try_gpu
from d2lzh_pytorch.nlp.train.train_seq2seq import train_seq2seq
from d2lzh_pytorch.nlp.load_data.load_nmt_data import load_data_nmt


def save_net(net):
    # 保存模型
    net = net.to("cpu")

    import datetime
    import os
    str_time = str(datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))

    # 重命名模型
    save_path = 'pretrain_BERT_classifier_%s.pt' % str_time

    torch.save(net, save_path)  # 全保存
    print("训练完毕, 模型 %s 已保存至当前路径" % save_path)
    print("模型大小是: %0.2f M" % (os.path.getsize(save_path) / 1024 / 1024))
    pass


def train_bahdanau():
    batch_size = 64
    num_steps = 10
    train_iter, src_vocab, tgt_vocab = load_data_nmt(batch_size, num_steps)

    # 超参数
    lr = 0.005
    num_epochs = 250
    device = try_gpu()

    # 网络
    net = get_net()

    # 开始训练
    train_seq2seq(net, train_iter,
                  lr,
                  num_epochs,
                  tgt_vocab,
                  device)

    # 保存模型
    save_net(net)
    pass


def main():
    # 开始训练
    train_bahdanau()
    pass


if __name__ == "__main__":
    main()
