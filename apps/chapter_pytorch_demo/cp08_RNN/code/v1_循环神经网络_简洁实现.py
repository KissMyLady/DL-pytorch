# coding:utf-8
# Author:mylady
# Datetime:2023/6/28 13:35

import torch_package
import os
from torch_package import nn
from torch_package.nn import functional as F
# from d2l import torch as d2l

import sys
sys.path.append("..")
import d2lzh_pytorch.torch as d2l

# 数据加载
from d2lzh_pytorch.nlp.load_data.load_time_machine import load_data_time_machine

# 模型加载
from d2lzh_pytorch.nlp.model.RNN_model import RNNModel

# 训练加载, 预测加载
from d2lzh_pytorch.nlp.train_model.train_ch8 import train_ch8, predict_ch8


# 保存模型
def saveModel(net):
    net = net.to("cpu")
    import datetime
    str_time = str(datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))
    save_path = 'concise_RNN_%s.pt' % str_time
    torch.save(net, save_path)  # 全保存 39M
    print("训练完毕, 模型 %s 已保存至当前路径" % save_path)
    pass


def main():
    # 加载数据
    batch_size = 32
    num_steps = 35

    BASE_PATH = "/mnt/g1t/ai_data/Datasets_on_HHD/NLP/text_data"
    txtPath = os.path.join(BASE_PATH, "贾平凹-山本.txt")
    stopwords_file = os.path.join(BASE_PATH, "stopwords.txt")

    # 返回封装的数据
    train_iter, vocab = load_data_time_machine(batch_size, 
                                               num_steps, 
                                               txtPath, 
                                               stopwords_file)


    # 定义模型
    device = d2l.try_gpu()
    num_hiddens = 256
    rnn_layer = nn.RNN(len(vocab), num_hiddens)
    # 循环神经网络模型
    net = RNNModel(rnn_layer, vocab_size=len(vocab))
    net = net.to(device)

    # 训练
    num_epochs = 2000
    lr = 0.1
    train_ch8(net, train_iter,
            vocab, lr,
            num_epochs,
            device)

    # 预测
    predict_ch8('保安队', 20, 
                net, vocab, device)
    
    saveModel(net)
    pass


if __name__ == '__main__':
    main()
