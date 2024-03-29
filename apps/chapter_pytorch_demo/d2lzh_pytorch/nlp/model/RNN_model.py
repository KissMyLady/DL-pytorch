import sys
sys.path.append("..")
sys.path.append("../..")

import torch
from torch import nn
from torch.nn import functional as F
from d2lzh_pytorch.myUtils import try_gpu
import os


# 循环神经网络的简洁实现
class RNNModel(nn.Module):
    """循环神经网络模型"""
    def __init__(self, rnn_layer, vocab_size, **kwargs):
        super(RNNModel, self).__init__(**kwargs)
        self.rnn = rnn_layer
        self.vocab_size = vocab_size
        self.num_hiddens = self.rnn.hidden_size
        # 如果RNN是双向的（之后将介绍），num_directions应该是2，否则应该是1
        if not self.rnn.bidirectional:
            self.num_directions = 1
            self.linear = nn.Linear(self.num_hiddens, self.vocab_size)
        else:
            self.num_directions = 2
            self.linear = nn.Linear(self.num_hiddens * 2, self.vocab_size)

    def forward(self, inputs, state):
        X = F.one_hot(inputs.T.long(), self.vocab_size)
        X = X.to(torch.float32)
        Y, state = self.rnn(X, state)
        # 全连接层首先将Y的形状改为(时间步数*批量大小,隐藏单元数)
        # 它的输出形状是(时间步数*批量大小,词表大小)。
        output = self.linear(Y.reshape((-1, Y.shape[-1])))
        return output, state

    def begin_state(self, device, batch_size=1):
        if not isinstance(self.rnn, nn.LSTM):
            # nn.GRU以张量作为隐状态
            zero_y = torch.zeros((self.num_directions * self.rnn.num_layers,
                                  batch_size, self.num_hiddens
                                 ), device=device)
            return zero_y
        else:
            # nn.LSTM以元组作为隐状态
            zery_y = (torch.zeros((self.num_directions * self.rnn.num_layers,
                                   batch_size, self.num_hiddens
                                   ), device=device),
                      torch.zeros((self.num_directions * self.rnn.num_layers,
                                   batch_size, self.num_hiddens), 
                                   device=device))
            return zery_y


# 返回RNN神经网络
def get_RNNModel():
    from d2lzh_pytorch.nlp.load_data.load_time_machine import load_data_time_machine

    batch_size = 32
    num_steps = 35
    BASE_PATH = "/mnt/g1t/ai_data/Datasets_on_HHD/NLP/text_data"

    # 加载
    txtPath = os.path.join(BASE_PATH, "贾平凹-山本.txt")
    stopwords_file = os.path.join(BASE_PATH, "stopwords.txt")

    # 返回封装的数据
    train_iter, vocab = load_data_time_machine(batch_size, num_steps, 
                                               txtPath, stopwords_file)

    device = try_gpu()

    num_hiddens = 256

    # RNN层
    rnn_layer = nn.RNN(len(vocab), num_hiddens)

    # RNN模型
    net = RNNModel(rnn_layer, vocab_size=len(vocab))
    return net


def test_1():
    
    pass


def main():
    pass


if __name__ == "__main__":
    main()
    pass
