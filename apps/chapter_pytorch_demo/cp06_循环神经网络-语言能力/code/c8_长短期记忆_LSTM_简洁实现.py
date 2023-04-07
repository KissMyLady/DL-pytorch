import numpy as np
import torch
from torch import nn, optim
import torch.nn.functional as F

# import sys
# sys.path.append("..")
from apps.chapter_pytorch_demo import d2lzh_pytorch as d2l

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# 加载歌词数据
(corpus_indices,
 char_to_idx,
 idx_to_char,
 vocab_size) = d2l.load_data_jay_lyrics(read_file_path=r"../../data/jaychou_lyrics.txt")

num_inputs = vocab_size
num_hiddens = 256
num_outputs = vocab_size

# 参数配置

num_epochs = 160
num_steps = 35
batch_size = 32
lr = 1e2
clipping_theta = 1e-2

pred_period = 40
pred_len = 50
prefixes = ['分开', '不分开']

# 长短期记忆
lstm_layer = nn.LSTM(input_size=vocab_size, hidden_size=num_hiddens)

# 模型
model = d2l.RNNModel(lstm_layer, vocab_size)


def test_1():
    import math
    math.exp(1548159.46875)
    pass


def main():
    # 训练
    d2l.train_and_predict_rnn_pytorch(model, num_hiddens, vocab_size, device,
                                      corpus_indices, idx_to_char, char_to_idx,
                                      num_epochs, num_steps,
                                      lr, clipping_theta, batch_size,
                                      pred_period,
                                      pred_len,
                                      prefixes)
    pass


if __name__ == '__main__':
    main()
