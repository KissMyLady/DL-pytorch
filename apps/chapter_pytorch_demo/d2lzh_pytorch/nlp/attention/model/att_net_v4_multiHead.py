# coding:utf-8
# Author:mylady
# 2023/7/28 14:14
import torch
import math
from torch import nn


def sequence_mask(X, valid_len, value=0):
    """
    Mask irrelevant entries in sequences.
    屏蔽序列中不相关的条目
    Defined in :numref:`sec_seq2seq_decoder`
    """
    maxlen = X.size(1)
    mask = torch.arange((maxlen),
                        dtype=torch.float32,
                        device=X.device
                        )[None, :] < valid_len[:, None]
    X[~mask] = value
    return X


def masked_softmax(X, valid_lens):
    """
    通过在最后一个轴上掩蔽元素来执行softmax操作
    """
    # X:3D张量，valid_lens:1D或2D张量
    if valid_lens is None:
        return nn.functional.softmax(X, dim=-1)
    else:
        shape = X.shape
        if valid_lens.dim() == 1:
            valid_lens = torch.repeat_interleave(valid_lens, shape[1])
        else:
            valid_lens = valid_lens.reshape(-1)
        # 最后一轴上被掩蔽的元素使用一个非常大的负值替换，从而其softmax输出为0
        X = sequence_mask(X.reshape(-1, shape[-1]),
                          valid_lens,
                          value=-1e6)
        res = nn.functional.softmax(X.reshape(shape), dim=-1)
        return res


# 缩放点积注意力
class DotProductAttention(nn.Module):
    def __init__(self, dropout, **kwargs):
        super(DotProductAttention, self).__init__(**kwargs)
        self.dropout = nn.Dropout(dropout)

    def forward(self, queries, keys, values, valid_lens=None):
        d = queries.shape[-1]
        scores = torch.bmm(queries, keys.transpose(1, 2)) / math.sqrt(d)

        self.attention_weights = masked_softmax(scores, valid_lens)

        y_1 = self.dropout(self.attention_weights)
        y = torch.bmm(y_1, values)
        return y


def transpose_qkv(X, num_heads):
    """
    为了多注意力头的并行计算而变换形状
    """
    X = X.reshape(X.shape[0], X.shape[1], num_heads, -1)
    X = X.permute(0, 2, 1, 3)
    return X.reshape(-1, X.shape[2], X.shape[3])


def transpose_output(X, num_heads):
    """逆转 transpose_qkv 函数的操作"""
    X = X.reshape(-1, num_heads, X.shape[1], X.shape[2])
    X = X.permute(0, 2, 1, 3)
    return X.reshape(X.shape[0], X.shape[1], -1)


# 多头注意力
class MultiHeadAttention(nn.Module):
    def __init__(self,
                 key_size=100, query_size=100, value_size=100,
                 num_hiddens=100, num_heads=5, dropout=0.5,
                 bias=False, **kwargs):
        super(MultiHeadAttention, self).__init__(**kwargs)
        self.num_heads = num_heads
        self.attention = DotProductAttention(dropout)  # 缩放点积注意力
        self.W_q = nn.Linear(query_size, num_hiddens, bias=bias)
        self.W_k = nn.Linear(key_size, num_hiddens, bias=bias)
        self.W_v = nn.Linear(value_size, num_hiddens, bias=bias)
        self.W_o = nn.Linear(num_hiddens, num_hiddens, bias=bias)

    def forward(self, queries, keys, values, valid_lens):
        # 输入变换形状
        queries = transpose_qkv(self.W_q(queries), self.num_heads)
        keys = transpose_qkv(self.W_k(keys), self.num_heads)
        values = transpose_qkv(self.W_v(values), self.num_heads)

        if valid_lens is not None:
            valid_lens = torch.repeat_interleave(valid_lens,
                                                 repeats=self.num_heads,
                                                 dim=0)
            pass

        output = self.attention(queries, keys, values, valid_lens)
        # 输出变换形状
        output_concat = transpose_output(output, self.num_heads)
        return self.W_o(output_concat)


def get_net():
    num_hiddens = 100
    num_heads = 5

    attention = MultiHeadAttention(key_size=num_hiddens, query_size=num_hiddens, value_size=num_hiddens,
                                   num_hiddens=num_hiddens, num_heads=num_heads,
                                   dropout=0.5)

    return attention


def main():
    pass


if __name__ == "__main__":
    main()
