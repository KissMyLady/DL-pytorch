# coding:utf-8
# Author:mylady
# Datetime:2023/4/23 8:05
import torch
from torch import nn


class Encoder(nn.Module):
    """
    编码器-解码器架构的基本编码器接口
    """
    def __init__(self, **kwargs):
        super(Encoder, self).__init__(**kwargs)

    def forward(self, X, *args):
        raise NotImplementedError


class Decoder(nn.Module):
    """
    编码器-解码器架构的基本解码器接口
    """

    def __init__(self, **kwargs):
        super(Decoder, self).__init__(**kwargs)

    def init_state(self, enc_outputs, *args):
        raise NotImplementedError

    def forward(self, X, state):
        raise NotImplementedError


class EncoderDecoder(nn.Module):
    """
    编码器-解码器架构的基类
    """
    def __init__(self, encoder, decoder, **kwargs):
        super(EncoderDecoder, self).__init__(**kwargs)
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, enc_X, dec_X, *args):
        enc_outputs = self.encoder(enc_X, *args)
        dec_state = self.decoder.init_state(enc_outputs, *args)
        y_hat = self.decoder(dec_X, dec_state)
        return y_hat


class Seq2SeqEncoder(Encoder):
    """
    用于序列到序列学习的循环神经网络编码器
    """

    def __init__(self, vocab_size, embed_size, num_hiddens, num_layers,
                 dropout=0, **kwargs):
        super(Seq2SeqEncoder, self).__init__(**kwargs)
        # 嵌入层
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.rnn = nn.GRU(embed_size,
                          num_hiddens,
                          num_layers,
                          dropout=dropout)

    def forward(self, X, *args):
        # 输出'X'的形状：(batch_size,num_steps,embed_size)
        X = self.embedding(X)
        # 在循环神经网络模型中，第一个轴对应于时间步
        X = X.permute(1, 0, 2)
        # 如果未提及状态，则默认为0
        output, state = self.rnn(X)
        # output的形状:(num_steps,batch_size,num_hiddens)
        # state的形状:(num_layers,batch_size,num_hiddens)
        return output, state


class Seq2SeqDecoder(Decoder):
    """
    用于序列到序列学习的循环神经网络解码器
    """

    def __init__(self, vocab_size, embed_size, num_hiddens,
                 num_layers, dropout=0, **kwargs):
        super(Seq2SeqDecoder, self).__init__(**kwargs)
        self.embedding = nn.Embedding(vocab_size, embed_size)

        self.rnn = nn.GRU(embed_size + num_hiddens,
                          num_hiddens,
                          num_layers,
                          dropout=dropout)
        self.dense = nn.Linear(num_hiddens, vocab_size)

    def init_state(self, enc_outputs, *args):
        return enc_outputs[1]

    def forward(self, X, state):
        # 输出'X'的形状：(batch_size,num_steps,embed_size)
        X = self.embedding(X).permute(1, 0, 2)
        # 广播context，使其具有与X相同的num_steps
        context = state[-1].repeat(X.shape[0], 1, 1)
        X_and_context = torch.cat((X, context), 2)
        output, state = self.rnn(X_and_context, state)
        output = self.dense(output).permute(1, 0, 2)
        # output的形状:(batch_size,num_steps,vocab_size)
        # state的形状:(num_layers,batch_size,num_hiddens)
        return output, state


# 返回模型
def get_EncoderDecoder_model():
    """
    返回一个 EncoderDecoder 模型
    """
    from d2lzh_pytorch.nlp.load_data.load_nmt_data import load_data_nmt

    batch_size = 64
    num_steps = 10

    # 数据加载
    train_iter, src_vocab, tgt_vocab = load_data_nmt(batch_size, num_steps)

    # 模型超参
    embed_size = 32
    num_hiddens = 32
    num_layers = 2
    dropout = 0.1

    # 编码器-解码器
    encoder = Seq2SeqEncoder(len(src_vocab), embed_size, num_hiddens, num_layers, dropout)
    decoder = Seq2SeqDecoder(len(tgt_vocab), embed_size, num_hiddens, num_layers, dropout)

    net = EncoderDecoder(encoder, decoder)
    return net


def main():
    pass


if __name__ == '__main__':
    # main()
    pass