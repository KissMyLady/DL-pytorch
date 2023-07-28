# coding:utf-8
# Author:mylady
# 2023/7/28 11:14
import torch
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


class Decoder(nn.Module):
    """
    The base decoder interface for the encoder-decoder architecture.
    编码器-解码器架构的基本解码器接口。
    Defined in :numref:`sec_encoder-decoder`
    """

    def __init__(self, **kwargs):
        super(Decoder, self).__init__(**kwargs)

    def init_state(self, enc_outputs, *args):
        raise NotImplementedError

    def forward(self, X, state):
        raise NotImplementedError


class Encoder(nn.Module):
    """
    The base encoder interface for the encoder-decoder architecture.
    编码器-解码器架构的基本编码器接口。
    """

    def __init__(self, **kwargs):
        super(Encoder, self).__init__(**kwargs)

    def forward(self, X, *args):
        raise NotImplementedError


class AdditiveAttention(nn.Module):
    def __init__(self, key_size, query_size, num_hiddens, dropout, **kwargs):
        super(AdditiveAttention, self).__init__(**kwargs)
        self.W_k = nn.Linear(key_size, num_hiddens, bias=False)
        self.W_q = nn.Linear(query_size, num_hiddens, bias=False)
        self.w_v = nn.Linear(num_hiddens, 1, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, queries, keys, values, valid_lens):
        queries, keys = self.W_q(queries), self.W_k(keys)
        features = queries.unsqueeze(2) + keys.unsqueeze(1)
        features = torch.tanh(features)
        scores = self.w_v(features).squeeze(-1)
        self.attention_weights = masked_softmax(scores, valid_lens)
        return torch.bmm(self.dropout(self.attention_weights), values)


# 带有注意力机制解码器的基本接口
class AttentionDecoder(Decoder):
    def __init__(self, **kwargs):
        super(AttentionDecoder, self).__init__(**kwargs)

    @property
    def attention_weights(self):
        raise NotImplementedError


class Seq2SeqAttentionDecoder(AttentionDecoder):

    def __init__(self, vocab_size, embed_size,
                 num_hiddens, num_layers, dropout=0, **kwargs):
        super(Seq2SeqAttentionDecoder, self).__init__(**kwargs)
        self.attention = AdditiveAttention(num_hiddens,
                                           num_hiddens,
                                           num_hiddens,
                                           dropout)
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.rnn = nn.GRU(embed_size + num_hiddens,
                          num_hiddens, num_layers,
                          dropout=dropout)
        self.dense = nn.Linear(num_hiddens, vocab_size)

    def init_state(self, enc_outputs, enc_valid_lens, *args):
        # outputs的形状为(batch_size，num_steps，num_hiddens).
        # hidden_state的形状为(num_layers，batch_size，num_hiddens)
        outputs, hidden_state = enc_outputs
        return (outputs.permute(1, 0, 2), hidden_state, enc_valid_lens)

    def forward(self, X, state):
        enc_outputs, hidden_state, enc_valid_lens = state
        X = self.embedding(X).permute(1, 0, 2)
        outputs, self._attention_weights = [], []
        for x in X:
            query = torch.unsqueeze(hidden_state[-1], dim=1)
            context = self.attention(query,
                                     enc_outputs,
                                     enc_outputs,
                                     enc_valid_lens)
            x = torch.cat((context, torch.unsqueeze(x, dim=1)), dim=-1)
            out, hidden_state = self.rnn(x.permute(1, 0, 2), hidden_state)
            outputs.append(out)
            self._attention_weights.append(self.attention.attention_weights)
            pass

        # 全连接层变换后，outputs的形状为
        # (num_steps,batch_size,vocab_size)
        outputs = self.dense(torch.cat(outputs, dim=0))

        res_y1 = outputs.permute(1, 0, 2)
        res_y2 = [enc_outputs,
                  hidden_state,
                  enc_valid_lens]
        return res_y1, res_y2

    @property
    def attention_weights(self):
        return self._attention_weights


class Seq2SeqEncoder(Encoder):
    """
    The RNN encoder for sequence to sequence learning.
    用于序列到序列学习的RNN编码器。
    Defined in :numref:`sec_seq2seq`
    """
    def __init__(self, vocab_size, embed_size, num_hiddens, num_layers, dropout=0, **kwargs):
        super(Seq2SeqEncoder, self).__init__(**kwargs)
        # Embedding layer
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.rnn = nn.GRU(embed_size, num_hiddens, num_layers, dropout=dropout)

    def forward(self, X, *args):
        # The output `X` shape: (`batch_size`, `num_steps`, `embed_size`)
        X = self.embedding(X)
        # In RNN models, the first axis corresponds to time steps
        X = X.permute(1, 0, 2)
        # When state is not mentioned, it defaults to zeros
        output, state = self.rnn(X)
        # `output` shape: (`num_steps`, `batch_size`, `num_hiddens`)
        # `state` shape: (`num_layers`, `batch_size`, `num_hiddens`)
        return output, state


class EncoderDecoder(nn.Module):
    """
    The base class for the encoder-decoder architecture.
    编码器-解码器体系结构的基类。
    Defined in :numref:`sec_encoder-decoder`
    """

    def __init__(self, encoder, decoder, **kwargs):
        super(EncoderDecoder, self).__init__(**kwargs)
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, enc_X, dec_X, *args):
        enc_outputs = self.encoder(enc_X, *args)
        dec_state = self.decoder.init_state(enc_outputs, *args)
        return self.decoder(dec_X, dec_state)


def get_net():
    embed_size = 32
    num_hiddens = 32
    num_layers = 2
    dropout = 0.1

    batch_size = 64
    num_steps = 10
    # train_iter, src_vocab, tgt_vocab = load_data_nmt(batch_size, num_steps)

    src_vocab = [i for i in range(184)]  # 184
    tgt_vocab = [i for i in range(201)]  # 201

    # 32 x 32 x 2
    encoder = Seq2SeqEncoder(len(src_vocab),
                             embed_size,
                             num_hiddens,
                             num_layers,
                             dropout)

    decoder = Seq2SeqAttentionDecoder(len(tgt_vocab),
                                      embed_size,
                                      num_hiddens,
                                      num_layers,
                                      dropout)

    net = EncoderDecoder(encoder, decoder)
    return net


def test_1():
    from torchsummary import summary
    net = get_net()
    print(net)

    summary(net)
    """
    EncoderDecoder(
      (encoder): Seq2SeqEncoder(
        (embedding): Embedding(184, 32)
        (rnn): GRU(32, 32, num_layers=2, dropout=0.1)
      )
      (decoder): Seq2SeqAttentionDecoder(
        (attention): AdditiveAttention(
          (W_k): Linear(in_features=32, out_features=32, bias=False)
          (W_q): Linear(in_features=32, out_features=32, bias=False)
          (w_v): Linear(in_features=32, out_features=1, bias=False)
          (dropout): Dropout(p=0.1, inplace=False)
        )
        (embedding): Embedding(201, 32)
        (rnn): GRU(64, 32, num_layers=2, dropout=0.1)
        (dense): Linear(in_features=32, out_features=201, bias=True)
      )
    )
    
    =================================================================
    Layer (type:depth-idx)                   Param #
    =================================================================
    ├─Seq2SeqEncoder: 1-1                    --
    |    └─Embedding: 2-1                    5,888
    |    └─GRU: 2-2                          12,672  # 怎么计算出来的?
    ├─Seq2SeqAttentionDecoder: 1-2           --
    |    └─AdditiveAttention: 2-3            --
    |    |    └─Linear: 3-1                  1,024
    |    |    └─Linear: 3-2                  1,024
    |    |    └─Linear: 3-3                  32
    |    |    └─Dropout: 3-4                 --
    |    └─Embedding: 2-4                    6,432
    |    └─GRU: 2-5                          15,744
    |    └─Linear: 2-6                       6,633
    =================================================================
    Total params: 49,449
    Trainable params: 49,449
    Non-trainable params: 0
    =================================================================

    """
    pass


def main():
    test_1()
    pass


if __name__ == "__main__":
    main()
