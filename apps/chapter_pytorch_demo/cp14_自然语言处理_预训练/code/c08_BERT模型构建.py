# Author: theOracle
# Datetime: 2023/4/20 12:59 AM 
import torch
from torch import nn
# from d2l import torch as d2l

import sys

sys.path.append("../..")
import d2lzh_pytorch.torch as d2l


def get_tokens_and_segments(tokens_a, tokens_b=None):
    """获取输入序列的词元及其片段索引"""
    tokens = ['<cls>'] + tokens_a + ['<sep>']
    # 0和1分别标记片段A和B
    segments = [0] * (len(tokens_a) + 2)
    if tokens_b is not None:
        tokens += tokens_b + ['<sep>']
        segments += [1] * (len(tokens_b) + 1)

    return tokens, segments


# BERT编码器
class BERTEncoder(nn.Module):

    def __init__(self, vocab_size, num_hiddens, norm_shape, ffn_num_input,
                 ffn_num_hiddens, num_heads, num_layers, dropout,
                 max_len=1000, key_size=768, query_size=768, value_size=768,
                 **kwargs):
        super(BERTEncoder, self).__init__(**kwargs)
        self.token_embedding = nn.Embedding(vocab_size, num_hiddens)
        self.segment_embedding = nn.Embedding(2, num_hiddens)
        self.blks = nn.Sequential()

        for i in range(num_layers):
            # Transform的Block
            encoderBlock = d2l.EncoderBlock(key_size, query_size, value_size, num_hiddens, norm_shape,
                                            ffn_num_input, ffn_num_hiddens, num_heads, dropout, True)
            # 叠加
            self.blks.add_module(f"{i}", encoderBlock)

        # 在BERT中，位置嵌入是可学习的，因此我们创建一个足够长的位置嵌入参数(可学习的位置编码)
        self.pos_embedding = nn.Parameter(torch.randn(1, max_len, num_hiddens))

    def forward(self, tokens, segments, valid_lens):
        # 在以下代码段中，X的形状保持不变：（批量大小，最大序列长度，num_hiddens）
        X = self.token_embedding(tokens) + \
            self.segment_embedding(segments)
        X = X + self.pos_embedding.data[:, :X.shape[1], :]
        for blk in self.blks:
            X = blk(X, valid_lens)
        return X


def res_encoded_X():
    vocab_size = 10000
    num_hiddens = 768
    ffn_num_hiddens = 1024
    num_heads = 4

    norm_shape = [768]
    ffn_num_input = 768
    num_layers = 2
    dropout = 0.2

    encoder = BERTEncoder(vocab_size, num_hiddens, norm_shape, ffn_num_input,
                          ffn_num_hiddens, num_heads, num_layers, dropout)

    # 这里模拟表示两个句子token输入
    tokens = torch.randint(0, vocab_size, (2, 8))
    segments = torch.tensor([[0, 0, 0, 0, 1, 1, 1, 1],
                             [0, 0, 0, 1, 1, 1, 1, 1]])

    # 模拟句子输入到编码器, 将句子编码
    encoded_X = encoder(tokens, segments, None)
    # 输出 [batch_size2 x 句子长度8 x num_hiddens 768] = [2, 8, 768]
    return encoded_X


def test_BERTEncoder():
    encoded_X = res_encoded_X()

    print(encoded_X.shape)
    print('将句子编码后的输出: \n', encoded_X)
    pass


class MaskLM(nn.Module):
    """BERT的掩蔽语言模型任务"""

    def __init__(self, vocab_size, num_hiddens, num_inputs=768, **kwargs):
        super(MaskLM, self).__init__(**kwargs)
        self.mlp = nn.Sequential(nn.Linear(num_inputs, num_hiddens),
                                 nn.ReLU(),
                                 nn.LayerNorm(num_hiddens),
                                 nn.Linear(num_hiddens, vocab_size))

    def forward(self, X, pred_positions):
        num_pred_positions = pred_positions.shape[1]
        pred_positions = pred_positions.reshape(-1)
        batch_size = X.shape[0]
        batch_idx = torch.arange(0, batch_size)
        # 假设batch_size=2，num_pred_positions=3
        # 那么batch_idx是np.array（[0,0,0,1,1,1]）
        batch_idx = torch.repeat_interleave(batch_idx, num_pred_positions)
        masked_X = X[batch_idx, pred_positions]
        masked_X = masked_X.reshape((batch_size, num_pred_positions, -1))
        mlm_Y_hat = self.mlp(masked_X)
        return mlm_Y_hat


def test_MaskLM():
    vocab_size = 10000
    num_hiddens = 768
    encoded_X = res_encoded_X()

    # 实例化
    mlm = MaskLM(vocab_size, num_hiddens)

    # 输入三个 对应预测位置
    mlm_positions = torch.tensor([[1, 5, 2],
                                  [6, 1, 5]])

    # 对于每个预测
    mlm_Y_hat = mlm(encoded_X, mlm_positions)

    print(mlm_Y_hat.shape)
    print(mlm_Y_hat)

    # 损失计算
    mlm_Y = torch.tensor([[7, 8, 9],
                          [10, 20, 30]])
    loss = nn.CrossEntropyLoss(reduction='none')
    mlm_l = loss(mlm_Y_hat.reshape((-1, vocab_size)), mlm_Y.reshape(-1))

    print('loss计算: ', mlm_l.shape)
    print('mlm_l: \n', mlm_l)
    pass


class NextSentencePred(nn.Module):
    """BERT的下一句预测任务"""

    def __init__(self, num_inputs, **kwargs):
        super(NextSentencePred, self).__init__(**kwargs)
        self.output = nn.Linear(num_inputs, 2)

    def forward(self, X):
        # X的形状：(batchsize,num_hiddens)
        return self.output(X)


def test_NextSentencePred():
    encoded_X = res_encoded_X()

    encoded_X = torch.flatten(encoded_X, start_dim=1)

    # NSP的输入形状:(batchsize，num_hiddens)
    nsp = NextSentencePred(encoded_X.shape[-1])

    nsp_Y_hat = nsp(encoded_X)

    print("nsp_Y_hat.shape: ", nsp_Y_hat.shape)
    print("nsp_Y_hat: \n", nsp_Y_hat)
    pass


# BERT模型
class BERTModel(nn.Module):

    def __init__(self, vocab_size, num_hiddens, norm_shape, ffn_num_input,
                 ffn_num_hiddens, num_heads, num_layers, dropout,
                 max_len=1000, key_size=768, query_size=768, value_size=768,
                 hid_in_features=768, mlm_in_features=768,
                 nsp_in_features=768):
        super(BERTModel, self).__init__()
        # BERT编码器
        self.encoder = BERTEncoder(vocab_size, num_hiddens, norm_shape,
                                   ffn_num_input, ffn_num_hiddens,
                                   num_heads, num_layers,
                                   dropout, max_len=max_len, key_size=key_size,
                                   query_size=query_size, value_size=value_size)

        self.hidden = nn.Sequential(nn.Linear(hid_in_features, num_hiddens),
                                    nn.Tanh())  # 双正切函数
        self.mlm = MaskLM(vocab_size, num_hiddens, mlm_in_features)
        self.nsp = NextSentencePred(nsp_in_features)

    def forward(self, tokens, segments, valid_lens=None, pred_positions=None):
        encoded_X = self.encoder(tokens, segments, valid_lens)
        if pred_positions is not None:
            mlm_Y_hat = self.mlm(encoded_X, pred_positions)
        else:
            mlm_Y_hat = None
        # 用于下一句预测的多层感知机分类器的隐藏层，0是“<cls>”标记的索引
        nsp_Y_hat = self.nsp(self.hidden(encoded_X[:, 0, :]))
        return encoded_X, mlm_Y_hat, nsp_Y_hat


def main():
    # test_BERTEncoder()  # 编码器测试
    test_MaskLM()
    # test_NextSentencePred()  # 下一句预测
    pass


if __name__ == '__main__':
    main()
