# coding:utf-8
# Author:mylady
# Datetime:2023/4/25 20:02
import torch
from torch import nn
from torch.nn import functional as F
import sys

sys.path.append(".")
sys.path.append("..")
sys.path.append("../..")
from d2lzh_pytorch.BERT_unit_v3.load_snli import load_data_snli
from d2lzh_pytorch.myUtils import try_all_gpus
from d2lzh_pytorch.glove.use_glove import TokenEmbedding
from d2lzh_pytorch.CNN.train_cnn import train_ch13


def mlp(num_inputs, num_hiddens, flatten):
    net = []
    net.append(nn.Dropout(0.2))
    net.append(nn.Linear(num_inputs, num_hiddens))
    net.append(nn.ReLU())
    if flatten:
        net.append(nn.Flatten(start_dim=1))
    net.append(nn.Dropout(0.2))
    net.append(nn.Linear(num_hiddens, num_hiddens))
    net.append(nn.ReLU())
    if flatten:
        net.append(nn.Flatten(start_dim=1))
    return nn.Sequential(*net)


class Attend(nn.Module):

    def __init__(self, num_inputs, num_hiddens, **kwargs):
        super(Attend, self).__init__(**kwargs)
        self.f = mlp(num_inputs, num_hiddens, flatten=False)

    def forward(self, A, B):
        # A/B的形状：（批量大小，序列A/B的词元数，embed_size）
        # f_A/f_B的形状：（批量大小，序列A/B的词元数，num_hiddens）
        f_A = self.f(A)
        f_B = self.f(B)
        # e的形状：（批量大小，序列A的词元数，序列B的词元数）
        e = torch.bmm(f_A, f_B.permute(0, 2, 1))
        # beta的形状：（批量大小，序列A的词元数，embed_size），
        # 意味着序列B被软对齐到序列A的每个词元(beta的第1个维度)
        beta = torch.bmm(F.softmax(e, dim=-1), B)
        # beta的形状：（批量大小，序列B的词元数，embed_size），
        # 意味着序列A被软对齐到序列B的每个词元(alpha的第1个维度)
        alpha = torch.bmm(F.softmax(e.permute(0, 2, 1), dim=-1), A)
        return beta, alpha


class Compare(nn.Module):

    def __init__(self, num_inputs, num_hiddens, **kwargs):
        super(Compare, self).__init__(**kwargs)
        self.g = mlp(num_inputs, num_hiddens, flatten=False)

    def forward(self, A, B, beta, alpha):
        V_A = self.g(torch.cat([A, beta], dim=2))
        V_B = self.g(torch.cat([B, alpha], dim=2))
        return V_A, V_B


class Aggregate(nn.Module):

    def __init__(self, num_inputs, num_hiddens, num_outputs, **kwargs):
        super(Aggregate, self).__init__(**kwargs)
        self.h = mlp(num_inputs, num_hiddens, flatten=True)
        self.linear = nn.Linear(num_hiddens, num_outputs)

    def forward(self, V_A, V_B):
        # 对两组比较向量分别求和
        V_A = V_A.sum(dim=1)
        V_B = V_B.sum(dim=1)
        # 将两个求和结果的连结送到多层感知机中
        Y_hat = self.linear(self.h(torch.cat([V_A, V_B], dim=1)))
        return Y_hat


class DecomposableAttention(nn.Module):

    def __init__(self, vocab, embed_size, num_hiddens,
                 num_inputs_attend=100,
                 num_inputs_compare=200,
                 num_inputs_agg=400, **kwargs):
        super(DecomposableAttention, self).__init__(**kwargs)
        self.embedding = nn.Embedding(len(vocab), embed_size)
        self.attend = Attend(num_inputs_attend, num_hiddens)
        self.compare = Compare(num_inputs_compare, num_hiddens)
        # 有3种可能的输出：蕴涵、矛盾和中性
        self.aggregate = Aggregate(num_inputs_agg, num_hiddens, num_outputs=3)

    def forward(self, X):
        premises, hypotheses = X
        A = self.embedding(premises)
        B = self.embedding(hypotheses)
        beta, alpha = self.attend(A, B)
        V_A, V_B = self.compare(A, B, beta, alpha)
        Y_hat = self.aggregate(V_A, V_B)
        return Y_hat


# 测试
def test_1(net, vocab):
    from d2lzh_pytorch.download_unit import download_extract
    from d2lzh_pytorch.BERT_unit_v3.load_snli import read_snli
    from d2lzh_pytorch.BERT_unit_v3.predict_utils import predict_snli

    # 使用测试的数据集进行测试
    data_dir = download_extract('SNLI')  # 857M

    train_data = read_snli(data_dir, is_train=False)

    sum_total = 0
    acc_sum = 0
    for x0, x1, y in zip(train_data[0],
                         train_data[1],
                         train_data[2]):
        # print('前提：', x0)
        # print('假设：', x1)
        # print('标签：', y)
        res = predict_snli(net, vocab,
                           x0.split(' '),
                           x1.split(' ')
                           )
        label = 'entailment' if y == 0 else 'contradiction' if y == 1 else 'neutral'
        if label == res:
            # 预测正确
            acc_sum += 1
            pass
        else:
            # 预测错误
            print('error: y: %s , y_hat: %s' % (label, res), '内容: %s \t %s' % (x0, x1))
            pass

        sum_total += 1
        if sum_total >= 200:
            break
    print('acc: %.4f' % (acc_sum / sum_total))
    pass


def run():
    batch_size = 256
    num_steps = 50

    train_iter, test_iter, vocab = load_data_snli(batch_size, num_steps)

    embed_size = 100
    num_hiddens = 200
    devices = try_all_gpus()

    net = DecomposableAttention(vocab, embed_size, num_hiddens)
    glove_embedding = TokenEmbedding('glove.6b.100d')
    embeds = glove_embedding[vocab.idx_to_token]
    net.embedding.weight.data.copy_(embeds)

    # 训练
    lr = 0.001
    num_epochs = 5

    trainer = torch.optim.Adam(net.parameters(), lr=lr)
    loss = nn.CrossEntropyLoss(reduction="none")

    # 训练
    train_ch13(net, train_iter, test_iter,
               loss,
               trainer,
               num_epochs,
               devices
               )

    # 预测
    test_1(net, vocab)
    pass


def main():
    pass


if __name__ == '__main__':
    main()
