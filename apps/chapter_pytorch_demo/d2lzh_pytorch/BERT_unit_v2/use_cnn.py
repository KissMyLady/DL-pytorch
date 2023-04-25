# coding:utf-8
# Author:mylady
# Datetime:2023/4/24 18:14
import torch
from torch import nn

import sys

sys.path.append(".")
sys.path.append("../..")
from d2lzh_pytorch.myUtils import try_all_gpus
from d2lzh_pytorch.BERT_unit_v2.load_imdb import load_data_imdb
from d2lzh_pytorch.glove.use_glove import TokenEmbedding
from d2lzh_pytorch.CNN.train_cnn import train_ch13


def corr1d(X, K):
    """一维卷积"""
    w = K.shape[0]
    Y = torch.zeros((X.shape[0] - w + 1))
    for i in range(Y.shape[0]):
        Y[i] = (X[i: i + w] * K).sum()
    return Y


def corr1d_multi_in(X, K):
    # 首先，遍历'X'和'K'的第0维（通道维）。然后，把它们加在一起
    return sum(corr1d(x, k) for x, k in zip(X, K))


class TextCNN(nn.Module):
    """cnn模型"""

    def __init__(self, vocab_size, embed_size,
                 kernel_sizes, num_channels, **kwargs):
        super(TextCNN, self).__init__(**kwargs)
        self.embedding = nn.Embedding(vocab_size, embed_size)
        # 这个嵌入层不需要训练
        self.constant_embedding = nn.Embedding(vocab_size, embed_size)
        self.dropout = nn.Dropout(0.5)
        self.decoder = nn.Linear(sum(num_channels), 2)
        # 最大时间汇聚层没有参数，因此可以共享此实例
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.relu = nn.ReLU()
        # 创建多个一维卷积层
        self.convs = nn.ModuleList()
        for c, k in zip(num_channels, kernel_sizes):
            self.convs.append(nn.Conv1d(2 * embed_size, c, k))

    def forward(self, inputs):
        # 沿着向量维度将两个嵌入层连结起来，
        # 每个嵌入层的输出形状都是（批量大小，词元数量，词元向量维度）连结起来
        embeddings = torch.cat((self.embedding(inputs),
                                self.constant_embedding(inputs)
                                ), dim=2)
        # 根据一维卷积层的输入格式，重新排列张量，以便通道作为第2维
        embeddings = embeddings.permute(0, 2, 1)
        # 每个一维卷积层在最大时间汇聚层合并后，获得的张量形状是（批量大小，通道数，1）
        # 删除最后一个维度并沿通道维度连结
        encoding = torch.cat([
            torch.squeeze(self.relu(self.pool(conv(embeddings))), dim=-1)
            for conv in self.convs], dim=1)

        outputs = self.decoder(self.dropout(encoding))
        return outputs


def init_weights(m):
    if type(m) in (nn.Linear, nn.Conv1d):
        nn.init.xavier_uniform_(m.weight)
    pass


def run():
    # 加载数据
    batch_size = 64

    train_iter, test_iter, vocab = load_data_imdb(batch_size)

    embed_size = 100
    kernel_sizes = [3, 4, 5]
    nums_channels = [100, 100, 100]

    devices = try_all_gpus()
    net = TextCNN(len(vocab),
                  embed_size,
                  kernel_sizes,
                  nums_channels)

    # 初始化参数
    net.apply(init_weights)

    # 加载预先训练的词向量
    glove_embedding = TokenEmbedding('glove.6b.100d')
    embeds = glove_embedding[vocab.idx_to_token]
    net.embedding.weight.data.copy_(embeds)
    net.constant_embedding.weight.data.copy_(embeds)
    net.constant_embedding.weight.requires_grad = False

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
    pass


def test_1(test_data, net, vocab):
    from d2lzh_pytorch.BERT_unit_v2.predict_utils import predict_sentiment
    sum_total = 0
    acc_sum = 0
    for x, y in zip(test_data[0], test_data[1]):
        predict_res = predict_sentiment(net, vocab, x)
        label = 'positive' if y == 1 else 'negative'
        if label == predict_res:
            acc_sum += 1
            pass
        else:
            print('预测错误了：', y, '内容review:', x[0:60], '预测结果: ', predict_res)
            pass
        sum_total += 1
        if sum_total >= 100:
            break
    print('acc: %.4f' % (acc_sum / sum_total))
    pass


def main():
    pass


if __name__ == '__main__':
    main()
