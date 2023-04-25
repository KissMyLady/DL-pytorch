# coding:utf-8
# Author:mylady
# Datetime:2023/4/24 22:45
import torch
from torch import nn

import time
import sys

sys.path.append("..")
from d2lzh_pytorch.myUtils import try_all_gpus
from d2lzh_pytorch.BERT_unit_v2.load_imdb import load_data_imdb, download_extract, read_imdb
from d2lzh_pytorch.glove.use_glove import TokenEmbedding

from d2lzh_pytorch.BERT_unit_v2.use_rnn import BiRNN, init_weights
from d2lzh_pytorch.BERT_unit_v2.predict_utils import predict_sentiment

from d2lzh_pytorch.CNN.train_cnn import train_ch13


# 预测
def test_1(net, vocab):
    sum_total = 0
    acc_sum = 0
    data_dir = download_extract('aclImdb', 'aclImdb')
    test_data = read_imdb(data_dir, is_train=False)
    print('测试集数目：', len(test_data[0]))
    for x, y in zip(test_data[0], test_data[1]):
        predict_res = predict_sentiment(net, vocab, x)
        label = 'positive' if y == 1 else 'negative'

        if label == predict_res:
            # 预测正确
            acc_sum += 1
            pass
        else:
            # 预测错误
            print('测试数据集 标签错误：', y, '内容review:', x[0:60], '预测结果: ', predict_res)
            pass
        sum_total += 1
        if sum_total >= 100:
            break
    print('acc: %.4f' % (acc_sum / sum_total))
    pass


def run():
    batch_size = 64
    # 加载数据
    train_iter, test_iter, vocab = load_data_imdb(batch_size)

    embed_size = 100
    num_hiddens = 100
    num_layers = 2

    devices = try_all_gpus()

    # 网络构建
    net = BiRNN(len(vocab),
                embed_size,
                num_hiddens,
                num_layers
                )

    net.apply(init_weights)
    glove_embedding = TokenEmbedding('glove.6b.100d')
    embeds = glove_embedding[vocab.idx_to_token]

    # 使用这些预训练的词向量来表示评论中的词元，并且在训练期间不要更新这些向量。
    net.embedding.weight.data.copy_(embeds)
    net.embedding.weight.requires_grad = False

    # 训练参数
    lr = 0.01
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


def main():
    pass


if __name__ == '__main__':
    main()
