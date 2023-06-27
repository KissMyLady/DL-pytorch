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


class BiRNN(nn.Module):
    def __init__(self, vocab_size, embed_size, num_hiddens, num_layers, **kwargs):
        super(BiRNN, self).__init__(**kwargs)
        self.embedding = nn.Embedding(vocab_size, embed_size)
        # 将bidirectional设置为True以获取双向循环神经网络
        self.encoder = nn.LSTM(embed_size,
                               num_hiddens,
                               num_layers=num_layers,
                               bidirectional=True)
        self.decoder = nn.Linear(4 * num_hiddens, 2)

    def forward(self, inputs):
        # inputs的形状是（批量大小，时间步数）
        # 因为长短期记忆网络要求其输入的第一个维度是时间维，
        # 所以在获得词元表示之前，输入会被转置。
        # 输出形状为（时间步数，批量大小，词向量维度）
        embeddings = self.embedding(inputs.T)
        self.encoder.flatten_parameters()
        # 返回上一个隐藏层在不同时间步的隐状态，
        # outputs的形状是（时间步数，批量大小，2*隐藏单元数）
        outputs, _ = self.encoder(embeddings)
        # 连结初始和最终时间步的隐状态，作为全连接层的输入，
        # 其形状为（批量大小，4*隐藏单元数）
        encoding = torch.cat((outputs[0], outputs[-1]), dim=1)
        outs = self.decoder(encoding)
        return outs


def init_weights(m):
    if type(m) == nn.Linear:
        nn.init.xavier_uniform_(m.weight)

    if type(m) == nn.LSTM:
        for param in m._flat_weights_names:
            if "weight" in param:
                nn.init.xavier_uniform_(m._parameters[param])
            pass
        pass


def main():
    # 加载数据
    batch_size = 64
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
    # 加载词向量
    glove_embedding = TokenEmbedding('glove.6b.100d')
    embeds = glove_embedding[vocab.idx_to_token]
    # 使用这些预训练的词向量来表示评论中的词元，并且在训练期间不要更新这些向量。
    net.embedding.weight.data.copy_(embeds)
    net.embedding.weight.requires_grad = False

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


if __name__ == '__main__':
    main()
