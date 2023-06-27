# coding:utf-8
# Author:mylady
# Datetime:2023/4/24 22:56
import torch
import sys

sys.path.append(".")
sys.path.append("..")
sys.path.append("../..")
from d2lzh_pytorch.myUtils import try_gpu


def predict_sentiment(net, vocab, sequence):
    """
    预测文本序列的情感
    """
    sequence = torch.tensor(vocab[sequence.split()], device=try_gpu())
    label = torch.argmax(net(sequence.reshape(1, -1)), dim=1)
    return 'positive' if label == 1 else 'negative'


def main():
    pass


if __name__ == '__main__':
    main()
