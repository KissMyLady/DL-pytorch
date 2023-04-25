# coding:utf-8
# Author:mylady
# Datetime:2023/4/25 20:06
import torch
from d2lzh_pytorch.myUtils import try_gpu


def predict_snli(net, vocab, premise, hypothesis):
    """预测前提和假设之间的逻辑关系

    entailment: 限定继承
    contradiction: 矛盾
    neutral: 自然
    """
    net.eval()
    premise = torch.tensor(vocab[premise], device=try_gpu())
    hypothesis = torch.tensor(vocab[hypothesis], device=try_gpu())
    label = torch.argmax(net([premise.reshape((1, -1)),
                              hypothesis.reshape((1, -1))]
                             ), dim=1)
    return 'entailment' if label == 0 else 'contradiction' if label == 1 else 'neutral'


def main():
    pass


if __name__ == '__main__':
    main()
