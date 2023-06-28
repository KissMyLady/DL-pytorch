import sys
sys.path.append("..")
sys.path.append("../..")

from .BERT_model import get_BERT_model
import math
from torch import nn
import torch


class BERTClassifier(nn.Module):

    def __init__(self, bert):
        super(BERTClassifier, self).__init__()
        self.encoder = bert.encoder
        self.hidden  = bert.hidden
        self.output  = nn.Linear(256, 3)

    def forward(self, inputs):
        tokens_X, segments_X, valid_lens_x = inputs
        encoded_X = self.encoder(tokens_X, segments_X, valid_lens_x)
        return self.output(self.hidden(encoded_X[:, 0, :]))


def get_BERTClassifier():
    # 获取原始bert
    bert = get_BERT_model()

    # 加入分类器
    net = BERTClassifier(bert)
    return net


def test_1():

    pass


def main():
    pass


if __name__ == "__main__":
    main()
    pass
