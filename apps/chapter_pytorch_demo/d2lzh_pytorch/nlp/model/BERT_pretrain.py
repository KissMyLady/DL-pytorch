# coding:utf-8
# Author:mylady
# Datetime:2023/6/29 12:30
import sys
sys.path.append("..")
sys.path.append("../..")

import os
import json

import torch
from torch import nn
from d2lzh_pytorch.myUtils import try_gpu
from d2lzh_pytorch.nlp.model.BERT_model import get_BERT_model
from d2lzh_pytorch.nlp.model.BERT_model import BERTModel
from d2lzh_pytorch.nlp.load_data.load_time_machine import Vocab


class BERTClassifier(nn.Module):
    
    def __init__(self, bert):
        super(BERTClassifier, self).__init__()
        self.encoder = bert.encoder
        self.hidden = bert.hidden
        self.output = nn.Linear(256, 3)

    def forward(self, inputs):
        tokens_X, segments_X, valid_lens_x = inputs
        encoded_X = self.encoder(tokens_X, segments_X, valid_lens_x)
        return self.output(self.hidden(encoded_X[:, 0, :]))


# 缩小版 bert.small.torch
def load_pretrained_model(data_dir, 
                          num_hiddens, ffn_num_hiddens,
                          num_heads, num_layers, 
                          dropout, max_len):
    devices = try_gpu()
    # data_dir = download_extract(pretrained_model)

    # 定义空词表以加载预定义词表
    vocab = Vocab()
    vocabPath = os.path.join(data_dir, 'vocab.json')
    vocab.idx_to_token = json.load(open(vocabPath))
    vocab.token_to_idx = {token: idx for idx, token in enumerate(vocab.idx_to_token)}

    # 加载bert模型
    # bert = get_BERT_model(len(vocab))
    print("预训练的BERT Vocab长度为 :%s" % (len(vocab)))

    bert = BERTModel(len(vocab),  
                     num_hiddens, norm_shape=[256],
                     ffn_num_input=256, ffn_num_hiddens=ffn_num_hiddens,
                     num_heads=4, num_layers=2, 
                     dropout=0.2,
                     max_len=max_len, 
                     key_size=256, query_size=256, value_size=256, 
                     hid_in_features=256,
                     mlm_in_features=256, 
                     nsp_in_features=256)

    filePath = os.path.join(data_dir, 'pretrained.params')
    bert.load_state_dict(torch.load(filePath))
    return bert, vocab



def get_pretrin_BERTClassifier(is_base=False):

    if is_base is True:
        """
        BERT缩小版
        pretrained.params  683M
        vocab.json         642K
        """
        data_dir = "/mnt/g1t/ai_data/Datasets_on_HHD/d2l_data/data/bert.base.torch"
        print("加载基础版的BERT")
    else:
        """
        BERT缩小版
        pretrained.params  123M
        vocab.json         642K
        """
        data_dir = "/mnt/g1t/ai_data/Datasets_on_HHD/d2l_data/data/bert.small.torch"
        print("加载缩小版的BERT")
        pass
    
    # 预训练的BERT加载
    bert, vocab = load_pretrained_model(data_dir, 
                                        num_hiddens=256, ffn_num_hiddens=512, 
                                        num_heads=4, num_layers=2, 
                                        dropout=0.1, max_len=512)

    # 套一层输出网络
    net = BERTClassifier(bert)

    # 封装返回
    return net, vocab


def test_1():
    pass


def main():
    pass


if __name__ == "__main__":
    #main()
    pass