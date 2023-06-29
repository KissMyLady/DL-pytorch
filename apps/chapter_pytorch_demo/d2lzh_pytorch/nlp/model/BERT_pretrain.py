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


# 基础版 bert.small.torch
def load_pretrained_model_BASE(data_dir, 
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
                     num_hiddens,
                     norm_shape=[768],
                     ffn_num_input=768, 
                     ffn_num_hiddens=ffn_num_hiddens,
                     num_heads=num_heads, 
                     num_layers=num_layers, 
                     dropout=0.2,
                     max_len=max_len, 
                     key_size=768, 
                     query_size=768,
                     value_size=768, 
                     hid_in_features=768,
                     mlm_in_features=768, 
                     nsp_in_features=768)

    filePath = os.path.join(data_dir, 'pretrained.params')
    bert.load_state_dict(torch.load(filePath))
    return bert, vocab


# 缩小版 bert.small.torch
def load_pretrained_model_small(data_dir, 
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
                     num_hiddens,
                     norm_shape=[256],
                     ffn_num_input=256, 
                     ffn_num_hiddens=ffn_num_hiddens,
                     num_heads=4, 
                     num_layers=2, 
                     dropout=0.2,
                     max_len=max_len, 
                     key_size=256, 
                     query_size=256,
                     value_size=256, 
                     hid_in_features=256,
                     mlm_in_features=256, 
                     nsp_in_features=256)

    filePath = os.path.join(data_dir, 'pretrained.params')
    bert.load_state_dict(torch.load(filePath))
    return bert, vocab


# 缩小版
def get_pretrin_BERTClassifier_small():
    data_dir = "/mnt/g1t/ai_data/Datasets_on_HHD/d2l_data/data/bert.small.torch"

    # 预训练的BERT加载
    bert, vocab = load_pretrained_model_small(data_dir, 
                                              num_hiddens=256, ffn_num_hiddens=512, 
                                              num_heads=4, num_layers=2, 
                                              dropout=0.1, max_len=512,
                                              )

    # 套一层输出网络
    net = BERTClassifier(bert)

    # 封装返回
    return net, vocab


# 更大的模型
def get_pretrin_BERTClassifier_BASE(**kwargs):
    """
    Fine-tune a much larger pretrained BERT model that is about as big as the original BERT base model 
    if your computational resource allows. Set arguments in the load_pretrained_model function as: 
        replacing 'bert.small' with 'bert.base',
         increasing values of num_hiddens=256, ffn_num_hiddens=512, num_heads=4, and num_blks=2 
                                       to 768,                 3072,          12,  and 12, respectively. 
         By increasing fine-tuning epochs (and possibly tuning other hyperparameters), 
         can you get a testing accuracy higher than 0.86?

    如果您的计算资源允许，可以微调一个更大的预训练伯特模型，该模型大约与原始的伯特基模型一样大。
    将 load_pretrained_model 函数中的参数设置为：将“bert.small”替换为“bert.base”，
    分别将 num_hiddens =256、ffn_num_hiddens =512、num_heads =4和 num_blks =2的值增加
                    到 768 、               3072 、         12和         12。
    通过增加微调epoch（并可能调整其他超参数），可以获得高于0.86的测试精度吗？
    
    """
    # 2023-6-29再次下载BERT模型
    # data_dir = "/mnt/g1t/ai_data/Datasets_on_HHD/d2l_data/data/bert.small.torch"
    data_dir = "/mnt/g1t/ai_data/Datasets_on_HHD/d2l_data/data/bert.base.torch"
    print("加载基础版的BERT")

    # 预训练的BERT加载
    # bert, vocab = load_pretrained_model_BASE(data_dir, 
    #                                          num_hiddens=256, ffn_num_hiddens=512, 
    #                                          num_heads=4, num_layers=2, 
    #                                          dropout=0.1, max_len=512,)
    
    # 模型获取
    bert, vocab = load_pretrained_model_BASE(data_dir, 
                                             num_hiddens=768, ffn_num_hiddens=3072, 
                                             num_heads=4, num_layers=2, 
                                             dropout=0.1, max_len=512,)
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