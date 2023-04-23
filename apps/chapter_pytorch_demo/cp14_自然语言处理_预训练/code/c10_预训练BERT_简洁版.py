# coding:utf-8
# Author:mylady
# Datetime:2023/4/24 7:26
import torch
from torch import nn

import time
import sys

sys.path.append("..")
from d2lzh_pytorch.attention.transformer_unit import EncoderBlock
from d2lzh_pytorch.myUtils import Timer, Accumulator, try_gpu, try_all_gpus
from d2lzh_pytorch.myPolt import Animator
from d2lzh_pytorch.BERT_unit.load_data import load_data_wiki
from d2lzh_pytorch.BERT_unit.BERT_model import BERTModel, get_tokens_and_segments
from d2lzh_pytorch.BERT_unit.BERT_train import train_bert


def main():
    batch_size = 512
    max_len = 64

    # 加载数据
    train_iter, vocab = load_data_wiki(batch_size, max_len)

    net = BERTModel(len(vocab),
                    num_hiddens=128,
                    norm_shape=[128],
                    ffn_num_input=128,
                    ffn_num_hiddens=256,
                    num_heads=2,
                    num_layers=2,
                    dropout=0.2,
                    key_size=128,
                    query_size=128,
                    value_size=128,
                    hid_in_features=128,
                    mlm_in_features=128,
                    nsp_in_features=128)

    devices = try_all_gpus()
    loss = nn.CrossEntropyLoss()
    num_steps = 50

    # 训练
    train_bert(train_iter,
               net,
               loss,
               len(vocab),
               devices,
               num_steps
               )

    # 模型保存
    PATH = 'BERT_train_chapter_14_%s.pt' % (time.time())
    torch.save(net, PATH)   # 保存整个模型  22M
    pass


if __name__ == '__main__':
    main()
