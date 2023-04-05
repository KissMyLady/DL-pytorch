import json
import multiprocessing
import os
import torch
from torch import nn

import sys
sys.path.append("..")
import d2lzh_pytorch.torch as d2l


d2l.DATA_HUB['bert.base'] = (
    d2l.DATA_URL + 'bert.base.torch.zip',
    '225d66f04cae318b841a13d32af3acc165f253ac'
)


def load_pretrained_model(pretrained_model, num_hiddens, ffn_num_hiddens,
                          num_heads, num_layers, dropout, max_len, devices):
    
    data_dir = d2l.download_extract(pretrained_model)
    
    # 定义空词表以加载预定义词表
    vocab = d2l.Vocab()
    
    vocab.idx_to_token = json.load(open(os.path.join(data_dir,'vocab.json')))
    vocab.token_to_idx = {
        token: idx for idx, token in enumerate(vocab.idx_to_token)
    }
    
    bert = d2l.BERTModel(len(vocab), 
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
    
    # 加载预训练BERT参数
    bert.load_state_dict(torch.load(os.path.join(data_dir,'pretrained.params')))
    return bert, vocab


def run():
    devices = d2l.try_all_gpus()
    # devices = [torch.device('cpu')]


    # 加载词向量
    bert, vocab = load_pretrained_model('bert.base', 
                                        num_hiddens=256, 
                                        ffn_num_hiddens=512, 
                                        num_heads=4,
                                        num_layers=2, 
                                        dropout=0.1, 
                                        max_len=512, 
                                        devices=devices)
    pass

def main():
    run()
    pass

if __name__ == "__main__":
    main()