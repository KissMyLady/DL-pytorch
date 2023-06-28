# coding:utf-8
# Author:mylady
# Datetime:2023/6/28 23:42
import sys
sys.path.append("../..")
# import d2lzh_pytorch.torch as d2l

import os
import torch
from torch.utils import data

# 引入词表 将字符串映射到从0开始的数字索引中
from d2lzh_pytorch.myUtils import download_extract
from d2lzh_pytorch.myUtils import DATA_HUB
from d2lzh_pytorch.nlp.load_data.load_time_machine import Vocab


#@save
DATA_HUB['aclImdb'] = (
    'http://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz',
    '01ada507287d82875905620988597833ad4e0903')

# data_dir = d2l.download_extract('aclImdb', 'aclImdb')


def tokenize(lines, token='word'):
    """Split text lines into word or character tokens.

    Defined in :numref:`sec_text_preprocessing`"""
    if token == 'word':
        return [line.split() for line in lines]
    elif token == 'char':
        return [list(line) for line in lines]
    else:
        print('ERROR: unknown token type: ' + token)


def truncate_pad(line, num_steps, padding_token):
    """Truncate or pad sequences.

    Defined in :numref:`sec_machine_translation`"""
    if len(line) > num_steps:
        return line[:num_steps]  # Truncate
    return line + [padding_token] * (num_steps - len(line))  # Pad


#@save
def read_imdb(data_dir, is_train):
    """读取IMDb评论数据集文本序列和标签"""
    data = list()
    labels = list()
    
    for label in ('pos', 'neg'):
        folder_name = os.path.join(data_dir, 'train' if is_train else 'test', label)
        for file in os.listdir(folder_name):
            with open(os.path.join(folder_name, file), 'rb') as f:
                review = f.read().decode('utf-8').replace('\n', '')
                data.append(review)
                labels.append(1 if label == 'pos' else 0)
                pass
            pass
        pass
    return data, labels


def load_array(data_arrays, batch_size, is_train=True):
    """Construct a PyTorch data iterator.
    Defined in `sec_linear_concise`
    
    定 义: 3.3 线性回归的简洁实现
    功 能: 构造一个PyTorch数据迭代器
    描 述: 每次加载 batch_size 个数据返回
    """
    dataset = data.TensorDataset(*data_arrays)
    next_data = data.DataLoader(dataset,
                                batch_size,
                                shuffle=is_train)
    return next_data


def load_data_imdb(batch_size, num_steps=500):
    """返回数据迭代器和IMDb评论数据集的词表"""
    # data_dir = download_extract('aclImdb', 'aclImdb')
    data_dir = "/mnt/g1t/ai_data/Datasets_on_HHD/data/aclImdb"
    train_data = read_imdb(data_dir, True)
    test_data  = read_imdb(data_dir, False)
    train_tokens = tokenize(train_data[0], token='word')
    test_tokens  = tokenize(test_data[0], token='word')

    # 词元化
    vocab = Vocab(train_tokens, min_freq=5)
    
    train_features = torch.tensor([truncate_pad(vocab[line], 
                                    num_steps, 
                                    vocab['<pad>']
                                    ) for line in train_tokens])
    
    test_features = torch.tensor([truncate_pad(vocab[line], num_steps, vocab['<pad>']) for line in test_tokens])

    # 数据返回
    train_iter = load_array((train_features, 
                             torch.tensor(train_data[1])),
                             batch_size)
    
    test_iter = load_array((test_features, 
                            torch.tensor(test_data[1])),
                            batch_size,
                            is_train=False)
    # 返回
    return train_iter, test_iter, vocab


def test_1():
    pass


def main():
    pass


if __name__ == "__main__":
    # main()
    pass