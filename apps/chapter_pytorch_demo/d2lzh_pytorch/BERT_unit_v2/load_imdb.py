# coding:utf-8
# Author:mylady
# Datetime:2023/4/24 18:02
import torch
import os
import sys

sys.path.append(".")
sys.path.append("../..")
from d2lzh_pytorch.load_chinese_txt_data import tokenize
from d2lzh_pytorch.load_Vocab import Vocab
from d2lzh_pytorch.seq.load_nmt_data import truncate_pad, load_array


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


def load_data_imdb(batch_size, num_steps=500):
    """返回数据迭代器和IMDb评论数据集的词表"""
    # data_dir = download_extract('aclImdb', 'aclImdb')
    data_dir = r"/home/mylady/code/python/DL-pytorch/apps/chapter_pytorch_demo/data/aclImdb"

    train_data = read_imdb(data_dir, True)
    test_data = read_imdb(data_dir, False)
    train_tokens = tokenize(train_data[0], token='word')
    test_tokens = tokenize(test_data[0], token='word')
    vocab = Vocab(train_tokens, min_freq=5)

    train_features = torch.tensor([
        truncate_pad(vocab[line],
                     num_steps,
                     vocab['<pad>']
                     ) for line in train_tokens]
    )

    test_features = torch.tensor([
        truncate_pad(vocab[line], num_steps, vocab['<pad>']) for line in test_tokens
    ])

    # 每次加载batch_size个数据返回
    train_iter = load_array((train_features,
                             torch.tensor(train_data[1])
                             ), batch_size)

    test_iter = load_array((test_features,
                            torch.tensor(test_data[1])),
                           batch_size, is_train=False)
    return train_iter, test_iter, vocab


def run():
    batch_size = 64

    # 使用
    train_iter, test_iter, vocab = load_data_imdb(batch_size)
    pass


def main():
    pass


if __name__ == '__main__':
    main()
