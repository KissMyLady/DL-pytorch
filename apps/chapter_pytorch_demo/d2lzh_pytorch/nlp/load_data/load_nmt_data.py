# coding:utf-8
# Author:mylady
# Datetime:2023/4/23 7:43
import sys
sys.path.append("..")
sys.path.append("../..")
sys.path.append("../../..")
sys.path.append("../../../..")

import d2lzh_pytorch.torch_package as d2l

import torch
from torch.utils import data

# 引入词表 将字符串映射到从0开始的数字索引中
from d2lzh_pytorch.nlp.load_data.load_time_machine import Vocab

reduce_sum = lambda x, *args, **kwargs: x.sum(*args, **kwargs)
astype = lambda x, *args, **kwargs: x.type(*args, **kwargs)
int32 = torch.int32


def read_data_nmt():
    """Load the English-French dataset.
    Defined in `sec_machine_translation`
    数据格式:
        Go.     Va !
        Hi.     Salut !
        Run!    Cours!
        Run!    Courez!
        Who?    Qui ?
        Wow!    Ça alors!
        Fire!   Au feu !
        Help!   À l'aide!
        Jump.   Saute.
    """
    # 注意防火墙.下载地址: http://d2l-data.s3-accelerate.amazonaws.com/fra-eng.zip
    # $ wget http://d2l-data.s3-accelerate.amazonaws.com/fra-eng.zip
    # $ unzip fra-eng.zip
    # data_dir = d2l.download_extract('fra-eng')

    # filePath = r"/mnt/g1t/ai_data/Datasets_on_HHD/NLP/fra-eng/fra.txt"
    filePath = r"X:\big_ai_files\d2l_data\data\fra-eng\fra.txt"

    with open(filePath, 'r', encoding='utf8') as f:
        return f.read()


def preprocess_nmt(text):
    """
    定 义: 9.5 机器翻译与数据集
    功 能: 预处理“英语－法语”数据集
    """

    def no_space(char, prev_char):
        return char in set(',.!?') and prev_char != ' '

    # 使用空格替换不间断空格
    # 使用小写字母替换大写字母
    text = text.replace('\u202f', ' ').replace('\xa0', ' ').lower()
    # Insert space between words and punctuation marks
    out = [' ' + char if i > 0 and no_space(char, text[i - 1]) else char
           for i, char in enumerate(text)]
    return ''.join(out)


def tokenize_nmt(text, num_examples=None):
    """Tokenize the English-French dataset.
    Defined in `sec_machine_translation`

    定 义: 9.5 机器翻译与数据集
    功 能: 词元化“英语－法语”数据数据集

    """
    source, target = [], []
    for i, line in enumerate(text.split('\n')):
        if num_examples and i > num_examples:
            break
        parts = line.split('\t')
        if len(parts) == 2:
            source.append(parts[0].split(' '))
            target.append(parts[1].split(' '))
    return source, target


def truncate_pad(line, num_steps, padding_token):
    """
    定 义: 9.5 机器翻译与数据集
    功 能: 截断或填充文本序列
    """
    if len(line) > num_steps:
        return line[:num_steps]  # 截断 Truncate
    return line + [padding_token] * (num_steps - len(line))  # 填充Pad


def build_array_nmt(lines, vocab, num_steps):
    """
    定 义: 9.5 机器翻译与数据集
    功 能: 将机器翻译的文本序列转换成小批量
    """
    lines = [vocab[l] for l in lines]
    lines = [l + [vocab['<eos>']] for l in lines]
    array = torch.tensor([truncate_pad(l, num_steps, vocab['<pad>']) for l in lines])
    valid_len = reduce_sum(astype(array != vocab['<pad>'], int32), 1)
    return array, valid_len


def load_array(data_arrays, batch_size, is_train=True):
    """
    定 义: 3.3 线性回归的简洁实现
    功 能: 构造一个PyTorch数据迭代器
    描 述: 每次加载 batch_size 个数据返回
    """
    dataset = data.TensorDataset(*data_arrays)
    next_data = data.DataLoader(dataset,
                                batch_size,
                                shuffle=is_train)
    return next_data


def load_data_nmt(batch_size, num_steps, num_examples=600):
    """
    功 能: 返回翻译数据集的迭代器和词表
    """
    # 预处理
    text = preprocess_nmt(read_data_nmt())

    # 词元化
    source, target = tokenize_nmt(text, num_examples)
    src_vocab = Vocab(source, min_freq=2, reserved_tokens=['<pad>', '<bos>', '<eos>'])
    tgt_vocab = Vocab(target, min_freq=2, reserved_tokens=['<pad>', '<bos>', '<eos>'])

    # 转换成小批量
    src_array, src_valid_len = build_array_nmt(source, src_vocab, num_steps)
    tgt_array, tgt_valid_len = build_array_nmt(target, tgt_vocab, num_steps)
    data_arrays = (src_array, src_valid_len, tgt_array, tgt_valid_len)

    # 加载数据返回
    data_iter = load_array(data_arrays, batch_size)
    return data_iter, src_vocab, tgt_vocab


def test_2():
    # 使用 load_data_nmt 加载数据
    train_iter, src_vocab, tgt_vocab = load_data_nmt(batch_size=2, num_steps=8)

    for X, X_valid_len, Y, Y_valid_len in train_iter:
        print('X:', X.type(torch.int32))
        print('X的有效长度:', X_valid_len)
        print('Y:', Y.type(torch.int32))
        print('Y的有效长度:', Y_valid_len)
        break
    pass


def test_3():
    batch_size = 64
    num_steps = 10

    train_iter, src_vocab, tgt_vocab = load_data_nmt(batch_size, num_steps)
    print("src_vocab 长度为: ", len(src_vocab))  # 184
    print("tgt_vocab 长度为: ", len(tgt_vocab))  # 201

    pass


def main():
    test_3()
    pass


if __name__ == '__main__':
    main()
    pass
