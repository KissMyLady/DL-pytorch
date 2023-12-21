# coding:utf-8
# Author:mylady
# Datetime:2023/4/7 14:33
import torch_package
import random

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# 随机采样
def data_iter_random(corpus_indices, batch_size, num_steps, device=None):
    # 减1是因为输出的索引x是相应输入的索引y加1
    num_examples = (len(corpus_indices) - 1) // num_steps
    epoch_size = num_examples // batch_size
    example_indices = list(range(num_examples))
    random.shuffle(example_indices)

    # 返回从pos开始的长为num_steps的序列
    def _data(pos):
        return corpus_indices[pos: pos + num_steps]

    for i in range(epoch_size):
        # 每次读取batch_size个随机样本
        i = i * batch_size
        batch_indices = example_indices[i: i + batch_size]
        X = [_data(j * num_steps) for j in batch_indices]
        Y = [_data(j * num_steps + 1) for j in batch_indices]

        a1 = torch.tensor(X, dtype=torch.float32, device=device)
        a2 = torch.tensor(Y, dtype=torch.float32, device=device)
        yield a1, a2


# 相邻采样
def data_iter_consecutive(corpus_indices, batch_size, num_steps, device=None):
    corpus_indices = torch.tensor(corpus_indices, dtype=torch.float32, device=device)
    data_len = len(corpus_indices)
    batch_len = data_len // batch_size
    indices = corpus_indices[0: batch_size * batch_len].view(batch_size, batch_len)
    epoch_size = (batch_len - 1) // num_steps

    for i in range(epoch_size):
        i = i * num_steps
        X = indices[:, i: i + num_steps]
        Y = indices[:, i + 1: i + num_steps + 1]
        yield X, Y


def test_1():
    # with zipfile.ZipFile('~/Datasets/RNN_data/jaychou_lyrics.txt.zip') as zin:
    # 读取歌词
    with open('../../data/RNN_data/jaychou_lyrics.txt', encoding='utf-8') as f:
        corpus_chars = f.read()  # .decode('utf-8')
    corpus_chars = corpus_chars.replace('\n', ' ').replace('\r', ' ')
    corpus_chars = corpus_chars[0:10000]

    # 建立字符索引
    idx_to_char = list(set(corpus_chars))
    char_to_idx = dict([(char, i) for i, char in enumerate(idx_to_char)])

    vocab_size = len(char_to_idx) # 1027
    corpus_indices = [char_to_idx[char] for char in corpus_chars]

    # 获取歌词txt的前40个字符串
    sample = corpus_indices[0:40]
    print('chars:', ''.join([idx_to_char[idx] for idx in sample]))
    print('indices:', sample)
    pass


def test_2():
    # 随机采样
    my_seq = list(range(30))
    for X, Y in data_iter_random(my_seq, batch_size=4, num_steps=6):
        print('X: ', X, '\nY:', Y, '\n')
    pass


# 相邻采样
def test_3():
    my_seq = list(range(30))
    for X, Y in data_iter_consecutive(my_seq, batch_size=4, num_steps=6):
        print('X: ', X, '\nY:', Y, '\n')
    pass


def main():
    # test_1()  # 读取歌词
    test_2()  # 随机采样
    # test_3()  # 相邻采样
    pass


if __name__ == '__main__':
    main()
