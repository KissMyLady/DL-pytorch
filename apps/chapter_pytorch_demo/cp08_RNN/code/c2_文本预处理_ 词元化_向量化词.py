# coding:utf-8
# Author:mylady
# Datetime:2023/3/24 15:20
import collections
import re
import jieba
from d2l import torch as d2l


# 加载英文小说
def read_time_machine():
    """
    将时间机器数据集加载到文本行的列表中
    """
    with open(d2l.download('time_machine'), 'r') as f:
        lines = f.readlines()
        pass
    return [re.sub('[^A-Za-z]+', ' ', line).strip().lower() for line in lines]


# 加载汉字小说
def read_time_machine_v2() -> list:
    """
    将时间机器数据集加载到文本行的列表中
    """
    with open("/mnt/325E98425E980131/ai_data/NLP_data/贾平凹-山本.txt",
              'r', encoding='utf8') as f:
        lines = f.readlines()
        pass

    # 加载停用词表
    stopwords_file = '/mnt/325E98425E980131/ai_data/NLP_data/stopwords.txt'
    with open(stopwords_file, "r") as words:
        stopwords = [i.strip() for i in words]
        pass

    stopwords.extend(['n', '.', '（', '）', '-', '——', '(', ')', ' ', '，'])
    textList = jieba.lcut(str(lines))
    # q_cut_str = " ".join(textList)

    q_cut_list = [i for i in textList if i not in stopwords]  # 去除停用词
    return q_cut_list


def load_data():
    with open("/mnt/325E98425E980131/ai_data/NLP_data/贾平凹-山本.txt",
              'r', encoding='utf8') as f:
        lines = f.readlines()
        pass

    # 加载停用词表
    stopwords_file = '/mnt/325E98425E980131/ai_data/NLP_data/stopwords.txt'
    with open(stopwords_file, "r") as words:
        stopwords = [i.strip() for i in words]
        pass

    stopwords.extend(['n', '.', '（', '）', '-', '——', '(', ')', ' ', '，'])
    textList = jieba.lcut(str(lines))  # .split()

    q_cut_list = [i for i in textList if i not in stopwords]  # 去除停用词

    q_cut_list[: 20]
    pass


def tokenize(lines, token='word'):  # @save
    """
    将文本行拆分为单词或字符词元
    """
    if token == 'word':
        return [line for line in lines]

    elif token == 'char':
        return [list(line) for line in lines]

    else:
        print('错误：未知词元类型：' + token)
    pass


class Vocab:
    """
    文本词表
    """
    def __init__(self, tokens=None, min_freq=0, reserved_tokens=None):
        if tokens is None:
            tokens = []

        if reserved_tokens is None:
            reserved_tokens = []

        # 按出现频率排序
        counter = count_corpus(tokens)
        self._token_freqs = sorted(counter.items(),
                                   key=lambda x: x[1],
                                   reverse=True
                                   )
        # 未知词元的索引为0
        self.idx_to_token = ['<unk>'] + reserved_tokens
        self.token_to_idx = {token: idx
                             for idx, token in enumerate(self.idx_to_token)
                             }

        for token, freq in self._token_freqs:
            if freq < min_freq:
                break
            if token not in self.token_to_idx:
                self.idx_to_token.append(token)
                self.token_to_idx[token] = len(self.idx_to_token) - 1

    def __len__(self):
        return len(self.idx_to_token)

    def __getitem__(self, tokens):
        if not isinstance(tokens, (list, tuple)):
            return self.token_to_idx.get(tokens, self.unk)
        return [self.__getitem__(token) for token in tokens]

    def to_tokens(self, indices):
        if not isinstance(indices, (list, tuple)):
            return self.idx_to_token[indices]
        return [self.idx_to_token[index] for index in indices]

    @property
    def unk(self):  # 未知词元的索引为0
        return 0

    @property
    def token_freqs(self):
        return self._token_freqs


def count_corpus(tokens):
    """
    统计词元的频率
    """
    # 这里的tokens是1D列表或2D列表
    if len(tokens) == 0 or isinstance(tokens[0], list):
        # 将词元列表展平成一个列表
        tokens = [token for line in tokens for token in line]

    return collections.Counter(tokens)


def main():
    lines = read_time_machine_v2()
    tokens = tokenize(lines, token='char')

    for i in range(30):
        print(tokens[i])
        pass

    # 词元化/向量化/
    vocab = Vocab(tokens)

    print(list(vocab.token_to_idx.items())[:30])

    for i in [0, 10]:
        print('文本:', tokens[i])
        print('索引:', vocab[tokens[i]])
    pass


if __name__ == '__main__':
    main()
