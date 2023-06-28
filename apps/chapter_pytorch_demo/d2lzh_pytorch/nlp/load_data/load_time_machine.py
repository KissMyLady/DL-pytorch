# coding:utf-8
# Author:mylady
# Datetime:2023/4/23 3:32
import collections
import re
import os
import random
import torch


# 字数统计, 返回字典 ({'词': '频率'}) 形式数据
def count_corpus(tokens):
    """Count token frequencies.

    Defined in :numref:`sec_text_preprocessing`"""
    # Here `tokens` is a 1D list or 2D list
    if len(tokens) == 0 or isinstance(tokens[0], list):
        # Flatten a list of token lists into a list of tokens
        tokens = [token for line in tokens for token in line]

    # python内置方法调用, 返回统计
    return collections.Counter(tokens)


"""
词元的类型是字符串，而模型需要的输入是数字，因此这种类型不方便模型使用。 现在，让我们[构建一个字典，通常也叫做词表（vocabulary），
 用来将字符串类型的词元映射到从0开始的数字索引中]。 我们先将训练集中的所有文档合并在一起，对它们的唯一词元进行统计， 得到的统计结
 果称之为语料（corpus）。 然后根据每个唯一词元的出现频率，为其分配一个数字索引。 很少出现的词元通常被移除，这可以降低复杂性。
 另外，语料库中不存在或已删除的任何词元都将映射到一个特定的未知词元“<unk>”。 我们可以选择增加一个列表，用于保存那些被保留的词元，
 例如：填充词元（“<pad>”）； 序列开始词元（“<bos>”）； 序列结束词元（“<eos>”）。
"""


class Vocab:
    """Vocabulary for text.
    名 称: 词汇表
    定 义: 8.2节 文本预处理封装此类  sec_text_preprocessing
    作 用: 将字符串映射到从0开始的数字索引中.
    调 用:
        vocab = Vocab(tokens)
        list(vocab.token_to_idx.items())[0:10]
    """

    def __init__(self, tokens=None, min_freq=0, reserved_tokens=None):
        if tokens is None:
            tokens = []
        if reserved_tokens is None:
            reserved_tokens = []
        # 按出现频率排序
        counter = count_corpus(tokens)  # 字数统计, 返回字典 ({}, {}) 形式数据
        self._token_freqs = sorted(counter.items(), key=lambda x: x[1], reverse=True)  # 按照频率排序
        # 未知 unknown 词元在0首位
        self.idx_to_token = ['<unk>'] + reserved_tokens
        self.token_to_idx = {token: idx for idx, token in enumerate(self.idx_to_token)}  # 初始化

        # 遍历字典
        for token, freq in self._token_freqs:
            # 很少出现的词被移除
            if freq < min_freq:
                break
            # 如果当前词还没遍历过
            if token not in self.token_to_idx:
                self.idx_to_token.append(token)  # 添加到 idx_to_token ['<unk>', 'he', 'is', 'pirate']
                self.token_to_idx[token] = len(self.idx_to_token) - 1  # 添加到 token_to_idx['1025', '999', '68', '12']

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
    def token_freqs(self):  # 返回频率字典
        return self._token_freqs


def read_time_machine():
    """Load the time machine dataset into a list of text lines.
    Defined in `sec_text_preprocessing`

    定 义: 8.2 文本预处理
    功 能: 将时间机器数据集加载到文本行的列表中

    """
    with open("time_machine.txt", 'r') as f:
        lines = f.readlines()
    res = [re.sub('[^A-Za-z]+', ' ', line).strip().lower() for line in lines]
    return res


#
def read_txt(txtPath=None,
             stopwords_file=None):
    """
    定 义: 8.2 文本预处理
    描 述: 自定义汉字文本加载, 将时间机器数据集加载到文本行的列表中
    功 能: 加载汉字小说
    """
    if txtPath is None or stopwords_file is None:
        assert Exception('加载错误. 读取文本与停词表路径不存在')

    # 加载文本数据
    with open(txtPath, 'r', encoding='utf8') as f:
        lines = f.readlines()

    # 加载停用词表
    with open(stopwords_file, "r", encoding='utf8') as words:
        stopwords = [i.strip() for i in words]

    import jieba
    stopwords.extend(['n', '.', '（', '）', '-', '——', '(', ')', ' ', '，'])
    textList = jieba.lcut(str(lines))
    # q_cut_str = " ".join(textList)

    q_cut_list = [i for i in textList if i not in stopwords]  # 去除停用词
    return q_cut_list


def tokenize(lines, token='word'):
    """Split text lines into word or character tokens.
    Defined in `sec_text_preprocessing`

    标 题: 词元化
    定 义: 8.2 文本预处理
    功 能: 将文本行拆分为单词或字符元

    """
    if token == 'word':
        return [line.split() for line in lines]
    elif token == 'char':
        return [list(line) for line in lines]
    else:
        print('ERROR: unknown token type: ' + token)


def load_corpus_time_machine(max_tokens=-1, txtPath="", stopwords_file=""):
    """Return token indices and the vocabulary of the time machine dataset.
    Defined in `sec_text_preprocessing`

    定 义: 8.2 文本预处理
    功能: 返回时光机器数据集的词元索引列表和词表

    """
    # 文本预处理
    lines = read_txt(txtPath=txtPath, stopwords_file=stopwords_file)

    # 词元化
    tokens = tokenize(lines, 'char')

    # 词汇表
    vocab = Vocab(tokens)

    # Since each text line in the time machine dataset is not necessarily a
    # sentence or a paragraph, flatten all the text lines into a single list
    corpus = [vocab[token] for line in tokens for token in line]
    if max_tokens > 0:
        corpus = corpus[:max_tokens]
    return corpus, vocab


def seq_data_iter_random(corpus, batch_size, num_steps):
    """Generate a minibatch of subsequences using random sampling.
    Defined in `sec_language_model`

    标 题: 随机采样
    定 义: 8.3 语言模型和数据集
    功 能: 使用随机抽样生成一个小批量子序列

    """
    # 从随机偏移量开始对序列进行分区, 随机范围包括 `num_steps - 1`
    # sequence
    corpus = corpus[random.randint(0, num_steps - 1):]
    # 减去1，是因为我们需要考虑标签
    num_subseqs = (len(corpus) - 1) // num_steps
    # 长度为num_steps的子序列的起始索引
    initial_indices = list(range(0, num_subseqs * num_steps, num_steps))
    # 在随机抽样的迭代过程中，
    # 来自两个相邻的、随机的、小批量中的子序列不一定在原始序列上相邻
    random.shuffle(initial_indices)

    def data(pos):
        # 返回从pos位置开始的长度为num_steps的序列
        return corpus[pos: pos + num_steps]

    num_batches = num_subseqs // batch_size
    for i in range(0, batch_size * num_batches, batch_size):
        # 在这里，initial_indices包含子序列的随机起始索引  subsequences
        initial_indices_per_batch = initial_indices[i: i + batch_size]
        X = [data(j) for j in initial_indices_per_batch]
        Y = [data(j + 1) for j in initial_indices_per_batch]
        yield torch.tensor(X), torch.tensor(Y)


def seq_data_iter_sequential(corpus, batch_size, num_steps):
    """Generate a minibatch of subsequences using sequential partitioning.
    Defined in `sec_language_model`

    标 题: 顺序分区
    定 义: 8.3 语言模型和数据集
    功 能: 使用顺序分区生成一个小批量子序列

    """
    # 从随机偏移量开始划分序列
    offset = random.randint(0, num_steps)
    num_tokens = ((len(corpus) - offset - 1) // batch_size) * batch_size
    Xs = torch.tensor(corpus[offset: offset + num_tokens])
    Ys = torch.tensor(corpus[offset + 1: offset + 1 + num_tokens])
    Xs, Ys = Xs.reshape(batch_size, -1), Ys.reshape(batch_size, -1)
    num_batches = Xs.shape[1] // num_steps
    for i in range(0, num_steps * num_batches, num_steps):
        X = Xs[:, i: i + num_steps]
        Y = Ys[:, i: i + num_steps]
        yield X, Y


class SeqDataLoader:
    """An iterator to load sequence data.

    定 义: 8.3 语言模型和数据集
    功 能: 加载序列数据的迭代器

    """

    def __init__(self, batch_size, num_steps, use_random_iter, max_tokens, txtPath, stopwords_file):
        """Defined in `sec_language_model`"""
        if use_random_iter:
            # 随机采样
            self.data_iter_fn = seq_data_iter_random
        else:
            # 默认使用顺序采样
            self.data_iter_fn = seq_data_iter_sequential
        
        # 返回时光机器数据集的词元索引列表和词表
        self.corpus, self.vocab = load_corpus_time_machine(max_tokens, txtPath, stopwords_file)
        self.batch_size, self.num_steps = batch_size, num_steps

    def __iter__(self):
        return self.data_iter_fn(self.corpus, self.batch_size, self.num_steps)


def load_data_time_machine(batch_size, num_steps, txtPath="", stopwords_file="",
                           use_random_iter=False, max_tokens=10000):
    """
    返回时光机器数据集的迭代器和词表
    使 用:
        batch_size = 32
        num_steps = 35
        txtPath = "贾平凹-山本.txt"
        stopwords_file = "stopwords.txt"
        train_iter, vocab = load_data_time_machine(batch_size, num_steps, txtPath, stopwords_file)
    """
    # 封装代码块: 加载序列数据的迭代器
    data_iter = SeqDataLoader(batch_size,
                              num_steps,
                              use_random_iter,
                              max_tokens,
                              txtPath,
                              stopwords_file,
                              )
    print('load vocabulary success. len(vocab): %s' % (len(data_iter.vocab)))
    return data_iter, data_iter.vocab



def test_1():
    """
    定 义: 8.3 语言模型和数据集
    功 能: 传入文本路径,停词表, 返回训练数据和词元
    """
    batch_size = 32
    num_steps = 35

    # BASE: /mnt/g1t/ai_data/Datasets_on_HHD/NLP/text_data
    BASE_PATH = "/mnt/g1t/ai_data/Datasets_on_HHD/NLP/text_data"

    # 加载
    txtPath = os.path.join(BASE_PATH, "贾平凹-山本.txt")
    stopwords_file = os.path.join(BASE_PATH, "stopwords.txt")

    # 返回封装的数据
    train_iter, vocab = load_data_time_machine(batch_size, num_steps, txtPath, stopwords_file)

    print(len(vocab))
    pass


def main():
    pass


if __name__ == '__main__':
    main()
