# coding:utf-8
# Author:mylady
# Datetime:2023/4/24 4:56
import math
import os
import random
import torch
# from d2l import torch as d2l

import sys

sys.path.append(".")
sys.path.append("../..")
from d2lzh_pytorch.load_Vocab import Vocab
from d2lzh_pytorch.load_chinese_txt_data import count_corpus


# d2l.DATA_HUB['ptb'] = (d2l.DATA_URL + 'ptb.zip', '319d85e578af0cdc590547f26231e4e31cdf1e42')


def read_ptb():
    """将PTB数据集加载到文本行的列表中"""
    filePath = r"Z:\python\DL-pytorch\apps\chapter_pytorch_demo\data\ptb\ptb.train.txt"
    with open(filePath, 'r', encoding='utf-8') as f:
        raw_text = f.read()

    return [line.split() for line in raw_text.split('\n')]


def subsample(sentences, vocab):
    """Subsample high-frequency words.
    Defined in `sec_word2vec_data`

    标 题: 下采样
    定 义: 14.3 用于预训练词嵌入的数据集
    描 述: 下采样高频词. 通过删除高频词来加速计算

    """
    # 排除未知词元'<unk>'
    sentences = [[token for token in line if vocab[token] != vocab.unk]
                 for line in sentences]
    # 字数统计, 返回字典
    counter = count_corpus(sentences)
    num_tokens = sum(counter.values())

    # 如果在下采样期间保留词元，则返回True
    def keep(token):
        return (random.uniform(0, 1) <
                math.sqrt(1e-4 / counter[token] * num_tokens))

    res_a1 = [[token for token in line if keep(token)] for line in sentences]
    return res_a1, counter


class RandomGenerator:
    """根据n个采样权重在{1,...,n}中随机抽取

    标 题: 负采样
    定 义: 14.3 用于预训练词嵌入的数据集
    功 能: 降低计算复杂度

    """

    def __init__(self, sampling_weights):
        # Exclude
        self.population = list(range(1, len(sampling_weights) + 1))
        self.sampling_weights = sampling_weights
        self.candidates = []
        self.i = 0

    def draw(self):
        if self.i == len(self.candidates):
            # 缓存k个随机采样结果
            self.candidates = random.choices(self.population,
                                             self.sampling_weights,
                                             k=10000)
            self.i = 0
        self.i += 1
        return self.candidates[self.i - 1]


def get_centers_and_contexts(corpus, max_window_size):
    """Return center words and context words in skip-gram.
    Defined in `sec_word2vec_data`

    标 题: 中心词和上下文词的提取
    定 义: 14.3 用于预训练词嵌入的数据集
    功 能: 返回跳元模型中的中心词和上下文词

    """
    centers, contexts = [], []
    for line in corpus:
        # 要形成“中心词-上下文词”对，每个句子至少需要有2个词
        if len(line) < 2:
            continue
        centers += line
        for i in range(len(line)):  # 上下文窗口中间i
            window_size = random.randint(1, max_window_size)
            indices = list(range(max(0, i - window_size),
                                 min(len(line), i + 1 + window_size)))
            # 从上下文词中排除中心词
            indices.remove(i)
            contexts.append([line[idx] for idx in indices])
    return centers, contexts


def get_negatives(all_contexts, vocab, counter, K):
    """
    定 义: 14.3 用于预训练词嵌入的数据集
    功 能: 返回负采样中的噪声词
    """
    # 索引为 1、2、...（索引0是词表中排除的未知标记）  采样概率计算
    sampling_weights = [counter[vocab.to_tokens(i)] ** 0.75 for i in range(1, len(vocab))]
    all_negatives = []

    generator = RandomGenerator(sampling_weights)

    # 遍历上下文
    for contexts in all_contexts:
        negatives = []
        # 5 倍
        while len(negatives) < len(contexts) * K:
            neg = generator.draw()
            # 噪声词不能是上下文词
            if neg not in contexts:
                negatives.append(neg)
        all_negatives.append(negatives)
    return all_negatives


def batchify(data):
    """
    定 义: 14.3 用于预训练词嵌入的数据集
    功 能: 返回带有负采样的跳元模型的小批量样本
    """
    max_len = max(len(c) + len(n) for _, c, n in data)

    centers = []  # 中心词
    contexts_negatives = []  # 上下文词+噪音
    masks = []  # 掩码
    labels = []  # 将上下文与噪声词区分

    for center, context, negative in data:
        cur_len = len(context) + len(negative)
        centers += [center]
        contexts_negatives += [context + negative + [0] * (max_len - cur_len)]
        # 掩码: 在计算损失时排除填充
        masks += [[1] * cur_len + [0] * (max_len - cur_len)]
        labels += [[1] * len(context) + [0] * (max_len - len(context))]

    # 返回
    res_a1 = torch.tensor(centers).reshape((-1, 1))
    res_a2 = torch.tensor(contexts_negatives)
    res_a3 = torch.tensor(masks)
    res_a4 = torch.tensor(labels)
    return (res_a1, res_a2, res_a3, res_a4)


class PTBDataset(torch.utils.data.Dataset):

    def __init__(self, centers, contexts, negatives):
        assert len(centers) == len(contexts) == len(negatives)
        self.centers = centers
        self.contexts = contexts
        self.negatives = negatives

    def __getitem__(self, index):
        return (self.centers[index], self.contexts[index],
                self.negatives[index])

    def __len__(self):
        return len(self.centers)


# 下载PTB数据集，然后将其加载到内存中
def load_data_ptb(batch_size, max_window_size, num_noise_words):
    num_workers = 4  # 获取进程数 d2l.get_dataloader_workers()

    # 将PTB数据集加载到文本行的列表中
    sentences = read_ptb()

    # 词元化
    vocab = Vocab(sentences, min_freq=10)

    # 下采样
    subsampled, counter = subsample(sentences, vocab)

    # 词元映射的索引
    corpus = [vocab[line] for line in subsampled]

    # “中心词-上下文词对”
    all_centers, all_contexts = get_centers_and_contexts(corpus, max_window_size)

    # 负采样
    all_negatives = get_negatives(all_contexts, vocab, counter, num_noise_words)

    # 数据集
    dataset = PTBDataset(all_centers, all_contexts, all_negatives)

    # 数据载入
    data_iter = torch.utils.data.DataLoader(dataset,
                                            batch_size,
                                            shuffle=True,
                                            collate_fn=batchify,
                                            num_workers=num_workers)
    return data_iter, vocab


def main():
    batch_size = 512
    max_window_size = 5
    num_noise_words = 5

    # 包导入
    data_iter, vocab = load_data_ptb(batch_size,
                                     max_window_size,
                                     num_noise_words)

    # 遍历数据
    for i, batch in enumerate(data_iter):
        center, context_negative, mask, label = [data for data in batch]
        print('')
        if i >= 1:
            break
    pass


if __name__ == '__main__':
    main()
