# coding:utf-8
# Author:mylady
# Datetime:2023/4/23 7:52
import collections


# 字数统计, 返回字典 ({'词': '频率'}) 形式数据
def count_corpus(tokens):
    """Count token frequencies.
    Defined in `sec_text_preprocessing`
    """
    # Here `tokens` is a 1D list or 2D list
    if len(tokens) == 0 or isinstance(tokens[0], list):
        # Flatten a list of token lists into a list of tokens
        tokens = [token for line in tokens for token in line]
    # python内置方法调用, 返回统计
    return collections.Counter(tokens)


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