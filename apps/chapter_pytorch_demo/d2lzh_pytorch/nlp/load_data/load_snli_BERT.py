import sys
sys.path.append("..")
sys.path.append("../..")

import torch
from d2lzh_pytorch import myUtils

from d2lzh_pytorch.nlp.load_data.load_time_machine import Vocab
from d2lzh_pytorch.nlp.load_data.load_snli import read_snli

import multiprocessing
import json
import os


def tokenize(lines, token='word'):
    """Split text lines into word or character tokens.
    Defined in :numref:`sec_text_preprocessing`"""
    if token == 'word':
        return [line.split() for line in lines]
    elif token == 'char':
        return [list(line) for line in lines]
    else:
        print('ERROR: unknown token type: ' + token)


def get_tokens_and_segments(tokens_a, tokens_b=None):
    """Get tokens of the BERT input sequence and their segment IDs.
       获取BERT输入序列的词元及其段IDs。
    Defined in :numref:`sec_bert`

    定 义: 14.8 来自Transformers的双向编码器表示（BERT）
    简 介: 获取输入序列的词元及其片段索引
    作 用: 将一个句子或两个句子作为输入, 然后返回BERT输入序列的标记及其相应的片段索引

    """
    tokens = ['<cls>'] + tokens_a + ['<sep>']
    # 0 and 1 are marking segment A and B, respectively
    # 返回[0, 0, .., 1, 1, ..] 分别标记片段A和B
    segments = [0] * (len(tokens_a) + 2)
    if tokens_b is not None:
        tokens += tokens_b + ['<sep>']
        segments += [1] * (len(tokens_b) + 1)
    return tokens, segments


class SNLIBERTDataset(torch.utils.data.Dataset):
    
    def __init__(self, dataset, max_len, vocab=None):
        all_premise_hypothesis_tokens = [
            [p_tokens, h_tokens] for p_tokens, h_tokens in zip(
            *[tokenize([s.lower() for s in sentences])
              for sentences in dataset[:2]])]

        self.labels = torch.tensor(dataset[2])
        self.vocab = vocab
        self.max_len = max_len
        
        (self.all_token_ids, 
         self.all_segments,
         self.valid_lens) = self._preprocess(all_premise_hypothesis_tokens)
        
        print('read ' + str(len(self.all_token_ids)) + ' examples')

    def _preprocess(self, all_premise_hypothesis_tokens):
        pool = multiprocessing.Pool(4)  # 使用4个进程
        out = pool.map(self._mp_worker, all_premise_hypothesis_tokens)
        all_token_ids = [token_ids for token_ids, segments, valid_len in out]
        all_segments = [segments for token_ids, segments, valid_len in out]
        valid_lens = [valid_len for token_ids, segments, valid_len in out]
        
        return (torch.tensor(all_token_ids, dtype=torch.long),
                torch.tensor(all_segments, dtype=torch.long),
                torch.tensor(valid_lens))

    def _mp_worker(self, premise_hypothesis_tokens):
        p_tokens, h_tokens = premise_hypothesis_tokens
        self._truncate_pair_of_tokens(p_tokens, h_tokens)
        tokens, segments = get_tokens_and_segments(p_tokens, h_tokens)
        token_ids = self.vocab[tokens] + [self.vocab['<pad>']] * (self.max_len - len(tokens))
        segments = segments + [0] * (self.max_len - len(segments))
        valid_len = len(tokens)
        return token_ids, segments, valid_len

    def _truncate_pair_of_tokens(self, p_tokens, h_tokens):
        # 为BERT输入中的'<CLS>'、'<SEP>'和'<SEP>'词元保留位置
        while len(p_tokens) + len(h_tokens) > self.max_len - 3:
            if len(p_tokens) > len(h_tokens):
                p_tokens.pop()
            else:
                h_tokens.pop()

    def __getitem__(self, idx):
        return (self.all_token_ids[idx], 
                self.all_segments[idx],
                self.valid_lens[idx]), self.labels[idx]

    def __len__(self):
        return len(self.all_token_ids)


def load_snli_bert(vocab):
    data_dir = "/mnt/g1t/ai_data/Datasets_on_HHD/d2l_data/data/bert.base.torch"

    # 定义空词表以加载预定义词表
    # vocab = Vocab()
    # vocabPath = os.path.join(data_dir, 'vocab.json')
    # vocab.idx_to_token = json.load(open(vocabPath))
    # vocab.token_to_idx = {token: idx for idx, token in enumerate(vocab.idx_to_token)}


    # 如果出现显存不足错误，请减少“batch_size”。在原始的BERT模型中，max_len=512
    batch_size = 512
    max_len = 128
    num_workers = myUtils.get_dataloader_workers()

    # data_dir = d2l.download_extract('SNLI')
    data_dir = "/mnt/g1t/ai_data/Datasets_on_HHD/d2l_data/data/snli_1.0"

    train_set = SNLIBERTDataset(read_snli(data_dir, True), max_len, vocab)
    test_set = SNLIBERTDataset(read_snli(data_dir, False), max_len, vocab)

    # 加载数据
    train_iter = torch.utils.data.DataLoader(train_set, 
                                            batch_size, 
                                            shuffle=True,
                                            num_workers=num_workers)

    test_iter = torch.utils.data.DataLoader(test_set, 
                                            batch_size,
                                            num_workers=num_workers)
    # len_vocab = len(vocab)
    return train_iter, test_iter #, len_vocab


def test_1():
    pass


def main():
    pass


if __name__ == "__main__":
    main()
    pass
