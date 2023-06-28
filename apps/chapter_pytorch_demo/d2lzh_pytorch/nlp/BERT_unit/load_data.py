# coding:utf-8
# Author:mylady
# Datetime:2023/4/24 6:09
import torch
from torch import nn

import random
import sys
import os

sys.path.append("..")
sys.path.append("../..")
from d2lzh_pytorch.BERT_unit.BERT_model import get_tokens_and_segments
from d2lzh_pytorch.load_chinese_txt_data import tokenize
from d2lzh_pytorch.load_Vocab import Vocab
from d2lzh_pytorch.download_unit import download_extract


def _read_wiki(data_dir):
    file_name = os.path.join(data_dir, 'wiki.train.tokens')
    # file_name = r"/home/mylady/code/python/DL-pytorch/apps/chapter_pytorch_demo/data/wikitext-2/wiki.train.tokens"
    with open(file_name, 'r') as f:
        lines = f.readlines()
    # 大写字母转换为小写字母
    paragraphs = [line.strip().lower().split(' . ')
                  for line in lines if len(line.split(' . ')) >= 2]
    random.shuffle(paragraphs)
    return paragraphs


# 生成二分类任务的训练样本
def _get_next_sentence(sentence, next_sentence, paragraphs):
    """Defined in `sec_bert-dataset`

    定 义: 14.9 用于预训练BERT的数据集
    描 述: 生成二分类任务的训练样本
    """
    if random.random() < 0.5:
        is_next = True
    else:
        # `paragraphs` is a list of lists of lists
        next_sentence = random.choice(random.choice(paragraphs))
        is_next = False
    return sentence, next_sentence, is_next


def _get_nsp_data_from_paragraph(paragraph, paragraphs, vocab, max_len):
    nsp_data_from_paragraph = []
    for i in range(len(paragraph) - 1):
        tokens_a, tokens_b, is_next = _get_next_sentence(
            paragraph[i], paragraph[i + 1], paragraphs)
        # 考虑1个'<cls>'词元和2个'<sep>'词元
        if len(tokens_a) + len(tokens_b) + 3 > max_len:
            continue
        tokens, segments = get_tokens_and_segments(tokens_a, tokens_b)
        nsp_data_from_paragraph.append((tokens, segments, is_next))
    return nsp_data_from_paragraph


def _replace_mlm_tokens(tokens, candidate_pred_positions, num_mlm_preds, vocab):
    """Defined in `sec_bert-dataset`

    定 义: 14.9 用于预训练BERT的数据集
    描 述: 从输入 paragraph 生成用于下一句预测的训练样本
    构 造: 为了从BERT输入序列生成遮蔽语言模型的训练样本，我们定义了以下 _replace_mlm_tokens 函数。
           的BERT输入序列的词元索引的列表（特殊词元在遮蔽语言模型任务中不被预测），以及 num_mlm_preds
           指示预测的数量（选择15%要预测的随机词元）。在 subsec_mlm中定义遮蔽语言模型任务之后，
           在每个预测位置，输入可以由特殊的“掩码”词元或随机词元替换，或者保持不变。最后，该函数返回可能
           替换后的输入词元、发生预测的词元索引和这些预测的标签。
    """
    # 为遮蔽语言模型的输入创建新的词元副本，其中输入可能包含替换的“<mask>”或随机词元
    mlm_input_tokens = [token for token in tokens]
    pred_positions_and_labels = []
    # 打乱后用于在遮蔽语言模型任务中获取15%的随机词元进行预测
    random.shuffle(candidate_pred_positions)
    for mlm_pred_position in candidate_pred_positions:
        if len(pred_positions_and_labels) >= num_mlm_preds:
            break
        masked_token = None
        # 80% of the time: replace the word with the '<mask>' token
        if random.random() < 0.8:
            masked_token = '<mask>'
        else:
            # 10%的时间：保持词不变
            if random.random() < 0.5:
                masked_token = tokens[mlm_pred_position]
            # 10%的时间：用随机词替换该词
            else:
                masked_token = random.choice(vocab.idx_to_token)
        mlm_input_tokens[mlm_pred_position] = masked_token
        pred_positions_and_labels.append((mlm_pred_position, tokens[mlm_pred_position]))
    return mlm_input_tokens, pred_positions_and_labels


def _get_mlm_data_from_tokens(tokens, vocab):
    candidate_pred_positions = []
    # tokens是一个字符串列表
    for i, token in enumerate(tokens):
        # 在遮蔽语言模型任务中不会预测特殊词元
        if token in ['<cls>', '<sep>']:
            continue
        candidate_pred_positions.append(i)
    # 遮蔽语言模型任务中预测15%的随机词元
    num_mlm_preds = max(1, round(len(tokens) * 0.15))
    mlm_input_tokens, pred_positions_and_labels = _replace_mlm_tokens(tokens,
                                                                      candidate_pred_positions,
                                                                      num_mlm_preds,
                                                                      vocab)
    pred_positions_and_labels = sorted(pred_positions_and_labels, key=lambda x: x[0])
    pred_positions = [v[0] for v in pred_positions_and_labels]
    mlm_pred_labels = [v[1] for v in pred_positions_and_labels]
    return vocab[mlm_input_tokens], pred_positions, vocab[mlm_pred_labels]


def _pad_bert_inputs(examples, max_len, vocab):
    """Defined in `subsec_prepare_mlm_data`

    定 义: 14.9 用于预训练BERT的数据集
    描 述: 将文本转换为预训练数据集
    构 造: 定义辅助函数 _pad_bert_inputs 来将特殊的“<mask>”词元附加到输入。它的参数 examples 包含
           来自两个预训练任务的辅助函数 _get_nsp_data_from_paragraph 和 _get_mlm_data_from_tokens 的输出。

    """
    max_num_mlm_preds = round(max_len * 0.15)
    all_token_ids, all_segments, valid_lens, = [], [], []
    all_pred_positions, all_mlm_weights, all_mlm_labels = [], [], []
    nsp_labels = []

    for (token_ids, pred_positions, mlm_pred_label_ids, segments, is_next) in examples:
        all_token_ids.append(torch.tensor(token_ids + [vocab['<pad>']] * (max_len - len(token_ids)), dtype=torch.long))
        all_segments.append(torch.tensor(segments + [0] * (max_len - len(segments)), dtype=torch.long))
        # valid_lens不包括'<pad>'的计数
        valid_lens.append(torch.tensor(len(token_ids), dtype=torch.float32))
        all_pred_positions.append(
            torch.tensor(pred_positions + [0] * (max_num_mlm_preds - len(pred_positions)), dtype=torch.long))
        # 填充词元的预测将通过乘以0权重在损失中过滤掉
        all_mlm_weights.append(
            torch.tensor([1.0] * len(mlm_pred_label_ids) + [0.0] * (max_num_mlm_preds - len(pred_positions)),
                         dtype=torch.float32))
        all_mlm_labels.append(torch.tensor(mlm_pred_label_ids + [0] * (max_num_mlm_preds - len(mlm_pred_label_ids)),
                                           dtype=torch.long))
        nsp_labels.append(torch.tensor(is_next, dtype=torch.long))

    return (all_token_ids, all_segments, valid_lens, all_pred_positions,
            all_mlm_weights, all_mlm_labels, nsp_labels)


class _WikiTextDataset(torch.utils.data.Dataset):
    """Defined in `subsec_prepare_mlm_data`

    定 义: 14.9 用于预训练BERT的数据集
    描 述:
    构 造: _WikiTextDataset 类为用于预训练BERT的 WikiText-2 数据集。通过实现 __getitem__ 函数，
          我们可以任意访问 WikiText-2 语料库的一对句子生成的预训练样本（遮蔽语言模型和下一句预测）样本。
    """

    def __init__(self, paragraphs, max_len):
        # Input `paragraphs[i]` is a list of sentence strings representing a
        # paragraph; while output `paragraphs[i]` is a list of sentences
        # representing a paragraph, where each sentence is a list of tokens
        # 输入paragraphs[i]是代表段落的句子字符串列表；
        # 而输出paragraphs[i]是代表段落的句子列表，其中每个句子都是词元列表
        paragraphs = [tokenize(paragraph, token='word') for paragraph in paragraphs]
        sentences = [sentence for paragraph in paragraphs for sentence in paragraph]
        self.vocab = Vocab(sentences, min_freq=5, reserved_tokens=['<pad>', '<mask>', '<cls>', '<sep>'])
        # Get data for the next sentence prediction task
        examples = []
        for paragraph in paragraphs:
            examples.extend(_get_nsp_data_from_paragraph(paragraph,
                                                         paragraphs,
                                                         self.vocab,
                                                         max_len))
        # Get data for the masked language model task
        # 获取遮蔽语言模型任务的数据
        examples = [(_get_mlm_data_from_tokens(tokens, self.vocab)
                     + (segments, is_next))
                    for tokens, segments, is_next in examples]
        # Pad inputs
        # 填充输入
        (self.all_token_ids, self.all_segments, self.valid_lens,
         self.all_pred_positions, self.all_mlm_weights,
         self.all_mlm_labels, self.nsp_labels) = _pad_bert_inputs(examples, max_len, self.vocab)

    def __getitem__(self, idx):
        return (self.all_token_ids[idx], self.all_segments[idx],
                self.valid_lens[idx], self.all_pred_positions[idx],
                self.all_mlm_weights[idx], self.all_mlm_labels[idx],
                self.nsp_labels[idx])

    def __len__(self):
        return len(self.all_token_ids)


def load_data_wiki(batch_size, max_len):
    """加载WikiText-2数据集"""
    num_workers = 4
    data_dir = download_extract('wikitext-2', 'wikitext-2')
    paragraphs = _read_wiki(data_dir)
    train_set = _WikiTextDataset(paragraphs, max_len)
    train_iter = torch.utils.data.DataLoader(train_set,
                                             batch_size,
                                             shuffle=True,
                                             num_workers=num_workers)
    return train_iter, train_set.vocab


def main():
    batch_size = 512
    max_len = 64

    # 加载数据
    train_iter, vocab = load_data_wiki(batch_size, max_len)
    names = ['conters', 'contexts_negatives', 'masks', 'labels']

    for batch in train_iter:
        for name, data in zip(names, batch):
            print(name, ' = ', data.shape)
        break
    print(list(vocab.token_to_idx.items())[0:10])

    pass


if __name__ == '__main__':
    main()
