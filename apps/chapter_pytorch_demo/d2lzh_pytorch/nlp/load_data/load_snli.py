import torch
from d2lzh_pytorch import myUtils

# 引入词表 将字符串映射到从0开始的数字索引中
from d2lzh_pytorch.nlp.load_data.load_time_machine import Vocab

import re
import os


def read_snli(data_dir, is_train):
    """
        Read the SNLI dataset into premises, hypotheses, and labels.
    """
    def extract_text(s):
        # Remove information that will not be used by us
        s = re.sub('\\(', '', s)
        s = re.sub('\\)', '', s)
        # Substitute two or more consecutive whitespace with space
        s = re.sub('\\s{2,}', ' ', s)
        return s.strip()

    label_set = {'entailment': 0, 'contradiction': 1, 'neutral': 2}
    file_name = os.path.join(data_dir, 
                            'snli_1.0_train.txt' if is_train else 'snli_1.0_test.txt')
    
    # 读取文本数据
    with open(file_name, 'r') as f:
        rows = [row.split('\t') for row in f.readlines()[1:]]
    
    # 转列表
    premises = [extract_text(row[1]) for row in rows if row[0] in label_set]
    hypotheses = [extract_text(row[2]) for row in rows if row[0] in label_set]
    labels = [label_set[row[0]] for row in rows if row[0] in label_set]
    return premises, hypotheses, labels


def tokenize(lines, token='word'):
    """
        Split text lines into word or character tokens.
    """
    if token == 'word':
        return [line.split() for line in lines]
    elif token == 'char':
        return [list(line) for line in lines]
    else:
        print('ERROR: unknown token type: ' + token)


def truncate_pad(line, num_steps, padding_token):
    """
        Truncate or pad sequences.
    """
    if len(line) > num_steps:
        return line[:num_steps]  # Truncate
    return line + [padding_token] * (num_steps - len(line))  # Pad


class SNLIDataset(torch.utils.data.Dataset):
    """
        A customized dataset to load the SNLI dataset.
    """
    def __init__(self, dataset, num_steps, vocab=None):
        self.num_steps = num_steps
        all_premise_tokens = tokenize(dataset[0])
        all_hypothesis_tokens = tokenize(dataset[1])
        if vocab is None:
            self.vocab = Vocab(all_premise_tokens + all_hypothesis_tokens,
                               min_freq=5, 
                               reserved_tokens=['<pad>'])
        else:
            self.vocab = vocab
        self.premises =   self._pad(all_premise_tokens)
        self.hypotheses = self._pad(all_hypothesis_tokens)
        self.labels = torch.tensor(dataset[2])
        print('read ' + str(len(self.premises)) + ' examples')

    def _pad(self, lines):
        return torch.tensor([truncate_pad(self.vocab[line], 
                                          self.num_steps, 
                                          self.vocab['<pad>']) for line in lines])

    def __getitem__(self, idx):
        return (self.premises[idx], self.hypotheses[idx]), self.labels[idx]

    def __len__(self):
        return len(self.premises)


def load_data_snli(batch_size, num_steps=50):
    """下载SNLI数据集并返回数据迭代器和词表"""
    num_workers = myUtils.get_dataloader_workers()
    # data_dir = myUtils.download_extract('SNLI')

    data_dir = "/mnt/g1t/ai_data/Datasets_on_HHD/d2l_data/data/snli_1.0"
    # 读取数据
    train_data = read_snli(data_dir, True)
    test_data  = read_snli(data_dir, False)

    # 清洗
    train_set  = SNLIDataset(train_data, num_steps)
    test_set   = SNLIDataset(test_data, num_steps, train_set.vocab)

    # 加载
    train_iter = torch.utils.data.DataLoader(train_set, batch_size,
                                             shuffle=True,
                                             num_workers=num_workers)
    test_iter = torch.utils.data.DataLoader(test_set, batch_size,
                                            shuffle=False,
                                            num_workers=num_workers)
    # 返回
    return train_iter, test_iter, train_set.vocab



def test_1():
    batch_size = 256
    num_steps = 50

    train_iter, test_iter, vocab = load_data_snli(batch_size, num_steps)
    
    for X, Y in train_iter:
        print(X[0].shape)
        print(X[1].shape)
        print(Y.shape)
        break
    pass


def main():
    pass


if __name__ == "__main__":
    main()
    pass