# coding:utf-8
# Author:mylady
# Datetime:2023/4/24 18:20
import os
import torch

import sys

sys.path.append(".")
sys.path.append("..")
sys.path.append("../..")

from d2lzh_pytorch.download_unit import download_extract
from d2lzh_pytorch.download_unit import DATA_HUB, DATA_URL


# @save
DATA_HUB['glove.6b.50d'] = (DATA_URL + 'glove.6B.50d.zip',
                            '0b8703943ccdb6eb788e6f091b8946e82231bc4d')

# @save
DATA_HUB['glove.6b.100d'] = (DATA_URL + 'glove.6B.100d.zip',
                             'cd43bfb07e44e6f27cbcc7bc9ae3d80284fdaf5a')

# @save
DATA_HUB['glove.42b.300d'] = (DATA_URL + 'glove.42B.300d.zip',
                              'b5116e234e9eb9076672cfeabf5469f3eec904fa')

# @save
DATA_HUB['wiki.en'] = (DATA_URL + 'wiki.en.zip',
                       'c1816da3821ae9f43899be655002f6c723e91b88')


# @save
class TokenEmbedding:
    """GloVe嵌入"""

    def __init__(self, embedding_name):
        self.idx_to_token, self.idx_to_vec = self._load_embedding(embedding_name)
        self.unknown_idx = 0
        self.token_to_idx = {token: idx for idx, token in
                             enumerate(self.idx_to_token)}

    def _load_embedding(self, embedding_name):
        idx_to_token, idx_to_vec = ['<unk>'], []
        data_dir = download_extract(embedding_name)
        # GloVe网站：https://nlp.stanford.edu/projects/glove/
        # fastText网站：https://fasttext.cc/
        with open(os.path.join(data_dir, 'vec.txt'), 'r') as f:
            for line in f:
                elems = line.rstrip().split(' ')
                token, elems = elems[0], [float(elem) for elem in elems[1:]]
                # 跳过标题信息，例如fastText中的首行
                if len(elems) > 1:
                    idx_to_token.append(token)
                    idx_to_vec.append(elems)
        idx_to_vec = [[0] * len(idx_to_vec[0])] + idx_to_vec
        return idx_to_token, torch.tensor(idx_to_vec)

    def __getitem__(self, tokens):
        indices = [self.token_to_idx.get(token, self.unknown_idx)
                   for token in tokens]
        vecs = self.idx_to_vec[torch.tensor(indices)]
        return vecs

    def __len__(self):
        return len(self.idx_to_token)


def knn(W, x, k):
    # 增加1e-9以获得数值稳定性
    cos = torch.mv(W, x.reshape(-1, )) / (
            torch.sqrt(torch.sum(W * W, axis=1) + 1e-9) *
            torch.sqrt((x * x).sum()))
    _, topk = torch.topk(cos, k=k)
    return topk, [cos[int(i)] for i in topk]


def get_similar_tokens(query_token, k, embed):
    topk, cos = knn(embed.idx_to_vec, embed[[query_token]], k + 1)
    for i, c in zip(topk[1:], cos[1:]):  # 排除输入词
        print(f'{embed.idx_to_token[int(i)]}：cosine相似度={float(c):.3f}')
        pass


def get_analogy(token_a, token_b, token_c, embed):
    vecs = embed[[token_a, token_b, token_c]]
    x = vecs[1] - vecs[0] + vecs[2]
    topk, cos = knn(embed.idx_to_vec, x, 1)
    return embed.idx_to_token[int(topk[0])]  # 删除未知词


def main():
    # 使用预先训练好的词向量
    glove_load = TokenEmbedding('glove.6b.50d')

    # glove_wikien =  TokenEmbedding('wiki.en')

    print("计算词的相似度")
    for word in ["chip", 'baby', 'girl', 'beautiful']:
        get_similar_tokens('chip', 10, glove_load)

    print("计算词类比")
    res_1 = get_analogy('man', 'woman', 'son', glove_load)
    res_2 = get_analogy('beijing', 'china', 'tokyo', glove_load)
    res_3 = get_analogy('bad', 'worst', 'big', glove_load)
    res_4 = get_analogy('do', 'did', 'go', glove_load)

    print("res_1", res_1)
    print("res_2", res_2)
    print("res_3", res_3)
    print("res_4", res_4)
    pass


if __name__ == '__main__':
    main()
