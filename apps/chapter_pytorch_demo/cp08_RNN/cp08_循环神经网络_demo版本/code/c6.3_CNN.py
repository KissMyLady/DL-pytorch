# coding:utf-8
# Author:mylady
# Datetime:2023/3/20 14:28
import torch_package
import random
import zipfile

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

read_url_list_path = "K:\code_big\jaychou_lyrics.txt"
read_url_list_path_new = "K:\code_big\jaychou_lyrics_new.txt"


# 清洗文本
def clearText():
    f = open(read_url_list_path, 'r', encoding='utf-8')
    f_new = open(read_url_list_path_new, 'w', encoding='utf-8')
    for line in f:
        if '\r' in line:
            line = line.replace('\r', ' ')
            # 用空格替代回车键
        if '\n' in line:
            line = line.replace('\n', ' ')
        f_new.write(line)

    f.close()
    f_new.close()
    pass


# 本函数已保存在d2lzh_pytorch包中方便以后使用
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
        yield torch.tensor(X, dtype=torch.float32, device=device),torch.tensor(Y, dtype=torch.float32, device=device)


def test_1_open():
    with zipfile.ZipFile('K:\code_big\jaychou_lyrics.txt.zip') as zin:
        with zin.open('jaychou_lyrics.txt') as f:
            corpus_chars = f.read().decode('utf-8')

    print(corpus_chars[:40])
    pass


def test_2():
    with open(read_url_list_path_new, encoding='utf8') as f:
        corpus_chars = f.read()

    print(corpus_chars[:40])
    corpus_chars = corpus_chars.replace('\n', ' ').replace('\r', ' ')
    corpus_chars = corpus_chars[0:10000]

    # 建立索引
    idx_to_char = list(set(corpus_chars))
    char_to_idx = dict([(char, i) for i, char in enumerate(idx_to_char)])
    vocab_size = len(char_to_idx)
    vocab_size  # 1027

    # 并打印前20个字符及其对应的索引
    corpus_indices = [char_to_idx[char] for char in corpus_chars]
    sample = corpus_indices[:20]

    print('chars:', ''.join([idx_to_char[idx] for idx in sample]))
    print('indices:', sample)

    my_seq = list(range(30))

    for X, Y in data_iter_random(my_seq, batch_size=2, num_steps=6):
        print('X: ', X, '\nY:', Y, '\n')
    pass


def main():
    test_1_open()
    # clearText()
    pass


if __name__ == '__main__':
    main()
