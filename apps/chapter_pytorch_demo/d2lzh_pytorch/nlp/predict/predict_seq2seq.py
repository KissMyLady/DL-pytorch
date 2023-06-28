import sys
sys.path.append(".")
sys.path.append("../..")

import torch
from torch import nn
import math
import collections


def truncate_pad(line, num_steps, padding_token):
    """
    定 义: 9.5 机器翻译与数据集
    功 能: 截断或填充文本序列
    """
    if len(line) > num_steps:
        return line[:num_steps]  # 截断 Truncate
    return line + [padding_token] * (num_steps - len(line))  # 填充Pad


def bleu(pred_seq, label_seq, k):
    """计算BLEU"""
    pred_tokens, label_tokens = pred_seq.split(' '), label_seq.split(' ')
    len_pred, len_label = len(pred_tokens), len(label_tokens)
    score = math.exp(min(0, 1 - len_label / len_pred))
    for n in range(1, k + 1):
        num_matches, label_subs = 0, collections.defaultdict(int)
        for i in range(len_label - n + 1):
            label_subs[' '.join(label_tokens[i: i + n])] += 1
        for i in range(len_pred - n + 1):
            if label_subs[' '.join(pred_tokens[i: i + n])] > 0:
                num_matches += 1
                label_subs[' '.join(pred_tokens[i: i + n])] -= 1
        score *= math.pow(num_matches / (len_pred - n + 1), math.pow(0.5, n))
    return score


def predict_seq2seq(net, src_sentence, src_vocab, tgt_vocab,
                    num_steps, device, save_attention_weights=False):
    """序列到序列模型的预测"""
    # 在预测时将net设置为评估模式
    net.eval()
    src_tokens = src_vocab[src_sentence.lower().split(' ')] + [src_vocab['<eos>']]
    enc_valid_len = torch.tensor([len(src_tokens)], device=device)

    # 截断或填充文本序列
    src_tokens = truncate_pad(src_tokens, num_steps, src_vocab['<pad>'])

    # 添加批量轴
    enc_X = torch.unsqueeze(torch.tensor(src_tokens,
                                         dtype=torch.long,
                                         device=device), dim=0)
    enc_outputs = net.encoder(enc_X, enc_valid_len)
    dec_state = net.decoder.init_state(enc_outputs, enc_valid_len)

    # 添加批量轴
    dec_X = torch.unsqueeze(torch.tensor([tgt_vocab['<bos>']],
                                         dtype=torch.long,
                                         device=device
                                         ), dim=0)

    output_seq, attention_weight_seq = [], []

    for _ in range(num_steps):
        Y, dec_state = net.decoder(dec_X, dec_state)
        # 我们使用具有预测最高可能性的词元，作为解码器在下一时间步的输入
        dec_X = Y.argmax(dim=2)
        pred = dec_X.squeeze(dim=0).type(torch.int32).item()
        # 保存注意力权重（稍后讨论）
        if save_attention_weights:
            attention_weight_seq.append(net.decoder.attention_weights)
        # 一旦序列结束词元被预测，输出序列的生成就完成了
        if pred == tgt_vocab['<eos>']:
            break
        output_seq.append(pred)
        pass

    return ' '.join(tgt_vocab.to_tokens(output_seq)), attention_weight_seq


def test_1():
    # 训练数据导入
    from d2lzh_pytorch.nlp.load_data.load_nmt_data import load_data_nmt
    
    # 加载训练数据
    train_iter, src_vocab, tgt_vocab = load_data_nmt(batch_size=2, num_steps=8)

    # 预测数据
    engs = ['go .', "i lost .", 'he\'s calm .', 'i\'m home .']
    fras = ['va !', 'j\'ai perdu .', 'il est calme .', 'je suis chez moi .']

    num_steps = 10

    net = net.to("cuda")
    device = myUtils.try_gpu()

    # 预测
    for eng, fra in zip(engs, fras):
        translation, attention_weight_seq = predict_seq2seq(net,
                                                            eng,
                                                            src_vocab,
                                                            tgt_vocab,
                                                            num_steps,
                                                            device)
        
        print(f'{eng} => {translation}, bleu {bleu(translation, fra, k=2):.3f}')
    pass

def main():
    pass


if __name__ == "__main__":
    main()
    pass
