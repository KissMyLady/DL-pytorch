# coding:utf-8
# Author:mylady
# 2023/7/28 14:42
import sys

sys.path.append("..")
sys.path.append("../..")
sys.path.append("../../..")
sys.path.append("../../../..")

from apps.chapter_pytorch_demo.d2lzh_pytorch.myUtils import try_gpu
from apps.chapter_pytorch_demo.d2lzh_pytorch.nlp.train.train_seq2seq import train_seq2seq
from apps.chapter_pytorch_demo.d2lzh_pytorch.nlp.load_data.load_nmt_data import load_data_nmt

from nlp.attention.model.att_net_v6_Transformer import get_net
from myUtils import save_net


def train_transformer():
    # 加载训练数据
    train_iter, src_vocab, tgt_vocab = load_data_nmt(batch_size=2, num_steps=8)

    # 加载Transformer模型
    net = get_net()

    lr = 0.005
    num_epochs = 300
    device = try_gpu()

    # 训练
    train_seq2seq(net, train_iter,
                  lr,
                  num_epochs,
                  tgt_vocab,
                  device)

    # 保存训练后的模型
    save_net(net, "attention_Transformer_net")
    pass


def user_model():
    import torch
    from apps.chapter_pytorch_demo.d2lzh_pytorch.nlp.predict.predict_seq2seq import predict_seq2seq
    from apps.chapter_pytorch_demo.d2lzh_pytorch.nlp.predict.predict_seq2seq import bleu

    # 加载训练数据
    train_iter, src_vocab, tgt_vocab = load_data_nmt(batch_size=2, num_steps=8)

    # 预测数据
    engs = ['go .', "i lost .", 'he\'s calm .', 'i\'m home .']
    fras = ['va !', 'j\'ai perdu .', 'il est calme .', 'je suis chez moi .']
    num_steps = 10

    net = torch.load("./attention_Transformer_net_2023-07-28_15-32-41.pt", map_location=torch.device('cpu'))

    device = try_gpu()

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
    # 训练transformer
    print("开始训练transformer >> ")

    train_transformer()

    print("训练结束 << ")
    pass


if __name__ == "__main__":
    user_model()
