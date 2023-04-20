# coding:utf-8
# Author:mylady
# Datetime:2023/4/20 0:32
import torch
from torch import nn
# from d2l import torch as d2l

import sys

sys.path.append("../..")
import d2lzh_pytorch.torch as d2l


def _get_batch_loss_bert(net, loss, vocab_size, tokens_X,
                         segments_X, valid_lens_x,
                         pred_positions_X, mlm_weights_X,
                         mlm_Y, nsp_y):
    # 前向传播
    _, mlm_Y_hat, nsp_Y_hat = net(tokens_X,
                                  segments_X,
                                  valid_lens_x.reshape(-1),
                                  pred_positions_X)

    # 计算遮蔽语言模型损失
    mlm_l = loss(mlm_Y_hat.reshape(-1, vocab_size),
                 mlm_Y.reshape(-1)
                 ) * mlm_weights_X.reshape(-1, 1)

    mlm_l = mlm_l.sum() / (mlm_weights_X.sum() + 1e-8)

    # 计算下一句子预测任务的损失
    nsp_l = loss(nsp_Y_hat, nsp_y)
    res_loss = mlm_l + nsp_l

    return mlm_l, nsp_l, res_loss


def train_bert(train_iter, net, loss, vocab_size, devices, num_steps):
    net = nn.DataParallel(net, device_ids=devices).to(devices[0])
    trainer = torch.optim.Adam(net.parameters(), lr=0.01)
    step = 0
    timer = d2l.Timer()

    animator = d2l.Animator(xlabel='step',
                            ylabel='loss',
                            xlim=[1, num_steps],
                            legend=['mlm', 'nsp'])
    # 遮蔽语言模型损失的和，下一句预测任务损失的和，句子对的数量，计数
    metric = d2l.Accumulator(4)
    num_steps_reached = False

    while step < num_steps and not num_steps_reached:

        for tokens_X, segments_X, valid_lens_x, \
            pred_positions_X, mlm_weights_X, mlm_Y, nsp_y in train_iter:
            # 复制到GPU
            tokens_X = tokens_X.to(devices[0])
            segments_X = segments_X.to(devices[0])
            valid_lens_x = valid_lens_x.to(devices[0])
            pred_positions_X = pred_positions_X.to(devices[0])
            mlm_weights_X = mlm_weights_X.to(devices[0])
            mlm_Y, nsp_y = mlm_Y.to(devices[0]), nsp_y.to(devices[0])

            trainer.zero_grad()
            timer.start()

            mlm_l, nsp_l, l = _get_batch_loss_bert(net,
                                                   loss,
                                                   vocab_size,
                                                   tokens_X,
                                                   segments_X,
                                                   valid_lens_x,
                                                   pred_positions_X,
                                                   mlm_weights_X,
                                                   mlm_Y,
                                                   nsp_y
                                                   )
            l.backward()
            trainer.step()
            metric.add(mlm_l, nsp_l, tokens_X.shape[0], 1)
            timer.stop()
            animator.add(step + 1,
                         (metric[0] / metric[3], metric[1] / metric[3]))
            step += 1
            timer.stop()

            if step == num_steps:
                num_steps_reached = True
                break

    # print('耗时: ', timer)
    print('耗时: ', timer.sum())

    loss_1 = metric[0] / metric[3]
    loss_2 = metric[1] / metric[3]
    p_sec = metric[2] / timer.sum()

    print(f'MLM loss {loss_1:.3f}', f'NSP loss {loss_2:.3f}')
    print(f'{p_sec:.1f} sentence pairs/sec on ', f'{str(devices)}')


# 加载数据
batch_size = 512
max_len = 64
train_iter, vocab = d2l.load_data_wiki(batch_size, max_len)

# 网络模型
net = d2l.BERTModel(len(vocab),
                    num_hiddens=128,
                    norm_shape=[128],
                    ffn_num_input=128,
                    ffn_num_hiddens=256,
                    num_heads=2,
                    num_layers=2,
                    dropout=0.2,
                    key_size=128,
                    query_size=128,
                    value_size=128,
                    hid_in_features=128,
                    mlm_in_features=128,
                    nsp_in_features=128
                    )
devices = d2l.try_all_gpus()
loss = nn.CrossEntropyLoss()


def run_train():
    num_steps = 50
    # 训练
    train_bert(train_iter,
               net,
               loss,
               len(vocab),
               devices,
               num_steps
               )
    pass


# 用BERT表示文本
def get_bert_encoding(net, tokens_a, tokens_b=None):
    tokens, segments = d2l.get_tokens_and_segments(tokens_a, tokens_b)

    token_ids = torch.tensor(vocab[tokens], device=devices[0]).unsqueeze(0)
    segments = torch.tensor(segments, device=devices[0]).unsqueeze(0)
    valid_len = torch.tensor(len(tokens), device=devices[0]).unsqueeze(0)

    encoded_X, _, _ = net(token_ids, segments, valid_len)
    return encoded_X


def test_1():
    tokens_a = ['a', 'crane', 'is', 'flying']

    encoded_text = get_bert_encoding(net, tokens_a)

    # 词元：'<cls>','a','crane','is','flying','<sep>'
    encoded_text_cls = encoded_text[:, 0, :]
    encoded_text_crane = encoded_text[:, 2, :]
    pass


def main():
    run_train()
    # test_1()
    pass


if __name__ == '__main__':
    main()
