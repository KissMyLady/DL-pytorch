import sys
sys.path.append(".")
sys.path.append("../..")

import torch
from torch import nn
import math
import collections

from d2lzh_pytorch import myUtils
from d2lzh_pytorch import myPolt
from d2lzh_pytorch.rnn_train_chinese import grad_clipping


def sequence_mask(X, valid_len, value=0):
    """
    在序列中屏蔽不相关的项
    """
    maxlen = X.size(1)
    mask = torch.arange((maxlen),
                        dtype=torch.float32,
                        device=X.device
                        )[None, :] < valid_len[:, None]
    X[~mask] = value
    return X


class MaskedSoftmaxCELoss(nn.CrossEntropyLoss):
    """
    带遮蔽的softmax交叉熵损失函数
    """

    # pred的形状：(batch_size,num_steps,vocab_size)
    # label的形状：(batch_size,num_steps)
    # valid_len的形状：(batch_size,)
    def forward(self, pred, label, valid_len):
        weights = torch.ones_like(label)
        # 封装模块, 在序列中屏蔽不相关的项
        weights = sequence_mask(weights, valid_len)
        self.reduction = 'none'
        unweighted_loss = super(MaskedSoftmaxCELoss, self).forward(pred.permute(0, 2, 1), label)
        weighted_loss = (unweighted_loss * weights).mean(dim=1)
        return weighted_loss


# 编码器-解码器的训练
def train_seq2seq(net, data_iter, lr, num_epochs, tgt_vocab, device):
    """训练序列到序列模型"""

    def xavier_init_weights(m):
        if type(m) == nn.Linear:
            nn.init.xavier_uniform_(m.weight)
        if type(m) == nn.GRU:
            for param in m._flat_weights_names:
                if "weight" in param:
                    nn.init.xavier_uniform_(m._parameters[param])

    net.apply(xavier_init_weights)
    net.to(device)

    # 优化器
    optimizer = torch.optim.Adam(net.parameters(), lr=lr)

    # 损失函数
    loss = MaskedSoftmaxCELoss()
    net.train()

    animator = myPolt.Animator(xlabel='epoch',
                               ylabel='loss', 
                               xlim=[10, num_epochs]
                               )

    # 训练轮次
    for epoch in range(num_epochs):
        timer = myUtils.Timer()

        # 训练损失总和，词元数量
        metric = myUtils.Accumulator(2)

        for batch in data_iter:
            optimizer.zero_grad()
            X, X_valid_len, Y, Y_valid_len = [x.to(device) for x in batch]
            bos = torch.tensor([tgt_vocab['<bos>']] * Y.shape[0],
                               device=device).reshape(-1, 1)

            dec_input = torch.cat([bos, Y[:, :-1]], 1)  # 强制教学
            Y_hat, _ = net(X, dec_input, X_valid_len)

            # 梯度下降
            l = loss(Y_hat, Y, Y_valid_len)
            l.sum().backward()  # 损失函数的标量进行“反向传播”
            grad_clipping(net, 1)  # 梯度裁剪
            num_tokens = Y_valid_len.sum()

            # 更新梯度
            optimizer.step()

            with torch.no_grad():
                metric.add(l.sum(), num_tokens)
            pass

        if (epoch + 1) % 10 == 0:
            animator.add(epoch + 1, (metric[0] / metric[1],))
            pass

    print(f'loss {metric[0] / metric[1]:.3f}, {metric[1] / timer.stop():.1f} '
          f'tokens/sec on {str(device)}')


# 训练
def test_1():
    # 训练数据导入
    from d2lzh_pytorch.nlp.load_data.load_nmt_data import load_data_nmt
    # 模型
    from d2lzh_pytorch.nlp.model.EncoderDecoder_model import get_EncoderDecoder_model

    # 加载训练数据
    train_iter, src_vocab, tgt_vocab = load_data_nmt(batch_size=2, num_steps=8)

    lr = 0.005
    num_epochs = 300
    device = myUtils.try_gpu()

    net = load_data_nmt()

    # 训练
    train_seq2seq(net,
                  train_iter,
                  lr,
                  num_epochs,
                  tgt_vocab,
                  device)
    pass

