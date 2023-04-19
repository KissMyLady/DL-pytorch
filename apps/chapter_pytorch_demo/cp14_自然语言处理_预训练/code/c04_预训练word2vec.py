import math
import torch
from torch import nn
# from d2l import torch as d2l

import sys
sys.path.append("../..")
import d2lzh_pytorch.torch as d2l


# 跳元模型
def skip_gram(center, contexts_and_negatives, embed_v, embed_u):
    v = embed_v(center)
    u = embed_u(contexts_and_negatives)
    pred = torch.bmm(v, u.permute(0, 2, 1))
    return pred


# 带掩码的二元交叉熵损失
class SigmoidBCELoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, inputs, target, mask=None):
        out = nn.functional.binary_cross_entropy_with_logits(inputs,
                                                             target,
                                                             weight=mask,
                                                             reduction="none")
        return out.mean(dim=1)


# 训练
def train(net, data_iter, lr, num_epochs, device=d2l.try_gpu()):
    # 二元交叉熵损失
    loss = SigmoidBCELoss()

    def init_weights(m):
        if type(m) == nn.Embedding:
            nn.init.xavier_uniform_(m.weight)

    net.apply(init_weights)
    net = net.to(device)
    # 梯度
    optimizer = torch.optim.Adam(net.parameters(), lr=lr)
    # 可视化配置
    animator = d2l.Animator(xlabel='epoch',
                            ylabel='loss',
                            xlim=[1, num_epochs])

    # 规范化的损失之和，规范化的损失数
    metric = d2l.Accumulator(2)

    timer = d2l.Timer()
    for epoch in range(num_epochs):
        num_batches = len(data_iter)
        for i, batch in enumerate(data_iter):
            optimizer.zero_grad()
            center, context_negative, mask, label = [data.to(device) for data in batch]
            
            pred = skip_gram(center, context_negative, net[0], net[1])
            l = (loss(pred.reshape(label.shape).float(),
                      label.float(),
                      mask) / mask.sum(axis=1) * mask.shape[1])

            l.sum().backward()
            optimizer.step()
            metric.add(l.sum(), l.numel())
            if (i + 1) % (num_batches // 5) == 0 or i == num_batches - 1:
                animator.add(epoch + (i + 1) / num_batches,
                             (metric[0] / metric[1],))
        timer.stop()

    print('Time consuming: %s s' % timer.final_time())
    print('耗时区间: %s s' % timer.interval_consume())
    print(f'loss {metric[0] / metric[1]:.3f}, '
          f'{metric[1] / timer.stop():.1f} tokens/sec on {str(device)}')


# 执行
def run():
    batch_size = 512
    max_window_size = 5
    num_noise_words = 5

    # 数据加载
    data_iter, vocab = d2l.load_data_ptb(batch_size,
                                         max_window_size,
                                         num_noise_words
                                        )

    # 初始化模型参数
    embed_size = 100
    # 嵌入层
    embed = nn.Embedding(num_embeddings=len(vocab), embedding_dim=embed_size)
    net = nn.Sequential(embed,
                        embed)

    # 训练
    lr = 0.002
    num_epochs = 5

    # 开始训练
    train(net, data_iter,
          lr,
          num_epochs
          )


def main():
    run()


if __name__ == '__main__':
    main()