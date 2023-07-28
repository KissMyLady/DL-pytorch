# coding:utf-8
# Author:mylady
import torch
from torch import nn


# 注意力汇聚：Nadaraya-Watson 核回归
class NWKernelRegression(nn.Module):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.w = nn.Parameter(torch.rand((1,), requires_grad=True))

    def forward(self, queries, keys, values):
        """
        @query: train_data
        """
        # queries 和 attention_weights 的形状为(查询个数，“键－值”对个数)
        queries = queries.repeat_interleave(
            keys.shape[1]
        ).reshape((-1, keys.shape[1]))

        # 计算
        y_1 = -((queries - keys) * self.w)
        y_2 = y_1 ** 2 / 2
        attention_weights = nn.functional.softmax(y_2, dim=1)

        # values的形状为(查询个数，“键－值”对个数)
        y = torch.bmm(attention_weights.unsqueeze(1),
                      values.unsqueeze(-1)
                      ).reshape(-1)
        return y


def get_attention():
    net = NWKernelRegression()
    return net


def main():
    pass


if __name__ == "__main__":
    main()
