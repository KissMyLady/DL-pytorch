# coding:utf-8
# Author:mylady
# Datetime:2023/4/23 4:35
import torch
from torch import nn
from torch.nn import functional as F


class RNNModel(nn.Module):
    """The RNN model.
    Defined in `sec_rnn-concise`

    定 义: 8.6 RNN 简洁实现
    功 能: 模块的封装. rnn_layer 为传入的 nn.RNN(len_vocab, num_hiddens) 这种RNN网络
    """

    def __init__(self, rnn_layer, vocab_size, **kwargs):
        super(RNNModel, self).__init__(**kwargs)
        self.rnn = rnn_layer
        self.vocab_size = vocab_size
        self.num_hiddens = self.rnn.hidden_size
        # If the RNN is bidirectional (to be introduced later),
        # `num_directions` should be 2, else it should be 1.
        if not self.rnn.bidirectional:
            self.num_directions = 1
            self.linear = nn.Linear(self.num_hiddens, self.vocab_size)
        else:
            self.num_directions = 2
            self.linear = nn.Linear(self.num_hiddens * 2, self.vocab_size)

    def forward(self, inputs, state):
        X = F.one_hot(inputs.T.long(), self.vocab_size)
        X = X.to(torch.float32)
        Y, state = self.rnn(X, state)
        # The fully connected layer will first change the shape of `Y` to
        # (`num_steps` * `batch_size`, `num_hiddens`). Its output shape is
        # (`num_steps` * `batch_size`, `vocab_size`).
        output = self.linear(Y.reshape((-1, Y.shape[-1])))
        return output, state

    def begin_state(self, device, batch_size=1):
        if not isinstance(self.rnn, nn.LSTM):
            # `nn.GRU` takes a tensor as hidden state
            res_3 = torch.zeros((self.num_directions * self.rnn.num_layers,batch_size, self.num_hiddens), device=device)
            return res_3
        else:
            # `nn.LSTM` takes a tuple of hidden states
            res_1 = torch.zeros((self.num_directions * self.rnn.num_layers, batch_size, self.num_hiddens), device=device)
            res_2 = torch.zeros((self.num_directions * self.rnn.num_layers, batch_size, self.num_hiddens), device=device)
            return res_1, res_2


def main():
    pass


if __name__ == '__main__':
    main()
