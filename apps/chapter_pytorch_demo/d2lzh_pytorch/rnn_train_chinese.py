# coding:utf-8
# Author:mylady
# Datetime:2023/4/23 4:05
import torch
from torch import nn
import math


import sys
sys.path.append(".")

from d2lzh_pytorch import myUtils
from d2lzh_pytorch import myPolt


reshape = lambda x, *args, **kwargs: x.reshape(*args, **kwargs)
size = lambda x, *args, **kwargs: x.numel(*args, **kwargs)


def sgd(params, lr, batch_size):
    """Minibatch stochastic gradient descent.
    Defined in `sec_linear_scratch`

    """
    with torch.no_grad():
        for param in params:
            param -= lr * param.grad / batch_size
            param.grad.zero_()


def grad_clipping(net, theta):
    """Clip the gradient.
    Defined in `sec_rnn_scratch`

    """
    if isinstance(net, nn.Module):
        params = [p for p in net.parameters() if p.requires_grad]
    else:
        params = net.params
    norm = torch.sqrt(sum(torch.sum((p.grad ** 2)) for p in params))
    if norm > theta:
        for param in params:
            param.grad[:] *= theta / norm


def train_epoch_ch8(net, train_iter, loss, updater, device, use_random_iter):
    """Train a net within one epoch (defined in Chapter 8).
    Defined in `sec_rnn_scratch`
    """
    state, timer = None, myUtils.Timer()
    metric = myUtils.Accumulator(2)  # Sum of training loss, no. of tokens
    for X, Y in train_iter:
        if state is None or use_random_iter:
            # Initialize `state` when either it is the first iteration or
            # using random sampling
            state = net.begin_state(batch_size=X.shape[0], device=device)
        else:
            if isinstance(net, nn.Module) and not isinstance(state, tuple):
                # `state` is a tensor for `nn.GRU`
                state.detach_()
            else:
                # `state` is a tuple of tensors for `nn.LSTM` and
                # for our custom scratch implementation
                for s in state:
                    s.detach_()
        y = Y.T.reshape(-1)
        X, y = X.to(device), y.to(device)
        y_hat, state = net(X, state)
        l = loss(y_hat, y.long()).mean()
        if isinstance(updater, torch.optim.Optimizer):
            updater.zero_grad()
            l.backward()
            grad_clipping(net, 1)
            updater.step()
        else:
            l.backward()
            grad_clipping(net, 1)
            # Since the `mean` function has been invoked
            updater(batch_size=1)
        metric.add(l * size(y), size(y))
    return math.exp(metric[0] / metric[1]), metric[1] / timer.stop()


def predict_ch8(prefix, num_preds, net, vocab, device):
    """Generate new characters following the `prefix`.

    Defined in :numref:`sec_rnn_scratch`"""
    state = net.begin_state(batch_size=1, device=device)
    outputs = [vocab[prefix[0]]]
    get_input = lambda: reshape(torch.tensor([outputs[-1]], device=device), (1, 1))
    for y in prefix[1:]:  # Warm-up period
        _, state = net(get_input(), state)
        outputs.append(vocab[y])
    for _ in range(num_preds):  # Predict `num_preds` steps
        y, state = net(get_input(), state)
        outputs.append(int(y.argmax(dim=1).reshape(1)))
    return ''.join([vocab.idx_to_token[i] for i in outputs])


def train_ch8(net, train_iter, vocab, lr, num_epochs, device,
              use_random_iter=False):
    """Train a model (defined in Chapter 8).

    Defined in :numref:`sec_rnn_scratch`"""
    times = myUtils.Timer()
    loss = nn.CrossEntropyLoss()
    animator = myPolt.Animator(xlabel='epoch', ylabel='perplexity',
                               legend=['train'], xlim=[10, num_epochs])
    # Initialize
    if isinstance(net, nn.Module):
        updater = torch.optim.SGD(net.parameters(), lr)
    else:
        updater = lambda batch_size: sgd(net.params, lr, batch_size)

    predict = lambda prefix: predict_ch8(prefix, 50, net, vocab, device)
    # Train and predict
    for epoch in range(num_epochs):
        ppl, speed = train_epoch_ch8(net,
                                     train_iter,
                                     loss,
                                     updater,
                                     device,
                                     use_random_iter)
        if (epoch + 1) % 10 == 0:
            # print(predict('说'))
            animator.add(epoch + 1, [ppl])

    times.stop()
    print('Time consuming: %8.4f 秒' % times.sum())
    print(f'perplexity {ppl:.1f}, {speed:.1f} tokens/sec on {str(device)}')
    print(predict('井掌柜'))


def main():
    pass


if __name__ == '__main__':
    main()
