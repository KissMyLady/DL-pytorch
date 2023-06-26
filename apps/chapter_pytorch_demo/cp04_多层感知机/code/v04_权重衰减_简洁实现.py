import d2lzh_pytorch.torch as d2l
import torch
from torch import nn

import sys
sys.path.append("..")
print(torch.__version__)


def train_concise(wd, num_inputs, train_iter, test_iter):

    net = nn.Sequential(nn.Linear(num_inputs, 1))

    for param in net.parameters():
        param.data.normal_()

    loss = nn.MSELoss(reduction='none')
    num_epochs, lr = 100, 0.003

    # 偏置参数没有衰减
    trainer = torch.optim.SGD([{"params":net[0].weight,'weight_decay': wd},
                               {"params":net[0].bias}], 
                               lr=lr)
    
    animator = d2l.Animator(xlabel='epochs', 
                            ylabel='loss', 
                            yscale='log',
                            xlim=[5, num_epochs], legend=['train', 'test'])
    for epoch in range(num_epochs):
        for X, y in train_iter:
            trainer.zero_grad()
            l = loss(net(X), y)
            l.mean().backward()
            trainer.step()

        if (epoch + 1) % 5 == 0:
            animator.add(epoch + 1,
                        (d2l.evaluate_loss(net, train_iter, loss),
                        d2l.evaluate_loss(net, test_iter, loss)))

    print('w的L2范数：', net[0].weight.norm().item())


def main():
    n_train = 20
    n_test = 100
    num_inputs = 200
    batch_size = 5

    true_w, true_b = d2l.ones((num_inputs, 1)) * 0.01, 0.05

    train_data = d2l.synthetic_data(true_w, true_b, n_train)
    train_iter = d2l.load_array(train_data, batch_size)

    test_data = d2l.synthetic_data(true_w, true_b, n_test)
    test_iter = d2l.load_array(test_data, batch_size, is_train=False)

    #@tab pytorch
    wd = 1
    train_concise(wd,
                  num_inputs, 
                  train_iter, 
                  test_iter)
