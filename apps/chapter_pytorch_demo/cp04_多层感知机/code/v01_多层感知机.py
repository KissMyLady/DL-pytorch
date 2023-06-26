import torch
from torch import nn
# from d2l import torch as d2l

import sys
sys.path.append("..")
import d2lzh_pytorch.torch as d2l

print(torch.__version__)


# 激活函数
def relu(X):
    a = torch.zeros_like(X)
    return torch.max(X, a)


loss = nn.CrossEntropyLoss(reduction='none')

def main():
    batch_size = 256
    rootPath = r"/mnt/g1t/ai_data/Datasets_on_HHD/FashionMNIST"
    train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size, root=rootPath)

    # 加载模型参数
    num_inputs  = 784
    num_hiddens = 256
    num_outputs = 10

    W1 = nn.Parameter(torch.randn(num_inputs, num_hiddens, requires_grad=True) * 0.01)
    b1 = nn.Parameter(torch.zeros(num_hiddens, requires_grad=True))

    W2 = nn.Parameter(torch.randn(num_hiddens, num_outputs, requires_grad=True) * 0.01)
    b2 = nn.Parameter(torch.zeros(num_outputs, requires_grad=True))
    params = [W1, b1, W2, b2]


    def net(X):
        X = X.reshape((-1, num_inputs))
        H = relu(X@W1 + b1)  # 这里“@”代表矩阵乘法
        return (H@W2 + b2)
    
    num_epochs = 10
    lr = 0.1

    # 优化算法
    updater = torch.optim.SGD(params, lr=lr)

    
    # 开始训练
    d2l.train_ch3(net, 
                  train_iter, 
                  test_iter,
                  loss, 
                  num_epochs,
                  updater)
    pass


if __name__ == "__main__":
    main()
