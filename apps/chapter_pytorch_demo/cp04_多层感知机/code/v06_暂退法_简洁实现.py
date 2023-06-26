import d2lzh_pytorch.torch as d2l
import math
import torch
from torch import nn
# from d2l import torch as d2l

import sys
sys.path.append("..")
print(torch.__version__)


def main():
    # 数据加载
    batch_size = 256
    rootPath = r"/mnt/g1t/ai_data/Datasets_on_HHD/FashionMNIST"
    train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size, 
                                                        root=rootPath)
    
    # 模型
    dropout1, dropout2 = 0.2, 0.5
    net = nn.Sequential(nn.Flatten(),
                        nn.Linear(784, 256),
                        nn.ReLU(),
                        # 在第一个全连接层之后添加一个dropout层
                        nn.Dropout(dropout1),
                        nn.Linear(256, 256),
                        nn.ReLU(),
                        # 在第二个全连接层之后添加一个dropout层
                        nn.Dropout(dropout2),
                        nn.Linear(256, 10))

    num_epochs = 10
    lr = 0.5
    # 损失函数
    loss = nn.CrossEntropyLoss(reduction='none')
    # 优化器
    trainer = torch.optim.SGD(net.parameters(), lr=lr)

    # 开始训练
    d2l.train_ch3(net, train_iter, test_iter, 
                  loss, 
                  num_epochs, 
                  trainer)


if __name__ == "__main__":
    main()
    pass
