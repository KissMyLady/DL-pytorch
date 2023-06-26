import d2lzh_pytorch.torch as d2l
import torch
from torch import nn
# from d2l import torch as d2l
import sys
sys.path.append("..")

print(torch.__version__)


def main():
    # 超参数
    batch_size= 256
    lr = 0.1
    num_epochs = 10

    # 加载训练数据
    rootPath = r"/mnt/g1t/ai_data/Datasets_on_HHD/FashionMNIST"
    train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size, root=rootPath)

    # 模型构建
    net = nn.Sequential(nn.Flatten(),
                        nn.Linear(784, 256),
                        nn.ReLU(),
                        nn.Linear(256, 10)
                    )

    # 模型初始化
    def init_weights(m):
        if type(m) == nn.Linear:
            nn.init.normal_(m.weight, std=0.01)
    net.apply(init_weights)

    # 损失函数
    loss = nn.CrossEntropyLoss(reduction='none')
    
    # 优化器
    trainer = torch.optim.SGD(net.parameters(), lr=lr)
    
    # 训练
    d2l.train_ch3(net, 
                  train_iter, 
                  test_iter, 
                  loss, 
                  num_epochs, 
                  trainer)
    pass


if __name__ == "__main__":
    main()
