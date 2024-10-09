import sys
sys.path.append("../..")
import d2lzh_pytorch.torch_package as d2l


import torch
from torch import nn
from d2lzh_pytorch.torch_package.utils import data
from d2lzh_pytorch.torch_package.nn import init
# from d2l import torch as d2l


# 升级版数据加载器
def load_array(data_arrays, batch_size, is_train=True):
    dataset = data.TensorDataset(*data_arrays)
    next_data = data.DataLoader(dataset, 
                                batch_size, 
                                shuffle=is_train)
    return next_data



def main():
    true_w = torch.tensor([2, -3.4])
    true_b = 4.2

    features, labels = d2l.synthetic_data(true_w, true_b, 1000)
    batch_size = 10

    # X, Y数据
    data_iter = load_array((features, labels), batch_size)

    # 模型
    net = nn.Sequential(
        nn.Linear(2, 1))

    # 损失
    loss = nn.MSELoss()

    # 优化算法
    trainer = torch.optim.SGD(net.parameters(), lr=0.03)

    # 初始化模型参数
    for layer in net:
       if isinstance(layer, nn.Linear):
           layer.weight.data.normal_(0, 0.01)
           layer.bias.data.normal_(0, 0.01)

    # 训练
    num_epochs = 3
    for epoch in range(num_epochs):
        for X, y in data_iter:
            l = loss(net(X), y)

            trainer.zero_grad()
            l.backward()
            trainer.step()

        y_heat = net(features)
        l = loss(y_heat, labels)
        print(f'epoch {epoch + 1}, loss {l:f}')
    pass


if __name__ == "__main__":
    main()
