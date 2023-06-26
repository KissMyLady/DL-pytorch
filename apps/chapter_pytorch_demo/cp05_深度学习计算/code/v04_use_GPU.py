import torch
from torch import nn


def test_1():
    if torch.cuda.is_available():
        print("GPU是否可用: \t", torch.cuda.is_available())
        print("GPU数量: \t",    torch.cuda.device_count())
        print("GPU索引号: \t",   torch.cuda.current_device())
        print("GPU名称: \t",     torch.cuda.get_device_name())
    else:
        print("warn: 当前服务器GPU不可用")

    x = torch.tensor([1, 2, 3])

    print("查询张量所在的设备: ", x.device)
    pass


# 在GPU上创建张量
def test_2():
    def try_gpu(i=0):
        if torch.cuda.device_count() >= i + 1:
            return torch.device(f'cuda:{i}')
        return torch.device('cpu')

    X = torch.ones(2, 3, device=try_gpu())
    print(X)
    pass


# 计算
def test_3():
    def try_gpu(i=0):
        if torch.cuda.device_count() >= i + 1:
            return torch.device(f'cuda:{i}')
        return torch.device('cpu')

    X = torch.ones(2, 3)
    Z = X.cuda(0)

    print("打印张量X: ", X)
    print("打印计算结果Z: ", Z)

    Y = torch.rand(2, 3, device=try_gpu(0))
    Y + Z
    pass


# 神经网络与GPU
def test_4():
    def try_gpu(i=0):
        if torch.cuda.device_count() >= i + 1:
            return torch.device(f'cuda:{i}')
        return torch.device('cpu')

    net = nn.Sequential(
        nn.Linear(3, 1)
    )

    # 将模型专业到GPU
    net = net.to(device=try_gpu())

    # 在GPU上创建张量
    X = torch.ones(2, 3, device=try_gpu(0))

    print(net(X))
    print(net[0].weight.data.device)
    pass


def main():
    test_4()
    pass


if __name__ == "__main__":
    main()
    pass
