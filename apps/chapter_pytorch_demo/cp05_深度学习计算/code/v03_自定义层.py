import torch
from torch import nn
from torch_package.nn import functional as F

print(torch.__version__)


def test_1():
    class CenteredLayer(nn.Module):

        def __init__(self):
            super().__init__()

        def forward(self, X):
            y1 = X - X.mean()
            return y1

    layer = CenteredLayer()
    X = torch.FloatTensor([1, 2, 3, 4, 5])
    print(X)
    print('计算结果: ', layer(X))
    pass


# 复杂度
def test_2():

    class CenteredLayer(nn.Module):

        def __init__(self):
            super().__init__()

        def forward(self, X):
            y1 = X - X.mean()
            return y1

    net = nn.Sequential(
        nn.Linear(8, 128),
        CenteredLayer()
    )

    # 这里使用自定义, 初始化初始化网络
    def init_normal(m):
        if type(m) == nn.Linear:
            nn.init.normal_(m.weight, mean=0, std=1)
            nn.init.zeros_(m.bias)

    # 初始化参数
    net.apply(init_normal)
    X = torch.rand(4, 8)
    Y = net(X)

    print(Y.shape, Y.mean())
    pass


# 带参数的层
def test_3():
    class MyLinear(nn.Module):
        def __init__(self, in_units, units):
            super().__init__()
            self.weight = nn.Parameter(torch.randn(in_units, units))
            self.bias = nn.Parameter(torch.randn(units,))

        def forward(self, X):
            linear = torch.matmul(X, self.weight.data) + self.bias.data
            y_hat = F.relu(linear)
            return y_hat

    linear = MyLinear(5, 3)
    print(linear.weight)

    print(linear.state_dict())

    X3 = torch.rand(2, 5)
    Y3 = linear(X3)
    print(Y3)
    pass


# 还可以使用自定义层
def test_4():
    class MyLinear(nn.Module):
        def __init__(self, in_units, units):
            super().__init__()
            self.weight = nn.Parameter(torch.randn(in_units, units))
            self.bias = nn.Parameter(torch.randn(units,))

        def forward(self, X):
            linear = torch.matmul(X, self.weight.data) + self.bias.data
            y_hat = F.relu(linear)
            return y_hat

    net = nn.Sequential(
        MyLinear(64, 8), 
        MyLinear(8, 1)
    )

    X4 = torch.rand(size=(10, 64))

    # 计算
    y_hat = net(X4)
    y_hat
    pass
