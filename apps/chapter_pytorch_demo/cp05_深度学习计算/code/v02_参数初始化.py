import torch
from torch import nn


# 内置初始化
def test_1():

    net = nn.Sequential(
        nn.Linear(4, 8),
        nn.ReLU(),
        nn.Linear(8, 1)
    )

    def init_normal(m):
        if type(m) == nn.Linear:
            nn.init.normal_(m.weight, mean=0, std=0.01)
            nn.init.zeros_(m.bias)

    net.apply(init_normal)

    net[0].weight.data[0], net[0].bias.data[0]
    pass


# 常量初始化
def test_2():

    net = nn.Sequential(
        nn.Linear(4, 8),
        nn.ReLU(),
        nn.Linear(8, 1)
    )

    def init_constant(m):
        if type(m) == nn.Linear:
            nn.init.constant_(m.weight, 1)
            nn.init.zeros_(m.bias)

    net.apply(init_constant)
    net[0].weight.data[0], net[0].bias.data[0]
    pass


# 指定地方初始化
def test_3():
    net = nn.Sequential(
        nn.Linear(4, 8),
        nn.ReLU(),
        nn.Linear(8, 1)
    )

    def init_xavier(m):
        if type(m) == nn.Linear:
            nn.init.xavier_uniform_(m.weight)

    def init_42(m):
        if type(m) == nn.Linear:
            nn.init.constant_(m.weight, 42)

    # 指定地方初始化
    net[0].apply(init_xavier)
    net[2].apply(init_42)
    pass


# 自定义初始化
def test_4():
    net = nn.Sequential(
        nn.Linear(4, 8),
        nn.ReLU(),
        nn.Linear(8, 1)
    )

    def my_init(m):
        if type(m) == nn.Linear:
            print("Init", *[(name, param.shape)
                  for name, param in m.named_parameters()][0])
            nn.init.uniform_(m.weight, -10, 10)
            m.weight.data *= m.weight.data.abs() >= 5

    net.apply(my_init)
    pass


# 参数绑定
def test_5():

    # 我们需要给共享层一个名称，以便可以引用它的参数
    shared = nn.Linear(8, 8)

    net = nn.Sequential(
        nn.Linear(4, 8), nn.ReLU(),
        shared,          nn.ReLU(),
        shared,          nn.ReLU(),
        nn.Linear(8, 1)
    )

    net(X)

    # 确保它们实际上是同一个对象，而不只是有相同的值
    print(net[2].weight.data[0] == net[4].weight.data[0])

    """
    这个例子表明第三个和第五个神经网络层的参数是绑定的。 
    它们不仅值相等，而且由相同的张量表示。 因此，如果我们改变其中一个参数，另一个参数也会改变。 

    这里有一个问题：
        当参数绑定时，梯度会发生什么情况？
    答案是 
        由于模型参数包含梯度，因此在反向传播期间第二个隐藏层 （即第三个神经网络层）
        和第三个隐藏层（即第五个神经网络层）的梯度会加在一起。
    """
    pass
