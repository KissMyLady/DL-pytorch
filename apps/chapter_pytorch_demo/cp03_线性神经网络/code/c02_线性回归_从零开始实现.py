# %matplotlib inline

import sys
sys.path.append("../..")
import d2lzh_pytorch.torch_package as d2l


import torch
import random


def synthetic_data(w, b, num_examples):  #@save
    """生成y=Xw+b+噪声"""
    X = torch.normal(0, 1, (num_examples, len(w)))
    y = torch.matmul(X, w) + b
    y += torch.normal(0, 0.01, y.shape)
    return X, y.reshape((-1, 1))


def data_iter(batch_size, features, labels):
    num_examples = len(features)
    indices = list(range(num_examples))
    # 这些样本是随机读取的，没有特定的顺序
    random.shuffle(indices)
    
    for i in range(0, num_examples, batch_size):
        batch_indices = torch.tensor(indices[i: min(i + batch_size, num_examples)])
        yield features[batch_indices], labels[batch_indices]


def linreg(X, w, b):
    """线性回归模型"""
    return torch.matmul(X, w) + b


def squared_loss(y_hat, y):
    """均方损失"""
    return (y_hat - y.reshape(y_hat.shape)) ** 2 / 2


def sgd(params, lr, batch_size):  #@save
    """小批量随机梯度下降"""
    with torch.no_grad():
        for param in params:
            param -= lr * param.grad / batch_size
            param.grad.zero_()

# 画图封装
def draw_plot(features, labels, w, b):
    # 画出拟合的参数
    xx_list_1 = []
    yy_list_1 = []
    
    xx_list_2 = []
    yy_list_2 = []
    
    for xx in range(-5, 5):
        # 计算轴线0
        y_head_1 = xx * w[0].item() + b.item()
        # print("输入x: %s, 输出y: %s" % (xx, y_head))
        xx_list_1.append(xx)
        yy_list_1.append(y_head_1)
        
        # 计算轴线1
        y_head_2 = xx * w[1].item() + b.item()
        # print("输入x: %s, 输出y: %s" % (xx, y_head))
        xx_list_2.append(xx)
        yy_list_2.append(y_head_2)


    d2l.set_figsize()
    d2l.plt.scatter(features[:, 1].detach().numpy(),
                    labels.detach().numpy(), 
                    1)
    d2l.plt.scatter(features[:, 0].detach().numpy(),
                    labels.detach().numpy(), 
                    1)
    d2l.plt.plot(
        xx_list_1, yy_list_1
    )
    d2l.plt.plot(
        xx_list_2, yy_list_2
    )
    
    d2l.plt.show()
    pass


def main():
    batch_size = 10
    true_w = torch.tensor([2, -3.4])
    true_b = 4.2

    # 生成数据
    features, labels = synthetic_data(true_w, true_b, 1000)
    w = torch.normal(0, 0.01, size=(2,1), requires_grad=True)
    b = torch.zeros(1, requires_grad=True)

    # 训练
    lr = 0.003
    num_epochs = 3
    net = linreg
    loss = squared_loss

    # 训练前，更新随机生成的 w 和 b绘制图像
    print("训练前, 参数w: ", w)
    print("训练前, 参数b: ", b)
    draw_plot(features, labels, w, b)

    # 开始训练
    for epoch in range(num_epochs):
        for X, y in data_iter(batch_size, features, labels):
            l = loss(net(X, w, b), y)  # X和y的小批量损失
            # 因为l形状是(batch_size,1)，而不是一个标量。l中的所有元素被加到一起，
            # 并以此计算关于[w,b]的梯度
            l.sum().backward()
            sgd([w, b], lr, batch_size)  # 使用参数的梯度更新参数

        print("训练中 .., 参数w: ", w)
        print("训练中 .., 参数b: ", b)
        draw_plot(features, labels, w, b)

        with torch.no_grad():
            train_l = loss(net(features, w, b), labels)
            print(f'epoch {epoch + 1}, loss {float(train_l.mean()):f}')
    
    # 训练完毕, 打印训练后的参数 w, b
    print(f'w的估计误差: {true_w - w.reshape(true_w.shape)}')
    print(f'b的估计误差: {true_b - b}')


if __name__ == "__main__":
    main()



